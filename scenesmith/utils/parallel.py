"""Fault-tolerant parallel execution utilities.

Provides process isolation for parallel task execution, ensuring that one task
crashing does not affect others. This is critical for long-running batch jobs
where ProcessPoolExecutor's "broken pool" behavior is problematic.
"""

import logging
import multiprocessing
import os
import queue
import signal
import traceback

from multiprocessing.connection import wait
from typing import Any, Callable

console_logger = logging.getLogger(__name__)


def _get_isolated_process_context() -> multiprocessing.context.BaseContext:
    """Select a clean process context without re-importing the ACP entrypoint."""
    available_methods = multiprocessing.get_all_start_methods()
    requested_method = os.environ.get("SCENEEXPERT_MP_START_METHOD", "").strip()

    if requested_method:
        if requested_method not in available_methods:
            raise RuntimeError(
                f"Unsupported SCENEEXPERT_MP_START_METHOD={requested_method!r}; "
                f"available methods: {available_methods}"
            )
        method = requested_method
    elif "forkserver" in available_methods:
        method = "forkserver"
    else:
        method = "spawn"

    if method == "forkserver":
        # Python preloads __main__ into forkserver by default. In this project
        # that executes main.py and imports bpy, which fails with missing _bpy
        # during ACP child bootstrap. An empty preload list keeps the server
        # clean; each worker imports only the pickled target module it needs.
        multiprocessing.set_forkserver_preload([])

    console_logger.info(f"Using multiprocessing start method: {method}")
    return multiprocessing.get_context(method)


def _get_signal_name(exit_code: int) -> str:
    """Get human-readable signal name from exit code.

    Negative exit codes indicate the process was killed by a signal.
    For example, -11 means SIGSEGV (segmentation fault).

    Args:
        exit_code: Process exit code (negative for signals).

    Returns:
        Signal name if exit_code is negative, otherwise empty string.
    """
    if exit_code >= 0:
        return ""
    signal_num = -exit_code
    try:
        return f" ({signal.Signals(signal_num).name})"
    except (ValueError, AttributeError):
        return ""


def _reset_worker_logging() -> None:
    """Reset logging handlers at start of worker process.

    Prevents file descriptor inheritance issues with fork(). When forking,
    child processes can inherit file handlers from the parent, causing logs
    to be written to wrong files.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()


def _worker_wrapper(
    target: Callable,
    kwargs: dict,
    task_id: str,
    result_queue: multiprocessing.Queue,
    return_values: bool,
) -> None:
    """Wrapper that runs target function and reports result to queue."""
    _reset_worker_logging()
    console_logger.debug(f"Worker {task_id} starting: {target.__name__}")
    try:
        result = target(**kwargs)
        if return_values:
            result_queue.put((task_id, True, result))
        else:
            result_queue.put((task_id, True, None))
    except Exception as e:
        # Preserve full traceback for debugging, not just str(e).
        error_msg = f"{e}\n{traceback.format_exc()}"
        console_logger.error(f"Worker {task_id} failed: {error_msg}")
        result_queue.put((task_id, False, error_msg))


def run_parallel_isolated(
    tasks: list[tuple[str, Callable, dict]],
    max_workers: int,
    return_values: bool = False,
) -> dict[str, tuple[bool, Any]]:
    """Run tasks in isolated processes with fault tolerance.

    Spawns up to max_workers processes at a time. As each completes, spawns the
    next task. One process crashing does not affect others.

    Unlike ProcessPoolExecutor, this function:
    - Spawns a fresh process per task (clean state, no accumulated resources)
    - Continues running other tasks if one crashes
    - Uses efficient wait() instead of polling

    Args:
        tasks: List of (task_id, target_function, kwargs) tuples. The target
            function will be called with **kwargs.
        max_workers: Maximum number of concurrent processes.
        return_values: If True, capture and return values from target functions.
            If False, only track success/failure status.

    Returns:
        Dict mapping task_id to (success: bool, result_or_error).
        For successful tasks: result_or_error is the return value (if
        return_values=True) or None (if return_values=False).
        For failed tasks: result_or_error is the error message string.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    # Never use a plain fork from a process that has initialized CUDA, Drake,
    # OpenMP, SQLite, httpx, or Agents SDK tracing. Linux uses a clean
    # forkserver; platforms without it fall back to spawn.
    mp_context = _get_isolated_process_context()
    result_queue = mp_context.Queue()
    pending = list(tasks)
    active: dict[int, tuple[multiprocessing.Process, str]] = {}
    results: dict[str, tuple[bool, Any]] = {}

    while pending or active:
        # Spawn processes up to max_workers.
        while len(active) < max_workers and pending:
            task_id, target, kwargs = pending.pop(0)
            p = mp_context.Process(
                target=_worker_wrapper,
                args=(target, kwargs, task_id, result_queue, return_values),
            )
            p.start()
            active[p.pid] = (p, task_id)
            console_logger.info(f"Started {task_id} (pid={p.pid})")

        # Wait for any process to finish (efficient, no busy polling).
        if active:
            sentinels = [proc.sentinel for proc, _ in active.values()]
            wait(sentinels, timeout=1.0)

        # Drain before join() so a large return value cannot fill the pipe and
        # block the worker's queue feeder thread during process shutdown.
        while True:
            try:
                result_task_id, success, result_or_error = result_queue.get_nowait()
                results[result_task_id] = (success, result_or_error)
                status = "completed" if success else f"failed: {result_or_error}"
                console_logger.info(f"{result_task_id} {status}")
            except queue.Empty:
                break

        # Join finished processes before the second drain. This guarantees the
        # feeder thread has flushed small, late-arriving result messages.
        finished: list[tuple[multiprocessing.Process, str]] = []
        for pid, (proc, task_id) in list(active.items()):
            if not proc.is_alive():
                proc.join()
                del active[pid]
                finished.append((proc, task_id))

        # Drain again after process shutdown. This second pass removes the race
        # where a process exits between the first get_nowait() and join().
        while True:
            try:
                result_task_id, success, result_or_error = result_queue.get_nowait()
                results[result_task_id] = (success, result_or_error)
                status = "completed" if success else f"failed: {result_or_error}"
                console_logger.info(f"{result_task_id} {status}")
            except queue.Empty:
                break

        # Any finished process that did not report a result crashed before the
        # Python wrapper could catch an exception (for example SIGSEGV/OOM).
        for proc, task_id in finished:
            if task_id not in results:
                signal_name = _get_signal_name(proc.exitcode)
                results[task_id] = (
                    False,
                    f"Process crashed (exitcode={proc.exitcode}{signal_name})",
                )
                console_logger.error(
                    f"{task_id} crashed (exitcode={proc.exitcode}{signal_name})"
                )

    result_queue.close()
    result_queue.join_thread()

    return results
