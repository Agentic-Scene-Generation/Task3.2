"""Release a stale Vite listener only when it belongs to this Web project."""

from __future__ import annotations

import os
import signal
import time

from pathlib import Path

try:
    from tools.critic_probe_web import _listening_tcp_pids
except ModuleNotFoundError:  # Running this file directly from the tools directory.
    from critic_probe_web import _listening_tcp_pids


DEV_SERVER_PORT = 5175
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _command_runs_this_vite(command: list[str], cwd: Path, web_root: Path) -> bool:
    expected_entry = (web_root / "node_modules" / "vite" / "bin" / "vite.js").resolve()
    for argument in command[1:]:
        candidate = Path(argument)
        if candidate.name != "vite.js":
            continue
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (cwd / candidate).resolve()
        )
        if resolved == expected_entry:
            return True
    return False


def _process_runs_this_vite(pid: int, web_root: Path = WEB_ROOT) -> bool:
    try:
        command = [
            value.decode("utf-8", errors="replace")
            for value in (Path("/proc") / str(pid) / "cmdline")
            .read_bytes()
            .split(b"\0")
            if value
        ]
        cwd = Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return False
    return _command_runs_this_vite(command, cwd, web_root.resolve())


def release_previous_dev_server(port: int = DEV_SERVER_PORT) -> None:
    listener_pids = _listening_tcp_pids(port)
    previous_pids = {
        pid
        for pid in listener_pids
        if pid != os.getpid() and _process_runs_this_vite(pid)
    }
    if not previous_pids:
        if listener_pids:
            raise RuntimeError(
                f"Port {port} is occupied by another program; refusing to stop it."
            )
        return

    print(f"Stopping previous Web dev server: {sorted(previous_pids)}", flush=True)
    for pid in previous_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not (_listening_tcp_pids(port) & previous_pids):
            return
        time.sleep(0.05)
    raise RuntimeError(f"Previous Web dev server did not release port {port}.")


if __name__ == "__main__":
    release_previous_dev_server()
