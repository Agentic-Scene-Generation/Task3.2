import asyncio
import copy
import csv
import faulthandler
import json
import logging
import os
import shutil
import time
import uuid

from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from agents import custom_span, trace
from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalServer,
)
from scenesmith.agent_utils.geometry_generation_server import GeometryGenerationServer
from scenesmith.agent_utils.house import HouseLayout, HouseScene, RoomGeometry
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalServer
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalServer
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalServer
from scenesmith.agent_utils.physical_feasibility import (
    apply_physical_feasibility_postprocessing,
)
from scenesmith.agent_utils.room import AgentType, ObjectType, RoomScene
from scenesmith.agent_utils.sceneeval_exporter import (
    SceneEvalExportConfig,
    SceneEvalExporter,
)
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.scene_expert.config_utils import resolve_scene_expert_stage_budget
from scenesmith.scene_expert.exceptions import StageValidationError
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.parallel import run_parallel_isolated
from scenesmith.utils.print_utils import bold_green, yellow
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

# SceneExpert hook runner (imported lazily to avoid circular imports at module level)
# TYPE_CHECKING block keeps the type hint available without a hard import.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenesmith.scene_expert.hooks import SceneExpertHookRunner

console_logger = logging.getLogger(__name__)

# Pipeline stages in execution order (derived from AgentType enum).
PIPELINE_STAGES = [agent.value for agent in AgentType]

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
}

_SCENE_STATUS_FILENAME = "scene_status.json"
_SCENE_SUCCESS_MARKER = "_SUCCESS"


def _write_scene_status(
    output_dir: Path,
    scene_id: int,
    prompt: str,
    status: str,
    attempt: int,
    error: str | None = None,
) -> None:
    """Atomically persist the lifecycle state of one scene task."""
    scene_dir = output_dir / f"scene_{scene_id:03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    status_path = scene_dir / _SCENE_STATUS_FILENAME
    payload = {
        "scene_id": scene_id,
        "prompt": prompt,
        "status": status,
        "attempt": attempt,
        "pid": os.getpid(),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    if error:
        payload["error"] = error
    temporary_path = status_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(status_path)


def _archive_failed_scene_attempt(
    output_dir: Path,
    scene_id: int,
    attempt: int,
) -> Path | None:
    """Move a failed partial scene aside before a clean-process retry."""
    scene_dir = output_dir / f"scene_{scene_id:03d}"
    if not scene_dir.exists():
        return None

    archive_root = output_dir / "failed_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = archive_root / (
        f"scene_{scene_id:03d}_attempt_{attempt:02d}_{timestamp}"
    )
    shutil.move(str(scene_dir), str(archive_path))
    return archive_path


def _is_retryable_scene_failure(error: str) -> bool:
    """Return whether a fresh process can plausibly recover this failure."""
    normalized = error.lower()
    transient_markers = (
        "sigsegv",
        "sigabrt",
        "sigkill",
        "exitcode=-11",
        "exitcode=-6",
        "exitcode=-9",
        "exitcode=137",
        "apitimeouterror",
        "request timed out",
        "connection reset",
        "connection refused",
        "stage failed deterministic validation",
    )
    return any(marker in normalized for marker in transient_markers)


def _is_repairable_stage_validation(error: StageValidationError) -> bool:
    """Classify layout/content failures that should not terminate ACP early."""
    text = " ".join(error.reasons).lower()
    terminal_markers = (
        "fatal asset retrieval setup error",
        "invalid room geometry",
        "room geometry is unavailable",
        "unrecoverable environment",
    )
    return not any(marker in text for marker in terminal_markers)


def _run_sceneexpert_placement_stage(
    *,
    stage: str,
    agent: Any,
    scene: RoomScene,
    run_once: Callable[[], Any],
) -> int:
    """Run a placement stage with bounded critic retry and full regeneration."""
    baseline_state = copy.deepcopy(scene.to_state_dict())
    stage_prompt = str(scene.text_description or "")
    budget = getattr(scene, "scene_expert_stage_budget", {}) or {}
    max_regenerations = max(
        0, int(budget.get("max_stage_regenerations", 0) or 0)
    )
    regeneration_attempt = 0
    critic_retry_attempted = False

    while True:
        try:
            asyncio.run(run_once())
            return regeneration_attempt
        except StageValidationError as exc:
            if not _is_repairable_stage_validation(exc):
                raise
            critic_only = bool(exc.reasons) and all(
                "visual critic did not produce a trustworthy score" in reason.lower()
                for reason in exc.reasons
            )
            if critic_only and not critic_retry_attempted:
                critic_retry_attempted = True
                console_logger.warning(
                    "%s output is hard-valid but unscored; retrying only the compact "
                    "final critic before considering regeneration",
                    stage,
                )
                retry_critic = getattr(agent, "retry_final_critic_evaluation", None)
                if callable(retry_critic):
                    asyncio.run(retry_critic())
                    return regeneration_attempt

            if regeneration_attempt >= max_regenerations:
                console_logger.error(
                    "%s stage remained invalid after %d full regeneration(s): %s",
                    stage,
                    regeneration_attempt,
                    "; ".join(exc.reasons),
                )
                raise

            regeneration_attempt += 1
            console_logger.warning(
                "%s stage failed its completion contract; restoring the stage "
                "input checkpoint and requesting a new agent design (%d/%d): %s",
                stage,
                regeneration_attempt,
                max_regenerations,
                "; ".join(exc.reasons),
            )
            scene.restore_from_state_dict(copy.deepcopy(baseline_state))
            scene.text_description = (
                f"{stage_prompt}\n\n"
                "# Mandatory Stage Regeneration\n"
                "The previous attempt was rejected. Create a genuinely new, "
                "bounded stage layout and do not finish with zero stage-native "
                "objects. If a requested asset failed, choose a semantically "
                "equivalent HSSD substitute with realistic natural proportions. "
                "Resolve all of: "
                + "; ".join(exc.reasons)
            )
            prepare_regeneration = getattr(agent, "prepare_stage_regeneration", None)
            if callable(prepare_regeneration):
                asyncio.run(prepare_regeneration(list(exc.reasons)))


def _root_error_summary(error: str, max_chars: int = 700) -> str:
    """Return one actionable root-cause line without replaying nested tracebacks."""
    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    if not lines:
        return "Unknown scene failure"
    root_line = lines[-1]
    if root_line.startswith("^") or root_line.startswith("File "):
        root_line = lines[0]
    if len(root_line) > max_chars:
        root_line = root_line[: max_chars - 3] + "..."
    return root_line


def _write_batch_summary(
    *,
    output_dir: Path,
    experiment_run_id: str,
    prompts_with_ids: list[tuple[int, str]],
    results: dict[str, tuple[bool, str | None]],
) -> None:
    summary_path = output_dir / "batch_summary.json"
    existing_scenes: dict[str, dict] = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("experiment_run_id") == experiment_run_id:
                existing_scenes = {
                    str(item.get("scene_id")): item
                    for item in existing.get("scenes", [])
                }
        except (OSError, ValueError, TypeError):
            existing_scenes = {}

    prompt_map = {scene_id: prompt for scene_id, prompt in prompts_with_ids}
    for scene_id, prompt in prompts_with_ids:
        task_id = f"scene_{scene_id:03d}"
        success, error = results.get(task_id, (False, "Missing worker result"))
        existing_scenes[task_id] = {
            "scene_id": task_id,
            "prompt": prompt_map[scene_id],
            "status": "completed" if success else "failed",
            "root_error": "" if success else _root_error_summary(str(error)),
            "scene_status_path": str(output_dir / task_id / _SCENE_STATUS_FILENAME),
        }

    scenes = [existing_scenes[key] for key in sorted(existing_scenes)]
    payload = {
        "experiment_run_id": experiment_run_id,
        "updated_at": datetime.now().astimezone().isoformat(),
        "total_scenes": len(scenes),
        "completed_scenes": sum(
            1 for item in scenes if item["status"] == "completed"
        ),
        "failed_scenes": sum(1 for item in scenes if item["status"] == "failed"),
        "scenes": scenes,
    }
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(summary_path)


def _get_retrieval_gpu_device() -> str | None:
    """Get GPU device for retrieval servers.

    If multiple GPUs available (as seen by PyTorch), returns the last
    logical GPU index to avoid competing with Blender and geometry
    generation (which use lower-indexed GPUs).

    This respects CUDA_VISIBLE_DEVICES - PyTorch remaps physical GPUs
    to logical indices 0, 1, 2, ... so we use the last logical index.

    Returns:
        Device string like "cuda:7" or None if single GPU / detection fails.
    """
    try:
        # Import torch inside function to avoid CUDA initialization before
        # ProcessPoolExecutor forks workers (fork-after-CUDA causes corruption).
        import torch

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # Use the last logical GPU for retrieval servers.
            return f"cuda:{gpu_count - 1}"
    except ImportError:
        pass
    return None


def _get_config_bool(cfg: DictConfig, key: str, default: bool = False) -> bool:
    """Read a nested OmegaConf bool without assuming every node exists."""
    value = OmegaConf.select(cfg, key, default=default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class RenderGPUAllocator:
    """Round-robin GPU allocator for distributing Blender rendering.

    Assigns GPUs in round-robin order for BlenderServer instances. This enables
    parallel scene generation without GPU memory exhaustion by spreading the
    rendering load across multiple GPUs.

    Thread-safe for concurrent allocation from multiple workers.
    """

    def __init__(self) -> None:
        self._gpus = self._detect_gpus()
        self._counter = 0
        self._lock = Lock()
        console_logger.info(f"RenderGPUAllocator initialized with GPUs: {self._gpus}")

    def _detect_gpus(self) -> list[int]:
        """Detect available GPU indices, respecting CUDA_VISIBLE_DEVICES."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Parse comma-separated GPU indices from CUDA_VISIBLE_DEVICES.
            try:
                return [int(x.strip()) for x in cuda_visible.split(",")]
            except ValueError:
                console_logger.warning(
                    f"Failed to parse CUDA_VISIBLE_DEVICES='{cuda_visible}', "
                    "falling back to device file detection"
                )

        # Detect from /dev/nvidia* device files.
        gpus = []
        for i in range(16):
            if Path(f"/dev/nvidia{i}").exists():
                gpus.append(i)
        return gpus if gpus else [0]  # Default to GPU 0 if none detected.

    def allocate(self) -> int:
        """Get next GPU in round-robin order.

        Returns:
            GPU device index for BlenderServer.
        """
        with self._lock:
            gpu = self._gpus[self._counter % len(self._gpus)]
            self._counter += 1
            return gpu

    @property
    def available_gpus(self) -> list[int]:
        """Get list of available GPU indices."""
        return self._gpus.copy()


def _reset_inherited_sdk_state() -> None:
    """Reset OpenAI Agents SDK state inherited via fork.

    After fork(), the child inherits corrupted SDK state:
    1. Active trace/span ContextVars - makes workers think they're in parent's trace
    2. BatchTraceProcessor with orphaned threading.Lock and dead background thread
    3. BackendSpanExporter with corrupted httpx.Client connections
    4. SQLiteSession thread-local connections - file descriptors shared with parent
       cause SIGABRT when SQLite detects cross-process connection reuse

    We clear all of these so workers start fresh. Workers can reinitialize
    tracing if needed.

    Must be called at the start of each worker function.
    """
    from agents.tracing import scope

    # Clear any inherited trace/span context so workers start fresh.
    scope._current_trace.set(None)
    scope._current_span.set(None)

    # Clear the corrupted processor from the provider's processor list.
    # After fork(), the BatchTraceProcessor has orphaned locks and dead background thread.
    # The provider holds a reference to it via _multi_processor._processors.
    # We clear that list so traces won't try to use the corrupted processor.
    # Traces will still work, just won't be exported (which is fine for subprocesses).
    try:
        from agents.tracing import setup as tracing_setup

        provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        if provider and hasattr(provider, "_multi_processor"):
            provider._multi_processor.set_processors([])
    except Exception:
        pass  # Best effort - don't crash on reset failure.

    # Close any SQLiteSession thread-local connections inherited via fork.
    # SQLiteSession lazily opens a sqlite3.Connection per thread and stores it in
    # threading.local(). After fork() the child inherits the parent's main-thread
    # connection (same file descriptor), so two processes share one SQLite handle.
    # SQLite's internal mutex state is then inconsistent and triggers SIGABRT.
    # We close and discard the connection so the child gets a fresh one on first use.
    try:
        from agents.memory.sqlite_session import SQLiteSession

        local = SQLiteSession.__init__.__globals__.get("threading")
        # Walk every live SQLiteSession instance via gc and close its thread-local conn.
        import gc

        for obj in gc.get_objects():
            if type(obj) is SQLiteSession and hasattr(obj, "_local"):
                conn = getattr(obj._local, "connection", None)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    obj._local.connection = None  # type: ignore[attr-defined]
        del local
    except Exception:
        pass  # Best effort.


def _load_prompts_from_csv(csv_path: str) -> list[tuple[int, str]]:
    """Load scene prompts from CSV file.

    Args:
        csv_path: Path to CSV file with columns: scene_index, prompt.

    Returns:
        List of (scene_id, prompt) tuples.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV has invalid format or data.
    """
    prompts_with_ids = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row.
        # Start at 2 (after header).
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"CSV row {row_num} has fewer than 2 columns: {row}")
            try:
                scene_id = int(row[0])
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: scene_index '{row[0]}' is not a valid integer"
                )
            prompt = row[1]
            prompts_with_ids.append((scene_id, prompt))
    return prompts_with_ids


def _export_scene_blend_file(
    scene: RoomScene, scene_dir: Path, cfg_dict: dict, name: str = "final_scene"
) -> None:
    """Export scene to a .blend file.

    Args:
        scene: The scene to export.
        scene_dir: Base directory for scene outputs.
        cfg_dict: Configuration dictionary.
        name: Name for the scene state subdirectory.
    """
    from scenesmith.agent_utils.rendering import save_scene_as_blend

    blend_output_path = scene_dir / "scene_states" / name / "scene.blend"
    try:
        rendering_cfg = cfg_dict.get("furniture_agent", {}).get("rendering", {})
        visualization_cfg = cfg_dict.get("experiment", {}).get(
            "stage_visualization", {}
        )
        snapshot_names = set(
            visualization_cfg.get(
                "checkpoints",
                [
                    "scene_after_furniture",
                    "scene_after_wall_objects",
                    "scene_after_ceiling_objects",
                    "final_scene",
                ],
            )
        )
        render_snapshots = bool(visualization_cfg.get("enabled", True)) and (
            name in snapshot_names
        )
        snapshot_rendering_cfg = None
        if render_snapshots:
            snapshot_payload = OmegaConf.to_container(
                OmegaConf.create(rendering_cfg), resolve=True
            )
            if bool(visualization_cfg.get("clean_annotations", True)):
                annotations = snapshot_payload.setdefault("annotations", {})
                annotations.update(
                    {
                        "enable_set_of_mark_labels": False,
                        "enable_bounding_boxes": False,
                        "enable_direction_arrows": False,
                        "enable_support_surface_debug": False,
                        "enable_convex_hull_debug": False,
                    }
                )
            snapshot_rendering_cfg = OmegaConf.create(snapshot_payload)
        save_scene_as_blend(
            scene=scene,
            output_path=blend_output_path,
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(
                rendering_cfg.get("blender_server_port_range", [8000, 8050])
            ),
            server_startup_delay=rendering_cfg.get("server_startup_delay", 0.1),
            port_cleanup_delay=rendering_cfg.get("port_cleanup_delay", 0.1),
            render_cfg=snapshot_rendering_cfg,
            render_output_dir=(
                scene_dir / "scene_states" / name / "renders"
                if render_snapshots
                else None
            ),
            rendering_mode=str(
                visualization_cfg.get("rendering_mode", "furniture")
            ),
            render_taa_samples=int(
                visualization_cfg.get(
                    "taa_samples",
                    rendering_cfg.get("taa_samples", 4),
                )
            ),
        )
    except Exception as e:
        console_logger.error(f"Failed to export .blend file: {e}")


async def _rescore_furniture_after_postprocessing(
    furniture_agent: StatefulFurnitureAgent,
    scene: RoomScene,
) -> None:
    """Re-score the canonical furniture state after projection/simulation.

    The furniture agent writes its normal scores during add_furniture(), before
    the experiment-level physical feasibility post-processing runs.  That
    post-processing can move or remove objects, and the wall stage receives this
    updated in-memory scene.  Re-scoring here keeps scene_renders/furniture,
    scene_states/furniture, and the scene handed to wall_mounted aligned.
    """
    console_logger.info(
        "Furniture post-processing changed the scene; re-scoring canonical "
        "post-processed furniture layout"
    )
    furniture_agent.scene = scene
    critic_tools = furniture_agent._create_critic_tools()
    furniture_agent.critic = furniture_agent._create_critic_agent(
        scene=scene,
        tools=critic_tools,
    )
    try:
        furniture_agent.rendering_manager.clear_cache()
    except Exception:
        console_logger.debug("Could not clear furniture render cache", exc_info=True)

    await furniture_agent._request_critique_impl(update_checkpoint=False)
    await furniture_agent._finalize_scene_and_scores()


def _sync_scene_room_geometry_from_layout(
    scene: RoomScene,
    house_layout: HouseLayout,
    room_id: str,
) -> None:
    """Ensure a RoomScene uses the latest geometry/material assets from layout."""
    latest_geometry = house_layout.get_room_geometry(room_id)
    if latest_geometry is None:
        return

    current_geometry = scene.room_geometry
    try:
        if (
            current_geometry is not None
            and current_geometry.content_hash() == latest_geometry.content_hash()
        ):
            return
    except Exception:
        console_logger.debug(
            "Could not compare room geometry hashes; synchronizing from layout",
            exc_info=True,
        )

    # Replace architectural wall objects so wall extraction/rendering sees the
    # same materialized room geometry that floor_plan saved in house_layout.json.
    for object_id, obj in list(scene.objects.items()):
        if obj.object_type == ObjectType.WALL:
            del scene.objects[object_id]
    scene.room_geometry = latest_geometry
    for wall in latest_geometry.walls:
        scene.add_object(wall)

    console_logger.info(
        "Synchronized room geometry for room %s from house_layout before "
        "downstream stage execution",
        room_id,
    )


def _fix_paths_in_json_file(
    json_path: Path, new_room_dir: Path, new_scene_dir: Path | None = None
) -> None:
    """Fix absolute paths in a JSON file to point to new directories.

    Scans JSON for any string values containing absolute paths and rebases them:
    - Room-level paths (generated_assets/, scene_renders/) → new_room_dir
    - Scene-level paths (room_geometry/, floor_plans/) → new_scene_dir

    Args:
        json_path: Path to JSON file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    if not json_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    with open(json_path) as f:
        data = json.load(f)

    def fix_path(value: str) -> str:
        """Fix a single path string if it's an absolute path."""
        if not value.startswith("/"):
            return value  # Already relative, no fix needed.

        # Room-level paths (relative to room directory).
        room_markers = ["generated_assets/", "scene_renders/", "scene_states/"]
        for marker in room_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_room_dir / marker.rstrip("/") / rel_path)

        # Scene-level paths (relative to scene directory).
        scene_markers = ["room_geometry/", "floor_plans/"]
        for marker in scene_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_scene_dir / marker.rstrip("/") / rel_path)

        return value  # Unknown pattern, leave as-is.

    def fix_paths_recursive(obj):
        """Recursively fix paths in a nested structure."""
        if isinstance(obj, dict):
            return {k: fix_paths_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_paths_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return fix_path(obj)
        return obj

    fixed_data = fix_paths_recursive(data)

    with open(json_path, "w") as f:
        json.dump(fixed_data, f, indent=2)

    console_logger.debug(f"Fixed paths in {json_path}")


def _fix_paths_in_yaml_file(
    yaml_path: Path, new_room_dir: Path, new_scene_dir: Path | None = None
) -> None:
    """Fix absolute paths in YAML file (e.g., scene.dmd.yaml Drake directives).

    Handles file:// URIs used in Drake model directives.

    Args:
        yaml_path: Path to YAML file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    import re

    if not yaml_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    content = yaml_path.read_text()

    def replace_path(match: re.Match) -> str:
        """Replace a file:// URI with the correct new path."""
        old_path = match.group(1)
        # Determine if room-level or scene-level path.
        if "/generated_assets/" in old_path or "/scene_renders/" in old_path:
            # Room-level: extract relative part after room_*/.
            rel_match = re.search(r"room_[^/]+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_room_dir / rel_match.group(1)}"
        elif "/room_geometry/" in old_path or "/floor_plans/" in old_path:
            # Scene-level: extract relative part after scene_*/.
            rel_match = re.search(r"scene_\d+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_scene_dir / rel_match.group(1)}"
        return match.group(0)

    new_content = re.sub(r"file://(/[^\s\"']+)", replace_path, content)
    yaml_path.write_text(new_content)
    console_logger.debug(f"Fixed paths in {yaml_path}")


def _copy_checkpoint_for_stage(
    source_scene_dir: Path, target_scene_dir: Path, start_stage: str
) -> None:
    """Copy only the checkpoint state needed to resume from start_stage.

    Unlike copytree of entire scene, this explicitly copies only required files:
    - Scene-level: room_geometry/, floor_plans/, house_layout.json
    - Room-level: checkpoint directory + referenced assets

    NOT copied (ensuring fresh start for resumed stage):
    - *.db (session files - agent starts fresh conversation)
    - scene_renders/ (render directories - counter starts at 0)
    - *.log (log files - clean logs for new run)
    - action_log.json (replay log - new run builds its own)

    Args:
        source_scene_dir: Path to source scene directory.
        target_scene_dir: Path to target scene directory.
        start_stage: Stage to resume from (determines what to copy).
    """
    if not source_scene_dir.exists():
        raise FileNotFoundError(
            f"Source scene directory not found: {source_scene_dir}. "
            f"Ensure resume_from_path points to an experiment with this scene."
        )

    console_logger.info(f"Copying checkpoint for {start_stage} from {source_scene_dir}")

    # Remove target if it exists (Hydra may have created it).
    if target_scene_dir.exists():
        shutil.rmtree(target_scene_dir)

    target_scene_dir.mkdir(parents=True, exist_ok=True)

    # Copy scene-level directories.
    shutil.copytree(
        source_scene_dir / "room_geometry",
        target_scene_dir / "room_geometry",
    )
    shutil.copytree(
        source_scene_dir / "floor_plans",
        target_scene_dir / "floor_plans",
    )
    # Materials directory contains textures referenced by floor/wall GLTFs.
    materials_dir = source_scene_dir / "materials"
    if materials_dir.exists():
        shutil.copytree(materials_dir, target_scene_dir / "materials")
    shutil.copy(
        source_scene_dir / "house_layout.json",
        target_scene_dir / "house_layout.json",
    )

    checkpoint_name = STAGE_CHECKPOINTS[start_stage]
    asset_dirs = STAGE_ASSET_DIRS[start_stage]

    # Copy room-level checkpoint state and assets.
    for room_dir in source_scene_dir.iterdir():
        if not room_dir.is_dir() or not room_dir.name.startswith("room_"):
            continue

        target_room = target_scene_dir / room_dir.name
        target_room.mkdir(parents=True, exist_ok=True)

        # Copy entire checkpoint directory for self-containment.
        # Includes scene_state.json, scene.dmd.yaml, and scene.blend.
        if checkpoint_name:
            source_state = room_dir / "scene_states" / checkpoint_name
            if source_state.exists():
                target_state = target_room / "scene_states" / checkpoint_name
                shutil.copytree(source_state, target_state)

                # Fix absolute paths in scene_state.json.
                _fix_paths_in_json_file(
                    json_path=target_state / "scene_state.json",
                    new_room_dir=target_room,
                    new_scene_dir=target_scene_dir,
                )

                # Fix absolute paths in scene.dmd.yaml (Drake directives).
                _fix_paths_in_yaml_file(
                    yaml_path=target_state / "scene.dmd.yaml",
                    new_room_dir=target_room,
                    new_scene_dir=target_scene_dir,
                )

        # Copy required asset directories.
        for asset_subdir in asset_dirs:
            source_assets = room_dir / "generated_assets" / asset_subdir
            if source_assets.exists():
                target_assets = target_room / "generated_assets" / asset_subdir
                shutil.copytree(source_assets, target_assets)

                # Fix absolute paths in asset_registry.json.
                asset_registry = target_assets / "asset_registry.json"
                if asset_registry.exists():
                    _fix_paths_in_json_file(
                        json_path=asset_registry,
                        new_room_dir=target_room,
                        new_scene_dir=target_scene_dir,
                    )

    console_logger.info(
        f"Copied checkpoint for {start_stage}: "
        f"checkpoint={checkpoint_name}, assets={asset_dirs}"
    )


def _generate_room(
    room_id: str,
    room_prompt: str,
    room_geometry: RoomGeometry,
    room_dir: Path,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str = "furniture",
    stop_stage: str = "manipuland",
    house_layout: HouseLayout | None = None,
    render_gpu_id: int | None = None,
    scene_expert_hooks: "SceneExpertHookRunner | None" = None,
) -> RoomScene:
    """Generate a single room with furniture, wall/ceiling objects, and manipulands.

    This is the core room generation function used by both single-room and
    multi-room (house) modes. It receives a pre-generated RoomGeometry from the
    HouseLayout and handles furniture, wall object, ceiling object, and
    manipuland placement.

    The room geometry is generated at the house level (by the floor plan generator)
    and passed in here. This ensures consistent handling for both single-room
    and multi-room modes.

    Pipeline stages run in order: furniture → wall_mounted → ceiling_mounted → manipuland
    (floor_plan stage is handled at house level before calling this function)

    State is always saved after each stage for resumability:
    - After furniture: scene_after_furniture.json
    - After wall_mounted: scene_after_wall_objects.json
    - After ceiling_mounted: scene_after_ceiling_objects.json
    - After manipuland: scene_after_manipulands.json (via final_scene logging)

    Args:
        room_id: Unique identifier for the room (e.g., "main", "living_room").
        room_prompt: Text description for the room.
        room_geometry: Pre-generated RoomGeometry from HouseLayout.
        room_dir: Directory for room outputs (e.g., scene_000/room_main/).
        logger: Logger instance for saving outputs.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        stop_stage: Stage to stop after ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        house_layout: Optional HouseLayout for door/window export in SceneEval.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        RoomScene with furniture, wall/ceiling objects, and (optionally) manipulands.
    """
    room_start_time = time.time()

    # Create scene and add walls and floor from room geometry.
    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_id,
        text_description=room_prompt,
        action_log_path=room_dir / "action_log.json",
    )
    for wall in room_geometry.walls:
        scene.add_object(wall)
    # Note: Floor is NOT added to scene.objects to avoid duplicate
    # collision geometry (room_geometry.sdf already contains floor).
    # Floor remains accessible via scene.room_geometry.floor for
    # manipuland placement queries.

    # Get stage index for comparison (room stages exclude floor_plan).
    # ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]
    room_stages = PIPELINE_STAGES[1:]
    start_idx = room_stages.index(start_stage) if start_stage in room_stages else 0

    # Load projection config (needed for furniture and final post-processing).
    projection_cfg = cfg_dict["experiment"]["projection"]

    # Furniture stage.
    if start_idx <= 0:  # Run furniture if starting from furniture or earlier.
        with custom_span("furniture_placement"):
            console_logger.info("Adding furniture to scene")
            start_time = time.time()
            if scene_expert_hooks:
                scene_expert_hooks.pre_stage("furniture", scene)
            furniture_agent = BaseExperiment.build_furniture_agent(
                cfg_dict=cfg_dict,
                compatible_agents=(
                    IndoorSceneGenerationExperiment.compatible_furniture_agents
                ),
                logger=logger,
                render_gpu_id=render_gpu_id,
            )
            try:
                recovery_cfg = cfg_dict["furniture_agent"].get(
                    "hard_constraint_recovery", {}
                )
                max_stage_regenerations = max(
                    0, int(recovery_cfg.get("max_stage_regenerations", 1) or 0)
                )
                continue_after_exhaustion = bool(
                    recovery_cfg.get(
                        "continue_after_repairable_exhaustion", True
                    )
                )
                empty_stage_state = scene.to_state_dict()
                stage_prompt = scene.text_description
                regeneration_attempt = 0
                critic_only_retry_attempted = False
                best_agent_candidate = None
                repairable_hard_exhausted = False
                capture_agent_candidate = getattr(
                    furniture_agent, "capture_agent_candidate", None
                )
                prefer_agent_candidate = getattr(
                    furniture_agent, "prefer_agent_candidate", None
                )
                should_regenerate_for_quality = getattr(
                    furniture_agent, "should_regenerate_for_quality", None
                )
                while True:
                    try:
                        asyncio.run(furniture_agent.add_furniture(scene=scene))
                        candidate = (
                            capture_agent_candidate()
                            if callable(capture_agent_candidate)
                            else None
                        )
                        if callable(prefer_agent_candidate):
                            best_agent_candidate = prefer_agent_candidate(
                                best_agent_candidate,
                                candidate,
                            )
                        should_regenerate, quality_reason = (
                            should_regenerate_for_quality(candidate)
                            if callable(should_regenerate_for_quality)
                            else (False, "quality fallback unsupported by this agent")
                        )
                        if (
                            should_regenerate
                            and regeneration_attempt < max_stage_regenerations
                        ):
                            regeneration_attempt += 1
                            console_logger.warning(
                                "Furniture agent candidate missed the trusted critic "
                                "target; restarting the full designer/critic stage "
                                "from the empty-room checkpoint (%d/%d): %s",
                                regeneration_attempt,
                                max_stage_regenerations,
                                quality_reason,
                            )
                            scene.restore_from_state_dict(empty_stage_state)
                            scene.text_description = (
                                f"{stage_prompt}\n\n"
                                "# Mandatory Quality Regeneration\n"
                                "The previous layout was physically valid but did "
                                "not meet the visual critic target. Propose a "
                                "genuinely new expert layout and address: "
                                f"{quality_reason}."
                            )
                            asyncio.run(
                                furniture_agent.prepare_stage_regeneration(
                                    [quality_reason]
                                )
                            )
                            continue
                        break
                    except StageValidationError as exc:
                        critic_only = bool(exc.reasons) and all(
                            "visual critic did not produce a trustworthy score"
                            in reason.lower()
                            for reason in exc.reasons
                        )
                        if critic_only and not critic_only_retry_attempted:
                            critic_only_retry_attempted = True
                            console_logger.warning(
                                "Furniture output is hard-valid but unscored; "
                                "retrying only the compact final critic before "
                                "discarding the layout"
                            )
                            try:
                                asyncio.run(
                                    furniture_agent.retry_final_critic_evaluation()
                                )
                                break
                            except StageValidationError as retry_exc:
                                exc = retry_exc

                        repairable = _is_repairable_stage_validation(exc)
                        if (
                            repairable
                            and regeneration_attempt < max_stage_regenerations
                        ):
                            regeneration_attempt += 1
                            console_logger.warning(
                                "Furniture layout remained invalid after local repair; "
                                "restarting the full designer/critic stage from the "
                                "empty-room checkpoint (%d/%d): %s",
                                regeneration_attempt,
                                max_stage_regenerations,
                                "; ".join(exc.reasons),
                            )
                            scene.restore_from_state_dict(empty_stage_state)
                            scene.text_description = (
                                f"{stage_prompt}\n\n"
                                "# Mandatory Stage Regeneration\n"
                                "The previous furniture layout was rejected by "
                                "deterministic validation. Design a genuinely new "
                                "layout from the empty room; do not incrementally "
                                "recreate the rejected arrangement. Resolve all of: "
                                + "; ".join(exc.reasons)
                            )
                            asyncio.run(
                                furniture_agent.prepare_stage_regeneration(
                                    exc.reasons
                                )
                            )
                            continue

                        if repairable and continue_after_exhaustion:
                            console_logger.warning(
                                "Furniture repair and stage regeneration were "
                                "exhausted. Persisting the diagnosed candidate and "
                                "continuing downstream instead of hard-failing ACP: %s",
                                "; ".join(exc.reasons),
                            )
                            asyncio.run(
                                furniture_agent.complete_repair_exhausted_stage(
                                    exc.reasons
                                )
                            )
                            repairable_hard_exhausted = True
                            break
                        raise

                # A deterministic relation layout remains a final comparison
                # candidate, never the normal generator.  Hard-recovery
                # exhaustion and missing critic evidence are not successful
                # agent outcomes, so they must not silently bypass this branch.
                latest_candidate = (
                    capture_agent_candidate(
                        allow_hard_invalid=repairable_hard_exhausted
                    )
                    if callable(capture_agent_candidate)
                    else None
                )
                if repairable_hard_exhausted:
                    comparison_candidate = latest_candidate
                else:
                    if callable(prefer_agent_candidate):
                        best_agent_candidate = prefer_agent_candidate(
                            best_agent_candidate,
                            latest_candidate,
                        )
                    comparison_candidate = best_agent_candidate
                restore_agent_candidate = getattr(
                    furniture_agent, "restore_agent_candidate", None
                )
                should_generate_fallback = getattr(
                    furniture_agent,
                    "should_generate_deterministic_fallback",
                    None,
                )
                if (
                    comparison_candidate is not None
                    and callable(restore_agent_candidate)
                    and callable(should_generate_fallback)
                ):
                    restore_agent_candidate(comparison_candidate)
                    should_fallback, fallback_reason = should_generate_fallback(
                        comparison_candidate,
                        regeneration_attempts=regeneration_attempt,
                        max_stage_regenerations=max_stage_regenerations,
                        repairable_hard_exhausted=repairable_hard_exhausted,
                    )
                    if should_fallback:
                        console_logger.warning(
                            "Pure-agent furniture workflow exhausted; generating "
                            "one separately rendered deterministic comparison "
                            "candidate: %s",
                            fallback_reason,
                        )
                        compare_deterministic_fallback = getattr(
                            furniture_agent,
                            "compare_deterministic_fallback",
                            None,
                        )
                        if callable(compare_deterministic_fallback):
                            asyncio.run(
                                compare_deterministic_fallback(
                                    agent_candidate=comparison_candidate,
                                    trigger=fallback_reason,
                                    regeneration_attempts=regeneration_attempt,
                                )
                            )
                    else:
                        persist_agent_best = getattr(
                            furniture_agent,
                            "persist_agent_best_candidate",
                            None,
                        )
                        if callable(persist_agent_best) and getattr(
                            scene, "scene_expert_stage_budget", None
                        ):
                            persist_agent_best(comparison_candidate)
                end_time = time.time()
                console_logger.info(
                    f"Furniture added to room {room_id} in "
                    f"{timedelta(seconds=end_time - start_time)}"
                )

                pre_postprocess_hash = scene.content_hash()
                pre_postprocess_state = scene.to_state_dict()

                # Furniture post-processing (projection + simulation).
                if projection_cfg["enabled"] and projection_cfg["furniture"]["enabled"]:
                    furniture_cfg = projection_cfg["furniture"]
                    sim_cfg = projection_cfg["simulation"]

                    # Log pre-projection state for debugging.
                    logger.log_scene(scene=scene, name="furniture_only_pre_projection")

                    console_logger.info(
                        "Running furniture post-processing (projection + simulation)"
                    )
                    postprocess_start_time = time.time()

                    # Determine HTML output path for simulation.
                    furniture_sim_html_path = None
                    if sim_cfg.get("save_html", False):
                        furniture_sim_html_path = (
                            logger.output_dir
                            / "simulation"
                            / "furniture_simulation.html"
                        )

                    # Get fallen furniture config from physics_validation.
                    physics_val_cfg = cfg_dict["furniture_agent"]["physics_validation"]
                    scene, projection_success, removed_ids = (
                        apply_physical_feasibility_postprocessing(
                            scene=scene,
                            weld_furniture=False,
                            projection_enabled=True,
                            projection_influence_distance=furniture_cfg[
                                "influence_distance"
                            ],
                            projection_solver_name=furniture_cfg["solver_name"],
                            projection_iteration_limit=furniture_cfg["iteration_limit"],
                            projection_time_limit_s=furniture_cfg["time_limit_s"],
                            projection_xy_only=furniture_cfg["xy_only"],
                            projection_fix_rotation=furniture_cfg["fix_rotation"],
                            simulation_enabled=sim_cfg["enabled"],
                            simulation_time_s=sim_cfg["simulation_time_s"],
                            simulation_time_step_s=sim_cfg["time_step_s"],
                            simulation_timeout_s=sim_cfg["timeout_s"],
                            simulation_html_path=furniture_sim_html_path,
                            remove_fallen_furniture=physics_val_cfg[
                                "remove_fallen_furniture"
                            ],
                            fallen_tilt_threshold_degrees=physics_val_cfg[
                                "fallen_tilt_threshold_degrees"
                            ],
                        )
                    )
                    postprocess_end_time = time.time()
                    if not projection_success:
                        console_logger.error(
                            "Furniture projection failed; restoring original positions"
                        )
                        scene.restore_from_state_dict(pre_postprocess_state)
                        removed_ids = []
                    else:
                        if removed_ids:
                            console_logger.info(
                                f"Removed {len(removed_ids)} fallen furniture item(s) "
                                f"during simulation: {removed_ids}"
                            )
                        console_logger.info(
                            f"Furniture post-processing completed for room {room_id} "
                            f"in {postprocess_end_time - postprocess_start_time:.2f} "
                            "seconds"
                        )

                if scene.content_hash() != pre_postprocess_hash:
                    try:
                        asyncio.run(
                            _rescore_furniture_after_postprocessing(
                                furniture_agent=furniture_agent,
                                scene=scene,
                            )
                        )
                    except Exception as e:
                        console_logger.error(
                            "Failed to re-score post-processed furniture layout: %s",
                            e,
                            exc_info=True,
                        )
            finally:
                # Always cleanup server subprocesses after all furniture-stage
                # scoring/rendering that depends on the agent's Blender server.
                furniture_agent.cleanup()

        # Always save state after furniture stage (unconditional for resumability).
        logger.log_scene(scene=scene, name="scene_after_furniture")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_furniture",
        )
        console_logger.info("Saved furniture checkpoint (scene_after_furniture)")
        if scene_expert_hooks:
            scene_expert_hooks.post_stage("furniture", scene, room_dir)
    elif start_idx == 1:
        # Starting from wall_objects - load scene from saved furniture state.
        console_logger.info("Loading scene from saved furniture state for wall_objects")
        furniture_state_path = (
            room_dir / "scene_states" / "scene_after_furniture" / "scene_state.json"
        )
        if not furniture_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'wall_objects' stage: furniture state not found at "
                f"{furniture_state_path}. Run with start_stage='furniture' first."
            )
        with open(furniture_state_path) as f:
            furniture_state = json.load(f)
        scene.restore_from_state_dict(furniture_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from furniture checkpoint"
        )

    # Check if we should stop after furniture stage.
    if stop_stage == "furniture":
        console_logger.info("Stopping after furniture stage as configured")
        return scene

    # Wall objects stage.
    if start_idx <= 1:  # Run wall_objects if starting from wall_objects or earlier.
        with custom_span("wall_object_placement"):
            console_logger.info("Adding wall-mounted objects to scene")
            start_time = time.time()

            # Load house_layout from parent directory (saved during floor plan stage).
            house_layout_path = room_dir.parent / "house_layout.json"
            if not house_layout_path.exists():
                raise FileNotFoundError(
                    f"Cannot run wall_objects stage: house_layout.json not found at "
                    f"{house_layout_path}. This should have been saved during floor "
                    f"plan generation."
                )
            with open(house_layout_path) as f:
                house_layout_dict = json.load(f)
            house_layout = HouseLayout.from_dict(
                house_layout_dict, house_dir=room_dir.parent
            )
            _sync_scene_room_geometry_from_layout(
                scene=scene,
                house_layout=house_layout,
                room_id=room_id,
            )

            if scene_expert_hooks:
                scene_expert_hooks.pre_stage("wall_mounted", scene)
            wall_agent = BaseExperiment.build_wall_agent(
                cfg_dict=cfg_dict,
                compatible_agents=IndoorSceneGenerationExperiment.compatible_wall_agents,
                logger=logger,
                house_layout=house_layout,
                ceiling_height=scene.room_geometry.wall_height,
                wall_thickness=scene.room_geometry.wall_thickness,
                render_gpu_id=render_gpu_id,
            )
            try:
                _run_sceneexpert_placement_stage(
                    stage="wall_mounted",
                    agent=wall_agent,
                    scene=scene,
                    run_once=lambda: wall_agent.add_wall_objects(scene=scene),
                )
            finally:
                # Always cleanup server subprocesses.
                wall_agent.cleanup()
            end_time = time.time()
            console_logger.info(
                f"Wall objects added to room {room_id} in "
                f"{timedelta(seconds=end_time - start_time)}"
            )

        # Always save state after wall_objects stage (unconditional for resumability).
        logger.log_scene(scene=scene, name="scene_after_wall_objects")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_wall_objects",
        )
        console_logger.info("Saved wall_objects checkpoint (scene_after_wall_objects)")
        if scene_expert_hooks:
            scene_expert_hooks.post_stage("wall_mounted", scene, room_dir)
    elif start_idx == 2:
        # Starting from ceiling_mounted - load scene from saved wall_objects state.
        console_logger.info("Loading scene from saved wall_objects state for ceiling")
        wall_objects_state_path = (
            room_dir / "scene_states" / "scene_after_wall_objects" / "scene_state.json"
        )
        if not wall_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'ceiling_mounted' stage: wall_objects state not "
                f"found at {wall_objects_state_path}. Run with "
                f"start_stage='wall_mounted' first."
            )
        with open(wall_objects_state_path) as f:
            wall_objects_state = json.load(f)
        scene.restore_from_state_dict(wall_objects_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from wall_objects checkpoint"
        )

    # Check if we should stop after wall_mounted stage.
    if stop_stage == AgentType.WALL_MOUNTED.value:
        console_logger.info("Stopping after wall_mounted stage as configured")
        return scene

    # Ceiling objects stage.
    if start_idx <= 2:  # Run ceiling if starting from ceiling or earlier.
        with custom_span("ceiling_object_placement"):
            console_logger.info("Adding ceiling-mounted objects to scene")
            start_time = time.time()

            if scene_expert_hooks:
                scene_expert_hooks.pre_stage("ceiling_mounted", scene)
            ceiling_agent = BaseExperiment.build_ceiling_agent(
                cfg_dict=cfg_dict,
                compatible_agents=(
                    IndoorSceneGenerationExperiment.compatible_ceiling_agents
                ),
                logger=logger,
                ceiling_height=room_geometry.wall_height,
                render_gpu_id=render_gpu_id,
            )
            try:
                _run_sceneexpert_placement_stage(
                    stage="ceiling_mounted",
                    agent=ceiling_agent,
                    scene=scene,
                    run_once=lambda: ceiling_agent.add_ceiling_objects(scene=scene),
                )
            finally:
                # Always cleanup server subprocesses.
                ceiling_agent.cleanup()
            end_time = time.time()
            console_logger.info(
                f"Ceiling objects added to room {room_id} in "
                f"{timedelta(seconds=end_time - start_time)}"
            )

        # Always save state after ceiling stage (unconditional for resumability).
        logger.log_scene(scene=scene, name="scene_after_ceiling_objects")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_ceiling_objects",
        )
        console_logger.info(
            "Saved ceiling_objects checkpoint (scene_after_ceiling_objects)"
        )
        if scene_expert_hooks:
            scene_expert_hooks.post_stage("ceiling_mounted", scene, room_dir)
    else:
        # Starting from manipulands - load scene from saved ceiling_objects state.
        console_logger.info("Loading scene from saved ceiling_objects state")
        ceiling_objects_state_path = (
            room_dir
            / "scene_states"
            / "scene_after_ceiling_objects"
            / "scene_state.json"
        )
        if not ceiling_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'manipuland' stage: ceiling_objects state not "
                f"found at {ceiling_objects_state_path}. Run with "
                f"start_stage='ceiling_mounted' first."
            )
        with open(ceiling_objects_state_path) as f:
            ceiling_objects_state = json.load(f)
        scene.restore_from_state_dict(ceiling_objects_state)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from ceiling_objects checkpoint"
        )

    # Check if we should stop after ceiling_mounted stage.
    if stop_stage == AgentType.CEILING_MOUNTED.value:
        console_logger.info("Stopping after ceiling_mounted stage as configured")
        return scene

    # Add manipulands.
    with custom_span("manipuland_placement"):
        console_logger.info("Adding manipulands to scene")
        start_time = time.time()
        if scene_expert_hooks:
            scene_expert_hooks.pre_stage("manipuland", scene)
        manipuland_agent = BaseExperiment.build_manipuland_agent(
            cfg_dict=cfg_dict,
            compatible_agents=(
                IndoorSceneGenerationExperiment.compatible_manipuland_agents
            ),
            logger=logger,
            render_gpu_id=render_gpu_id,
        )
        _run_sceneexpert_placement_stage(
            stage="manipuland",
            agent=manipuland_agent,
            scene=scene,
            run_once=lambda: manipuland_agent.add_manipulands(scene=scene),
        )
        end_time = time.time()
        console_logger.info(
            f"Manipulands added to room {room_id} in "
            f"{timedelta(seconds=end_time - start_time)}"
        )

    # Final post-processing (projection + simulation).
    if projection_cfg["enabled"] and projection_cfg["final"]["enabled"]:
        final_cfg = projection_cfg["final"]
        sim_cfg = projection_cfg["simulation"]

        # Log pre-projection state for debugging.
        logger.log_scene(scene=scene, name="final_scene_pre_projection")

        console_logger.info("Running final post-processing (projection + simulation)")
        start_time = time.time()
        pre_final_postprocess_state = scene.to_state_dict()

        # Determine HTML output path for simulation.
        final_sim_html_path = None
        if sim_cfg.get("save_html", False):
            final_sim_html_path = (
                logger.output_dir / "simulation" / "final_simulation.html"
            )

        # Final post-processing: weld_furniture=True means only manipulands move.
        # Fallen furniture removal is not needed here (furniture is welded).
        # Get fallen manipuland config from manipuland_agent physics_validation.
        manipuland_physics_cfg = cfg_dict["manipuland_agent"]["physics_validation"]
        scene, projection_success, removed_ids = (
            apply_physical_feasibility_postprocessing(
                scene=scene,
                weld_furniture=True,
                projection_enabled=True,
                projection_influence_distance=final_cfg["influence_distance"],
                projection_solver_name=final_cfg["solver_name"],
                projection_iteration_limit=final_cfg["iteration_limit"],
                projection_time_limit_s=final_cfg["time_limit_s"],
                projection_xy_only=final_cfg["xy_only"],
                projection_fix_rotation=final_cfg["fix_rotation"],
                simulation_enabled=sim_cfg["enabled"],
                simulation_time_s=sim_cfg["simulation_time_s"],
                simulation_time_step_s=sim_cfg["time_step_s"],
                simulation_timeout_s=sim_cfg["timeout_s"],
                simulation_html_path=final_sim_html_path,
                remove_fallen_furniture=False,
                remove_fallen_manipulands=manipuland_physics_cfg[
                    "remove_fallen_manipulands"
                ],
                fallen_manipuland_floor_z=manipuland_physics_cfg[
                    "fallen_manipuland_floor_z"
                ],
                fallen_manipuland_near_floor_z=manipuland_physics_cfg[
                    "fallen_manipuland_near_floor_z"
                ],
                fallen_manipuland_z_displacement=manipuland_physics_cfg[
                    "fallen_manipuland_z_displacement"
                ],
            )
        )
        end_time = time.time()
        if not projection_success:
            console_logger.error(
                "Final projection failed; restoring original positions"
            )
            scene.restore_from_state_dict(pre_final_postprocess_state)
            removed_ids = []
        else:
            if removed_ids:
                console_logger.info(
                    f"Removed {len(removed_ids)} fallen manipuland(s) during "
                    f"final simulation: {removed_ids}"
                )
            console_logger.info(
                f"Final post-processing completed for room {room_id} in "
                f"{end_time - start_time:.2f} seconds"
            )

    # Log and export final scene.
    logger.log_scene(scene=scene, name="final_scene")
    _export_scene_blend_file(
        scene=scene, scene_dir=room_dir, cfg_dict=cfg_dict, name="final_scene"
    )
    if scene_expert_hooks:
        scene_expert_hooks.post_stage("manipuland", scene, room_dir)

    # Export to SceneEval format if enabled.
    sceneeval_cfg = cfg_dict["experiment"]["sceneeval_export"]
    if sceneeval_cfg["enabled"]:
        export_config = SceneEvalExportConfig(
            asset_id_prefix=sceneeval_cfg["asset_id_prefix"]
        )
        exporter = SceneEvalExporter(
            scene=scene,
            scene_dir=room_dir,
            config=export_config,
            house_layout=house_layout,
        )
        exporter.export()

    console_logger.info(
        f"Room {room_id} generation completed successfully in "
        f"{timedelta(seconds=time.time() - room_start_time)}"
    )

    return scene


def _run_sequential_room_generation(
    house_layout: HouseLayout,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    render_gpu_id: int | None = None,
    scene_expert_hooks: "SceneExpertHookRunner | None" = None,
) -> dict[str, RoomScene]:
    """Generate rooms sequentially (existing behavior).

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        logger: Logger for output routing.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.
        scene_expert_hooks: Optional SceneExpert hook runner for pre/post-stage
            hooks (memory retrieval, StageBrief injection, verification, tracing).

    Returns:
        Dictionary mapping room_id to RoomScene.
    """
    rooms: dict[str, RoomScene] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        with custom_span(f"room_{room_id}_generation"):
            with logger.room_context(room_id) as room_dir:
                console_logger.info(f"Generating room '{room_id}': {room_spec.prompt}")
                room_scene = _generate_room(
                    room_id=room_id,
                    room_prompt=room_spec.prompt,
                    room_geometry=room_geometry,
                    room_dir=room_dir,
                    logger=logger,
                    cfg_dict=cfg_dict,
                    start_stage=start_stage,
                    stop_stage=stop_stage,
                    house_layout=house_layout,
                    render_gpu_id=render_gpu_id,
                    scene_expert_hooks=scene_expert_hooks,
                )
                rooms[room_id] = room_scene
    return rooms


def _generate_floor_plan_worker(
    prompt: str,
    scene_dir: str,
    cfg_dict: dict,
    experiment_run_id: str | None,
    render_gpu_id: int | None = None,
) -> None:
    """Run floor plan generation in isolated subprocess.

    This function runs in a separate process to ensure all fork-unsafe state
    (SQLiteSession locks, tracing threads) is destroyed when the subprocess
    exits, before we fork room workers.

    Args:
        prompt: Scene description prompt.
        scene_dir: Path to scene output directory (as string).
        cfg_dict: Configuration dictionary.
        experiment_run_id: Unique ID for this experiment run.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.
    """
    # Reset any SDK state inherited via fork (defense in depth).
    _reset_inherited_sdk_state()

    faulthandler.enable()

    scene_path = Path(scene_dir)
    logger = ConsoleLogger(output_dir=scene_path)

    # Use FileLoggingContext to capture floor plan logs to scene.log.
    log_path = scene_path / "scene.log"
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(f"Floor plan worker started for scene: {scene_dir}")

        # Create trace metadata for this floor plan generation.
        trace_metadata = {"scene_dir": scene_dir, "prompt": prompt}
        if experiment_run_id:
            trace_metadata["experiment_run_id"] = experiment_run_id

        with trace(workflow_name="floor_plan_generation", metadata=trace_metadata):
            with custom_span("floor_plan_generation"):
                floor_plan_agent = BaseExperiment.build_floor_plan_agent(
                    cfg_dict=cfg_dict,
                    compatible_agents=(
                        IndoorSceneGenerationExperiment.compatible_floor_plan_agents
                    ),
                    logger=logger,
                    render_gpu_id=render_gpu_id,
                )
                configure_runtime_budget = getattr(
                    floor_plan_agent, "configure_stage_runtime_budget", None
                )
                if callable(configure_runtime_budget):
                    configure_runtime_budget(
                        resolve_scene_expert_stage_budget(cfg_dict, "floor_plan")
                    )
                try:
                    house_layout = asyncio.run(
                        floor_plan_agent.generate_house_layout(
                            prompt=prompt,
                            output_dir=scene_path / "floor_plans",
                        )
                    )
                finally:
                    floor_plan_agent.cleanup()

                # Save to disk for parent to load.
                house_layout_path = scene_path / "house_layout.json"
                with open(house_layout_path, "w") as f:
                    json.dump(house_layout.to_dict(scene_dir=scene_path), f, indent=2)
                console_logger.info(f"Saved house layout to {house_layout_path}")


def _generate_room_worker(
    room_id: str,
    room_prompt: str,
    room_geometry_dict: dict,
    room_dir: str,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    scene_id: int,
    experiment_run_id: str | None = None,
    house_layout_dict: dict | None = None,
    render_gpu_id: int | None = None,
) -> dict:
    """Worker function for parallel room generation.

    Runs in a subprocess. All args must be picklable (no Path, no complex objects).

    Note on tracing: Room traces are INDEPENDENT from parent scene trace because
    ProcessPoolExecutor creates separate processes. We include scene_id in metadata
    to enable correlation via trace queries.

    Args:
        room_id: Unique identifier for the room.
        room_prompt: Text description for the room.
        room_geometry_dict: Serialized RoomGeometry dictionary.
        room_dir: Path to room output directory (as string).
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        scene_id: Parent scene ID for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        house_layout_dict: Optional serialized HouseLayout for door/window export.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        Dict containing scene_state and metadata for reconstruction.
    """
    # Reset any SDK state inherited via fork (defense in depth).
    _reset_inherited_sdk_state()

    room_dir_path = Path(room_dir)

    faulthandler.enable()

    log_path = room_dir_path / "room.log"

    # Create logger for this room (logs to file, not stdout).
    room_logger = ConsoleLogger(output_dir=room_dir_path)

    # Reconstruct RoomGeometry from serialized dict.
    room_geometry = RoomGeometry.from_dict(room_geometry_dict, scene_dir=room_dir_path)

    # Reconstruct HouseLayout from serialized dict (if provided).
    house_layout = None
    if house_layout_dict:
        house_layout = HouseLayout.from_dict(
            house_layout_dict, house_dir=room_dir_path.parent
        )

    # Use FileLoggingContext to capture logs to room.log.
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(
            f"Worker started for room '{room_id}' with room prompt '{room_prompt}'"
        )

        # Create trace metadata for this room.
        trace_metadata = {
            "room_id": room_id,
            "parent_scene_id": f"scene_{scene_id:03d}",
            "experiment_name": cfg_dict["name"],
            "room_dir": str(room_dir_path),
            "room_prompt": room_prompt,
        }
        if experiment_run_id:
            trace_metadata["experiment_run_id"] = experiment_run_id

        with trace(
            workflow_name=f"scene_{scene_id:03d}_room_{room_id}",
            metadata=trace_metadata,
        ):
            room_scene = _generate_room(
                room_id=room_id,
                room_prompt=room_prompt,
                room_geometry=room_geometry,
                room_dir=room_dir_path,
                logger=room_logger,
                cfg_dict=cfg_dict,
                start_stage=start_stage,
                stop_stage=stop_stage,
                house_layout=house_layout,
                render_gpu_id=render_gpu_id,
            )

        console_logger.info(f"Worker completed for room '{room_id}'")

    # Return serializable result for cross-process transfer.
    return {
        "scene_state": room_scene.to_state_dict(),
        "room_id": room_scene.room_id,
        "text_description": room_scene.text_description,
    }


def _reconstruct_room_scene(worker_result: dict, scene_dir: Path) -> RoomScene:
    """Reconstruct RoomScene from worker result dict.

    Args:
        worker_result: Dict containing scene_state from worker.
        scene_dir: Path to room directory for path resolution.

    Returns:
        Reconstructed RoomScene.
    """
    scene_state = worker_result["scene_state"]

    # Reconstruct RoomGeometry first (needed for RoomScene constructor).
    room_geometry = RoomGeometry.from_dict(
        scene_state["room_geometry"], scene_dir=scene_dir
    )

    # Create RoomScene with required fields.
    room_scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=scene_dir,
        room_id=worker_result["room_id"],
        text_description=worker_result.get("text_description", ""),
        action_log_path=scene_dir / "action_log.json",
    )

    # Restore objects and other state.
    room_scene.restore_from_state_dict(scene_state)

    return room_scene


def _run_parallel_room_generation(
    house_layout: HouseLayout,
    output_dir: Path,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    max_workers: int,
    scene_id: int,
    experiment_run_id: str | None = None,
    render_gpu_id: int | None = None,
) -> dict[str, RoomScene]:
    """Generate rooms in parallel with fault tolerance.

    Uses isolated processes per room instead of a shared executor pool.
    This ensures that if one room crashes, other rooms continue running.

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        output_dir: Base output directory for the scene.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        max_workers: Maximum number of concurrent room processes.
        scene_id: Scene identifier for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        render_gpu_id: GPU device ID for Blender rendering. When set, uses
            bubblewrap to isolate the BlenderServer to this GPU.

    Returns:
        Dictionary mapping room_id to RoomScene.

    Raises:
        RuntimeError: If any room generation fails.
    """
    console_logger.info("Running room generation in parallel")

    # Build task list.
    tasks: list[tuple[str, Callable, dict]] = []
    room_dirs: dict[str, Path] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        # Create room directory (must exist before worker starts).
        room_dir = output_dir / f"room_{room_id}"
        room_dir.mkdir(parents=True, exist_ok=True)
        room_dirs[room_id] = room_dir

        console_logger.info(f"Queued room '{room_id}' (logs → {room_dir / 'room.log'})")

        kwargs = {
            "room_id": room_id,
            "room_prompt": room_spec.prompt,
            "room_geometry_dict": room_geometry.to_dict(scene_dir=room_dir),
            "room_dir": str(room_dir),
            "cfg_dict": cfg_dict,
            "start_stage": start_stage,
            "stop_stage": stop_stage,
            "scene_id": scene_id,
            "experiment_run_id": experiment_run_id,
            "house_layout_dict": house_layout.to_dict(scene_dir=output_dir),
            "render_gpu_id": render_gpu_id,
        }
        tasks.append((room_id, _generate_room_worker, kwargs))

    # Run with fault tolerance and get return values.
    results = run_parallel_isolated(
        tasks=tasks, max_workers=max_workers, return_values=True
    )

    # Reconstruct RoomScenes from worker results.
    rooms: dict[str, RoomScene] = {}
    failures: list[tuple[str, str]] = []
    for room_id, (success, result_or_error) in results.items():
        room_dir = room_dirs[room_id]
        if success:
            rooms[room_id] = _reconstruct_room_scene(
                worker_result=result_or_error, scene_dir=room_dir
            )
            console_logger.info(f"Room '{room_id}' completed successfully")
        else:
            console_logger.error(f"Room '{room_id}' failed: {result_or_error}")
            failures.append((room_id, result_or_error))

    if failures:
        error_msg = "; ".join(f"Room '{rid}': {err}" for rid, err in failures)
        raise RuntimeError(f"Room generation failures: {error_msg}")

    return rooms


class IndoorSceneGenerationExperiment(BaseExperiment):
    """An experiment that generates indoor scenes."""

    compatible_floor_plan_agents = {
        "stateful_floor_plan_agent": StatefulFloorPlanAgent,
    }
    compatible_furniture_agents = {
        "stateful_furniture_agent": StatefulFurnitureAgent,
    }
    compatible_manipuland_agents = {
        "stateful_manipuland_agent": StatefulManipulandAgent,
    }
    compatible_wall_agents = {
        "stateful_wall_agent": StatefulWallAgent,
    }
    compatible_ceiling_agents = {
        "stateful_ceiling_agent": StatefulCeilingAgent,
    }

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.geometry_server: GeometryGenerationServer | None = None
        self.hssd_server: HssdRetrievalServer | None = None
        self.objaverse_server: ObjaverseRetrievalServer | None = None
        self.articulated_server: ArticulatedRetrievalServer | None = None
        self.materials_server: MaterialsRetrievalServer | None = None

    def __del__(self):
        """Ensure servers are stopped when experiment is destroyed."""
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.warning("Stopping geometry server in destructor")
            try:
                self.geometry_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop geometry server in destructor: {e}"
                )

        if self.hssd_server and self.hssd_server.is_running():
            console_logger.warning("Stopping HSSD server in destructor")
            try:
                self.hssd_server.stop()
            except Exception as e:
                console_logger.error(f"Failed to stop HSSD server in destructor: {e}")

        if self.articulated_server and self.articulated_server.is_running():
            console_logger.warning("Stopping articulated server in destructor")
            try:
                self.articulated_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop articulated server in destructor: {e}"
                )

        if self.materials_server and self.materials_server.is_running():
            console_logger.warning("Stopping materials server in destructor")
            try:
                self.materials_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop materials server in destructor: {e}"
                )

    def _start_geometry_server(self) -> None:
        """Start geometry generation server (if general_asset_source == 'generated')."""
        # Only start if at least one agent uses generated strategy.
        furniture_uses_generated = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "generated"
        )
        manipuland_uses_generated = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "generated"
        )

        if not (furniture_uses_generated or manipuland_uses_generated):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.geometry_generation_server

        # Determine backend - use furniture agent config (they should match).
        backend = self.cfg.furniture_agent.asset_manager.get("backend", "hunyuan3d")

        # Prepare SAM3D config if using SAM3D backend.
        sam3d_config = None
        if backend == "sam3d":
            sam3d_cfg = self.cfg.furniture_agent.asset_manager.sam3d
            sam3d_config = {
                "sam3_checkpoint": str(sam3d_cfg.sam3_checkpoint),
                "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
            }

        console_logger.info(
            f"Starting geometry generation server ({backend}) on "
            f"{server_config.host}:{server_config.port}"
        )

        self.geometry_server = GeometryGenerationServer(
            host=server_config.host,
            port=server_config.port,
            backend=backend,
            sam3d_config=sam3d_config,
            log_file=self.output_dir / "experiment.log",
        )

        self.geometry_server.start()
        self.geometry_server.wait_until_ready(timeout_s=30.0)
        console_logger.info("Geometry generation server ready")

    def _stop_geometry_server(self) -> None:
        """Stop the geometry generation server."""
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.info("Stopping geometry generation server...")
            self.geometry_server.stop()
            console_logger.info("Geometry generation server stopped")
            self.geometry_server = None

    def _start_hssd_server(self) -> None:
        """Start HSSD retrieval server (if general_asset_source == 'hssd')."""
        # Only start if at least one agent uses HSSD strategy.
        furniture_uses_hssd = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "hssd"
        )
        manipuland_uses_hssd = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "hssd"
        )
        wall_uses_hssd = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "hssd"
        )
        ceiling_uses_hssd = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "hssd"
        )

        if not (
            furniture_uses_hssd
            or manipuland_uses_hssd
            or wall_uses_hssd
            or ceiling_uses_hssd
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.hssd_retrieval_server
        # Get HSSD data configuration from asset manager config.
        hssd_config = self.cfg.furniture_agent.asset_manager.hssd

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting HSSD retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.hssd_server = HssdRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            hssd_data_path=str(hssd_config.data_path),
            hssd_preprocessed_path=str(hssd_config.preprocessed_path),
            hssd_top_k=hssd_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.hssd_server.start()
        # Longer timeout for CLIP loading.
        self.hssd_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("HSSD retrieval server ready")

    def _stop_hssd_server(self) -> None:
        """Stop the HSSD retrieval server."""
        if self.hssd_server and self.hssd_server.is_running():
            console_logger.info("Stopping HSSD retrieval server...")
            self.hssd_server.stop()
            console_logger.info("HSSD retrieval server stopped")
            self.hssd_server = None

    def _start_objaverse_server(self) -> None:
        """Start Objaverse retrieval server (if general_asset_source == 'objaverse')."""
        # Only start if at least one agent uses objaverse strategy.
        furniture_uses_objaverse = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "objaverse"
        )
        manipuland_uses_objaverse = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "objaverse"
        )
        wall_uses_objaverse = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "objaverse"
        )
        ceiling_uses_objaverse = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "objaverse"
        )

        if not (
            furniture_uses_objaverse
            or manipuland_uses_objaverse
            or wall_uses_objaverse
            or ceiling_uses_objaverse
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.objaverse_retrieval_server
        # Get Objaverse data configuration from asset manager config.
        objaverse_config = self.cfg.furniture_agent.asset_manager.objaverse

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting Objaverse retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.objaverse_server = ObjaverseRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,
            objaverse_data_path=str(objaverse_config.data_path),
            objaverse_preprocessed_path=str(objaverse_config.preprocessed_path),
            objaverse_top_k=objaverse_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.objaverse_server.start()
        # Longer timeout for CLIP loading.
        self.objaverse_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Objaverse retrieval server ready")

    def _stop_objaverse_server(self) -> None:
        """Stop the Objaverse retrieval server."""
        if self.objaverse_server and self.objaverse_server.is_running():
            console_logger.info("Stopping Objaverse retrieval server...")
            self.objaverse_server.stop()
            console_logger.info("Objaverse retrieval server stopped")
            self.objaverse_server = None

    def _start_articulated_server(self) -> None:
        """Start articulated retrieval server (if articulated strategy is enabled)."""
        # Check if articulated strategy is enabled for any agent.
        furniture_articulated_enabled = (
            self.cfg.furniture_agent.asset_manager.router.strategies.articulated.enabled
        )
        manipuland_articulated_enabled = (
            self.cfg.manipuland_agent.asset_manager.router.strategies.articulated.enabled
        )
        wall_articulated_enabled = (
            self.cfg.wall_agent.asset_manager.router.strategies.articulated.enabled
        )
        ceiling_articulated_enabled = (
            self.cfg.ceiling_agent.asset_manager.router.strategies.articulated.enabled
        )

        if not (
            furniture_articulated_enabled
            or manipuland_articulated_enabled
            or wall_articulated_enabled
            or ceiling_articulated_enabled
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.articulated_retrieval_server

        # Get articulated data configuration from furniture agent config.
        articulated_config = self.cfg.furniture_agent.asset_manager.articulated

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting articulated retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.articulated_server = ArticulatedRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            articulated_config=articulated_config,
            clip_device=retrieval_device,
        )

        self.articulated_server.start()
        # Longer timeout for CLIP loading.
        self.articulated_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Articulated retrieval server ready")

    def _stop_articulated_server(self) -> None:
        """Stop the articulated retrieval server."""
        if self.articulated_server and self.articulated_server.is_running():
            console_logger.info("Stopping articulated retrieval server...")
            self.articulated_server.stop()
            console_logger.info("Articulated retrieval server stopped")
            self.articulated_server = None

    def _start_materials_server(self) -> None:
        """Start materials retrieval server."""
        materials_enabled_paths = (
            "floor_plan_agent.materials.use_retrieval_server",
            "furniture_agent.asset_manager.router.strategies.thin_covering.enabled",
            "manipuland_agent.asset_manager.router.strategies.thin_covering.enabled",
            "wall_agent.asset_manager.router.strategies.thin_covering.enabled",
            "ceiling_agent.asset_manager.router.strategies.thin_covering.enabled",
        )
        if not any(
            _get_config_bool(self.cfg, path) for path in materials_enabled_paths
        ):
            console_logger.info(
                "Materials retrieval disabled by config; skipping materials server"
            )
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.materials_retrieval_server

        retrieval_device = _get_retrieval_gpu_device()
        console_logger.info(
            f"Starting materials retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device or 'default'})"
        )

        self.materials_server = MaterialsRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            materials_config=server_config,  # Pass DictConfig directly.
            clip_device=retrieval_device,
        )

        self.materials_server.start()
        # Longer timeout for CLIP loading.
        self.materials_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Materials retrieval server ready")

    def _stop_materials_server(self) -> None:
        """Stop the materials retrieval server."""
        if self.materials_server and self.materials_server.is_running():
            console_logger.info("Stopping materials retrieval server...")
            self.materials_server.stop()
            console_logger.info("Materials retrieval server stopped")
            self.materials_server = None

    @staticmethod
    def _generate_single_scene(
        prompt: str,
        scene_id: int,
        output_dir: Path,
        cfg_dict: dict,
        capture_logs: bool = False,
        experiment_run_id: str | None = None,
        render_gpu_id: int | None = None,
        attempt: int = 1,
    ) -> None:
        """Generate a single scene (static method for parallel execution).

        Pipeline stages run in order:
        floor_plan → furniture → wall_mounted → ceiling_mounted → manipulands
        Use config pipeline.start_stage and pipeline.stop_stage to control execution.

        Args:
            prompt: Scene description.
            scene_id: Scene identifier.
            output_dir: Base output directory for the experiment.
            cfg_dict: Configuration as dictionary.
            capture_logs: If True, suppress stdout and only write to file.
            experiment_run_id: Unique ID for this experiment run.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.
            attempt: One-based clean-process attempt number for this scene.
        """
        # Reset any SDK state inherited via fork (defense in depth).
        _reset_inherited_sdk_state()

        faulthandler.enable()

        scene_generation_start_time = time.time()

        # Create scene directory.
        scene_dir = output_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / _SCENE_SUCCESS_MARKER).unlink(missing_ok=True)
        _write_scene_status(
            output_dir=output_dir,
            scene_id=scene_id,
            prompt=prompt,
            status="running",
            attempt=attempt,
        )

        # Always create log file.
        log_path = scene_dir / "scene.log"

        # Log start message before potential suppression.
        if capture_logs:
            console_logger.info(
                f"Scene {scene_id:03d} started (logs → {log_path})\n"
                f"Prompt: {prompt}"
            )
        else:
            console_logger.info(
                f"Scene {scene_id:03d} started (debug mode)\nPrompt: {prompt}"
            )

        # Create a logger for this scene.
        logger = ConsoleLogger(output_dir=scene_dir)
        scene_expert_hooks = None

        # Get pipeline stage configuration.
        pipeline_cfg = cfg_dict["experiment"]["pipeline"]
        start_stage = pipeline_cfg["start_stage"]
        stop_stage = pipeline_cfg["stop_stage"]

        # Validate stages.
        if start_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid start_stage '{start_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )
        if stop_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid stop_stage '{stop_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )

        start_idx = PIPELINE_STAGES.index(start_stage)
        stop_idx = PIPELINE_STAGES.index(stop_stage)
        if start_idx > stop_idx:
            raise ValueError(
                f"start_stage '{start_stage}' cannot be after stop_stage '{stop_stage}'"
            )

        console_logger.info(
            f"Pipeline: start_stage='{start_stage}', stop_stage='{stop_stage}'"
        )

        # Handle resume from checkpoint if resume_from_path is specified.
        resume_from_path = pipeline_cfg.get("resume_from_path")
        if resume_from_path and start_stage != "floor_plan":
            source_experiment_dir = Path(resume_from_path)
            if not source_experiment_dir.exists():
                raise FileNotFoundError(
                    f"resume_from_path does not exist: {resume_from_path}"
                )
            _copy_checkpoint_for_stage(
                source_scene_dir=source_experiment_dir / f"scene_{scene_id:03d}",
                target_scene_dir=scene_dir,
                start_stage=start_stage,
            )

        with FileLoggingContext(log_file_path=log_path, suppress_stdout=capture_logs):
            try:
                # Create trace metadata for this scene.
                trace_metadata = {
                    "scene_id": f"scene_{scene_id:03d}",
                    "experiment_name": cfg_dict["name"],
                    "scene_dir": str(scene_dir),
                    "prompt": prompt,
                }
                if experiment_run_id:
                    trace_metadata["experiment_run_id"] = experiment_run_id

                console_logger.info(f"Generating scene for prompt: {prompt}")

                # Single trace wraps entire scene generation (floor plan + rooms).
                with trace(
                    workflow_name=f"scene_{scene_id:03d}_generation",
                    metadata=trace_metadata,
                ):
                    # Build SceneExpert hook runner before floor_plan so fast
                    # memory and StageBrief can guide the house-level layout too.
                    from scenesmith.scene_expert.hooks import build_hook_runner

                    scene_expert_hooks = build_hook_runner(
                        prompt=prompt,
                        scene_id=scene_id,
                        output_dir=output_dir,
                        cfg_dict=cfg_dict,
                    )
                    floor_plan_prompt = prompt
                    if scene_expert_hooks and start_stage == "floor_plan":
                        floor_plan_prompt = scene_expert_hooks.pre_floor_plan()

                    # Stage 1: Floor plan generation (or load from saved state).
                    if start_stage == "floor_plan":
                        # Run floor plan in subprocess to isolate fork-unsafe SDK
                        # state (SQLiteSession locks, tracing threads). The subprocess
                        # saves results to disk and exits cleanly before we fork room
                        # workers.
                        console_logger.info(
                            "Generating house layout (in isolated subprocess)"
                        )
                        layout_start_time = time.time()

                        # Run floor plan generation in isolated subprocess.
                        results = run_parallel_isolated(
                            tasks=[
                                (
                                    "floor_plan",
                                    _generate_floor_plan_worker,
                                    {
                                        "prompt": floor_plan_prompt,
                                        "scene_dir": str(scene_dir),
                                        "cfg_dict": cfg_dict,
                                        "experiment_run_id": experiment_run_id,
                                        "render_gpu_id": render_gpu_id,
                                    },
                                )
                            ],
                            max_workers=1,
                        )

                        # Check for failure.
                        success, error = results["floor_plan"]
                        if not success:
                            raise RuntimeError(f"Floor plan generation failed: {error}")

                        # Load result from disk (subprocess saved it).
                        house_layout_path = scene_dir / "house_layout.json"
                        with open(house_layout_path) as f:
                            house_layout_dict = json.load(f)
                        house_layout = HouseLayout.from_dict(
                            house_layout_dict, house_dir=scene_dir
                        )

                        layout_end_time = time.time()
                        console_logger.info(
                            f"House layout generated in "
                            f"{timedelta(seconds=layout_end_time - layout_start_time)}"
                        )
                        if scene_expert_hooks:
                            scene_expert_hooks.post_floor_plan(scene_dir)
                    else:
                        # Load house layout from saved state.
                        house_layout_path = scene_dir / "house_layout.json"
                        if not house_layout_path.exists():
                            raise FileNotFoundError(
                                f"Cannot start from '{start_stage}' stage: "
                                f"house_layout.json not found at {house_layout_path}. "
                                "Run with start_stage='floor_plan' first."
                            )
                        console_logger.info(
                            f"Loading house layout from {house_layout_path}"
                        )
                        with open(house_layout_path) as f:
                            house_layout_dict = json.load(f)
                        house_layout = HouseLayout.from_dict(
                            house_layout_dict, house_dir=scene_dir
                        )

                    # Check if we should stop after floor_plan stage.
                    if stop_stage == "floor_plan":
                        console_logger.info(
                            "Stopping after floor_plan stage as configured"
                        )
                        if scene_expert_hooks:
                            scene_expert_hooks.finalize(final_scene_path=str(scene_dir))
                        console_logger.info(
                            "Scene generation completed successfully in "
                            f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                        )
                        _write_scene_status(
                            output_dir=output_dir,
                            scene_id=scene_id,
                            prompt=prompt,
                            status="completed",
                            attempt=attempt,
                        )
                        (scene_dir / _SCENE_SUCCESS_MARKER).write_text(
                            "completed\n", encoding="utf-8"
                        )
                        return

                    # Stages 2-4: Furniture, wall objects, and manipulands (per-room).
                    # Determine room-level start/stop stages.
                    room_start_stage = (
                        "furniture" if start_stage == "floor_plan" else start_stage
                    )
                    room_stop_stage = stop_stage

                    # Generate rooms (parallel or sequential based on config).
                    parallel_rooms = pipeline_cfg["parallel_rooms"]
                    max_parallel_rooms = pipeline_cfg["max_parallel_rooms"]
                    num_rooms = len(house_layout.room_ids)

                    # Only use parallel if enabled, max_workers > 1, and multiple rooms.
                    use_parallel = (
                        parallel_rooms and max_parallel_rooms > 1 and num_rooms > 1
                    )
                    if scene_expert_hooks and use_parallel:
                        console_logger.warning(
                            "SceneExpert hooks are per-scene and not thread-safe for "
                            "parallel room generation; disabling parallel_rooms for "
                            "this scene."
                        )
                        use_parallel = False

                    if use_parallel:
                        rooms = _run_parallel_room_generation(
                            house_layout=house_layout,
                            output_dir=scene_dir,
                            cfg_dict=cfg_dict,
                            start_stage=room_start_stage,
                            stop_stage=room_stop_stage,
                            max_workers=max_parallel_rooms,
                            scene_id=scene_id,
                            experiment_run_id=experiment_run_id,
                            render_gpu_id=render_gpu_id,
                        )
                    else:
                        rooms = _run_sequential_room_generation(
                            house_layout=house_layout,
                            logger=logger,
                            cfg_dict=cfg_dict,
                            start_stage=room_start_stage,
                            stop_stage=room_stop_stage,
                            render_gpu_id=render_gpu_id,
                            scene_expert_hooks=scene_expert_hooks,
                        )

                    # Build HouseScene from generated rooms.
                    house_scene = HouseScene(layout=house_layout, rooms=rooms)

                    # SceneExpert: finalize trace + memory update after all rooms done.
                    if scene_expert_hooks:
                        scene_expert_hooks.finalize(
                            final_scene_path=str(scene_dir / "combined_house")
                        )

                    # Assemble house with intermediate snapshots filtered by object type.
                    # Each snapshot includes objects from completed stages only.
                    # Note: Thin coverings keep their agent's object_type (FURNITURE,
                    # WALL_MOUNTED, MANIPULAND) so they're included automatically.
                    snapshots = [
                        ("combined_house_after_furniture", [ObjectType.FURNITURE]),
                        (
                            "combined_house_after_wall_objects",
                            [ObjectType.FURNITURE, ObjectType.WALL_MOUNTED],
                        ),
                        (
                            "combined_house_after_ceiling",
                            [
                                ObjectType.FURNITURE,
                                ObjectType.WALL_MOUNTED,
                                ObjectType.CEILING_MOUNTED,
                            ],
                        ),
                        ("combined_house", None),  # Final: all objects.
                    ]

                    # Map stop_stage to number of snapshots to create.
                    stage_to_count = {
                        "furniture": 1,
                        AgentType.WALL_MOUNTED.value: 2,
                        AgentType.CEILING_MOUNTED.value: 3,
                    }
                    snapshot_count = stage_to_count.get(stop_stage, len(snapshots))

                    for name, types in snapshots[:snapshot_count]:
                        house_scene.assemble(
                            cfg=cfg_dict, output_name=name, include_object_types=types
                        )

                    console_logger.info(
                        "Scene generation completed successfully in "
                        f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                    )

            except Exception as e:
                if scene_expert_hooks:
                    scene_expert_hooks.save_partial_trace(error=str(e))
                _write_scene_status(
                    output_dir=output_dir,
                    scene_id=scene_id,
                    prompt=prompt,
                    status="failed",
                    attempt=attempt,
                    error=str(e),
                )
                console_logger.error(f"Scene generation failed: {e}")
                raise

        _write_scene_status(
            output_dir=output_dir,
            scene_id=scene_id,
            prompt=prompt,
            status="completed",
            attempt=attempt,
        )
        (scene_dir / _SCENE_SUCCESS_MARKER).write_text("completed\n", encoding="utf-8")

    def _run_serial_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
    ) -> None:
        """Run scenes in YAML order, each in a fresh isolated process."""
        console_logger.info(
            "Running scene generation serially with per-scene process isolation"
        )
        failed_scenes: list[tuple[int, str]] = []
        for scene_id, prompt in prompts_with_ids:
            try:
                self._run_isolated_scene_generation(
                    prompts_with_ids=[(scene_id, prompt)],
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                    num_workers=1,
                    capture_logs=False,
                )
            except RuntimeError as error:
                console_logger.error(
                    f"Scene {scene_id:03d} failed; continuing serial batch: {error}"
                )
                failed_scenes.append((scene_id, str(error)))

        if failed_scenes:
            failure_details = "\n".join(
                f"  - scene_{scene_id:03d}: {error}"
                for scene_id, error in failed_scenes
            )
            raise RuntimeError(
                f"{len(failed_scenes)}/{len(prompts_with_ids)} scene(s) failed:\n"
                f"{failure_details}"
            )

    def _run_parallel_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
        num_workers: int,
    ) -> None:
        """Run scene generation in parallel with fault tolerance.

        Uses isolated processes per scene instead of a shared executor pool.
        This ensures that if one scene crashes (e.g., GPU OOM), other scenes
        continue running unaffected.

        Raises:
            RuntimeError: If any scene generation fails.
        """
        console_logger.info(
            f"Running scene generation with {num_workers} isolated workers"
        )
        self._run_isolated_scene_generation(
            prompts_with_ids=prompts_with_ids,
            cfg_dict=cfg_dict,
            experiment_run_id=experiment_run_id,
            num_workers=num_workers,
            capture_logs=True,
        )

    def _run_isolated_scene_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
        num_workers: int,
        capture_logs: bool,
    ) -> None:
        """Run complete scene tasks with clean-process retry semantics."""
        retry_budget = max(
            0, int(cfg_dict["experiment"].get("scene_retry_attempts", 1))
        )

        gpu_allocator = RenderGPUAllocator()
        pending: dict[str, tuple[int, str, int]] = {}
        for scene_id, prompt in prompts_with_ids:
            render_gpu_id = gpu_allocator.allocate()
            task_id = f"scene_{scene_id:03d}"
            pending[task_id] = (scene_id, prompt, render_gpu_id)
            console_logger.info(f"Queued {task_id} (GPU {render_gpu_id}): {prompt}")

        final_results: dict[str, tuple[bool, str | None]] = {}
        attempt = 1
        while pending:
            tasks: list[tuple[str, Callable, dict]] = []
            for task_id, (scene_id, prompt, render_gpu_id) in pending.items():
                kwargs = {
                    "prompt": prompt,
                    "scene_id": scene_id,
                    "output_dir": self.output_dir,
                    "cfg_dict": cfg_dict,
                    "capture_logs": capture_logs,
                    "experiment_run_id": experiment_run_id,
                    "render_gpu_id": render_gpu_id,
                    "attempt": attempt,
                }
                tasks.append(
                    (
                        task_id,
                        IndoorSceneGenerationExperiment._generate_single_scene,
                        kwargs,
                    )
                )

            results = run_parallel_isolated(tasks=tasks, max_workers=num_workers)
            retry_pending: dict[str, tuple[int, str, int]] = {}
            for task_id, metadata in pending.items():
                scene_id, prompt, _ = metadata
                success, result_or_error = results[task_id]
                if success:
                    final_results[task_id] = (True, None)
                    console_logger.info(f"Completed {task_id} on attempt {attempt}")
                    continue

                error = str(result_or_error)
                _write_scene_status(
                    output_dir=self.output_dir,
                    scene_id=scene_id,
                    prompt=prompt,
                    status="failed",
                    attempt=attempt,
                    error=error[-8000:],
                )
                can_retry = attempt <= retry_budget and _is_retryable_scene_failure(
                    error
                )
                if can_retry:
                    archive_path = _archive_failed_scene_attempt(
                        output_dir=self.output_dir,
                        scene_id=scene_id,
                        attempt=attempt,
                    )
                    console_logger.warning(
                        f"{task_id} failed with a transient/native error; "
                        f"retrying in a fresh process ({attempt}/{retry_budget}). "
                        f"Partial output archived at {archive_path}"
                    )
                    retry_pending[task_id] = metadata
                else:
                    final_results[task_id] = (False, error)
                    console_logger.error(
                        f"{task_id} failed permanently after attempt {attempt}: "
                        f"{_root_error_summary(error)}"
                    )

            pending = retry_pending
            attempt += 1

        _write_batch_summary(
            output_dir=self.output_dir,
            experiment_run_id=experiment_run_id,
            prompts_with_ids=prompts_with_ids,
            results=final_results,
        )

        failed_scenes = [
            (task_id, error)
            for task_id, (success, error) in final_results.items()
            if not success
        ]
        if failed_scenes:
            failure_details = "\n".join(
                f"  - {task_id}: {_root_error_summary(str(error))}"
                for task_id, error in failed_scenes
            )
            raise RuntimeError(
                f"{len(failed_scenes)}/{len(prompts_with_ids)} scene(s) failed:\n"
                f"{failure_details}"
            )

    def generate_scenes(self) -> None:
        """Generate scenes with parallel support."""
        # Load prompts from CSV or YAML config.
        csv_path = self.cfg.experiment.csv_path
        if csv_path:
            prompts_with_ids = _load_prompts_from_csv(csv_path)
            console_logger.info(
                f"Loaded {len(prompts_with_ids)} prompts from CSV: {csv_path}"
            )
        else:
            prompts = self.cfg.experiment.prompts
            prompts_with_ids = list(enumerate(prompts))

        num_workers = min(self.cfg.experiment.num_workers, len(prompts_with_ids))

        # Online memory banks and their numpy indexes are shared across scenes.
        # Until writes use per-scene deltas followed by a parent-side merge,
        # concurrent harness_memory/full scenes can interleave JSONL updates or
        # retrieve a stale index. Preserve correctness by serializing those
        # modes; disabled/harness_only experiments remain scene-parallel.
        scene_expert_mode = OmegaConf.select(
            self.cfg, "scene_expert.mode", default="disabled"
        )
        if num_workers > 1 and scene_expert_mode in {"harness_memory", "full"}:
            console_logger.warning(
                f"scene_expert.mode={scene_expert_mode!r} uses a shared online "
                "memory bank; forcing experiment.num_workers=1"
            )
            num_workers = 1

        # Get pipeline stage configuration.
        pipeline_cfg = self.cfg.experiment.pipeline
        start_stage = pipeline_cfg.start_stage
        stop_stage = pipeline_cfg.stop_stage
        parallel_rooms = pipeline_cfg.parallel_rooms

        # Validate mutual exclusion: parallel scenes vs parallel rooms.
        if parallel_rooms and num_workers > 1:
            raise ValueError(
                "Cannot use both parallel rooms and parallel scenes. "
                "Set num_workers=1 to use parallel_rooms, or set parallel_rooms=false."
            )

        # Generate experiment run ID for trace filtering.
        experiment_run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        console_logger.info(f"Starting scene generation with {num_workers} workers")
        console_logger.info(f"Processing {len(prompts_with_ids)} scenes")
        console_logger.info(f"Experiment run ID: {experiment_run_id}")
        console_logger.info(
            f"Pipeline stages: start='{start_stage}', stop='{stop_stage}'"
        )

        # Convert config to dictionary for static method.
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)

        try:
            # Start GPU servers (CUDA init happens here).
            self._start_geometry_server()
            self._start_hssd_server()
            self._start_objaverse_server()
            self._start_articulated_server()
            self._start_materials_server()

            if num_workers == 1:
                self._run_serial_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                )
            else:
                self._run_parallel_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                    num_workers=num_workers,
                )

            console_logger.info("All scenes completed")

            # Log clear completion message.
            console_logger.info("=" * 60)
            console_logger.info(bold_green("ALL SCENES COMPLETED!"))
            console_logger.info("=" * 60)
            console_logger.info(yellow("Press Ctrl+C to exit the script."))
            console_logger.info("=" * 60)

        finally:
            # Stop GPU servers.
            self._stop_materials_server()
            self._stop_articulated_server()
            self._stop_objaverse_server()
            self._stop_hssd_server()
            self._stop_geometry_server()

    def evaluate_scenes(self) -> None:
        """
        Evaluate previously generated scenes.
        """
        raise NotImplementedError
