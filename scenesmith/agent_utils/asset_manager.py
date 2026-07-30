import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import RigidTransform

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
)
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.asset_router.dataclasses import (
    ArticulatedGeometry,
    AssetItem,
    GeneratedGeometry,
    ModificationInfo,
    ValidationResult,
)
from scenesmith.agent_utils.asset_runtime import (
    ASSET_SEMANTIC_CONTRACT_VERSION,
    AssetRuntimeGate,
    semantic_asset_family,
)
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenesmith.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalResult,
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.image_generation import (
    AssetOperationType,
    create_image_generator,
)
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalClient
from scenesmith.agent_utils.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.mesh_frame import (
    ASSET_FRAME_CONTRACT_VERSION,
    CANONICAL_DIMENSION_ORDER,
    CANONICAL_FRONT_AXIS,
    CANONICAL_UP_AXIS,
    axis_agnostic_uniform_fit_exists,
    choose_uniform_scale_for_contract,
    gltf_y_up_bounds_to_scene_z_up,
    hssd_dimension_shape_error,
    scene_dimensions_to_gltf_y_up,
    uniform_scale_shape_error,
    validate_uniform_dimension_fit,
)
from scenesmith.agent_utils.mesh_physics_analyzer import (
    MeshPhysicsAnalysis,
    analyze_mesh_orientation_and_material,
    build_deterministic_hssd_physics,
    get_front_axis_from_image_number,
)
from scenesmith.agent_utils.mesh_utils import (
    load_mesh_as_trimesh,
    remove_mesh_floaters,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalClient
from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
    ObjaverseRetrievalServerRequest,
)
from scenesmith.agent_utils.retrieval_errors import FatalRetrievalError
from scenesmith.agent_utils.room import AgentType, ObjectType, SceneObject, UniqueID
from scenesmith.agent_utils.sdf_generator import (
    add_self_collision_filter,
    generate_drake_sdf,
)
from scenesmith.agent_utils.sdf_mesh_utils import combine_sdf_meshes_at_joint_angles
from scenesmith.agent_utils.thin_covering_generator import (
    generate_thin_covering_sdf,
    infer_thin_covering_shape,
)
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.utils.logging import BaseLogger

if TYPE_CHECKING:
    from scenesmith.agent_utils.asset_router import AssetRouter
    from scenesmith.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)

_RETRYABLE_ASSET_VALIDATION_FAILURES = frozenset(
    {"rendering", "length", "invalid_json", "infrastructure"}
)


def _asset_validation_is_retryable(validation: ValidationResult) -> bool:
    """Classify transport/render/format failures separately from semantic rejection."""
    failure_kind = str(getattr(validation, "failure_kind", "") or "")
    if failure_kind in _RETRYABLE_ASSET_VALIDATION_FAILURES:
        return True
    # Backward compatibility for cached/test results written before failure_kind
    # became part of the validation contract.
    reason = str(validation.reason or "")
    return reason.startswith(("Rendering failed", "Validation call failed"))


def _enforce_critical_hssd_validation_contract(
    validation: ValidationResult,
    *,
    critical_family: bool,
) -> ValidationResult:
    """Require explicit standalone evidence before admitting critical assets."""
    if not validation.is_acceptable or not critical_family:
        return validation
    if (
        validation.contains_architectural_context is False
        and validation.requested_object_is_dominant is True
    ):
        return validation
    return ValidationResult(
        is_acceptable=False,
        reason=(
            "Critical HSSD admission response omitted explicit standalone asset "
            "evidence; both contains_architectural_context=false and "
            "requested_object_is_dominant=true are required"
        ),
        suggestions=[
            "Retry semantic validation or retrieve another standalone candidate"
        ],
        front_view_image_index=validation.front_view_image_index,
        orientation_confidence=validation.orientation_confidence,
        contains_architectural_context=validation.contains_architectural_context,
        requested_object_is_dominant=validation.requested_object_is_dominant,
        failure_kind="invalid_contract",
    )


@dataclass
class AssetPathConfig:
    """Configuration for asset file paths and metadata."""

    description: str
    """Description of the object."""

    short_name: str
    """Short name for the object."""

    image_path: Path | None
    """Path to the generated image."""

    geometry_path: Path
    """Path to the generated 3D geometry."""

    sdf_dir: Path
    """Directory containing the generated SDF file."""


@dataclass
class AssetGenerationRequest:
    """Request for generating scene assets (furniture, manipulands, etc.)."""

    object_descriptions: list[str]
    """List of object descriptions to generate."""

    short_names: list[str]
    """List of short names for filesystem-safe file naming."""

    object_type: ObjectType
    """Type of objects to generate (FURNITURE, MANIPULAND, etc.)."""

    desired_dimensions: list[list[float]]
    """Desired dimensions (width, depth, height) in meters for each object.
    Agent must predict dimensions considering scene context.
    Must match the length of object_descriptions.
    """

    style_context: str | None = None
    """Style context for consistency (e.g., 'modern minimalist kitchen')."""

    operation_type: AssetOperationType = AssetOperationType.INITIAL
    """Type of generation operation."""

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling on servers.

    When multiple scenes generate assets concurrently, passing scene_id ensures
    fair GPU time allocation across scenes in the geometry and HSSD servers.
    """


def _align_hssd_request_dimensions(
    request: AssetGenerationRequest,
) -> AssetGenerationRequest:
    """Pad omitted optional HSSD dimensions instead of indexing past the request."""
    if len(request.desired_dimensions) == len(request.object_descriptions):
        return request
    if len(request.desired_dimensions) > len(request.object_descriptions):
        raise ValueError(
            "HSSD request has more dimension entries "
            f"({len(request.desired_dimensions)}) than object descriptions "
            f"({len(request.object_descriptions)})"
        )
    original_count = len(request.desired_dimensions)
    aligned_dimensions = [
        (
            list(request.desired_dimensions[index])
            if index < original_count and request.desired_dimensions[index] is not None
            else []
        )
        for index in range(len(request.object_descriptions))
    ]
    console_logger.warning(
        "Aligned %d HSSD descriptions with %d dimension entries; missing optional "
        "dimensions will use semantic retrieval without a size hint",
        len(request.object_descriptions),
        original_count,
    )
    return AssetGenerationRequest(
        object_descriptions=list(request.object_descriptions),
        short_names=list(request.short_names),
        object_type=request.object_type,
        desired_dimensions=aligned_dimensions,
        style_context=request.style_context,
        operation_type=request.operation_type,
        scene_id=request.scene_id,
    )


def _optional_hssd_dimension_contract(
    dimensions: list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """Normalize an omitted size hint while rejecting malformed dimensions."""
    if dimensions is None or len(dimensions) == 0:
        return None
    if len(dimensions) != 3:
        raise ValueError(
            "HSSD desired dimensions must be omitted or contain exactly "
            f"[width, depth, height]; received {list(dimensions)}"
        )
    return [float(value) for value in dimensions]


@dataclass
class FailedAsset:
    """Information about a failed asset generation."""

    index: int
    """Index of the failed asset in the original request."""

    description: str
    """Description of the object that failed to generate."""

    error_message: str
    """Error message describing why generation failed."""


@dataclass
class AssetGenerationResult:
    """Result of asset generation with potential partial success."""

    successful_assets: list[SceneObject]
    """List of successfully generated scene objects."""

    failed_assets: list[FailedAsset]
    """List of assets that failed during generation."""

    modification_info: ModificationInfo | None = None
    """Set when router modified the original request (split composites or filtered
    items). Contains original description, resulting items, and any discarded
    manipulands (furniture agent only). None when router is disabled or request
    was not modified.
    """

    @property
    def has_failures(self) -> bool:
        """Check if any assets failed to generate."""
        return len(self.failed_assets) > 0

    @property
    def all_succeeded(self) -> bool:
        """Check if all assets were generated successfully."""
        return len(self.failed_assets) == 0


class AssetManager:
    """Manages 3D asset acquisition for scene generation.

    Supports two acquisition strategies configured via `general_asset_source`:
    - "generated": Text-to-3D generation (text → image → 3D mesh)
    - "hssd": Retrieval from HSSD library

    Has two operating modes based on `router.enabled` config:

    **Router path** (router.enabled=True):
    - LLM analyzes requests to split composites and select strategies
    - Parallel HTTP calls for generation/retrieval (thread-safe)
    - Sequential bpy operations for mesh processing (main thread)
    - VLM validation with retry loop for quality control

    **Non-router path** (router.enabled=False):
    - Direct dispatch to generation or retrieval based on config
    - Batch processing without LLM analysis
    - Simpler but less flexible

    Both paths produce simulation-ready Drake SDF files with:
    - Canonical orientation (Z-up, Y-forward)
    - Convex decomposition collision geometry (CoACD or V-HACD)
    - VLM-estimated physics properties (material, mass)

    Maintains style consistency through conversational context and includes
    an asset registry to track generated assets for reuse.
    """

    def __init__(
        self,
        logger: BaseLogger,
        vlm_service: VLMService,
        blender_server: "BlenderServer | None",
        collision_client: ConvexDecompositionClient | None,
        cfg: DictConfig,
        agent_type: AgentType,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        objaverse_server_host: str = "127.0.0.1",
        objaverse_server_port: int = 7009,
    ) -> None:
        """Initialize the asset manager.

        Args:
            logger: Logger instance for tracking operations.
            vlm_service: VLM service instance for mesh physics analysis.
            blender_server: Blender server instance for multi-view rendering.
            collision_client: Client for collision geometry generation via convex
                decomposition. Can be None for checkpoint loading (no collision
                generation needed).
            cfg: Configuration with asset_manager settings.
            agent_type: Agent type for directory organization. Assets will be
                stored in generated_assets/{agent_type.value}/.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
            articulated_server_host: Host for articulated retrieval server.
            articulated_server_port: Port for articulated retrieval server.
            materials_server_host: Host for materials retrieval server.
            materials_server_port: Port for materials retrieval server.
            objaverse_server_host: Host for Objaverse retrieval server.
            objaverse_server_port: Port for Objaverse retrieval server.
        """
        self.output_dir = logger.output_dir
        self.logger = logger
        self.cfg = cfg
        self.agent_type = agent_type

        # Extract config values.
        self.num_side_views_for_physics_analysis = (
            cfg.asset_manager.num_side_views_for_physics_analysis
        )
        self.side_view_elevation_degrees = cfg.asset_manager.side_view_elevation_degrees
        self.min_mesh_dimension_meters = cfg.asset_manager.min_mesh_dimension_meters
        self.mesh_relative_dimension_threshold = (
            cfg.asset_manager.mesh_relative_dimension_threshold
        )
        # Store collision geometry configuration.
        self.collision_method = cfg.collision_geometry.method
        self.collision_coacd_cfg = cfg.collision_geometry.coacd
        self.collision_vhacd_cfg = cfg.collision_geometry.vhacd

        self.vlm_service = vlm_service
        self.blender_server = blender_server
        self.collision_client = collision_client
        self.image_generator = create_image_generator(
            backend=cfg.asset_manager.image_generation.backend,
            config=cfg.asset_manager.image_generation,
        )

        # Create agent-specific subdirectories for organization.
        generated_assets_dir = self.output_dir / "generated_assets" / agent_type.value
        self.images_dir = generated_assets_dir / "images"
        self.geometry_dir = generated_assets_dir / "geometry"
        self.sdf_dir = generated_assets_dir / "sdf"
        self.debug_dir = generated_assets_dir / "debug"

        for dir_path in [
            self.images_dir,
            self.geometry_dir,
            self.sdf_dir,
            self.debug_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry with auto-save to enable incremental persistence.
        registry_path = generated_assets_dir / "asset_registry.json"
        self.registry = AssetRegistry(auto_save_path=registry_path)

        # Initialize strategy-specific clients.
        self.general_asset_source = cfg.asset_manager.general_asset_source
        if self.general_asset_source not in ["generated", "hssd", "objaverse"]:
            raise ValueError(f"Unknown asset source: {self.general_asset_source}")

        # Initialize geometry generation client if source is "generated".
        self.geometry_client: GeometryGenerationClient | None = None
        if self.general_asset_source == "generated":
            console_logger.info("Initializing geometry generation client")
            self.geometry_client = GeometryGenerationClient(
                host=geometry_server_host, port=geometry_server_port
            )

        # Initialize HSSD client if source is "hssd".
        self.hssd_client: HssdRetrievalClient | None = None
        if self.general_asset_source == "hssd":
            console_logger.info("Initializing HSSD retrieval client")
            self.hssd_client = HssdRetrievalClient(
                host=hssd_server_host, port=hssd_server_port
            )

        # Initialize Objaverse client if source is "objaverse".
        self.objaverse_client: ObjaverseRetrievalClient | None = None
        if self.general_asset_source == "objaverse":
            console_logger.info("Initializing Objaverse retrieval client")
            self.objaverse_client = ObjaverseRetrievalClient(
                host=objaverse_server_host, port=objaverse_server_port
            )

        # Initialize articulated retrieval client if articulated strategy is enabled.
        self.articulated_client: ArticulatedRetrievalClient | None = None
        articulated_enabled = cfg.asset_manager.router.strategies.articulated.enabled
        if articulated_enabled:
            console_logger.info("Initializing articulated retrieval client")
            self.articulated_client = ArticulatedRetrievalClient(
                host=articulated_server_host, port=articulated_server_port
            )

        # Initialize materials retrieval client if thin_covering strategy is enabled.
        self.materials_client: MaterialsRetrievalClient | None = None
        thin_covering_enabled = (
            cfg.asset_manager.router.strategies.thin_covering.enabled
        )
        thin_covering_cfg = cfg.asset_manager.router.strategies.thin_covering
        procedural_floor_covering_enabled = bool(
            getattr(thin_covering_cfg, "procedural_fallback_enabled", True)
        )
        if thin_covering_enabled:
            console_logger.info("Initializing materials retrieval client")
            self.materials_client = MaterialsRetrievalClient(
                host=materials_server_host, port=materials_server_port
            )

        # Initialize asset router if enabled in config.
        self.router: "AssetRouter | None" = None
        if cfg.asset_manager.router.enabled:
            console_logger.info("Initializing asset router for LLM-advised generation")
            self.router = AssetRouter(
                agent_type=agent_type,
                vlm_service=vlm_service,
                cfg=cfg,
                blender_server=blender_server,
            )
        self._thin_covering_router = self.router
        if (
            (thin_covering_enabled or procedural_floor_covering_enabled)
            and self._thin_covering_router is None
            and agent_type == AgentType.FURNITURE
        ):
            if not thin_covering_enabled:
                console_logger.info(
                    "Initializing local procedural floor-covering router without "
                    "the materials retrieval service"
                )
            self._thin_covering_router = AssetRouter(
                agent_type=agent_type,
                vlm_service=vlm_service,
                cfg=cfg,
                blender_server=blender_server,
            )
        # Direct HSSD semantic admission is independent from request routing and
        # the thin-covering/materials service. ACP commonly disables both; tying
        # validation to either one silently admitted category-mismatched assets.
        semantic_validation_cfg = getattr(
            getattr(cfg.asset_manager, "hssd", None),
            "semantic_validation",
            None,
        )
        self._asset_validation_router = self.router or self._thin_covering_router
        if (
            self.general_asset_source == "hssd"
            and bool(getattr(semantic_validation_cfg, "enabled", False))
            and self._asset_validation_router is None
        ):
            console_logger.info(
                "Initializing independent HSSD semantic validation router"
            )
            self._asset_validation_router = AssetRouter(
                agent_type=agent_type,
                vlm_service=vlm_service,
                cfg=cfg,
                blender_server=blender_server,
            )

        # Track duplicate requests from the last generate_assets call.
        self.last_duplicate_info: dict[str, list[int]] | None = None
        self._fatal_asset_error: str | None = None
        self._runtime_gate = AssetRuntimeGate()
        self._collision_geometry_cache: dict[str, list[trimesh.Trimesh]] = {}
        self._collision_cache_lock = threading.Lock()
        # Reuse the direct HSSD semantic call as an orientation calibration call.
        # Cache semantic decisions by dataset ID + requested family, and retain
        # the selected asset's calibration by dataset ID for canonicalization.
        self._direct_hssd_validation_results: dict[str, ValidationResult] = {}
        self._direct_hssd_semantic_cache: dict[str, ValidationResult] = {}
        self._execution_clock: object | None = None
        # Native HSSD client default. SceneExpert may replace it only through
        # the explicit execution-control plane.
        self._asset_acquisition_timeout_seconds = 3600
        self._execution_control_enabled = False
        self._asset_validation_runtime: dict[str, object] = {}
        self._direct_hssd_admission_states: dict[str, str] = {}
        self._reuse_only = False

    def configure_runtime_budget(
        self,
        *,
        stage: str,
        budget: dict,
        required_objects: list[str],
        execution_clock: object | None = None,
    ) -> None:
        """Configure per-stage acquisition limits supplied by SceneExpert."""
        self._execution_clock = execution_clock
        self._execution_control_enabled = bool(
            budget.get("execution_control_enabled", bool(budget))
        )
        self._asset_validation_runtime = dict(budget)
        self._asset_acquisition_timeout_seconds = (
            max(
                30,
                int(budget.get("asset_acquisition_timeout_seconds", 300) or 300),
            )
            if self._execution_control_enabled
            else 3600
        )
        self._runtime_gate.configure(
            stage=stage,
            budget=budget,
            required_objects=required_objects,
        )
        console_logger.info(
            "Asset runtime budget configured for %s: required=%s, requests=%d, "
            "optional_families=%d, assets_per_request=%d, retries_per_family=%d",
            stage,
            sorted(self._runtime_gate.required_families),
            self._runtime_gate.max_asset_requests,
            self._runtime_gate.max_optional_families,
            self._runtime_gate.max_assets_per_request,
            self._runtime_gate.max_retries_per_family,
        )

    def set_reuse_only(self, enabled: bool) -> None:
        """Restrict a placement continuation to already admitted assets."""
        self._reuse_only = bool(enabled)

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 50) -> str:
        """Sanitize a name for use as a filename.

        Args:
            name: Name to sanitize.
            max_length: Maximum length for the filename.

        Returns:
            Filesystem-safe filename string.
        """
        # Replace problematic characters with underscores.
        sanitized = re.sub(r"[^\w\-_.]", "_", name)
        # Remove consecutive underscores.
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim to max length.
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip("_")
        return sanitized

    def _generate_collision_geometry(self, mesh_path: Path) -> list[trimesh.Trimesh]:
        """Generate collision geometry using the configured convex decomposition method.

        Args:
            mesh_path: Path to the mesh file (GLTF/GLB/OBJ).

        Returns:
            List of convex trimesh objects from the decomposition.

        Raises:
            RuntimeError: If collision client is not available.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        stat = mesh_path.stat()
        method_cfg = (
            self.collision_coacd_cfg
            if self.collision_method == "coacd"
            else self.collision_vhacd_cfg
        )
        cache_key = "|".join(
            (
                str(mesh_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                str(self.collision_method),
                repr(method_cfg),
            )
        )
        with self._collision_cache_lock:
            cached = self._collision_geometry_cache.get(cache_key)
        if cached is not None:
            console_logger.info(
                "Collision geometry cache hit for %s (%d piece(s))",
                mesh_path.name,
                len(cached),
            )
            return [piece.copy() for piece in cached]

        # Build parameter dict based on method.
        if self.collision_method == "coacd":
            pieces = self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="coacd",
                threshold=self.collision_coacd_cfg.threshold,
                max_convex_hull=self.collision_coacd_cfg.max_convex_hull,
                preprocess_mode=self.collision_coacd_cfg.preprocess_mode,
                preprocess_resolution=self.collision_coacd_cfg.preprocess_resolution,
                resolution=self.collision_coacd_cfg.resolution,
                mcts_nodes=self.collision_coacd_cfg.mcts_nodes,
                mcts_iterations=self.collision_coacd_cfg.mcts_iterations,
                mcts_max_depth=self.collision_coacd_cfg.mcts_max_depth,
                pca=self.collision_coacd_cfg.pca,
                merge=self.collision_coacd_cfg.merge,
                decimate=self.collision_coacd_cfg.decimate,
                max_ch_vertex=self.collision_coacd_cfg.max_ch_vertex,
                extrude=self.collision_coacd_cfg.extrude,
                extrude_margin=self.collision_coacd_cfg.extrude_margin,
                apx_mode=self.collision_coacd_cfg.apx_mode,
                seed=self.collision_coacd_cfg.seed,
            )
        else:
            # V-HACD method.
            pieces = self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="vhacd",
                max_convex_hulls=self.collision_vhacd_cfg.max_convex_hulls,
                vhacd_resolution=self.collision_vhacd_cfg.resolution,
                max_recursion_depth=self.collision_vhacd_cfg.max_recursion_depth,
                max_num_vertices_per_ch=self.collision_vhacd_cfg.max_num_vertices_per_ch,
                min_volume_percent_error=self.collision_vhacd_cfg.min_volume_percent_error,
                shrink_wrap=self.collision_vhacd_cfg.shrink_wrap,
                fill_mode=self.collision_vhacd_cfg.fill_mode,
                min_edge_length=self.collision_vhacd_cfg.min_edge_length,
                find_best_plane=self.collision_vhacd_cfg.find_best_plane,
            )
        with self._collision_cache_lock:
            self._collision_geometry_cache[cache_key] = [
                piece.copy() for piece in pieces
            ]
            while len(self._collision_geometry_cache) > 64:
                self._collision_geometry_cache.pop(
                    next(iter(self._collision_geometry_cache))
                )
        return pieces

    def _validate_sam3d_config(self) -> None:
        """Validate SAM3D configuration at startup.

        Raises:
            ValueError: If SAM3D configuration is invalid or missing required fields.
            FileNotFoundError: If checkpoint files do not exist.
        """
        if "sam3d" not in self.cfg.asset_manager:
            raise ValueError(
                "SAM3D backend selected but 'sam3d' configuration is missing. "
                "Add 'sam3d' section to asset_manager config."
            )

        sam3d_cfg = self.cfg.asset_manager.sam3d

        # Validate required checkpoint fields.
        required_fields = ["sam3_checkpoint", "sam3d_checkpoint"]
        for field in required_fields:
            if field not in sam3d_cfg:
                raise ValueError(f"SAM3D configuration missing required field: {field}")

        # Validate checkpoint files exist.
        sam3_checkpoint = Path(sam3d_cfg.sam3_checkpoint)
        sam3d_checkpoint = Path(sam3d_cfg.sam3d_checkpoint)

        if not sam3_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {sam3_checkpoint}. "
                f"Run 'bash scripts/install_sam3d.sh' to download checkpoints."
            )

        if not sam3d_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM 3D Objects checkpoint not found: {sam3d_checkpoint}. "
                f"Run 'bash scripts/install_sam3d.sh' to download checkpoints."
            )

        # Validate mode field.
        mode = sam3d_cfg.mode
        if mode not in ["foreground", "object_description"]:
            raise ValueError(
                f"Invalid SAM3D mode: {mode}. "
                "Must be 'foreground' or 'object_description'."
            )

        # Validate threshold.
        threshold = sam3d_cfg.threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Invalid SAM3D threshold: {threshold}. Must be between 0.0 and 1.0."
            )

        console_logger.info(
            f"SAM3D configuration validated successfully (mode={mode}, "
            f"threshold={threshold})"
        )

    def _analyze_mesh_physics(
        self,
        *,
        mesh_path: Path,
        asset_source: str,
        object_name: str,
        debug_output_dir: Path,
    ) -> MeshPhysicsAnalysis:
        """Use deterministic HSSD metadata unless explicitly configured for VLM."""
        is_hssd = asset_source == "hssd"
        hssd_mode = str(
            getattr(
                self.cfg.asset_manager,
                "hssd_physics_analysis_mode",
                "deterministic",
            )
        ).lower()
        if is_hssd and hssd_mode == "deterministic":
            physics = build_deterministic_hssd_physics(
                mesh_path, object_name=object_name
            )
            console_logger.info(
                "Using deterministic HSSD physics for %s: material=%s, mass=%.1fkg",
                object_name,
                physics.material,
                physics.mass_kg,
            )
            return physics

        return analyze_mesh_orientation_and_material(
            mesh_path=mesh_path,
            vlm_service=self.vlm_service,
            cfg=self.cfg,
            elevation_degrees=self.side_view_elevation_degrees,
            blender_server=self.blender_server,
            num_side_views=self.num_side_views_for_physics_analysis,
            prompt_type="hssd" if is_hssd else "generated",
            include_vertical_views=not is_hssd,
            debug_output_dir=debug_output_dir,
        )

    def _is_deterministic_floor_covering(
        self, description: str, short_name: str, object_type: ObjectType
    ) -> bool:
        """Return whether a non-router furniture request should bypass HSSD."""
        return (
            self.agent_type == AgentType.FURNITURE
            and object_type == ObjectType.FURNITURE
            and self._thin_covering_router is not None
            and semantic_asset_family(description, short_name) == "rug"
        )

    def _generate_deterministic_floor_covering(
        self, request: AssetGenerationRequest, index: int
    ) -> SceneObject:
        """Generate a correctly sized rug without router analysis or VLM validation."""
        if self._thin_covering_router is None:
            raise RuntimeError("Thin-covering strategy is not available")
        item = AssetItem(
            description=request.object_descriptions[index],
            short_name=request.short_names[index],
            dimensions=list(request.desired_dimensions[index]),
            object_type=request.object_type,
            strategies=["thin_covering"],
            thin_covering_type="tileable",
        )
        generated = (
            self._thin_covering_router.generate_thin_covering_without_validation(
                item,
                materials_client=self.materials_client,
                image_generator=self.image_generator,
                geometry_dir=self.geometry_dir,
                debug_dir=self.debug_dir,
                scene_id=request.scene_id,
            )
        )
        if generated is None:
            raise RuntimeError(
                f"No procedural material available for floor covering '{item.description}'"
            )
        return self._convert_generated_to_scene_object(
            item=item, generated=generated, request=request
        )

    def _hssd_semantic_validation_settings(
        self, description: str, short_name: str
    ) -> tuple[bool, int, bool, float, int, float]:
        """Resolve targeted direct-HSSD validation without enabling LLM routing."""
        hssd_cfg = getattr(self.cfg.asset_manager, "hssd", None)
        validation_cfg = getattr(hssd_cfg, "semantic_validation", None)
        if validation_cfg is None:
            return False, 1, False, 180.0, 0, 0.55
        try:
            enabled = bool(validation_cfg.get("enabled", False))
            families = {
                str(value).lower()
                for value in list(validation_cfg.get("families", []) or [])
            }
            max_candidates = max(1, int(validation_cfg.get("max_candidates", 2) or 2))
            use_lenient = bool(validation_cfg.get("use_lenient", False))
            timeout_seconds = float(
                validation_cfg.get("timeout_seconds", 180.0) or 180.0
            )
            max_retries = max(0, int(validation_cfg.get("max_retries", 0) or 0))
            min_orientation_confidence = float(
                validation_cfg.get("min_orientation_confidence", 0.55) or 0.55
            )
        except Exception:
            enabled = bool(getattr(validation_cfg, "enabled", False))
            families = {
                str(value).lower()
                for value in list(getattr(validation_cfg, "families", []) or [])
            }
            max_candidates = max(
                1, int(getattr(validation_cfg, "max_candidates", 2) or 2)
            )
            use_lenient = bool(getattr(validation_cfg, "use_lenient", False))
            timeout_seconds = float(
                getattr(validation_cfg, "timeout_seconds", 180.0) or 180.0
            )
            max_retries = max(0, int(getattr(validation_cfg, "max_retries", 0) or 0))
            min_orientation_confidence = float(
                getattr(validation_cfg, "min_orientation_confidence", 0.55) or 0.55
            )

        if bool(getattr(self, "_execution_control_enabled", False)):
            runtime = dict(getattr(self, "_asset_validation_runtime", {}) or {})
            max_candidates = max(
                1,
                min(
                    4,
                    int(
                        runtime.get(
                            "asset_validation_max_candidates",
                            max_candidates,
                        )
                        or max_candidates
                    ),
                ),
            )
            timeout_seconds = max(
                1.0,
                float(
                    runtime.get(
                        "asset_validation_timeout_seconds",
                        timeout_seconds,
                    )
                    or timeout_seconds
                ),
            )
            # Retries are scheduled once per family by the candidate
            # transaction; individual HTTP calls never own hidden retries.
            max_retries = 0
        else:
            timeout_seconds = float(
                getattr(
                    self.cfg.asset_manager,
                    "hssd_vlm_timeout_seconds",
                    timeout_seconds,
                )
                or timeout_seconds
            )
            max_retries = max(
                0,
                int(
                    getattr(
                        self.cfg.asset_manager,
                        "hssd_vlm_max_retries",
                        0,
                    )
                    or 0
                ),
            )

        family = semantic_asset_family(description, short_name)
        return (
            enabled and family in families,
            max_candidates,
            use_lenient,
            max(1.0, timeout_seconds),
            max_retries,
            max(0.0, min(1.0, min_orientation_confidence)),
        )

    def _hssd_validation_config_value(self, key: str, default: object) -> object:
        hssd_cfg = getattr(self.cfg.asset_manager, "hssd", None)
        validation_cfg = getattr(hssd_cfg, "semantic_validation", None)
        if validation_cfg is None:
            return default
        try:
            return validation_cfg.get(key, default)
        except Exception:
            return getattr(validation_cfg, key, default)

    def _is_critical_hssd_family(self, family: str) -> bool:
        """Return whether a family requires verified semantic admission.

        Runtime-required families extend the configured high-impact set. They
        must never replace it: a scene that explicitly requires a blanket still
        needs a selected bed or sofa to pass the configured critical contract.
        """
        configured = {
            str(value).lower()
            for value in list(
                self._hssd_validation_config_value("critical_families", []) or []
            )
        }
        required_families = set(
            getattr(getattr(self, "_runtime_gate", None), "required_families", set())
            or set()
        )
        return family in configured or family in required_families

    def _optional_hssd_candidate_is_ambiguous(
        self,
        candidates: list[HssdRetrievalResult],
        *,
        proportion_match_found: bool,
    ) -> bool:
        if not candidates or not proportion_match_found:
            return True
        minimum_score = float(
            self._hssd_validation_config_value(
                "optional_min_similarity_score",
                0.28,
            )
            or 0.28
        )
        minimum_margin = float(
            self._hssd_validation_config_value(
                "optional_min_similarity_margin",
                0.04,
            )
            or 0.04
        )
        selected_score = float(candidates[0].similarity_score)
        competing_scores = sorted(
            (float(candidate.similarity_score) for candidate in candidates[1:]),
            reverse=True,
        )
        return selected_score < minimum_score or (
            bool(competing_scores)
            and selected_score - competing_scores[0] < minimum_margin
        )

    def _hssd_validation_cache_path(
        self,
        *,
        candidate_id: str,
        family: str,
        use_lenient: bool,
    ) -> Path | None:
        configured_dir = str(
            self._hssd_validation_config_value("cache_dir", "") or ""
        ).strip()
        cache_dir_value = os.environ.get(
            "SCENEEXPERT_ASSET_VALIDATION_CACHE_DIR",
            configured_dir,
        ).strip()
        if not cache_dir_value:
            return None
        cache_dir = Path(cache_dir_value).expanduser()
        agent_type = getattr(getattr(self, "agent_type", None), "value", "unknown")
        cache_key = hashlib.sha256(
            (
                f"hssd-semantic-v{ASSET_SEMANTIC_CONTRACT_VERSION}|"
                f"{candidate_id}|{family}|role={agent_type}|"
                f"lenient={int(use_lenient)}"
            ).encode("utf-8")
        ).hexdigest()
        return cache_dir / f"{cache_key}.json"

    def _load_persistent_hssd_validation(
        self,
        *,
        candidate_id: str,
        family: str,
        use_lenient: bool,
    ) -> ValidationResult | None:
        cache_path = self._hssd_validation_cache_path(
            candidate_id=candidate_id,
            family=family,
            use_lenient=use_lenient,
        )
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return ValidationResult(
                is_acceptable=bool(payload["is_acceptable"]),
                reason=str(payload.get("reason", "cached validation")),
                suggestions=list(payload.get("suggestions", []) or []),
                front_view_image_index=payload.get("front_view_image_index"),
                orientation_confidence=payload.get("orientation_confidence"),
                contains_architectural_context=payload.get(
                    "contains_architectural_context"
                ),
                requested_object_is_dominant=payload.get(
                    "requested_object_is_dominant"
                ),
            )
        except Exception as exc:
            console_logger.warning(
                "Ignoring unreadable HSSD semantic cache %s: %s",
                cache_path,
                exc,
            )
            return None

    def _save_persistent_hssd_validation(
        self,
        *,
        candidate_id: str,
        family: str,
        use_lenient: bool,
        validation: ValidationResult,
    ) -> None:
        cache_path = self._hssd_validation_cache_path(
            candidate_id=candidate_id,
            family=family,
            use_lenient=use_lenient,
        )
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ASSET_SEMANTIC_CONTRACT_VERSION,
            "candidate_id": candidate_id,
            "family": family,
            "is_acceptable": validation.is_acceptable,
            "reason": validation.reason,
            "suggestions": validation.suggestions,
            "front_view_image_index": validation.front_view_image_index,
            "orientation_confidence": validation.orientation_confidence,
            "contains_architectural_context": (
                validation.contains_architectural_context
            ),
            "requested_object_is_dominant": validation.requested_object_is_dominant,
        }
        temporary = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(cache_path)

    def _calibrated_hssd_front_axis(self, mesh_id: str) -> str | None:
        """Return a trusted front axis from the already-paid semantic VLM call."""
        validation = getattr(self, "_direct_hssd_validation_results", {}).get(mesh_id)
        if validation is None or validation.front_view_image_index is None:
            return None
        _, _, _, _, _, min_confidence = self._hssd_semantic_validation_settings(
            mesh_id, mesh_id
        )
        confidence = float(validation.orientation_confidence or 0.0)
        if confidence < min_confidence:
            console_logger.warning(
                "Ignoring low-confidence HSSD front calibration for %s (%.2f < %.2f)",
                mesh_id,
                confidence,
                min_confidence,
            )
            return None
        try:
            raw_axis = get_front_axis_from_image_number(
                image_number=int(validation.front_view_image_index),
                num_side_views=4,
                include_diagonal_views=False,
                include_vertical_views=True,
            )
        except ValueError:
            console_logger.warning(
                "Ignoring invalid HSSD front-view image index %s for %s",
                validation.front_view_image_index,
                mesh_id,
            )
            return None
        if raw_axis.lower() in {"z", "-z"}:
            return None
        return raw_axis.upper() if raw_axis.startswith("-") else f"+{raw_axis.upper()}"

    def _resolve_hssd_orientation(
        self,
        *,
        result: HssdRetrievalResult,
        gltf_path: Path,
        description: str,
        short_name: str,
        desired_dimensions: list[float] | tuple[float, ...] | None,
        default_physics: MeshPhysicsAnalysis,
    ) -> tuple[str, str, float | None]:
        """Resolve functional front using dataset, VLM, then family geometry.

        The final fallback only chooses between horizontal axes by dimension
        compatibility.  It never infers front from "the longest side" alone and
        therefore remains well-defined for near-cubic assets.
        """
        dataset_front = str(getattr(result, "front_axis", "") or "").upper()
        dataset_up = str(getattr(result, "up_axis", "") or "").upper()
        if dataset_front in {"+X", "-X", "+Y", "-Y"} and dataset_up == "+Z":
            return dataset_front, "hssd_dataset_front", 1.0

        calibrated_front = self._calibrated_hssd_front_axis(result.hssd_id)
        if calibrated_front is not None:
            validation = getattr(self, "_direct_hssd_validation_results", {}).get(
                result.hssd_id
            )
            return (
                calibrated_front,
                "semantic_multiview",
                float(getattr(validation, "orientation_confidence", 0.0) or 0.0),
            )

        family = semantic_asset_family(description, short_name)
        if family in {"table", "rug", "plant"}:
            return "+Y", "symmetric_front_irrelevant", None

        fallback_front = str(default_physics.front_axis or "+Y").upper()
        if fallback_front not in {"+X", "-X", "+Y", "-Y"}:
            fallback_front = "+Y"
        if desired_dimensions is not None:
            mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
            bbox_min, bbox_max = gltf_y_up_bounds_to_scene_z_up(mesh.bounds)
            source_dimensions = bbox_max - bbox_min
            target = np.asarray(desired_dimensions, dtype=float)
            as_is_error = uniform_scale_shape_error(source_dimensions, target)
            swapped_error = uniform_scale_shape_error(
                source_dimensions[[1, 0, 2]],
                target,
            )
            if swapped_error + 1e-6 < as_is_error:
                fallback_front = "+X"
            else:
                fallback_front = "+Y"
        console_logger.warning(
            "Using family geometry front fallback for %s (%s): %s",
            short_name,
            family,
            fallback_front,
        )
        return fallback_front, "family_geometry_fallback", 0.25

    def _select_direct_hssd_candidate(
        self,
        *,
        candidates: list[HssdRetrievalResult],
        description: str,
        short_name: str,
        desired_dimensions: list[float] | tuple[float, ...] | None = None,
        excluded_candidate_ids: set[str] | None = None,
        validation_deadline: float | None = None,
    ) -> HssdRetrievalResult:
        """Select a visually valid candidate for silhouette-critical furniture.

        The non-router path intentionally avoids an LLM request-analysis call, but
        taking CLIP rank 1 blindly cannot reject a wall frame mislabeled as a bed.
        For a small configurable set of high-impact furniture families, validate
        the already retrieved top candidates directly with the existing VLM.
        """
        desired_dimensions = _optional_hssd_dimension_contract(desired_dimensions)
        excluded = set(excluded_candidate_ids or set())
        if not hasattr(self, "_direct_hssd_admission_states"):
            self._direct_hssd_admission_states = {}
        candidates = [
            candidate for candidate in candidates if candidate.hssd_id not in excluded
        ]
        if not candidates:
            raise ValueError("No results returned from HSSD server")

        family = semantic_asset_family(description, short_name)
        critical_family = self._is_critical_hssd_family(family)
        proportion_match_found = True
        if desired_dimensions is not None:
            # Raw server extents do not have semantic axis order. Use them only
            # to rank the bounded candidate set; post-canonical dimensions own
            # the actual admission decision.
            candidates = sorted(
                candidates,
                key=lambda candidate: hssd_dimension_shape_error(
                    candidate.size,
                    desired_dimensions,
                ),
            )
            min_ratio, max_ratio = self._uniform_dimension_fit_bounds(
                critical_family=critical_family,
            )
            proportion_match_found = any(
                axis_agnostic_uniform_fit_exists(
                    candidate.size,
                    desired_dimensions,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                )
                for candidate in candidates
            )

        (
            enabled,
            max_candidates,
            use_lenient,
            timeout_seconds,
            max_retries,
            _,
        ) = self._hssd_semantic_validation_settings(description, short_name)
        validate_ambiguous_optional = bool(
            self._hssd_validation_config_value(
                "validate_ambiguous_optional",
                False,
            )
        )
        if not enabled and not critical_family and not validate_ambiguous_optional:
            return candidates[0]
        if not critical_family and not self._optional_hssd_candidate_is_ambiguous(
            candidates,
            proportion_match_found=proportion_match_found,
        ):
            console_logger.info(
                "Accepted optional HSSD candidate %s for '%s' from deterministic "
                "dimension/CLIP evidence; VLM validation was not required",
                candidates[0].hssd_id,
                description,
            )
            return candidates[0]

        validation_router = getattr(
            self,
            "_asset_validation_router",
            getattr(self, "_thin_covering_router", None),
        )
        if validation_router is None:
            console_logger.warning(
                "Direct HSSD semantic validation is enabled for '%s' but no "
                "validation router is available; using the dimension-ranked candidate",
                description,
            )
            return candidates[0]

        infrastructure_failures = 0
        attempted_candidates = 0
        transient_candidate: HssdRetrievalResult | None = None
        considered = candidates[: min(4, max_candidates)]
        validation_started = time.monotonic()
        configured_total_seconds = timeout_seconds
        execution_control_enabled = bool(
            getattr(self, "_execution_control_enabled", False)
        )
        runtime_validation = dict(getattr(self, "_asset_validation_runtime", {}) or {})
        if execution_control_enabled:
            configured_total_seconds = runtime_validation.get(
                "asset_validation_total_timeout_seconds",
                self._hssd_validation_config_value(
                    "total_timeout_seconds",
                    timeout_seconds * (2 if critical_family else 1),
                ),
            )
        total_validation_seconds = max(
            timeout_seconds,
            float(configured_total_seconds or timeout_seconds),
        )
        family_retry_count = (
            max(
                0,
                int(
                    runtime_validation.get(
                        "asset_validation_family_retries",
                        1,
                    )
                    or 0
                ),
            )
            if critical_family and execution_control_enabled
            else (1 if critical_family and max_retries > 0 else 0)
        )
        family_retry_semantic_rejected = False
        max_output_tokens = (
            max(
                1,
                int(
                    runtime_validation.get(
                        "asset_validation_max_output_tokens",
                        512,
                    )
                    or 512
                ),
            )
            if execution_control_enabled
            else None
        )
        retry_max_output_tokens = (
            max(
                int(max_output_tokens or 1),
                int(
                    runtime_validation.get(
                        "asset_validation_retry_max_output_tokens",
                        max(1024, int(max_output_tokens or 512) * 2),
                    )
                    or max(1024, int(max_output_tokens or 512) * 2)
                ),
            )
            if execution_control_enabled
            else None
        )
        active_max_output_tokens = max_output_tokens
        if validation_deadline is not None:
            total_validation_seconds = min(
                total_validation_seconds,
                max(0.0, validation_deadline - validation_started),
            )
        for candidate_index, candidate in enumerate(considered):
            validation_cache = getattr(self, "_direct_hssd_semantic_cache", None)
            if validation_cache is None:
                validation_cache = {}
                self._direct_hssd_semantic_cache = validation_cache
            validation_cache_key = f"{candidate.hssd_id}|{family}"
            validation = validation_cache.get(validation_cache_key)
            if validation is None:
                validation = self._load_persistent_hssd_validation(
                    candidate_id=candidate.hssd_id,
                    family=family,
                    use_lenient=use_lenient,
                )
                if validation is not None:
                    validation_cache[validation_cache_key] = validation
            mesh_path = Path(candidate.mesh_path)
            validation_dir = (
                self.debug_dir
                / short_name
                / f"hssd_{candidate_index:02d}_{candidate.hssd_id[:12]}_validation"
            )
            if validation is None:
                # Distinct candidates are more informative than repeatedly
                # calling the same timed-out candidate. HTTP retries are zero;
                # one explicit family retry is reserved below.
                allowed_retries = 0
                remaining_seconds = total_validation_seconds - (
                    time.monotonic() - validation_started
                )
                if remaining_seconds <= 1.0:
                    break
                remaining_candidates = len(considered) - candidate_index
                fair_slots = remaining_candidates + family_retry_count
                per_attempt_timeout = min(
                    timeout_seconds,
                    remaining_seconds / max(1, fair_slots),
                )
                validation = validation_router.validate_asset(
                    mesh_path=mesh_path,
                    description=description,
                    output_dir=validation_dir,
                    use_lenient=use_lenient,
                    timeout_seconds=max(1.0, per_attempt_timeout),
                    max_retries=allowed_retries,
                    max_output_tokens=active_max_output_tokens,
                )
                if getattr(validation, "failure_kind", None) == "length":
                    active_max_output_tokens = retry_max_output_tokens
                if not _asset_validation_is_retryable(validation):
                    # Infrastructure outcomes are retryable and must not poison
                    # the semantic cache for a later isolated stage retry.
                    validation_cache[validation_cache_key] = validation
                    self._save_persistent_hssd_validation(
                        candidate_id=candidate.hssd_id,
                        family=family,
                        use_lenient=use_lenient,
                        validation=validation,
                    )
            contracted_validation = _enforce_critical_hssd_validation_contract(
                validation,
                critical_family=critical_family,
            )
            if contracted_validation is not validation:
                validation = contracted_validation
                validation_cache[validation_cache_key] = validation
                self._save_persistent_hssd_validation(
                    candidate_id=candidate.hssd_id,
                    family=family,
                    use_lenient=use_lenient,
                    validation=validation,
                )
            if validation.is_acceptable:
                orientation_results = getattr(
                    self, "_direct_hssd_validation_results", None
                )
                if orientation_results is None:
                    orientation_results = {}
                    self._direct_hssd_validation_results = orientation_results
                orientation_results[candidate.hssd_id] = validation
                self._direct_hssd_admission_states[candidate.hssd_id] = "vlm_verified"
                console_logger.info(
                    "Direct HSSD semantic validation selected candidate %s for '%s'",
                    candidate.hssd_id,
                    description,
                )
                return candidate

            attempted_candidates += 1
            reason = str(validation.reason or "")
            if _asset_validation_is_retryable(validation):
                infrastructure_failures += 1
                if transient_candidate is None:
                    transient_candidate = candidate
            console_logger.warning(
                "Rejected HSSD candidate %s for '%s': %s",
                candidate.hssd_id,
                description,
                reason,
            )

        if (
            critical_family
            and transient_candidate is not None
            and family_retry_count > 0
        ):
            remaining_seconds = total_validation_seconds - (
                time.monotonic() - validation_started
            )
            if remaining_seconds > 1.0:
                retry_dir = (
                    self.debug_dir
                    / short_name
                    / f"hssd_family_retry_{transient_candidate.hssd_id[:12]}"
                )
                retry_validation = validation_router.validate_asset(
                    mesh_path=Path(transient_candidate.mesh_path),
                    description=description,
                    output_dir=retry_dir,
                    use_lenient=use_lenient,
                    timeout_seconds=max(1.0, min(timeout_seconds, remaining_seconds)),
                    max_retries=0,
                    max_output_tokens=active_max_output_tokens,
                )
                retry_validation = _enforce_critical_hssd_validation_contract(
                    retry_validation,
                    critical_family=critical_family,
                )
                if retry_validation.is_acceptable:
                    self._direct_hssd_validation_results[
                        transient_candidate.hssd_id
                    ] = retry_validation
                    self._direct_hssd_admission_states[transient_candidate.hssd_id] = (
                        "vlm_verified_after_family_retry"
                    )
                    return transient_candidate
                if not _asset_validation_is_retryable(retry_validation):
                    family_retry_semantic_rejected = True
                    retry_cache_key = f"{transient_candidate.hssd_id}|{family}"
                    self._direct_hssd_semantic_cache[retry_cache_key] = retry_validation
                    self._save_persistent_hssd_validation(
                        candidate_id=transient_candidate.hssd_id,
                        family=family,
                        use_lenient=use_lenient,
                        validation=retry_validation,
                    )

        if (
            attempted_candidates > 0
            and infrastructure_failures == attempted_candidates
            and not family_retry_semantic_rejected
        ):
            if not critical_family and bool(
                getattr(
                    self.cfg.asset_manager,
                    "hssd_vlm_fallback_on_transient_error",
                    True,
                )
            ):
                console_logger.warning(
                    "Optional HSSD VLM validation was unavailable for '%s'; "
                    "accepting deterministic dimension/CLIP candidate %s",
                    description,
                    candidates[0].hssd_id,
                )
                return candidates[0]
            raise TimeoutError(
                "HSSD semantic validation infrastructure was unavailable for "
                f"all {len(considered)} candidate(s) of '{description}'"
            )
        if attempted_candidates == 0:
            if not critical_family:
                return candidates[0]
            raise TimeoutError(
                "HSSD semantic validation had no executable time remaining for "
                f"'{description}'"
            )

        raise ValueError(
            f"All {len(considered)} HSSD candidates failed visual semantic "
            f"validation for '{description}'"
        )

    def _direct_hssd_candidate_count(self, description: str, short_name: str) -> int:
        enabled, max_candidates, _, _, _, _ = self._hssd_semantic_validation_settings(
            description, short_name
        )
        family = semantic_asset_family(description, short_name)
        if enabled or self._is_critical_hssd_family(family):
            return min(4, max_candidates)
        hssd_cfg = getattr(self.cfg.asset_manager, "hssd", None)
        try:
            return min(4, max(1, int(hssd_cfg.get("dimension_candidates", 3) or 3)))
        except Exception:
            return min(
                4,
                max(1, int(getattr(hssd_cfg, "dimension_candidates", 3) or 3)),
            )

    def _uniform_dimension_fit_bounds(
        self,
        *,
        critical_family: bool = False,
    ) -> tuple[float, float]:
        """Return bounded residual tolerance for approximate LLM dimensions."""
        if self.agent_type in {AgentType.WALL_MOUNTED, AgentType.CEILING_MOUNTED}:
            # Mounted-object prompts often specify a target footprint while HSSD
            # preserves a natural frame depth or pendant drop. Keep the guard
            # bounded, but do not reject a semantic match for a 2-3x axial
            # difference in an approximate designer estimate.
            return 0.30, 2.50
        if critical_family:
            return (
                float(
                    self._hssd_validation_config_value(
                        "critical_min_dimension_ratio",
                        0.75,
                    )
                    or 0.75
                ),
                float(
                    self._hssd_validation_config_value(
                        "critical_max_dimension_ratio",
                        1.35,
                    )
                    or 1.35
                ),
            )
        return 0.50, 1.75

    def _configured_asset_size_bounds(
        self,
        *,
        description: str,
        short_name: str,
    ) -> tuple[list[float], list[float]] | None:
        """Resolve the final hard-size contract used by furniture verification."""

        if self.agent_type != AgentType.FURNITURE:
            return None
        safety_cfg = getattr(self.cfg, "furniture_safety_controller", None)
        all_bounds = getattr(safety_cfg, "size_bounds", None)
        if all_bounds is None:
            return None

        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            f"{short_name} {description}".lower(),
        ).strip("_")
        family = semantic_asset_family(description, short_name)
        keys = []
        if "twin_bed" in normalized_name or "single_bed" in normalized_name:
            keys.append("twin_bed")
        keys.extend((family, short_name))
        for key in dict.fromkeys(value for value in keys if value):
            try:
                bounds_cfg = all_bounds.get(key)
            except Exception:
                bounds_cfg = getattr(all_bounds, key, None)
            if bounds_cfg is None:
                continue
            try:
                minimum = list(bounds_cfg.get("min", []) or [])
                maximum = list(bounds_cfg.get("max", []) or [])
            except Exception:
                minimum = list(getattr(bounds_cfg, "min", []) or [])
                maximum = list(getattr(bounds_cfg, "max", []) or [])
            if len(minimum) == len(maximum) == 3:
                return (
                    [float(value) for value in minimum],
                    [float(value) for value in maximum],
                )
        return None

    def _validate_configured_asset_size(
        self,
        *,
        dimensions: np.ndarray,
        description: str,
        short_name: str,
    ) -> None:
        """Reject assets that the downstream furniture verifier must reject."""

        configured = self._configured_asset_size_bounds(
            description=description,
            short_name=short_name,
        )
        if configured is None:
            return
        minimum, maximum = configured
        below = any(
            float(value) + 1e-3 < lower for value, lower in zip(dimensions, minimum)
        )
        above = any(
            float(value) - 1e-3 > upper for value, upper in zip(dimensions, maximum)
        )
        if below or above:
            raise ValueError(
                "Asset dimensions violate the shared furniture admission "
                f"contract: actual={np.asarray(dimensions).round(3).tolist()}, "
                f"expected min={minimum}, max={maximum}"
            )

    def _normalized_requested_dimensions(
        self,
        *,
        desired_dimensions: list[float] | tuple[float, ...],
        description: str,
        short_name: str,
    ) -> np.ndarray:
        """Clamp a designer request to the shared real-world family bounds."""
        normalized = np.asarray(desired_dimensions, dtype=float)
        configured = self._configured_asset_size_bounds(
            description=description,
            short_name=short_name,
        )
        if configured is not None:
            minimum, maximum = configured
            normalized = np.clip(
                normalized,
                np.asarray(minimum, dtype=float),
                np.asarray(maximum, dtype=float),
            )
        return normalized

    def _scale_and_measure_canonical_mesh(
        self,
        *,
        canonical_path: Path,
        final_path: Path,
        desired_dimensions: list[float] | tuple[float, ...] | None,
        description: str = "",
        short_name: str = "",
    ) -> tuple[Path, np.ndarray, np.ndarray, float]:
        """Scale a Y-up canonical mesh and expose its SceneSmith Z-up bounds."""
        applied_scale = 1.0
        if desired_dimensions is not None:
            canonical_mesh = load_mesh_as_trimesh(canonical_path, force_merge=True)
            source_min, source_max = gltf_y_up_bounds_to_scene_z_up(
                canonical_mesh.bounds
            )
            source_dimensions = source_max - source_min
            family = semantic_asset_family(description, short_name)
            critical_family = self._is_critical_hssd_family(family)
            min_ratio, max_ratio = self._uniform_dimension_fit_bounds(
                critical_family=critical_family,
            )
            configured = self._configured_asset_size_bounds(
                description=description,
                short_name=short_name,
            )
            minimum, maximum = configured if configured is not None else (None, None)
            applied_scale, normalized_target = choose_uniform_scale_for_contract(
                source_dimensions,
                desired_dimensions,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                minimum_dimensions=minimum,
                maximum_dimensions=maximum,
                # A designer dimension is an approximate preference. Once a
                # family-specific real-world envelope exists, that envelope is
                # the hard contract and the target only chooses the closest
                # feasible *uniform* scale.
                enforce_requested_ratio=configured is None,
            )
            # Preserve exact source proportions: the scaler receives the dimensions
            # produced by one chosen scalar, never independent per-axis targets.
            gltf_dimensions = scene_dimensions_to_gltf_y_up(
                source_dimensions * applied_scale
            )
            final_path, applied_scale = scale_mesh_uniformly_to_dimensions(
                mesh_path=canonical_path,
                desired_dimensions=gltf_dimensions,
                output_path=final_path,
                min_dimension_meters=self.min_mesh_dimension_meters,
                relative_threshold=self.mesh_relative_dimension_threshold,
            )
        else:
            canonical_path.replace(final_path)

        mesh = load_mesh_as_trimesh(final_path, force_merge=True)
        bbox_min, bbox_max = gltf_y_up_bounds_to_scene_z_up(mesh.bounds)
        if desired_dimensions is not None:
            actual_dimensions = bbox_max - bbox_min
            if configured is None:
                validate_uniform_dimension_fit(
                    actual_dimensions,
                    normalized_target,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                )
            if configured is not None:
                self._validate_configured_asset_size(
                    dimensions=actual_dimensions,
                    description=description,
                    short_name=short_name,
                )
            console_logger.info(
                "Canonical asset dimensions: requested=%s, normalized=%s, "
                "actual=%s, uniform_scale=%.3f",
                list(desired_dimensions),
                normalized_target.round(4).tolist(),
                actual_dimensions.round(4).tolist(),
                applied_scale,
            )
        return final_path, bbox_min, bbox_max, applied_scale

    def _prepare_hssd_candidate(
        self,
        *,
        result: HssdRetrievalResult,
        request: AssetGenerationRequest,
        index: int,
        config: AssetPathConfig,
        candidate_attempt: int,
    ) -> SceneObject:
        """Prepare one HSSD candidate as an atomic admission transaction."""
        description = request.object_descriptions[index]
        short_name = request.short_names[index]
        desired_dimensions = _optional_hssd_dimension_contract(
            request.desired_dimensions[index]
        )
        server_mesh_path = Path(result.mesh_path)
        mesh_id = result.hssd_id

        if server_mesh_path.suffix.lower() == ".glb":
            gltf_path = server_mesh_path.with_suffix(".gltf")
            if not gltf_path.exists():
                if not server_mesh_path.exists():
                    raise FileNotFoundError(
                        f"Retrieved mesh file missing: {server_mesh_path}"
                    )
                self.blender_server.convert_glb_to_gltf(
                    input_path=server_mesh_path,
                    output_path=gltf_path,
                    export_yup=True,
                )
        else:
            gltf_path = server_mesh_path

        debug_dir = self.debug_dir / short_name
        console_logger.info(
            "Resolving HSSD physics metadata for %s (mode=%s, candidate=%s)",
            short_name,
            getattr(
                self.cfg.asset_manager,
                "hssd_physics_analysis_mode",
                "deterministic",
            ),
            mesh_id,
        )
        default_physics = self._analyze_mesh_physics(
            mesh_path=gltf_path,
            asset_source="hssd",
            object_name=short_name,
            debug_output_dir=debug_dir,
        )
        front_axis, front_source, front_confidence = self._resolve_hssd_orientation(
            result=result,
            gltf_path=gltf_path,
            description=description,
            short_name=short_name,
            desired_dimensions=desired_dimensions,
            default_physics=default_physics,
        )
        physics_analysis = MeshPhysicsAnalysis(
            up_axis=CANONICAL_UP_AXIS,
            front_axis=front_axis,
            material=default_physics.material,
            mass_kg=default_physics.mass_kg,
            mass_range_kg=default_physics.mass_range_kg,
        )
        console_logger.info(
            "HSSD frame resolved for %s: source_up=%s, source_front=%s, source=%s",
            short_name,
            physics_analysis.up_axis,
            physics_analysis.front_axis,
            front_source,
        )

        canonical_path = config.sdf_dir / f"{config.short_name}_canonical.gltf"
        canonicalize_mesh(
            gltf_path=gltf_path,
            output_path=canonical_path,
            up_axis=physics_analysis.up_axis,
            front_axis=physics_analysis.front_axis,
            blender_server=self.blender_server,
            object_type=request.object_type,
        )
        final_gltf_path, bbox_min, bbox_max, applied_scale = (
            self._scale_and_measure_canonical_mesh(
                canonical_path=canonical_path,
                final_path=config.sdf_dir / f"{config.short_name}.gltf",
                desired_dimensions=desired_dimensions,
                description=description,
                short_name=short_name,
            )
        )

        collision_pieces = self._generate_collision_geometry(final_gltf_path)
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_drake_sdf(
            visual_mesh_path=final_gltf_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics_analysis,
            output_path=sdf_path,
            asset_name=config.short_name,
            mesh_frame="gltf_y_up",
        )

        normalized_target = (
            self._normalized_requested_dimensions(
                desired_dimensions=desired_dimensions,
                description=description,
                short_name=short_name,
            ).tolist()
            if desired_dimensions is not None
            else None
        )
        return self._create_scene_object(
            config=config,
            object_type=request.object_type,
            sdf_path=sdf_path,
            final_gltf_path=final_gltf_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            additional_metadata={
                "asset_source": "hssd",
                "hssd_mesh_id": mesh_id,
                "asset_frame_contract_version": ASSET_FRAME_CONTRACT_VERSION,
                "front_definition": "functional_outward",
                "source_front_axis": physics_analysis.front_axis,
                "canonical_front_axis": CANONICAL_FRONT_AXIS,
                "canonical_up_axis": CANONICAL_UP_AXIS,
                "canonical_dimension_order": list(CANONICAL_DIMENSION_ORDER),
                "front_axis_source": front_source,
                "front_axis_confidence": front_confidence,
                "requested_dimensions": (
                    list(desired_dimensions) if desired_dimensions is not None else None
                ),
                "normalized_requested_dimensions": normalized_target,
                "actual_dimensions": (bbox_max - bbox_min).tolist(),
                "uniform_scaling_only": True,
                "source_to_canonical_scale": float(applied_scale),
                "candidate_attempt": candidate_attempt,
                "semantic_admission_state": self._direct_hssd_admission_states.get(
                    mesh_id,
                    "deterministic",
                ),
            },
            # The final glTF and SDF already contain applied_scale.
            # SceneObject.scale_factor is reserved for later mutations.
            scale_factor=1.0,
        )

    @staticmethod
    def _is_candidate_specific_hssd_failure(error: Exception) -> bool:
        """Separate bad-candidate evidence from shared service failures."""
        if isinstance(error, (ValueError, FileNotFoundError)):
            return True
        message = str(error).lower()
        if any(
            token in message
            for token in (
                "timeout",
                "timed out",
                "connection",
                "server unavailable",
                "server is not running",
            )
        ):
            return False
        return any(
            token in message
            for token in (
                "degenerate",
                "blank",
                "empty mesh",
                "invalid mesh",
                "no geometry",
                "cannot satisfy the uniform dimension contract",
            )
        )

    def _retrieve_hssd_assets(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Retrieve assets from HSSD library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        if self.hssd_client is None:
            raise RuntimeError("HSSD retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )
        request = _align_hssd_request_dimensions(request)

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from HSSD server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Rugs and carpets are 2D coverings, not freestanding HSSD furniture.
        # Route these obvious cases deterministically even when the LLM router is
        # disabled, avoiding both semantic mismatches and an extra model call.
        floor_covering_indices = [
            index
            for index, (description, short_name) in enumerate(
                zip(request.object_descriptions, request.short_names)
            )
            if self._is_deterministic_floor_covering(
                description, short_name, request.object_type
            )
        ]
        floor_covering_index_set = set(floor_covering_indices)
        hssd_indices = [
            index
            for index in range(len(request.object_descriptions))
            if index not in floor_covering_index_set
        ]

        # Create batch requests for HSSD server with client-specified output dirs.
        retrieval_requests = [
            HssdRetrievalServerRequest(
                # Retrieve by the canonical functional family. Style and
                # placement adjectives remain soft evidence for bounded VLM
                # reranking; they must not collapse library recall.
                object_description=(
                    semantic_asset_family(
                        request.object_descriptions[index],
                        request.short_names[index],
                    ).replace("_", " ")
                    or request.object_descriptions[index]
                ),
                object_type=request.object_type.value,
                desired_dimensions=(
                    tuple(request.desired_dimensions[index])
                    if request.desired_dimensions[index]
                    else None
                ),
                output_dir=str(asset_path_configs[index].sdf_dir),
                scene_id=request.scene_id,
                num_candidates=self._direct_hssd_candidate_count(
                    request.object_descriptions[index],
                    request.short_names[index],
                ),
            )
            for index in hssd_indices
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        retrieval_responses = (
            self.hssd_client.retrieve_objects(
                retrieval_requests,
                timeout_s=self._asset_acquisition_timeout_seconds,
            )
            if retrieval_requests
            else []
        )
        for retrieval_index, response in retrieval_responses:
            index = hssd_indices[retrieval_index]
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing HSSD response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                max_candidates = min(
                    4,
                    self._direct_hssd_candidate_count(desc, short_name),
                    len(response.results),
                )
                candidates = list(response.results[:max_candidates])
                desired_dimensions = _optional_hssd_dimension_contract(
                    request.desired_dimensions[index]
                )
                excluded_candidate_ids: set[str] = set()
                candidate_errors: list[str] = []
                _, _, _, timeout_seconds, _, _ = (
                    self._hssd_semantic_validation_settings(desc, short_name)
                )
                transaction_seconds = max(
                    timeout_seconds,
                    float(
                        (
                            getattr(self, "_asset_validation_runtime", {}).get(
                                "asset_validation_total_timeout_seconds",
                                timeout_seconds,
                            )
                            if bool(getattr(self, "_execution_control_enabled", False))
                            else timeout_seconds
                        )
                        or timeout_seconds
                    ),
                )
                validation_deadline = time.monotonic() + transaction_seconds

                scene_obj: SceneObject | None = None
                for candidate_attempt in range(1, max_candidates + 1):
                    try:
                        result = self._select_direct_hssd_candidate(
                            candidates=candidates,
                            description=desc,
                            short_name=short_name,
                            desired_dimensions=desired_dimensions,
                            excluded_candidate_ids=excluded_candidate_ids,
                            validation_deadline=validation_deadline,
                        )
                    except Exception as selection_error:
                        candidate_errors.append(f"selection: {selection_error}")
                        break
                    try:
                        scene_obj = self._prepare_hssd_candidate(
                            result=result,
                            request=request,
                            index=index,
                            config=config,
                            candidate_attempt=candidate_attempt,
                        )
                        break
                    except Exception as candidate_error:
                        if not self._is_candidate_specific_hssd_failure(
                            candidate_error
                        ):
                            raise
                        excluded_candidate_ids.add(result.hssd_id)
                        candidate_errors.append(f"{result.hssd_id}: {candidate_error}")
                        console_logger.warning(
                            "Rolling back HSSD candidate %s for '%s' after "
                            "post-admission preparation failed: %s",
                            result.hssd_id,
                            desc,
                            candidate_error,
                        )

                if scene_obj is None:
                    failure_detail = (
                        "All bounded HSSD candidate transactions failed for "
                        f"'{desc}': {' | '.join(candidate_errors)}"
                    )
                    if any(
                        marker in failure_detail.lower()
                        for marker in (
                            "timeout",
                            "no executable time",
                            "infrastructure",
                        )
                    ):
                        failure_kind = "validation_unavailable"
                    elif "visual semantic validation" in failure_detail.lower():
                        failure_kind = "semantic_mismatch"
                    elif not candidates:
                        failure_kind = "retrieval_empty"
                    else:
                        failure_kind = "candidate_preparation"
                    console_logger.warning(
                        "HSSD asset unavailable after bounded candidate admission "
                        "(kind=%s, description=%r): %s",
                        failure_kind,
                        desc,
                        failure_detail,
                    )
                    failed_assets.append(
                        FailedAsset(
                            index=index,
                            description=desc,
                            error_message=f"[{failure_kind}] {failure_detail}",
                        )
                    )
                    continue
                successful_objects.append(scene_obj)
                console_logger.info(
                    "HSSD asset transaction committed: %s (candidate %s)",
                    config.short_name,
                    scene_obj.metadata.get("hssd_mesh_id"),
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process HSSD asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        for index in floor_covering_indices:
            desc = request.object_descriptions[index]
            try:
                successful_objects.append(
                    self._generate_deterministic_floor_covering(request, index)
                )
                console_logger.info(
                    "Generated deterministic floor covering: %s",
                    request.short_names[index],
                )
            except Exception as e:
                console_logger.error(
                    "Failed to generate floor covering '%s': %s", desc, e, exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def _retrieve_objaverse_assets(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Retrieve assets from Objaverse (ObjectThor) library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        if self.objaverse_client is None:
            raise RuntimeError("Objaverse retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from Objaverse server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Create batch requests for Objaverse server with client-specified output dirs.
        retrieval_requests = [
            ObjaverseRetrievalServerRequest(
                object_description=desc,
                object_type=request.object_type.value,
                desired_dimensions=tuple(dims) if dims else None,
                output_dir=str(config.sdf_dir),
                scene_id=request.scene_id,
            )
            for desc, dims, config in zip(
                request.object_descriptions,
                request.desired_dimensions,
                asset_path_configs,
            )
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        for index, response in self.objaverse_client.retrieve_objects(
            retrieval_requests
        ):
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing Objaverse response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                # Server returns mesh path (already exported to our output_dir).
                if not response.results:
                    raise ValueError("No results returned from Objaverse server")

                result = response.results[0]  # Get top result.
                server_mesh_path = Path(result.mesh_path)
                mesh_id = result.objaverse_uid

                # Server exported to our specified output_dir, convert GLB to GLTF if
                # needed. Uses BlenderServer for crash isolation.
                if server_mesh_path.suffix.lower() == ".glb":
                    # Server exported GLB, convert to GLTF with Y-up coordinates.
                    # Keep the GLB because duplicate requests may legitimately
                    # reference the same retrieved mesh in the same batch.
                    gltf_path = server_mesh_path.with_suffix(".gltf")
                    if not gltf_path.exists():
                        if not server_mesh_path.exists():
                            raise FileNotFoundError(
                                f"Retrieved mesh file missing: {server_mesh_path}"
                            )
                        self.blender_server.convert_glb_to_gltf(
                            input_path=server_mesh_path,
                            output_path=gltf_path,
                            export_yup=True,
                        )
                else:
                    # Already GLTF, use as-is.
                    gltf_path = server_mesh_path

                # Run VLM analysis for orientation, material and mass estimation.
                console_logger.info(
                    f"Running VLM analysis for Objaverse orientation/material/mass: "
                    f"{short_name}"
                )
                vlm_physics = analyze_mesh_orientation_and_material(
                    mesh_path=gltf_path,
                    vlm_service=self.vlm_service,
                    cfg=self.cfg,
                    elevation_degrees=self.side_view_elevation_degrees,
                    blender_server=self.blender_server,
                    num_side_views=self.num_side_views_for_physics_analysis,
                    prompt_type="generated",  # Full VLM analysis (not pre-canonicalized).
                    include_vertical_views=True,
                    debug_output_dir=self.debug_dir / short_name,
                )
                console_logger.info(
                    f"VLM analysis complete: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis}, material={vlm_physics.material}, "
                    f"mass={vlm_physics.mass_kg}kg"
                )

                # Use VLM's orientation, material, and mass determination.
                physics_analysis = MeshPhysicsAnalysis(
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    material=vlm_physics.material,
                    mass_kg=vlm_physics.mass_kg,
                    mass_range_kg=vlm_physics.mass_range_kg,
                )

                # Canonicalize mesh orientation to align with scenesmith canonical
                # (Z-up, Y-forward).
                console_logger.info(
                    f"Canonicalizing Objaverse mesh: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis} → +Y"
                )
                canonical_path = config.sdf_dir / f"{config.short_name}_canonical.gltf"
                canonicalize_mesh(
                    gltf_path=gltf_path,
                    output_path=canonical_path,
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    blender_server=self.blender_server,
                    object_type=request.object_type,
                )

                final_gltf_path, bbox_min, bbox_max, _ = (
                    self._scale_and_measure_canonical_mesh(
                        canonical_path=canonical_path,
                        final_path=config.sdf_dir / f"{config.short_name}.gltf",
                        desired_dimensions=request.desired_dimensions[index],
                    )
                )

                # Generate collision geometry via collision server.
                collision_pieces = self._generate_collision_geometry(final_gltf_path)

                sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=final_gltf_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics_analysis,
                    output_path=sdf_path,
                    asset_name=config.short_name,
                    mesh_frame="gltf_y_up",
                )

                # Create SceneObject using shared helper.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={
                        "asset_source": "objaverse",
                        "objaverse_mesh_id": mesh_id,
                        "requested_dimensions": list(request.desired_dimensions[index]),
                        "actual_dimensions": (bbox_max - bbox_min).tolist(),
                    },
                )

                successful_objects.append(scene_obj)

                console_logger.info(
                    f"Objaverse asset retrieved successfully: {config.short_name}"
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process Objaverse asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def _generate_assets_with_model(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using text-to-3D model (Hunyuan3D).

        This method handles the complete generation pipeline:
        - Style change detection and registry reset
        - Request validation (descriptions vs short names, dimensions)
        - Duplicate detection and deduplication
        - Asset path creation
        - Image generation via VLM
        - Mesh generation via geometry server
        - Asset processing and conversion

        Args:
            request: Asset generation request with descriptions and parameters.

        Returns:
            AssetGenerationResult with generated scene objects and metadata.
        """
        # Validate request.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        # Validate desired_dimensions.
        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        # Detect duplicates based on (description, desired_dimensions).
        unique_items: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for i, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_items:
                # This is a duplicate.
                original_idx = unique_items[key]
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(i)
                console_logger.warning(
                    f"Duplicate detected at index {i}: '{desc}' with dimensions "
                    f"{dims} (same as index {original_idx})"
                )
            else:
                # This is unique.
                unique_items[key] = i

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        # Log summary if duplicates found.
        if duplicate_indices:
            total_duplicates = sum(
                len(indices) for indices in duplicate_indices.values()
            )
            console_logger.warning(
                f"Found {total_duplicates} duplicate request(s) across "
                f"{len(duplicate_indices)} description(s). Generating only unique items."
            )

        # Build unique request lists.
        unique_indices = sorted(unique_items.values())
        unique_descriptions = [request.object_descriptions[i] for i in unique_indices]
        unique_short_names = [request.short_names[i] for i in unique_indices]
        unique_dimensions = [request.desired_dimensions[i] for i in unique_indices]

        # Create reduced request with only unique items.
        unique_request = AssetGenerationRequest(
            object_descriptions=unique_descriptions,
            short_names=unique_short_names,
            object_type=request.object_type,
            desired_dimensions=unique_dimensions,
            style_context=request.style_context,
            operation_type=request.operation_type,
            scene_id=request.scene_id,
        )

        # Create asset path configurations.
        asset_paths_configs = self._create_asset_paths(
            object_descriptions=unique_request.object_descriptions,
            short_names=unique_request.short_names,
        )

        # Generate images for all assets.
        self._generate_images(
            request=unique_request, asset_paths_configs=asset_paths_configs
        )

        # Convert images to 3D assets and create SceneObjects.
        successful_objects, failed_assets = self._process_assets_to_scene_objects(
            request=unique_request, asset_path_configs=asset_paths_configs
        )

        console_logger.info(
            f"Asset generation completed: {len(successful_objects)} unique objects "
            f"created, {len(failed_assets)} failed"
        )
        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def generate_assets(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        """Acquire assets under a service deadline, outside LLM active time.

        HSSD queueing, canonicalization, collision decomposition, and SDF
        generation are blocking tool work. They must remain bounded, but must
        not consume the designer/planner inference lease needed to place the
        admitted assets afterwards.
        """
        clock = self._execution_clock
        pause = getattr(clock, "pause_for_external_operation", None)
        resume = getattr(clock, "resume_from_external_operation", None)
        pause_token = pause("asset_acquisition") if callable(pause) else None
        try:
            return self._generate_assets_impl(request)
        finally:
            if callable(resume):
                resume(pause_token)

    def _generate_assets_impl(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate scene assets using configured source (generated or hssd).

        If router is enabled, analyzes requests to split composites and filter
        items before dispatching to the configured asset source.

        Args:
            request: Asset generation request with descriptions and context.

        Returns:
            AssetGenerationResult with successful assets and failure information.
        """
        console_logger.info(
            f"Starting {request.object_type.value} asset acquisition for "
            f"{len(request.object_descriptions)} items using "
            f"'{self.general_asset_source}' source. Router is "
            f"{'enabled' if self.router is not None else 'disabled'}."
        )

        if self._fatal_asset_error:
            return self._fatal_generation_result(request, self._fatal_asset_error)

        if self._reuse_only:
            successful_assets: list[SceneObject] = []
            failed_assets: list[FailedAsset] = []
            seen_families: set[str] = set()
            for index, description in enumerate(request.object_descriptions):
                short_name = (
                    request.short_names[index]
                    if index < len(request.short_names)
                    else ""
                )
                family = semantic_asset_family(description, short_name)
                cached = self._runtime_gate.success_cache.get(family, [])
                if cached and family not in seen_families:
                    successful_assets.append(cached[0])
                    seen_families.add(family)
                elif not cached:
                    failed_assets.append(
                        FailedAsset(
                            index=index,
                            description=description,
                            error_message=(
                                "Placement continuation is reuse-only and has no "
                                f"admitted real asset for family '{family}'."
                            ),
                        )
                    )
            return AssetGenerationResult(
                successful_assets=successful_assets,
                failed_assets=failed_assets,
            )

        original_request = request
        gate_plan = None
        prefetched_assets: list[SceneObject] = []
        gate_failures: list[FailedAsset] = []
        original_indices: list[int] = list(range(len(request.object_descriptions)))
        if self._runtime_gate.enabled and (
            len(request.object_descriptions)
            == len(request.short_names)
            == len(request.desired_dimensions)
        ):
            gate_plan = self._runtime_gate.plan(
                request.object_descriptions,
                request.short_names,
            )
            prefetched_assets = list(gate_plan.cached_assets)
            gate_failures = [
                FailedAsset(
                    index=failure.index,
                    description=failure.description,
                    error_message=failure.reason,
                )
                for failure in gate_plan.failures
            ]
            original_indices = gate_plan.allowed_indices
            if not original_indices:
                console_logger.info(
                    "Asset runtime gate served %d cached asset(s) and blocked/deferred "
                    "%d request item(s); no acquisition call needed",
                    len(prefetched_assets),
                    len(gate_failures),
                )
                return AssetGenerationResult(
                    successful_assets=prefetched_assets,
                    failed_assets=gate_failures,
                )
            request = AssetGenerationRequest(
                object_descriptions=[
                    original_request.object_descriptions[index]
                    for index in original_indices
                ],
                short_names=[
                    original_request.short_names[index] for index in original_indices
                ],
                object_type=original_request.object_type,
                desired_dimensions=[
                    original_request.desired_dimensions[index]
                    for index in original_indices
                ],
                style_context=original_request.style_context,
                operation_type=original_request.operation_type,
                scene_id=original_request.scene_id,
            )

        try:
            # If router is enabled, analyze and potentially modify the request.
            if self.router is not None:
                result = self._generate_assets_with_router(request)

            # Dispatch based on asset source (router disabled).
            elif self.general_asset_source == "hssd":
                result = self._retrieve_hssd_assets(request)
            elif self.general_asset_source == "objaverse":
                result = self._retrieve_objaverse_assets(request)
            elif self.general_asset_source == "generated":
                result = self._generate_assets_with_model(request)
            else:
                # This should never happen due to __init__ validation.
                raise ValueError(f"Unknown asset source: {self.general_asset_source}")
        except FatalRetrievalError as e:
            self._fatal_asset_error = str(e)
            result = self._fatal_generation_result(request, str(e))

        if gate_plan is None:
            return result

        allowed_families = [
            gate_plan.families_by_index[index] for index in original_indices
        ]
        allowed_family_set = set(allowed_families)
        for result_index, asset in enumerate(result.successful_assets):
            inferred_family = semantic_asset_family(
                str(getattr(asset, "description", "")),
                str(getattr(asset, "name", "")),
            )
            if inferred_family not in allowed_family_set:
                inferred_family = allowed_families[
                    min(result_index, len(allowed_families) - 1)
                ]
            self._runtime_gate.remember_success(inferred_family, asset)

        remapped_failures = []
        for failure in result.failed_assets:
            relative_index = int(failure.index)
            original_index = (
                original_indices[relative_index]
                if 0 <= relative_index < len(original_indices)
                else relative_index
            )
            remapped_failures.append(
                FailedAsset(
                    index=original_index,
                    description=failure.description,
                    error_message=failure.error_message,
                )
            )
        return AssetGenerationResult(
            successful_assets=prefetched_assets + result.successful_assets,
            failed_assets=gate_failures + remapped_failures,
            modification_info=result.modification_info,
        )

    def _fatal_generation_result(
        self, request: AssetGenerationRequest, error_message: str
    ) -> AssetGenerationResult:
        """Return a deterministic failure result without calling the router again."""
        console_logger.error(
            "Fatal asset retrieval setup error; skipping asset generation: "
            f"{error_message}"
        )
        return AssetGenerationResult(
            successful_assets=[],
            failed_assets=[
                FailedAsset(
                    index=index,
                    description=description,
                    error_message=(
                        "Fatal asset retrieval setup error. Stop retrying "
                        f"generate_assets until the environment is fixed: {error_message}"
                    ),
                )
                for index, description in enumerate(request.object_descriptions)
            ],
        )

    def _generate_assets_with_router(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using router for LLM-advised analysis and validation.

        Two-phase processing for thread safety:

        **Phase 1 - Parallel (thread-safe HTTP calls):**
        1. Validate request and check for style changes
        2. Deduplicate by (description, dimensions) to save LLM calls
        3. LLM analysis per unique item (split composites, select strategies)
        4. Parallel generation/retrieval via geometry or HSSD server
        5. VLM validation with retry loop (configured max_retries per strategy)

        **Phase 2 - Sequential (main thread, uses bpy):**
        6. GLB→GLTF conversion, floater removal, mesh canonicalization
        7. CoACD collision geometry, SDF generation
        8. Build SceneObjects and modification_info

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with modification_info if request was modified.
        """
        # Validate request lengths.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        all_items: list[AssetItem] = []
        all_discarded_manipulands: list[str] = []
        original_descriptions: list[str] = []
        had_modifications = False
        failed_assets: list[FailedAsset] = []

        # Pre-analysis deduplication: group by (description, dimensions) to save LLM calls.
        # Track duplicates for tool feedback (same format as _generate_assets_with_model).
        unique_requests: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for idx, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_requests:
                # Track duplicate.
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(idx)
            else:
                unique_requests[key] = idx

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        if len(unique_requests) < len(request.object_descriptions):
            console_logger.info(
                f"Pre-analysis deduplication: {len(request.object_descriptions)} requests "
                f"-> {len(unique_requests)} unique"
            )

        # Parallel analysis: LLM API calls are thread-safe.
        configured_workers = self.cfg.asset_manager.router.parallel_workers
        max_workers = min(configured_workers, len(unique_requests))

        console_logger.info(
            f"Analyzing {len(unique_requests)} requests in parallel "
            f"with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.router.analyze_request,
                    description=desc,
                    dimensions=list(dims),
                ): (idx, desc)
                for (desc, dims), idx in unique_requests.items()
            }

            for future in as_completed(futures):
                idx, desc = futures[future]
                try:
                    analysis = future.result()

                    if analysis.error:
                        console_logger.warning(
                            f"Router rejected '{desc}': {analysis.error}"
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx,
                                description=desc,
                                error_message=analysis.error,
                            )
                        )
                        continue

                    # Validate item types match this agent.
                    type_error = self.router.validate_item_types(analysis.items)
                    if type_error:
                        console_logger.warning(
                            f"Router type validation failed: {type_error}"
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=type_error
                            )
                        )
                        continue

                    # Collect items and track modifications.
                    all_items.extend(analysis.items)

                    if analysis.was_modified:
                        had_modifications = True
                        original_descriptions.append(
                            analysis.original_description or desc
                        )
                        if analysis.discarded_manipulands:
                            all_discarded_manipulands.extend(
                                analysis.discarded_manipulands
                            )

                except Exception as e:
                    console_logger.error(
                        f"Analysis failed for '{desc}': {e}", exc_info=True
                    )
                    failed_assets.append(
                        FailedAsset(index=idx, description=desc, error_message=str(e))
                    )

        if not all_items:
            console_logger.warning("Router returned no items to generate")
            return AssetGenerationResult(
                successful_assets=[],
                failed_assets=failed_assets,
                modification_info=None,
            )

        # Deduplicate items by description (same description = generate once).
        unique_items: dict[str, AssetItem] = {}
        for item in all_items:
            if item.description not in unique_items:
                unique_items[item.description] = item
        console_logger.info(
            f"Router produced {len(unique_items)} unique items from "
            f"{len(request.object_descriptions)} requests"
        )

        # Generate/retrieve using router. Handles multiple asset sources internally.
        result = self._generate_items_with_validation(
            unique_items=unique_items, request=request
        )

        # Build modification_info if request was modified.
        modification_info = None
        if had_modifications:
            modification_info = ModificationInfo(
                original_description=", ".join(original_descriptions),
                resulting_descriptions=[
                    item.description for item in unique_items.values()
                ],
                discarded_manipulands=(
                    all_discarded_manipulands if all_discarded_manipulands else None
                ),
            )

        # Combine failed assets from analysis phase with those from generation phase.
        all_failed = failed_assets + result.failed_assets

        return AssetGenerationResult(
            successful_assets=result.successful_assets,
            failed_assets=all_failed,
            modification_info=modification_info,
        )

    def _generate_items_with_validation(
        self, unique_items: dict[str, "AssetItem"], request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate items with overlapped generation and conversion.

        Generates geometry via parallel HTTP calls (thread-safe) and converts each
        mesh to a simulation asset immediately as it completes. This overlaps
        GPU-bound generation with CPU-bound conversion for better resource utilization.

        The main thread runs the as_completed loop and handles conversion (bpy),
        while worker threads continue fetching geometry in parallel.

        Args:
            unique_items: Dict of description -> AssetItem to generate.
            request: Original request (for style_context, object_type).

        Returns:
            AssetGenerationResult with successful assets and failures.
        """
        failed_assets: list[FailedAsset] = []
        successful_assets: list[SceneObject] = []

        configured_workers = self.cfg.asset_manager.router.parallel_workers
        items_list = list(unique_items.items())
        max_workers = min(configured_workers, len(items_list))

        console_logger.info(
            f"Generating {len(items_list)} items with {max_workers} parallel workers "
            "(overlapping generation with conversion)"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_geometry_with_validation,
                    item=item,
                    request=request,
                ): (idx, desc, item)
                for idx, (desc, item) in enumerate(items_list)
            }

            reported_indices: set[int] = set()
            for future in as_completed(futures):
                idx, desc, item = futures[future]
                try:
                    if future.cancelled():
                        continue
                    generated = future.result()
                    if generated is None:
                        console_logger.warning(f"All attempts exhausted for '{desc}'")
                        failed_assets.append(
                            FailedAsset(
                                index=idx,
                                description=desc,
                                error_message="All generation/retrieval attempts exhausted",
                            )
                        )
                        reported_indices.add(idx)
                        continue

                    console_logger.info(
                        f"Geometry acquired for '{desc}', converting..."
                    )

                    # Convert immediately while other geometries are still generating.
                    # This runs on main thread (bpy) while workers fetch next geometry.
                    try:
                        # Handle ArticulatedGeometry (SDF assets) vs GeneratedGeometry.
                        if isinstance(generated, ArticulatedGeometry):
                            scene_obj = self._convert_articulated_to_scene_object(
                                articulated=generated, request=request
                            )
                        else:
                            scene_obj = self._convert_generated_to_scene_object(
                                item=item, generated=generated, request=request
                            )
                        successful_assets.append(scene_obj)
                        reported_indices.add(idx)
                        console_logger.info(f"Successfully converted asset: '{desc}'")
                    except Exception as e:
                        console_logger.error(
                            f"Mesh conversion failed for '{desc}': {e}", exc_info=True
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=str(e)
                            )
                        )
                        reported_indices.add(idx)

                except FatalRetrievalError as e:
                    fatal_message = str(e)
                    self._fatal_asset_error = fatal_message
                    console_logger.error(
                        "Fatal asset retrieval setup error; cancelling remaining "
                        f"asset work for this batch: {fatal_message}"
                    )

                    for pending in futures:
                        if pending is not future:
                            pending.cancel()

                    for (
                        _pending_future,
                        (pending_idx, pending_desc, _),
                    ) in futures.items():
                        if pending_idx in reported_indices:
                            continue
                        failed_assets.append(
                            FailedAsset(
                                index=pending_idx,
                                description=pending_desc,
                                error_message=(
                                    "Fatal asset retrieval setup error. Stop "
                                    "retrying generate_assets until the "
                                    f"environment is fixed: {fatal_message}"
                                ),
                            )
                        )
                        reported_indices.add(pending_idx)
                    break

                except Exception as e:
                    console_logger.error(
                        f"Geometry generation failed for '{desc}': {e}", exc_info=True
                    )
                    failed_assets.append(
                        FailedAsset(index=idx, description=desc, error_message=str(e))
                    )
                    reported_indices.add(idx)

        console_logger.info(
            f"Router generation completed: {len(successful_assets)} success, "
            f"{len(failed_assets)} failed"
        )

        return AssetGenerationResult(
            successful_assets=successful_assets, failed_assets=failed_assets
        )

    def _generate_geometry_with_validation(
        self, item: AssetItem, request: AssetGenerationRequest
    ) -> GeneratedGeometry | ArticulatedGeometry | None:
        """Generate/retrieve validated geometry for a single item. Thread-safe.

        This method only performs HTTP-based operations (geometry server, HSSD server,
        BlenderServer for validation rendering) and is safe to call from worker threads.

        Args:
            item: The asset item to generate/retrieve.
            request: Original request (for style_context).

        Returns:
            GeneratedGeometry or ArticulatedGeometry if successful,
            None if all strategies/candidates exhausted.
        """
        return self.router.generate_with_validation(
            item=item,
            geometry_client=self.geometry_client,
            image_generator=self.image_generator,
            images_dir=self.images_dir,
            geometry_dir=self.geometry_dir,
            debug_dir=self.debug_dir,
            style_context=request.style_context,
            hssd_client=self.hssd_client,
            objaverse_client=self.objaverse_client,
            articulated_client=self.articulated_client,
            materials_client=self.materials_client,
            scene_id=request.scene_id,
        )

    def _convert_generated_to_scene_object(
        self,
        item: "AssetItem",
        generated: "GeneratedGeometry",
        request: AssetGenerationRequest,
    ) -> SceneObject:
        """Convert validated geometry to SceneObject. Must run on main thread.

        This method uses bpy for GLB→GLTF conversion and must be called from the
        main thread, not from ThreadPoolExecutor workers.

        Args:
            item: The asset item that was generated.
            generated: The validated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.

        Raises:
            Exception: If mesh conversion or SDF generation fails.
        """
        # Derive base_name from geometry path (already has unique timestamp or HSSD ID).
        base_name = generated.geometry_path.stem

        config = AssetPathConfig(
            description=item.description,
            short_name=item.short_name,
            image_path=generated.image_path,
            geometry_path=generated.geometry_path,
            sdf_dir=self.sdf_dir / base_name,
        )
        config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Thin coverings use simplified conversion: no VLM analysis.
        # Wall thin coverings (paintings, posters) get collision geometry.
        if generated.asset_source == "thin_covering":
            is_wall_covering = request.object_type == ObjectType.WALL_MOUNTED

            # Only add collision for wall coverings (paintings, posters).
            collision_dims = None
            collision_shape = "rectangular"
            if is_wall_covering and item.dimensions:
                # Wall covering dims: (width, depth, height) where depth is thickness.
                thickness = (
                    self.cfg.asset_manager.router.strategies.thin_covering.thickness_m
                )
                collision_dims = (item.dimensions[0], thickness, item.dimensions[2])
                collision_shape = infer_thin_covering_shape(item.description)

            sdf_path, final_gltf_path, bbox_min, bbox_max = (
                self._convert_thin_covering_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    collision_dims=collision_dims,
                    collision_shape=collision_shape,
                )
            )
            initial_scale = 1.0  # Thin coverings don't scale the mesh.
        else:
            # Convert validated geometry to simulation asset (physics analysis, SDF).
            sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale = (
                self._convert_mesh_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    object_type=request.object_type,
                    desired_dimensions=item.dimensions,
                    asset_source=generated.asset_source,
                )
            )

        # Build additional metadata using explicit asset_source from GeneratedGeometry.
        additional_metadata = {"asset_source": generated.asset_source}
        if generated.hssd_id is not None:
            additional_metadata["hssd_mesh_id"] = generated.hssd_id
        if generated.asset_source == "hssd":
            # HSSD support-surface annotations use source-asset coordinates.
            # Runtime geometry is already canonicalized and scaled.
            additional_metadata["source_to_canonical_scale"] = float(initial_scale)

        # Add thin_covering-specific metadata for physics validation.
        if generated.asset_source == "thin_covering":
            additional_metadata["width_m"] = item.dimensions[0]
            additional_metadata["depth_m"] = item.dimensions[1]
            additional_metadata["shape"] = infer_thin_covering_shape(item.description)
            # Wall coverings use Drake collision; floor/manipuland use 2D OBB overlap.
            additional_metadata["is_wall_covering"] = (
                request.object_type == ObjectType.WALL_MOUNTED
            )

        # Keep original object_type - thin coverings are identified via asset_source
        # metadata, not object_type. This preserves semantic category (FURNITURE,
        # WALL_MOUNTED, MANIPULAND) for stage-based filtering in snapshots.
        object_type = request.object_type

        # Create SceneObject.
        return self._create_scene_object(
            config=config,
            object_type=object_type,
            sdf_path=sdf_path,
            final_gltf_path=final_gltf_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            additional_metadata=additional_metadata,
            scale_factor=1.0,
        )

    def _convert_articulated_to_scene_object(
        self, articulated: ArticulatedGeometry, request: AssetGenerationRequest
    ) -> SceneObject:
        """Convert articulated retrieval result to SceneObject.

        Unlike generated assets, articulated objects already have:
        - Pre-processed SDF with links and joints
        - Bounding box at default pose (joints=0)
        - No need for VLM analysis or mesh canonicalization

        We combine the visual meshes at default pose for geometry_path (needed
        for collision checks, support surface extraction, snapping).

        Args:
            articulated: The articulated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.
        """
        item = articulated.item
        safe_name = self._sanitize_filename(item.short_name)
        timestamp = int(time.time())
        base_name = f"{safe_name}_{timestamp}"

        # Create output directory for combined geometry.
        output_dir = self.geometry_dir / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy articulated SDF directory to output for replay and export.
        # The SDF references meshes via relative paths, so we copy the entire directory.
        source_sdf_dir = articulated.sdf_path.parent
        dest_sdf_dir = self.sdf_dir / base_name
        console_logger.info(
            f"Copying articulated SDF directory from {source_sdf_dir} to {dest_sdf_dir}"
        )
        shutil.copytree(source_sdf_dir, dest_sdf_dir)
        copied_sdf_path = dest_sdf_dir / articulated.sdf_path.name

        # Add self-collision filtering if enabled.
        if self.cfg.asset_manager.articulated.enable_self_collision_filtering:
            add_self_collision_filter(copied_sdf_path)

        # Fix ArtVIP texture paths: GLTF files reference textures with relative paths,
        # but textures are in *_meshes/ subdirectories. Copy textures to parent dir.
        for meshes_subdir in dest_sdf_dir.glob("*_meshes"):
            for texture_file in meshes_subdir.glob("*.png"):
                dest_texture = dest_sdf_dir / texture_file.name
                if not dest_texture.exists():
                    shutil.copy2(texture_file, dest_texture)

        # Combine SDF visual meshes at default pose (joints=0) for geometry operations.
        console_logger.info(
            f"Combining articulated meshes at default pose for '{item.description}'"
        )
        combined_mesh = combine_sdf_meshes_at_joint_angles(
            copied_sdf_path, use_max_angles=False
        )

        # Save combined mesh as GLTF for collision checks, snapping, etc.
        combined_gltf_path = output_dir / f"{safe_name}_combined.gltf"
        combined_mesh.export(combined_gltf_path)

        console_logger.info(
            f"Articulated asset combined mesh saved to {combined_gltf_path}"
        )

        # Build metadata for provenance tracking.
        metadata = {
            "asset_source": "articulated",
            "articulated_source": articulated.source,
            "articulated_id": articulated.object_id,
            "is_articulated": True,
            "generation_timestamp": time.time(),
        }

        # Create SceneObject with copied SDF path and combined geometry.
        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(item.short_name),
            object_type=request.object_type,
            name=item.short_name,
            description=item.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=combined_gltf_path,
            sdf_path=copied_sdf_path,
            image_path=None,  # No generated image for articulated assets.
            bbox_min=np.array(articulated.bounding_box_min),
            bbox_max=np.array(articulated.bounding_box_max),
            metadata=metadata,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        console_logger.info(
            f"Articulated asset registered: {item.short_name} "
            f"(source={articulated.source}, id={articulated.object_id})"
        )

        return scene_obj

    def _create_asset_paths(
        self, object_descriptions: list[str], short_names: list[str]
    ) -> list[AssetPathConfig]:
        """Create file paths and identifiers for each asset to be generated.

        Args:
            object_descriptions: List of object descriptions to generate.
            short_names: List of short names for filesystem-safe file naming.

        Returns:
            List of AssetPathConfig objects containing asset paths and metadata.
        """
        asset_paths = []
        batch_stamp = time.time_ns()
        for index, (desc, short_name) in enumerate(
            zip(object_descriptions, short_names)
        ):
            # Use sanitized short name for file naming.
            safe_name = self._sanitize_filename(short_name)
            base_name = f"{safe_name}_{index:03d}_{batch_stamp}"

            asset_paths.append(
                AssetPathConfig(
                    description=desc,
                    short_name=short_name,
                    image_path=self.images_dir / f"{base_name}.png",
                    geometry_path=self.geometry_dir / f"{base_name}.glb",
                    sdf_dir=self.sdf_dir / base_name,
                )
            )
        return asset_paths

    def _generate_images(
        self,
        request: AssetGenerationRequest,
        asset_paths_configs: list[AssetPathConfig],
    ) -> None:
        """Generate images for all assets using the image generator.

        Args:
            request: Asset generation request with style and operation details.
            asset_paths_configs: List of asset path configurations.
        """
        style_prompt = request.style_context or "Modern style"
        console_logger.info(f"Generating {len(request.object_descriptions)} images")
        console_logger.debug(f"Style prompt: {style_prompt}")

        output_paths = [config.image_path for config in asset_paths_configs]

        start_time = time.time()
        self.image_generator.generate_images(
            style_prompt=style_prompt,
            object_descriptions=request.object_descriptions,
            output_paths=output_paths,
        )

        elapsed = time.time() - start_time
        console_logger.info(
            f"Generated {len(request.object_descriptions)} images in "
            f"{elapsed:.2f} seconds"
        )

    def _process_assets_to_scene_objects(
        self, request: AssetGenerationRequest, asset_path_configs: list[AssetPathConfig]
    ) -> tuple[list[SceneObject], list[FailedAsset]]:
        """Convert generated images to 3D assets and create SceneObjects.

        Uses batch processing to optimize GPU utilization by pipelining geometry
        generation and Drake SDF conversion. Handles failures gracefully by
        collecting failed assets instead of raising exceptions, allowing all
        generated geometries to be processed.

        Args:
            request: Asset generation request.
            asset_path_configs: List of asset path configurations.

        Returns:
            Tuple of (successful_objects, failed_assets). The successful_objects
            list contains SceneObject instances ready for placement. The failed_assets
            list contains FailedAsset instances with error details.
        """
        if not asset_path_configs:
            return [], []

        # Create Drake asset directories for all configs.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Prepare batch geometry generation requests.
        geometry_requests = []
        for config in asset_path_configs:
            expected_filename = config.geometry_path.name

            # Extract backend configuration.
            backend = self.cfg.asset_manager.backend

            # Prepare SAM3D config if backend is sam3d.
            sam3d_config = None
            if backend == "sam3d":
                sam3d_cfg = self.cfg.asset_manager.sam3d
                mode = sam3d_cfg.mode
                sam3d_config = {
                    "sam3_checkpoint": str(sam3d_cfg.sam3_checkpoint),
                    "sam3d_checkpoint": str(sam3d_cfg.sam3d_checkpoint),
                    "mode": mode,
                    "text_prompt": getattr(sam3d_cfg, "text_prompt", None),
                    "threshold": sam3d_cfg.threshold,
                }
                # Pass object description for "object_description" mode.
                # Uses the same description that generated the image for
                # semantic-aware segmentation.
                if mode == "object_description":
                    sam3d_config["object_description"] = config.description

            geometry_request = GeometryGenerationServerRequest(
                image_path=str(config.image_path),
                output_dir=str(self.geometry_dir),
                prompt=config.description,
                debug_folder=str(self.debug_dir),
                output_filename=expected_filename,
                backend=backend,
                sam3d_config=sam3d_config,
                scene_id=request.scene_id,
            )
            geometry_requests.append(geometry_request)

        console_logger.info(
            f"Submitting batch geometry generation for {len(geometry_requests)} assets"
        )

        # Initialize result tracking.
        scene_objects = []
        failed_assets = []

        # Process batch results as they stream back.
        # This enables pipelining: Drake conversion for asset N while GPU processes
        # asset N+1.
        for index, result in self.geometry_client.generate_geometries(
            geometry_requests
        ):
            # Handle geometry generation failures.
            if isinstance(result, GeometryGenerationError):
                console_logger.error(
                    f"Geometry generation failed for asset {index + 1}/"
                    f"{len(asset_path_configs)} ({asset_path_configs[index].description}): "
                    f"{result.error_message}"
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=result.error_message,
                    )
                )
                continue

            try:
                config = asset_path_configs[index]
                server_geometry_path = Path(result.geometry_path)

                console_logger.info(
                    f"Converting asset {index + 1}/{len(asset_path_configs)} to Drake "
                    f"format: {config.description}"
                )

                # Process mesh: VLM → canonicalize → scale → collision → SDF.
                sdf_path, final_gltf_path, bbox_min, bbox_max, _ = (
                    self._convert_mesh_to_simulation_asset(
                        geometry_path=server_geometry_path,
                        config=config,
                        object_type=request.object_type,
                        desired_dimensions=request.desired_dimensions[index],
                    )
                )

                # Create the SceneObject.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={"asset_source": "generated"},
                    scale_factor=1.0,
                )

                scene_objects.append(scene_obj)
                console_logger.info(
                    f"Successfully generated asset {index + 1}/{len(asset_path_configs)}: "
                    f"{config.description}"
                )

            except Exception as e:
                # Log failure but continue processing remaining assets.
                console_logger.error(
                    f"Failed to process asset {index + 1}/{len(asset_path_configs)} "
                    f"({asset_path_configs[index].description}): {e}",
                    exc_info=True,
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=str(e),
                    )
                )

        # Log summary.
        if failed_assets:
            console_logger.warning(
                f"Asset generation completed with {len(failed_assets)} failure(s) "
                f"and {len(scene_objects)} success(es)"
            )
        else:
            console_logger.info(
                f"Successfully processed all {len(scene_objects)} assets"
            )

        return scene_objects, failed_assets

    def _convert_mesh_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        object_type: ObjectType,
        desired_dimensions: list[float] | None = None,
        asset_source: str = "generated",
    ) -> tuple[Path, Path, np.ndarray, np.ndarray, float]:
        """Convert mesh to a simulatable Drake SDF.

        Pipeline:
        - Convert GLB → Y-up GLTF (enables VLM analysis in Blender's Z-up space)
        - Remove mesh floaters (disconnected components below volume threshold)
        - VLM analysis → orientation + material + mass (in Blender coords)
        - Canonicalize in Blender → rotate to canonical orientation + placement
          (Y-up GLTF input → Z-up GLTF output for Drake)
        - Scale to desired dimensions (if provided)
        - Collision → CoACD decomposition
        - SDF → Drake format with physics properties

        Multi-view images used for VLM physics analysis are saved to
        generated_assets/debug/<base_name>/ where <base_name> follows the pattern
        {sanitized_short_name}_{timestamp} (e.g., "office_desk_A_1759997032").

        Args:
            geometry_path: Path to raw GLB mesh from Hunyuan3D or HSSD.
            config: Asset path configuration.
            object_type: Type of object (determines placement strategy).
            desired_dimensions: Optional dimensions (width, depth, height) from agent.
            asset_source: Source of the asset ("generated" or "hssd"). HSSD assets
                use specialized VLM prompts and skip vertical views since they're
                already upright.

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max, scale_factor).
            The scale_factor is the uniform scaling applied during mesh scaling
            (1.0 if no scaling was applied). This is needed to correctly scale
            HSSD pre-computed support surfaces.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Processing mesh ({geometry_path}) to simulation asset "
            f"(object_type={object_type.value})"
        )

        # Convert GLB to Y-up GLTF (enables VLM analysis in Blender's Z-up space).
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Remove floaters from mesh before VLM analysis.
        console_logger.info("Removing disconnected mesh floaters")
        remove_mesh_floaters(
            mesh_path=gltf_path,
            output_path=gltf_path,
            distance_threshold=self.cfg.asset_manager.floater_distance_threshold,
        )

        # Resolve orientation, material, and mass. Generated assets use the VLM;
        # HSSD assets use deterministic metadata unless explicitly overridden.
        # Keep a debug directory available for the VLM path.
        # Use geometry_path stem to match asset naming pattern (e.g., "desk_A_1234567890").
        debug_dir = self.debug_dir / config.geometry_path.stem

        # HSSD assets use specialized prompts and skip vertical views since they're
        # already upright (Z-up). Generated assets need full orientation analysis.
        is_hssd = asset_source == "hssd"
        prompt_type = "hssd" if is_hssd else "generated"

        console_logger.info(
            "Resolving mesh physics (asset_source=%s, prompt_type=%s)",
            asset_source,
            prompt_type,
        )
        physics_analysis = self._analyze_mesh_physics(
            mesh_path=gltf_path,
            asset_source=asset_source,
            object_name=config.short_name,
            debug_output_dir=debug_dir,
        )

        console_logger.info(
            f"Mesh physics complete: up={physics_analysis.up_axis}, "
            f"front={physics_analysis.front_axis}, material={physics_analysis.material}, "
            f"mass={physics_analysis.mass_kg}kg"
        )

        # Canonicalize mesh in Blender. The file remains standard glTF Y-up;
        # SceneSmith conversion happens only at the dimensions/bounds boundary.
        canonical_path = config.sdf_dir / f"{config.short_name}_canonical.gltf"
        canonicalize_mesh(
            gltf_path=gltf_path,
            output_path=canonical_path,
            up_axis=physics_analysis.up_axis,
            front_axis=physics_analysis.front_axis,
            blender_server=self.blender_server,
            object_type=object_type,
        )

        # Scale mesh to desired dimensions (if provided). The returned source
        # scale is provenance for precomputed HSSD support surfaces only; it is
        # already baked into the final visual and collision geometry.
        final_gltf_path, bbox_min, bbox_max, applied_scale = (
            self._scale_and_measure_canonical_mesh(
                canonical_path=canonical_path,
                final_path=config.sdf_dir / f"{config.short_name}.gltf",
                desired_dimensions=desired_dimensions,
            )
        )
        initial_scale = applied_scale if is_hssd else 1.0

        # Generate collision geometry via convex decomposition server.
        collision_pieces = self._generate_collision_geometry(final_gltf_path)

        # Generate Drake SDF.
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_drake_sdf(
            visual_mesh_path=final_gltf_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics_analysis,
            output_path=sdf_path,
            asset_name=config.short_name,
            mesh_frame="gltf_y_up",
        )

        console_logger.info(
            f"Drake SDF complete: SDF at {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale

    def _convert_thin_covering_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        collision_dims: tuple[float, float, float] | None = None,
        collision_shape: str = "rectangular",
    ) -> tuple[Path, Path, np.ndarray, np.ndarray]:
        """Convert thin covering mesh to Drake SDF (simplified pipeline).

        Thin coverings are static decorative objects that don't require:
        - VLM orientation analysis (already correctly oriented)
        - Canonicalization (already in correct pose)
        - Collision geometry for floor/manipuland coverings (decorative only)

        Wall thin coverings (paintings, posters) DO get collision geometry so
        Drake can detect furniture collisions.

        Pipeline:
        - Convert GLB → GLTF with separate textures (for Drake)
        - Generate static SDF (with optional collision for wall coverings)
        - Compute bounding box from mesh

        Args:
            geometry_path: Path to thin covering GLB file.
            config: Asset path configuration.
            collision_dims: Optional (width, depth, height) for collision geometry.
                Used for wall thin coverings.
            collision_shape: Shape of collision ("rectangular" or "circular").

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max).
        """
        console_logger.info(f"Processing thin covering ({geometry_path}) to static SDF")

        # Convert GLB to GLTF with separate textures for Drake.
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Generate static SDF (with optional collision geometry for wall coverings).
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_thin_covering_sdf(
            visual_mesh_path=gltf_path,
            output_path=sdf_path,
            model_name=config.short_name,
            collision_dims=collision_dims,
            collision_shape=collision_shape,
        )

        # Load mesh for bounding box calculation.
        mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
        # Trimesh exposes standard glTF coordinates. Scene placement is Z-up.
        bbox_min, bbox_max = gltf_y_up_bounds_to_scene_z_up(mesh.bounds)

        console_logger.info(
            f"Thin covering SDF complete: {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, gltf_path, bbox_min, bbox_max

    def _find_sdf_file(self, sdf_dir: Path) -> Path:
        """Find the generated SDF file in the asset directory.

        Args:
            sdf_dir: Directory containing the generated SDF file.

        Returns:
            Path to the SDF file.

        Raises:
            RuntimeError: If no SDF file or multiple SDF files are found.
        """
        # First try direct search in the directory.
        sdf_files = list(sdf_dir.glob("*.sdf"))

        # If not found, search recursively (create_drake_asset_from_geometry creates
        # nested dirs).
        if not sdf_files:
            sdf_files = list(sdf_dir.glob("**/*.sdf"))

        if not sdf_files:
            raise RuntimeError(f"No SDF file generated in {sdf_dir}")
        if len(sdf_files) > 1:
            raise RuntimeError(f"Multiple SDF files generated in {sdf_dir}")
        return sdf_files[0].absolute()

    def _create_scene_object(
        self,
        config: AssetPathConfig,
        object_type: ObjectType,
        sdf_path: Path,
        final_gltf_path: Path,
        bbox_min: np.ndarray | None = None,
        bbox_max: np.ndarray | None = None,
        additional_metadata: dict | None = None,
        scale_factor: float = 1.0,
    ) -> SceneObject:
        """Convert assets to SceneObject (supports both generated and HSSD).

        Args:
            config: Asset path configuration containing metadata and paths.
            object_type: Type of object.
            sdf_path: Path to the generated SDF file.
            final_gltf_path: Path to the final scaled GLTF mesh file.
            bbox_min: Minimum corner of object-frame bounding box.
            bbox_max: Maximum corner of object-frame bounding box.
            additional_metadata: Optional metadata to merge into the object's
                metadata dict. Useful for HSSD assets to add {"asset_source": "hssd"}.
            scale_factor: Runtime-only scale not already baked into the final
                glTF/SDF. HSSD source scaling belongs in
                ``metadata['source_to_canonical_scale']``.

        Returns:
            Complete SceneObject ready for scene placement.
        """
        # Base metadata common to all assets.
        metadata = {"generation_timestamp": time.time()}

        # Merge additional metadata (for HSSD: {"asset_source": "hssd"}).
        if additional_metadata:
            metadata.update(additional_metadata)

        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(config.short_name),
            object_type=object_type,
            name=config.short_name,
            description=config.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=final_gltf_path,
            sdf_path=sdf_path,
            image_path=config.image_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            metadata=metadata,
            scale_factor=scale_factor,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        return scene_obj

    def get_asset_by_id(self, asset_id: UniqueID) -> SceneObject | None:
        """Get a registered asset by ID.

        Args:
            asset_id: Unique identifier of the asset.

        Returns:
            SceneObject if found, None otherwise.
        """
        return self.registry.get(asset_id)

    def list_available_assets(self) -> list[SceneObject]:
        """List all assets available for reuse.

        Returns:
            List of all registered SceneObjects.
        """
        return [
            asset
            for asset in self.registry.list_all()
            if self._runtime_gate.is_asset_admitted(asset)
        ]

    @staticmethod
    def _asset_identity_signatures(asset: SceneObject) -> set[str]:
        signatures: set[str] = set()
        for attribute in ("sdf_path", "geometry_path"):
            value = getattr(asset, attribute, None)
            if value:
                signatures.add(f"{attribute}:{Path(value)}")
        metadata = getattr(asset, "metadata", {}) or {}
        mesh_id = metadata.get("hssd_mesh_id")
        if mesh_id:
            signatures.add(f"hssd_mesh_id:{mesh_id}")
        return signatures

    def invalidate_assets(
        self,
        assets: list[SceneObject],
        *,
        reason: str,
    ) -> int:
        """Revoke assets that failed the final scene geometry contract.

        Semantic VLM admission alone is insufficient for required furniture:
        the canonical mesh must also satisfy the exact dimensions used by the
        downstream hard verifier. Revocation updates both the registry view and
        the per-stage semantic cache so regeneration can acquire a different
        HSSD mesh instead of replaying the rejected one.
        """

        invalid_signatures: set[str] = set()
        invalid_families: set[str] = set()
        for asset in assets:
            invalid_signatures.update(self._asset_identity_signatures(asset))
            invalid_families.add(
                semantic_asset_family(
                    str(getattr(asset, "description", "")),
                    str(getattr(asset, "name", "")),
                )
            )

        revoked = 0
        for registered in self.registry.list_all():
            if not (self._asset_identity_signatures(registered) & invalid_signatures):
                continue
            metadata = getattr(registered, "metadata", None)
            if metadata is None:
                metadata = {}
                registered.metadata = metadata
            if metadata.get("asset_admission_failed", False):
                continue
            metadata["asset_admission_failed"] = True
            metadata["asset_admission_failure_reason"] = str(reason)
            revoked += 1

        for family in invalid_families:
            self._runtime_gate.invalidate_family(family)

        if revoked and self.registry.auto_save_path:
            self.registry.save_to_file(self.registry.auto_save_path)
        console_logger.warning(
            "Revoked %d asset template(s) from families %s: %s",
            revoked,
            sorted(invalid_families),
            reason,
        )
        return revoked

    def _extract_bounds_from_visual_mesh(
        self, sdf_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract AABB from the visual GLTF mesh after conversion.

        Args:
            sdf_path: Path to the SDF file.

        Returns:
            Tuple of (bbox_min, bbox_max) arrays.

        Raises:
            FileNotFoundError: If GLTF file is not found.
            ValueError: If mesh cannot be loaded or is invalid.
        """
        # Pattern: {sdf_dir}/{asset_name}/{asset_name}.gltf
        gltf_path = sdf_path.with_suffix(".gltf")

        if not gltf_path.exists():
            raise FileNotFoundError(
                f"Visual GLTF not found at expected path: {gltf_path}"
            )

        # Load mesh using trimesh.
        mesh = trimesh.load(gltf_path, force="mesh")

        # Handle Scene objects (multiple meshes).
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.Trimesh()
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
            mesh = combined_mesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load valid mesh from {gltf_path}")

        # Extract bounds.
        bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        bbox_min, bbox_max = gltf_y_up_bounds_to_scene_z_up(bounds)

        console_logger.debug(
            f"Extracted bounds from {gltf_path}: min={bbox_min}, max={bbox_max}"
        )

        return bbox_min, bbox_max

    def clear_asset_registry(self) -> None:
        """Clear the asset registry."""
        self.registry.clear()
        self._runtime_gate.clear_success_cache()
