"""Dataclasses for the asset router module."""

from dataclasses import dataclass, field
from pathlib import Path

from scenesmith.agent_utils.room import ObjectType


@dataclass
class AssetItem:
    """Single asset to generate (from LLM analysis)."""

    description: str
    """The asset description to use for generation."""

    short_name: str
    """Short name for the asset (lowercase_with_underscores)."""

    dimensions: list[float]
    """Dimensions [width, depth, height] in meters."""

    object_type: ObjectType
    """Type: FURNITURE, MANIPULAND, or EITHER."""

    strategies: list[str]
    """Strategy chain to try, e.g. ["articulated", "generated"]."""

    thin_covering_type: str | None = None
    """Type of thin covering texture: "tileable" or "single_image".
    Only set when strategies includes "thin_covering".
    - "tileable": Pattern repeats across surface (rugs, carpets).
    - "single_image": One image spans entire surface (posters, paintings)."""


@dataclass
class AnalysisResult:
    """LLM analysis output from request analysis."""

    items: list[AssetItem]
    """List of assets to generate."""

    original_description: str | None
    """Set if the request was modified (split or filtered)."""

    discarded_manipulands: list[str] | None
    """Manipulands filtered out by furniture agent."""

    error: str | None = None
    """Set if the request was rejected."""

    @property
    def was_modified(self) -> bool:
        """True if original request was changed (split or filtered)."""
        return (
            self.original_description is not None
            or self.discarded_manipulands is not None
        )


@dataclass
class ModificationInfo:
    """Feedback to designer when request was modified."""

    original_description: str
    """What was originally requested."""

    resulting_descriptions: list[str]
    """What was actually generated."""

    discarded_manipulands: list[str] | None = None
    """What manipulands were filtered out (furniture agent only)."""


@dataclass
class ValidationResult:
    """Output from VLM validation of generated asset."""

    is_acceptable: bool
    """Whether the asset passes validation."""

    reason: str
    """Explanation for the decision (logged for debugging)."""

    suggestions: list[str] = field(default_factory=list)
    """Suggestions if rejected (what to try differently)."""


@dataclass
class GeneratedGeometry:
    """Result of validated geometry generation or retrieval."""

    geometry_path: Path
    """Path to the validated geometry file (GLB/GLTF)."""

    item: AssetItem
    """The item that was generated/retrieved."""

    asset_source: str
    """Source strategy that produced this geometry (e.g., 'generated', 'hssd',
    'articulated', 'thin_covering'). Used to track provenance in asset metadata."""

    image_path: Path | None = None
    """Path to the reference image (for generated assets, None for HSSD)."""

    hssd_id: str | None = None
    """HSSD object ID (for HSSD assets, None for others)."""

    objaverse_uid: str | None = None
    """Objaverse/ObjectThor unique identifier (for objaverse assets, None for others)."""


@dataclass
class ArticulatedGeometry:
    """Result of articulated object retrieval (pre-processed SDF).

    Unlike GeneratedGeometry which contains a single mesh, articulated objects
    have multi-link SDF files with joints (doors, drawers, etc.).
    """

    sdf_path: Path
    """Path to the articulated SDF file."""

    item: AssetItem
    """The item that was retrieved."""

    source: str
    """Data source: 'partnet_mobility' or 'artvip'."""

    object_id: str
    """Object ID within the source dataset (for provenance tracking)."""

    bounding_box_min: list[float]
    """Bounding box minimum [x, y, z] at default pose (joints=0)."""

    bounding_box_max: list[float]
    """Bounding box maximum [x, y, z] at default pose (joints=0)."""
