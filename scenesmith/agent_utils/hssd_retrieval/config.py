"""Configuration for HSSD retrieval system."""

import logging

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class HssdZvecConfig:
    """Configuration for Zvec-backed HSSD retrieval."""

    collection_path: Path
    """Path to the local Zvec collection directory."""

    base_url: str
    """Base URL for the llama.cpp embedding server."""

    embedding_dimension: int = 2048
    """Expected text embedding dimension."""

    embedding_field: str = "embedding"
    """Vector field name in the Zvec collection."""

    top_k_factor: int = 4
    """Multiplier to fetch a larger semantic pool before bbox re-ranking."""

    media_marker: str = "<__media__>"
    """Media marker kept for parity with the indexing script config."""

    timeout_seconds: float = 120.0
    """Timeout for llama.cpp embedding requests."""

    embd_normalize: int = 2
    """Normalization mode passed to llama.cpp /embeddings."""

    request_retries: int = 2
    """Number of retries for embedding requests."""

    retry_sleep_seconds: float = 1.0
    """Base retry backoff for embedding requests."""

    def __post_init__(self) -> None:
        """Validate config values."""
        self.collection_path = Path(self.collection_path)

        if not self.collection_path.exists():
            raise FileNotFoundError(
                f"HSSD Zvec collection path does not exist: {self.collection_path}"
            )

        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        if self.top_k_factor <= 0:
            raise ValueError("top_k_factor must be positive")


@dataclass
class HssdConfig:
    """Configuration for HSSD asset retrieval."""

    data_path: Path
    """Path to HSSD models directory (containing objects/ subdirectory)."""

    preprocessed_path: Path
    """Path to preprocessed data (indices, embeddings)."""

    retrieval_backend: str = "clip"
    """Semantic retrieval backend: 'clip' or 'embedding'."""

    use_top_k: int = 5
    """Number of top semantic candidates to consider before size ranking."""

    object_type_mapping: dict[str, str] | None = None
    """Map scenesmith ObjectType to HSSD categories."""

    zvec: HssdZvecConfig | None = None
    """Optional Zvec retrieval configuration for retrieval_backend='embedding'."""

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        self.data_path = Path(self.data_path)
        self.preprocessed_path = Path(self.preprocessed_path)
        self.retrieval_backend = str(self.retrieval_backend).lower()

        if not self.data_path.exists():
            raise FileNotFoundError(f"HSSD data path does not exist: {self.data_path}")

        if not self.preprocessed_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data path does not exist: {self.preprocessed_path}"
            )

        if self.retrieval_backend not in {"clip", "embedding"}:
            raise ValueError(
                "HSSD retrieval_backend must be one of: 'clip', 'embedding'"
            )

        if self.object_type_mapping is None:
            self.object_type_mapping = {
                "FURNITURE": "large_objects",
                "MANIPULAND": "small_objects",
                "WALL_MOUNTED": "wall_objects",
                "CEILING_MOUNTED": "ceiling_objects",
            }

        if self.retrieval_backend == "embedding" and self.zvec is None:
            raise ValueError(
                "HSSD retrieval_backend='embedding' requires hssd.zvec config"
            )

        console_logger.info(
            f"HSSD config initialized:\n"
            f"  data_path: {self.data_path}\n"
            f"  preprocessed_path: {self.preprocessed_path}\n"
            f"  retrieval_backend: {self.retrieval_backend}\n"
            f"  top_k: {self.use_top_k}"
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "HssdConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: HSSD config subtree (cfg.asset_manager.hssd).

        Returns:
            HssdConfig instance.
        """
        zvec_cfg = None
        if "zvec" in cfg and cfg.zvec is not None:
            zvec_cfg = HssdZvecConfig(
                collection_path=Path(cfg.zvec.collection_path),
                base_url=str(cfg.zvec.base_url),
                embedding_dimension=int(cfg.zvec.embedding_dimension),
                embedding_field=str(cfg.zvec.embedding_field),
                top_k_factor=int(cfg.zvec.top_k_factor),
                media_marker=str(cfg.zvec.media_marker),
                timeout_seconds=float(cfg.zvec.timeout_seconds),
                embd_normalize=int(cfg.zvec.embd_normalize),
                request_retries=int(cfg.zvec.request_retries),
                retry_sleep_seconds=float(cfg.zvec.retry_sleep_seconds),
            )

        return cls(
            data_path=Path(cfg.data_path),
            preprocessed_path=Path(cfg.preprocessed_path),
            retrieval_backend=str(getattr(cfg, "retrieval_backend", "clip")),
            use_top_k=cfg.use_top_k,
            object_type_mapping=dict(cfg.object_type_mapping),
            zvec=zvec_cfg,
        )
