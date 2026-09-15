"""HSSD object library retrieval system.

Adapted from HSM (https://arxiv.org/abs/2503.16848).
"""

__all__ = [
    "HssdConfig",
    "HssdRetriever",
]


def __getattr__(name: str):
    """Avoid loading CLIP/trimesh retrieval dependencies for sibling modules."""
    if name == "HssdConfig":
        from scenesmith.agent_utils.hssd_retrieval.config import HssdConfig

        return HssdConfig
    if name == "HssdRetriever":
        from scenesmith.agent_utils.hssd_retrieval.retrieval import HssdRetriever

        return HssdRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
