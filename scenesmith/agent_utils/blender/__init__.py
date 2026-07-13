"""Blender client API with bpy-dependent components loaded lazily."""

from typing import Any

from .params import RenderParams
from .server_manager import BlenderServer

__all__ = ["RenderParams", "BlenderRenderer", "BlenderRenderApp", "BlenderServer"]


def __getattr__(name: str) -> Any:
    """Import in-process Blender components only when explicitly requested."""
    if name == "BlenderRenderer":
        from .renderer import BlenderRenderer

        return BlenderRenderer
    if name == "BlenderRenderApp":
        from .server_app import BlenderRenderApp

        return BlenderRenderApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
