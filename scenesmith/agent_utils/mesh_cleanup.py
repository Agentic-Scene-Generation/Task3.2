"""Context-free mesh cleanup helpers used inside Blender subprocesses."""

from __future__ import annotations


def merge_duplicate_vertices(
    mesh_objects: object,
    *,
    distance: float = 0.0001,
    bmesh_module: object | None = None,
) -> int:
    """Merge duplicate vertices without Blender selection or mode operators.

    Imported GLTF assets may contain hidden mesh nodes. ``bpy.ops.object.mode_set``
    rejects a hidden active object, so canonicalization must operate directly on
    mesh data. Shared mesh datablocks are processed once.

    Returns:
        Number of unique mesh datablocks processed.
    """

    if bmesh_module is None:
        import bmesh as bmesh_module

    processed_meshes: set[int] = set()
    processed_count = 0
    for obj in mesh_objects:
        mesh = getattr(obj, "data", None)
        if getattr(obj, "type", None) != "MESH" or mesh is None:
            continue

        pointer = getattr(mesh, "as_pointer", None)
        mesh_key = int(pointer()) if callable(pointer) else id(mesh)
        if mesh_key in processed_meshes:
            continue
        processed_meshes.add(mesh_key)

        editable_mesh = bmesh_module.new()
        try:
            editable_mesh.from_mesh(mesh)
            vertices = list(editable_mesh.verts)
            if vertices:
                bmesh_module.ops.remove_doubles(
                    editable_mesh,
                    verts=vertices,
                    dist=distance,
                )
                editable_mesh.to_mesh(mesh)
                update = getattr(mesh, "update", None)
                if callable(update):
                    update()
            processed_count += 1
        finally:
            editable_mesh.free()

    return processed_count
