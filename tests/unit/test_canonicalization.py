"""Regression tests for context-free Blender mesh cleanup."""

import unittest

from scenesmith.agent_utils.mesh_cleanup import merge_duplicate_vertices


class _FakeMesh:
    def __init__(self, pointer: int) -> None:
        self.pointer = pointer
        self.updated = 0

    def as_pointer(self) -> int:
        return self.pointer

    def update(self) -> None:
        self.updated += 1


class _FakeObject:
    def __init__(self, mesh: _FakeMesh, *, hidden: bool = False) -> None:
        self.type = "MESH"
        self.data = mesh
        self.hide_viewport = hidden


class _FakeEditableMesh:
    def __init__(self) -> None:
        self.verts = [object(), object()]
        self.source = None
        self.target = None
        self.freed = False

    def from_mesh(self, mesh: _FakeMesh) -> None:
        self.source = mesh

    def to_mesh(self, mesh: _FakeMesh) -> None:
        self.target = mesh

    def free(self) -> None:
        self.freed = True


class _FakeBmeshOps:
    def __init__(self) -> None:
        self.calls = []

    def remove_doubles(self, editable_mesh, *, verts, dist) -> None:
        self.calls.append((editable_mesh, verts, dist))


class _FakeBmeshModule:
    def __init__(self) -> None:
        self.created = []
        self.ops = _FakeBmeshOps()

    def new(self) -> _FakeEditableMesh:
        editable_mesh = _FakeEditableMesh()
        self.created.append(editable_mesh)
        return editable_mesh


class TestMergeDuplicateVertices(unittest.TestCase):
    def test_hidden_meshes_are_processed_without_selection_context(self) -> None:
        shared_mesh = _FakeMesh(pointer=101)
        other_mesh = _FakeMesh(pointer=202)
        bmesh_module = _FakeBmeshModule()

        count = merge_duplicate_vertices(
            [
                _FakeObject(shared_mesh, hidden=True),
                _FakeObject(shared_mesh),
                _FakeObject(other_mesh, hidden=True),
            ],
            bmesh_module=bmesh_module,
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(bmesh_module.ops.calls), 2)
        self.assertEqual(shared_mesh.updated, 1)
        self.assertEqual(other_mesh.updated, 1)
        self.assertTrue(all(item.freed for item in bmesh_module.created))


if __name__ == "__main__":
    unittest.main()
