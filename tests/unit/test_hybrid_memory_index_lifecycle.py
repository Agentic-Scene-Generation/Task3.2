import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scenesmith.scene_expert.memory.hybrid_retriever import (
    HybridMemoryRetriever,
    console_logger,
)
from scenesmith.scene_expert.memory.index import NumpyMemoryIndex
from scenesmith.scene_expert.memory.schemas import FailureCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import SceneTaskSpec


class _DummyEmbedder:
    model_dir = Path("/models/bge-m3")
    model_id = "BAAI/bge-m3"
    device = "cpu"
    batch_size = 8
    max_length = 512

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls += 1
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _living_room_task() -> SceneTaskSpec:
    return SceneTaskSpec(
        room_type="living_room",
        style="modern",
        required_large_objects=["television", "tv_stand"],
    )


class HybridMemoryIndexLifecycleTest(unittest.TestCase):
    def test_empty_banks_are_normal_and_do_not_warn(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = FastMemoryStore(str(memory_dir))
            embedder = _DummyEmbedder()
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=embedder,
                require_indexes=True,
                auto_build_indexes=True,
            )

            with patch.object(console_logger, "warning") as warning:
                pack = retriever.retrieve(_living_room_task(), "furniture")

            warning.assert_not_called()
            self.assertEqual([], pack.success_hints)
            self.assertEqual([], pack.failure_hints)
            self.assertEqual([], pack.skill_texts)
            self.assertEqual(0, embedder.encode_calls)

    def test_runtime_records_build_and_refresh_cached_index(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            store = FastMemoryStore(str(memory_dir))
            embedder = _DummyEmbedder()
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=embedder,
                max_success=0,
                max_failure=2,
                max_skills=0,
                require_indexes=True,
                auto_build_indexes=True,
            )

            # Cache the normal empty-bank state before the MemoryWriter grows it.
            self.assertEqual(
                [],
                retriever.retrieve(
                    _living_room_task(),
                    "furniture",
                ).failure_hints,
            )

            external_store = FastMemoryStore(str(memory_dir))
            external_store.add_failure_case(
                FailureCase(
                    failure_id="failure_tv_support_001",
                    room_type="living_room",
                    stage="furniture",
                    object="television",
                    failure_type="support_invalid",
                    bad_pattern="television is not on its stand",
                    repair_action="place television on the tv stand",
                )
            )
            first_pack = retriever.retrieve(_living_room_task(), "furniture")
            self.assertEqual(1, len(first_pack.failure_hints))
            self.assertGreater(embedder.encode_calls, 0)

            # A second write makes the already cached one-row index stale. The
            # next retrieval must rebuild it instead of silently hiding memory.
            external_store.add_failure_case(
                FailureCase(
                    failure_id="failure_tv_support_002",
                    room_type="living_room",
                    stage="furniture",
                    object="tv_stand",
                    failure_type="missing_support_surface",
                    bad_pattern="support surface was not resolved",
                    repair_action="refresh support surfaces before placement",
                )
            )
            self.assertEqual(1, len(store.failure_cases))
            second_pack = retriever.retrieve(_living_room_task(), "furniture")

            self.assertEqual(2, len(second_pack.failure_hints))
            self.assertEqual(external_store.revision, second_pack.memory_bank_revision)
            self.assertEqual(external_store.bank_id, second_pack.memory_bank_id)
            index = NumpyMemoryIndex.for_bank(
                memory_dir / "indexes",
                "failure",
                "furniture",
            )
            index.load()
            self.assertEqual(2, len(index.metadata))


if __name__ == "__main__":
    unittest.main()
