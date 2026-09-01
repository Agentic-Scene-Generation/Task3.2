import json
import shutil
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.schemas import MemoryUpdateOp, SuccessCase
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import SceneTaskSpec


def _success(
    case_id: str,
    run_id: str,
    evidence_ref: str,
    *,
    status: str = "active",
) -> SuccessCase:
    return SuccessCase(
        case_id=case_id,
        room_type="bedroom",
        style="modern",
        stage="furniture",
        task_signature=["bed", "nightstand"],
        successful_pattern=["anchor bed before balancing nightstands"],
        positive_guidance=["preserve both bedside approach paths"],
        source="llm",
        source_run_id=run_id,
        source_task_id=f"task_{run_id}",
        prompt_fingerprint=run_id,
        evidence_refs=[evidence_ref],
        status=status,
        quality_score=0.85,
        confidence=0.8,
    )


class FastMemoryStoreV2Test(unittest.TestCase):
    def test_snapshot_identity_is_path_independent_and_content_sensitive(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            copied = root / "copied"
            store = FastMemoryStore(str(source))
            store.add_success_case(_success("case_1", "run_1", "trace_1"))
            first = store.snapshot_identity()
            shutil.copytree(source, copied)

            copied_store = FastMemoryStore(str(copied), read_only=True)
            copied_identity = copied_store.snapshot_identity()
            self.assertEqual(
                first["content_fingerprint"],
                copied_identity["content_fingerprint"],
            )
            self.assertNotEqual(first["memory_dir"], copied_identity["memory_dir"])

            store.add_success_case(
                _success("case_2", "run_2", "trace_2").model_copy(
                    update={"successful_pattern": ["keep the desk facing the board"]}
                )
            )
            self.assertNotEqual(
                first["content_fingerprint"],
                store.snapshot_identity()["content_fingerprint"],
            )

    def test_read_only_store_rejects_every_persistent_mutation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writable = FastMemoryStore(temp_dir)
            writable.add_success_case(_success("case_1", "run_1", "trace_1"))
            frozen = FastMemoryStore(temp_dir, read_only=True)
            before = frozen.snapshot_identity()

            with self.assertRaisesRegex(RuntimeError, "frozen read-only"):
                frozen.append_event({"event_type": "should_not_write"})
            with self.assertRaisesRegex(RuntimeError, "frozen read-only"):
                frozen.apply_updates([])

            self.assertEqual(before, frozen.snapshot_identity())

    def test_manifest_is_initialized_and_one_batch_increments_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = FastMemoryStore(temp_dir)
            initial_bank_id = store.bank_id
            summary = store.apply_updates(
                [
                    MemoryUpdateOp(
                        op="ADD",
                        memory_type="success_case",
                        content=_success("case_1", "run_1", "trace_1").model_dump(),
                    ),
                    MemoryUpdateOp(
                        op="ADD",
                        memory_type="success_case",
                        content=_success(
                            "case_2",
                            "run_2",
                            "trace_2",
                        )
                        .model_copy(
                            update={
                                "successful_pattern": [
                                    "reserve wardrobe clearance before decoration"
                                ]
                            }
                        )
                        .model_dump(),
                    ),
                ]
            )

            manifest = json.loads(
                (Path(temp_dir) / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertTrue(initial_bank_id)
        self.assertEqual(initial_bank_id, manifest["bank_id"])
        self.assertEqual(1, summary["revision"])
        self.assertEqual(1, manifest["revision"])
        self.assertEqual(1, manifest["bank_revisions"]["success"])
        self.assertEqual(2, manifest["counts"]["active_success"])

    def test_same_run_is_idempotent_and_cross_run_evidence_merges(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first = FastMemoryStore(temp_dir)
            first.add_success_case(_success("case_1", "run_1", "trace_1"))
            revision_after_first = first.revision

            self.assertFalse(
                first.add_success_case(_success("case_1", "run_1", "trace_1"))
            )
            self.assertEqual(revision_after_first, first.revision)

            second = FastMemoryStore(temp_dir)
            merged = second.add_success_case(
                _success("case_other_id", "run_2", "trace_2")
            )
            self.assertTrue(merged)
            self.assertEqual(1, len(second.success_cases))
            self.assertEqual(2, second.success_cases[0].observation_count)
            self.assertEqual(
                ["trace_1", "trace_2"], second.success_cases[0].evidence_refs
            )
            self.assertEqual(["run_1", "run_2"], second.success_cases[0].source_run_ids)

            self.assertEqual(1, len(first.success_cases))
            self.assertTrue(first.refresh_if_changed())
            self.assertEqual(2, first.success_cases[0].observation_count)

    def test_quarantined_records_never_enter_active_view(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = FastMemoryStore(temp_dir)
            store.add_success_case(
                _success(
                    "case_quarantined",
                    "run_1",
                    "trace_1",
                    status="quarantined",
                )
            )

            self.assertEqual(1, len(store.success_cases))
            self.assertEqual([], store.active_success_cases)
            self.assertEqual(0, store.manifest["counts"]["active_success"])

    def test_retrieval_excludes_only_memories_supported_by_the_same_task(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = FastMemoryStore(temp_dir)
            same_task = _success("same", "run_1", "trace_1").model_copy(
                update={
                    "source_task_id": "task_current",
                    "source_task_ids": ["task_current"],
                }
            )
            other_task = _success("other", "run_2", "trace_2").model_copy(
                update={
                    "successful_pattern": ["reserve a clear wardrobe approach lane"],
                    "source_task_id": "task_other",
                    "source_task_ids": ["task_other"],
                }
            )
            store.add_success_case(same_task)
            store.add_success_case(other_task)
            retriever = MemoryRetriever(
                store,
                max_success=3,
                exclude_source_task_id="task_current",
            )

            pack = retriever.retrieve(
                SceneTaskSpec(
                    room_type="bedroom",
                    style="modern",
                    required_large_objects=["bed", "nightstand"],
                ),
                "furniture",
            )

            self.assertEqual(["other"], pack.success_case_ids)
            self.assertEqual({"other": ["task_other"]}, pack.retrieved_source_task_ids)
            self.assertEqual({"other": ["run_2"]}, pack.retrieved_source_run_ids)


if __name__ == "__main__":
    unittest.main()
