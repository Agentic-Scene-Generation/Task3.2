import unittest

from scenesmith.scene_expert.memory.retriever import _tokenize
from scenesmith.scene_expert.memory.schemas import FailureCase, MemoryUpdateOp, SuccessCase
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport


class SceneExpertMemoryTest(unittest.TestCase):
    def test_chinese_aliases_expand_to_english_tokens(self) -> None:
        tokens = set(_tokenize("卧室里需要一张床、两个床头柜和一个衣柜"))

        self.assertIn("bedroom", tokens)
        self.assertIn("bed", tokens)
        self.assertIn("nightstand", tokens)
        self.assertIn("bedside_table", tokens)
        self.assertIn("wardrobe", tokens)

    def test_success_case_fallback_embedding_text_is_structured(self) -> None:
        record = SuccessCase(
            case_id="success_bedroom_001",
            room_type="bedroom",
            style="modern",
            stage="furniture",
            task_signature=["bed", "nightstand", "wardrobe"],
            successful_pattern=["bed centered on main wall"],
            scores={"semantic": 0.9, "physics": 0.8},
        )

        text = build_embedding_text(record)

        self.assertIn("memory_type=success", text)
        self.assertIn("stage=furniture", text)
        self.assertIn("room_type=bedroom", text)
        self.assertIn("required_objects=bed, nightstand, wardrobe", text)
        self.assertIn("success_pattern=bed centered on main wall", text)

    def test_failure_case_uses_negative_constraint_in_hint_and_embedding_text(
        self,
    ) -> None:
        record = FailureCase(
            failure_id="fail_mesh_001",
            room_type="bedroom",
            stage="furniture",
            failure_type="deterministic_asset_error",
            bad_pattern="candidate mesh cannot be loaded",
            negative_constraint="do not retry the same missing HSSD mesh",
            critic_check="verify the replacement asset file exists",
            repair_action="mark candidate invalid and retrieve a different asset",
            repair_verified=True,
            is_deterministic=True,
            scope="stage",
        )

        self.assertIn(
            "do not retry the same missing HSSD mesh",
            record.to_hint_text(),
        )

        text = build_embedding_text(record)

        self.assertIn("memory_type=failure", text)
        self.assertIn("scope=stage", text)
        self.assertIn("is_deterministic=true", text)
        self.assertIn(
            "negative_constraint=do not retry the same missing HSSD mesh",
            text,
        )

    def test_memory_writer_gates_low_quality_success_and_keeps_failure(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        full_report = FullVerifyReport(overall_score=0.4, pass_scene=False)
        ops = [
            MemoryUpdateOp(
                op="ADD",
                memory_type="success_case",
                content={
                    "case_id": "success_low_score",
                    "room_type": "bedroom",
                    "stage": "furniture",
                    "task_signature": ["bed"],
                    "successful_pattern": ["bed exists"],
                    "scores": {"semantic": 0.4},
                },
            ),
            MemoryUpdateOp(
                op="ADD",
                memory_type="failure_case",
                content={
                    "failure_id": "fail_missing_mesh",
                    "room_type": "bedroom",
                    "stage": "furniture",
                    "failure_type": "missing_mesh",
                    "bad_pattern": "HSSD candidate file missing",
                    "failure_reason": "missing mesh file",
                    "repair_action": "retrieve another asset",
                    "repair_verified": False,
                },
            ),
        ]

        filtered = writer._gate_and_enrich_ops(ops, full_report)

        self.assertEqual(1, len(filtered))
        failure = filtered[0]
        self.assertEqual("failure_case", failure.memory_type)
        self.assertIs(True, failure.content["is_deterministic"])
        self.assertEqual("stage", failure.content["scope"])
        self.assertTrue(failure.content["embedding_text"])


if __name__ == "__main__":
    unittest.main()
