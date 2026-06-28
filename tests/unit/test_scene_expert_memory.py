import unittest

import json
import sys
import types

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.build_memory_index import build_memory_indexes
from scenesmith.scene_expert.memory.embedding import (
    SceneMemoryEmbedder,
    resolve_memory_embedding_model_dir,
)
from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
from scenesmith.scene_expert.memory.index import NumpyMemoryIndex
from scenesmith.scene_expert.memory.retriever import _tokenize
from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    MemoryUpdateOp,
    Skill,
    SuccessCase,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    SceneTaskSpec,
    StageVerifyReport,
)
from scenesmith.scene_expert.task_compiler import _fallback_spec_from_prompt
from scenesmith.scene_expert.verifier import (
    FullVerifier,
    StageVerifier,
    _map_scenesmith_scores,
)


class SceneExpertMemoryTest(unittest.TestCase):
    def test_task_compiler_fallback_preserves_required_bedroom_objects(self) -> None:
        spec = _fallback_spec_from_prompt(
            "A bedroom with a bed, two nightstands, and a wardrobe in the corner."
        )

        self.assertEqual("bedroom", spec.room_type)
        self.assertEqual(
            ["bed", "nightstand", "nightstand", "wardrobe"],
            spec.required_large_objects,
        )
        self.assertIn("sleeping_zone", spec.functional_zones)

    def test_furniture_stage_verifier_fails_hard_missing_and_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_dir = root / "scene_states" / "furniture"
            scores_dir.mkdir(parents=True)
            (scores_dir / "scores.yaml").write_text(
                "\n".join(
                    [
                        "Realism:",
                        "  grade: 4",
                        "  comment: collision detected with wall",
                        "Functionality:",
                        "  grade: 3",
                        "  comment: room incomplete",
                        "Layout Plausibility:",
                        "  grade: 7",
                        "Prompt Following:",
                        "  grade: 4",
                        "  comment: missing primary bed",
                        "Summary: bed missing and collision detected",
                    ]
                ),
                encoding="utf-8",
            )

            report = StageVerifier(pass_threshold=0.6).verify(
                stage="furniture",
                stage_output_dir=str(root),
                task_spec=SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=["bed", "nightstand", "nightstand", "wardrobe"],
                ),
                scene_state_info={"object_names": ["nightstand_0", "nightstand_1", "wardrobe_0"]},
            )

            self.assertFalse(report.pass_stage)
            issue_types = {issue.issue_type for issue in report.issues}
            self.assertIn("missing_object", issue_types)
            self.assertIn("physics_collision", issue_types)

    def test_layout_plausibility_maps_to_scene_expert_category(self) -> None:
        mapped = _map_scenesmith_scores(
            {
                "Layout Plausibility": 4,
                "Layout": 9,
                "Realism": 8,
            }
        )

        self.assertAlmostEqual(0.4, mapped["plausibility"])
        self.assertAlmostEqual(0.85, mapped["aesthetic"])

    def test_full_verifier_gates_low_plausibility_even_with_high_average(
        self,
    ) -> None:
        report = StageVerifyReport(
            stage="furniture",
            pass_stage=True,
            scores={
                "semantic": 1.0,
                "aesthetic": 1.0,
                "plausibility": 0.4,
                "physics": 1.0,
                "interaction": 1.0,
            },
        )

        full_report = FullVerifier(pass_threshold=0.7).verify([report])

        self.assertAlmostEqual(0.4, full_report.plausibility_score)
        self.assertAlmostEqual(0.88, full_report.overall_score)
        self.assertFalse(full_report.pass_scene)

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

    def test_memory_writer_extracts_reasoning_content_and_markdown_json(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        message = types.SimpleNamespace(
            content=None,
            model_dump=lambda: {
                "content": None,
                "model_extra": {
                    "reasoning_content": (
                        "```json\n"
                        '{"updates":[{"op":"NOOP","memory_type":"success_case","content":{}}]}'
                        "\n```"
                    )
                },
            },
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        raw = writer._extract_response_text(response)
        data = writer._parse_json_payload(raw)

        self.assertEqual(1, len(data["updates"]))
        self.assertEqual("NOOP", data["updates"][0]["op"])

    def test_memory_writer_builds_conservative_fallback_success_ops(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        trace_summary = "\n".join(
            [
                "Trace: trace_000001",
                "Prompt: A bedroom with a bed, two nightstands, and a wardrobe.",
                "Stages:",
                "  [furniture] objective='Complete furniture' verify=PASS "
                "scores=(semantic=0.90, aesthetic=0.80, physics=0.90)",
            ]
        )
        full_report = FullVerifyReport(overall_score=0.8, pass_scene=True)

        ops = writer._fallback_success_ops(trace_summary, full_report)
        filtered = writer._gate_and_enrich_ops(ops, full_report)

        self.assertEqual(1, len(filtered))
        op = filtered[0]
        self.assertEqual("ADD", op.op)
        self.assertEqual("success_case", op.memory_type)
        self.assertEqual("furniture", op.content["stage"])
        self.assertEqual("bedroom", op.content["room_type"])
        self.assertIn("bed", op.content["required_objects"])
        self.assertTrue(op.content["embedding_text"])

    def test_embedding_model_dir_resolves_to_bge_m3_under_models_dir(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SCENEEXPERT_MODELS_DIR": "/models",
            },
            clear=True,
        ):
            self.assertEqual(
                Path("/models/bge-m3"),
                resolve_memory_embedding_model_dir(),
            )

        with patch.dict(
            "os.environ",
            {
                "SCENEEXPERT_MODELS_DIR": "/models",
                "SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR": "/custom/bge-m3",
            },
            clear=True,
        ):
            self.assertEqual(
                Path("/custom/bge-m3"),
                resolve_memory_embedding_model_dir(),
            )

    def test_embedder_pins_flagembedding_to_single_device(self) -> None:
        calls: dict[str, object] = {}

        class DummyBGEM3FlagModel:
            def __init__(self, model_dir: str, **kwargs: object) -> None:
                calls["model_dir"] = model_dir
                calls["kwargs"] = kwargs

            def encode(self, texts: list[str], **kwargs: object) -> dict[str, object]:
                return {"dense_vecs": [[1.0, 0.0] for _ in texts]}

        fake_module = types.SimpleNamespace(BGEM3FlagModel=DummyBGEM3FlagModel)
        with TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
                embedder = SceneMemoryEmbedder(model_dir=tmp, device="cpu")
                matrix = embedder.encode(["bedroom furniture"])

        self.assertEqual(str(Path(tmp)), calls["model_dir"])
        kwargs = calls["kwargs"]
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(["cpu"], kwargs["devices"])
        self.assertEqual((1, 2), matrix.shape)

    def test_numpy_memory_index_searches_normalized_vectors(self) -> None:
        with TemporaryDirectory() as tmp:
            index = NumpyMemoryIndex.for_bank(Path(tmp), "success", "furniture")
            index.build(
                vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata=[
                    {"memory_id": "bed_case"},
                    {"memory_id": "sofa_case"},
                ],
                manifest={"embedding_model_dir": "/models/bge-m3"},
            )

            loaded = NumpyMemoryIndex.for_bank(Path(tmp), "success", "furniture")
            results = loaded.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)

            self.assertEqual(1, len(results))
            self.assertEqual("bed_case", results[0][1]["memory_id"])

    def test_build_memory_indexes_writes_numpy_files_with_fallback_text(self) -> None:
        class DummyEmbedder:
            def __init__(self) -> None:
                self.texts: list[str] = []

            def encode(self, texts: list[str]) -> np.ndarray:
                self.texts.extend(texts)
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            record = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                task_signature=["bed", "nightstand"],
                successful_pattern=["bed centered on main wall"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                record.model_dump_json() + "\n",
                encoding="utf-8",
            )

            embedder = DummyEmbedder()
            summaries = build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success",),
                embedder=embedder,
            )

            self.assertEqual(1, len(summaries))
            self.assertEqual(1, summaries[0]["count"])
            self.assertIn("memory_type=success", embedder.texts[0])

            index = NumpyMemoryIndex.for_bank(
                memory_dir / "indexes",
                "success",
                "furniture",
            )
            index.load()
            self.assertEqual((1, 2), index.vectors.shape)
            self.assertEqual(
                "success_bedroom_001",
                index.metadata[0]["memory_id"],
            )
            self.assertEqual(
                str(Path("/models/bge-m3")),
                index.manifest["embedding_model_dir"],
            )

    def test_hybrid_retriever_reads_numpy_indexes_and_reranks_memory(self) -> None:
        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                del texts
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                style="modern",
                stage="furniture",
                required_objects=["bed", "nightstand", "wardrobe"],
                successful_pattern=["bed centered on main wall"],
                positive_guidance=["use bed as the anchor"],
                placement_reference=["bed_1 (bed): x=0.0, y=0.0, yaw=0"],
                scores={"semantic": 0.9, "aesthetic": 0.8, "physics": 0.9},
            )
            failure = FailureCase(
                failure_id="fail_asset_001",
                room_type="kitchen",
                stage="furniture",
                failure_type="missing_mesh",
                bad_pattern="HSSD candidate file missing",
                negative_constraint="do not retry the same missing HSSD file",
                repair_action="retrieve another asset",
                is_deterministic=True,
                scope="stage",
            )
            skill = Skill(
                skill_name="arrange_bedroom_anchor",
                stage="furniture",
                room_types=["bedroom"],
                required_objects=["bed", "nightstand"],
                preconditions=["bedroom furniture stage"],
                procedure=["place bed first", "place nightstands beside bed"],
                failure_avoidance=["do not block the wardrobe"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            (memory_dir / "failure_cases.jsonl").write_text(
                failure.model_dump_json() + "\n",
                encoding="utf-8",
            )
            (memory_dir / "skills.jsonl").write_text(
                skill.model_dump_json() + "\n",
                encoding="utf-8",
            )

            build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success", "failure", "skill"),
                embedder=DummyEmbedder(),
            )
            store = FastMemoryStore(str(memory_dir))
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=DummyEmbedder(),
                max_success=1,
                max_failure=1,
                max_skills=1,
                require_indexes=True,
            )
            task_spec = SceneTaskSpec(
                room_type="bedroom",
                style="modern",
                required_large_objects=["bed", "nightstand", "wardrobe"],
                functional_zones=["sleeping_zone", "storage_zone"],
            )

            pack = retriever.retrieve(task_spec, "furniture")

            self.assertEqual(1, len(pack.success_hints))
            self.assertIn("use bed as the anchor", pack.success_hints[0])
            self.assertIn("Reference Layout", pack.placement_reference)
            self.assertEqual(1, len(pack.failure_hints))
            self.assertIn("do not retry", pack.failure_hints[0])
            self.assertEqual(1, len(pack.skill_texts))
            self.assertIn("arrange_bedroom_anchor", pack.skill_texts[0])

    def test_hybrid_retriever_writes_timing_jsonl(self) -> None:
        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                del texts
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            timing_path = Path(tmp) / "scene_expert" / "timing" / "memory_retrieval.jsonl"
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                required_objects=["bed"],
                positive_guidance=["place bed first"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success",),
                embedder=DummyEmbedder(),
            )
            store = FastMemoryStore(str(memory_dir))
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=DummyEmbedder(),
                max_success=1,
                max_failure=0,
                max_skills=0,
                require_indexes=True,
                timing_path=timing_path,
            )

            retriever.retrieve(
                SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=["bed"],
                ),
                "furniture",
            )

            self.assertTrue(timing_path.exists())
            timing = json.loads(timing_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("hybrid", timing["retriever_type"])
            self.assertIn("embedding_encode_sec", timing)
            self.assertIn("index_load_sec", timing)
            self.assertIn("vector_search_sec", timing)
            self.assertIn("rerank_sec", timing)
            self.assertIn("total_sec", timing)


if __name__ == "__main__":
    unittest.main()
