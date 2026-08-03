import ast
import logging
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from scenesmith.scene_expert.runtime_state import candidate_state_hash


def _load_completion_compatibility_agent() -> type:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "scenesmith"
        / "manipuland_agents"
        / "stateful_manipuland_agent.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    stateful_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StatefulManipulandAgent"
    )
    method = next(
        node
        for node in stateful_class.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_manipuland_stage_contract_satisfied"
    )
    compatibility_class = ast.ClassDef(
        name="_CompletionCompatibilityAgent",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.ImportFrom(
                module="scenesmith.scene_expert.runtime_state",
                names=[ast.alias(name="candidate_state_hash")],
                level=0,
            ),
            compatibility_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"console_logger": logging.getLogger(__name__)}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_CompletionCompatibilityAgent"]


CompletionCompatibilityAgent = _load_completion_compatibility_agent()


class _HashScene:
    def __init__(self, scene_hash: str) -> None:
        self.scene_hash = scene_hash

    def content_hash(self) -> str:
        return self.scene_hash

    def to_state_dict(self) -> dict:
        return {"objects": {"candidate": {"state": self.scene_hash}}}


class StatefulManipulandCompletionTest(unittest.IsolatedAsyncioTestCase):
    async def test_postprocessed_stage_is_rescored_before_target_search(self) -> None:
        agent = CompletionCompatibilityAgent()
        agent.scene = _HashScene("postprocessed")
        agent._stage_trusted_score_available = True
        agent._stage_visual_scores = [0.8]
        agent._last_scored_scene_hash = "before-postprocessing"
        agent._last_score_provenance = {"score_source": "vlm_critic"}
        agent.previous_scores = SimpleNamespace()
        agent._stage_budget_value = lambda key, default: (
            0.6 if key == "min_visual_score" else default
        )
        agent._evaluate_current_hard_state = lambda: SimpleNamespace(hard_valid=True)
        agent._normalized_visual_score = lambda _scores: 0.85

        async def rescore(*, update_checkpoint: bool) -> None:
            self.assertFalse(update_checkpoint)
            agent._last_scored_scene_hash = candidate_state_hash(agent.scene)

        agent._request_critique_impl = AsyncMock(side_effect=rescore)

        satisfied = await agent._manipuland_stage_contract_satisfied(
            minimum=1,
            placed_count=2,
        )

        self.assertTrue(satisfied)
        agent._request_critique_impl.assert_awaited_once()
        self.assertEqual([0.8, 0.85], agent._stage_visual_scores)

    async def test_empty_later_target_cannot_override_satisfied_stage(self) -> None:
        agent = CompletionCompatibilityAgent()
        agent.scene = _HashScene("scored")
        agent._stage_trusted_score_available = True
        agent._stage_visual_scores = [0.8]
        agent._last_scored_scene_hash = candidate_state_hash(agent.scene)
        agent._stage_budget_value = lambda key, default: (
            0.6 if key == "min_visual_score" else default
        )
        agent._evaluate_current_hard_state = lambda: SimpleNamespace(hard_valid=True)
        agent._request_critique_impl = AsyncMock()

        satisfied = await agent._manipuland_stage_contract_satisfied(
            minimum=1,
            placed_count=2,
        )

        self.assertTrue(satisfied)
        agent._request_critique_impl.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
