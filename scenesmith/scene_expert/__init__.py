"""SceneExpert: outer-layer expert scheduling for SceneSmith.

SceneExpert wraps SceneSmith's 5-stage 3D scene generation pipeline with:
- Structured task understanding (TaskCompiler)
- Deterministic stage control (Harness FSM)
- Experience-based memory retrieval (Fast Memory)
- Expert planning hints per stage (StageBrief / Global Planner)
- Rule-based quality verification (Verifier)
- Automatic repair loops (Repair Controller)
- Full trace logging and memory updating

MVP implements only the online closed-loop (no offline SFT/DPO training).
"""

__all__ = ["SceneExpertPipeline"]


def __getattr__(name: str):
    if name == "SceneExpertPipeline":
        from scenesmith.scene_expert.pipeline import SceneExpertPipeline

        return SceneExpertPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
