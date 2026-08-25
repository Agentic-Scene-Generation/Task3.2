"""Run both SceneExpert compilers against the TEST_OPENAI_* endpoint.

This is intentionally opt-in and uses TEST_* variables so it cannot affect a
normal SceneSmith replay or accidentally reuse production credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from scenesmith.scene_expert.task_compiler import TaskCompiler
from scenesmith.scenebenchmark_critic.intent_compiler import IntentCompiler


def main() -> None:
    load_dotenv(Path(__file__).parents[1] / ".env")
    base_url = os.environ["TEST_OPENAI_BASE_URL"]
    api_key = os.environ["TEST_OPENAI_API_KEY"]
    model = os.environ["TEST_OPENAI_MODEL"]
    prompt = "A compact bedroom with one bed and two nightstands."

    task = TaskCompiler(model=model, api_base_url=base_url, api_key=api_key)
    print("TaskCompiler:", task.compile(prompt).model_dump_json())

    intent = IntentCompiler(model=model, api_base_url=base_url, api_key=api_key)
    print("IntentCompiler:", intent.compile(prompt))


if __name__ == "__main__":
    main()
