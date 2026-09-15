from __future__ import annotations

import json
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pytest

from scenesmith.scene_expert.evaluation_inputs import FrozenCompiledInputStore


def _payload() -> dict:
    return {
        "task_spec": {"room_type": "classroom", "style": "modern"},
        "intent_contract": {"schema_version": "intent.v1", "constraints": []},
        "compiler_metadata": {"task_compiler_trace": {"status": "ok"}},
    }


def test_pair_arms_reuse_one_compiled_input_bundle(tmp_path: Path) -> None:
    store = FrozenCompiledInputStore(tmp_path)
    calls = 0

    def producer() -> dict:
        nonlocal calls
        calls += 1
        return _payload()

    first, first_identity = store.load_or_create(
        scene_id=41,
        prompt="A modern classroom.",
        producer=producer,
    )
    second, second_identity = store.load_or_create(
        scene_id=41,
        prompt="A modern classroom.",
        producer=producer,
    )

    assert calls == 1
    assert first == second
    assert first_identity["source"] == "materialized"
    assert second_identity["source"] == "frozen_replay"
    assert (
        first_identity["compiled_input_fingerprint"]
        == second_identity["compiled_input_fingerprint"]
    )
    case_dir = store.case_dir(scene_id=41, prompt="A modern classroom.")
    assert {path.name for path in case_dir.iterdir()} == {
        "scene_task_spec.json",
        "intent_contract.json",
        "compiler_metadata.json",
        "fingerprint.json",
    }


def test_parallel_pair_arms_compile_once(tmp_path: Path) -> None:
    store = FrozenCompiledInputStore(tmp_path, poll_interval_sec=0.01)
    calls = 0
    calls_lock = Lock()

    def producer() -> dict:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return _payload()

    def load() -> tuple[dict, dict]:
        return store.load_or_create(
            scene_id=95,
            prompt="A shared classroom.",
            producer=producer,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: load(), range(2)))

    assert calls == 1
    assert {identity["source"] for _payload_value, identity in results} == {
        "materialized",
        "frozen_replay",
    }


def test_corrupt_frozen_bundle_fails_closed(tmp_path: Path) -> None:
    store = FrozenCompiledInputStore(tmp_path)
    store.load_or_create(scene_id=3, prompt="A bedroom.", producer=_payload)
    case_dir = store.case_dir(scene_id=3, prompt="A bedroom.")
    (case_dir / "scene_task_spec.json").write_text(
        json.dumps({"room_type": "kitchen"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        store.load_or_create(scene_id=3, prompt="A bedroom.", producer=_payload)
