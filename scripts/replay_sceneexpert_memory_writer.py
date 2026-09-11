#!/usr/bin/env python3
"""Replay only MemoryWriter against archived evidence into a NEW isolated bank.

No generation, critic, shared-base mutation, training, Git checks or old-bank writes.
Prefer exact memory_writer_input.json; old runs can use trace + memory_activity.
The --dry-run path projects requests without contacting any model service.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.service_diagnostics import (
    connection_diagnostics,
    safe_endpoint,
)
from scenesmith.scene_expert.structured_llm import SceneExpertStructuredLLMClient


def build_service_client(base_url: str):
    """Use one client for readiness and writing; loopback never goes via proxy."""
    import httpx

    from openai import OpenAI

    parts = urlsplit(base_url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "Use an http(s) API base URL without credentials, query or fragment"
        )
    local = parts.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    http_client = httpx.Client(trust_env=not local)
    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        max_retries=0,
        http_client=http_client,
    ), ("direct_loopback" if local else "environment_proxy")


def wait_for_service(client, model: str, wait_seconds: float) -> dict:
    """Bounded readiness probe; no chat calls, candidates or bank mutations."""
    deadline = time.monotonic() + wait_seconds
    result = {"ready": False, "expected_model": model, "attempts": []}
    while True:
        probe_timeout = max(0.1, min(5.0, deadline - time.monotonic()))
        try:
            models = client.with_options(
                timeout=probe_timeout, max_retries=0
            ).models.list()
            ids = [str(item.id) for item in models.data]
            result["available_models"] = ids
            if model not in ids:
                result["failure_kind"] = "model_mismatch"
                result["hint"] = (
                    "Endpoint responded but does not serve the requested model."
                )
                return result
            result["ready"] = True
            return result
        except Exception as exc:
            details = connection_diagnostics(exc)
            status = getattr(exc, "status_code", None)
            details["http_status"] = status
            result["attempts"].append(details)
            result["failure_kind"] = details["kind"]
            # Credentials/wrong routes will not recover by waiting.
            if status in {400, 401, 403, 404}:
                result["failure_kind"] = f"http_{status}"
                return result
            if time.monotonic() >= deadline:
                result["hint"] = (
                    "Start the model in this same CCI instance, confirm API port, and rerun. "
                    "CCI restarts and ACP cleanup stop previously launched services."
                )
                return result
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))


def save_summary(destination: Path, summary: dict) -> None:
    """Checkpoint even preflight/failure outcomes into the isolated directory."""
    target = destination / "replay_summary.json"
    temporary = destination / "replay_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def execute_replay(
    payload: dict,
    destination: Path,
    model: str,
    base_url: str,
    *,
    dry_run: bool = False,
    service_wait_seconds: float = 10.0,
) -> tuple[dict, int]:
    """Run after local input validation; a failed extraction never creates a bank."""
    report = FullVerifyReport.model_validate(payload["full_report"])
    summary = {
        "replay_source": payload["replay_source"],
        "model": model,
        "generation_rerun": False,
        "hostname": socket.gethostname(),
        "api_base_url": safe_endpoint(base_url),
        "phase": "preflight",
    }
    save_summary(destination, summary)
    client = None
    try:
        if dry_run:
            # No SDK construction, credentials, HTTP transports or model probes.
            writer = MemoryWriter(
                model=model, debug_dir=destination / "audit", llm_client=object()
            )
            writer._build_user_message(
                trace_summary=payload["trace_summary"],
                full_report=report,
                related_old_memory=payload["related_old_memory"],
                evidence_payload=payload["evidence"],
            )
            summary.update(
                {
                    "dry_run": True,
                    "phase": "projected",
                    "projection": writer._prompt_attempts,
                }
            )
            return summary, 0
        client, proxy_policy = build_service_client(base_url)
        summary["proxy_policy"] = proxy_policy
        summary["service_preflight"] = wait_for_service(
            client, model, service_wait_seconds
        )
        if not summary["service_preflight"]["ready"]:
            summary.update(
                {
                    "phase": "preflight_failed",
                    "writer_called": False,
                    "bank_created": False,
                    "spatial_records": 0,
                }
            )
            return summary, 2
        summary.update(
            {
                "phase": "writer_running",
                "writer_called": True,
                "skill_bootstrap_enabled": False,
            }
        )
        save_summary(destination, summary)
        structured_client = SceneExpertStructuredLLMClient(model=model, client=client)
        writer = MemoryWriter(
            model=model,
            debug_dir=destination / "audit",
            llm_client=structured_client,
            skill_bootstrap_enabled=False,
        )
        # Replay validates the LLM extraction path, not deterministic bootstrap.
        # Override a possible inherited bootstrap env flag only on this instance.
        writer._skill_bootstrap_enabled = False
        ops = writer.write(
            payload["trace_summary"],
            report,
            related_old_memory=payload["related_old_memory"],
            evidence_payload=payload["evidence"],
        )
        summary["writer"] = writer.last_trace
        if not writer.last_trace.get("success"):
            summary.update(
                {"phase": "writer_failed", "bank_created": False, "spatial_records": 0}
            )
            return summary, 1
        bank = FastMemoryStore(destination / "bank")
        applied = bank.apply_updates(ops)
        writer.record_store_result(applied)
        spatial_count = sum(
            bool(r.placement_experience)
            for r in [*bank.success_cases, *bank.failure_cases]
        )
        summary.update(
            {
                "phase": (
                    "completed" if spatial_count else "completed_no_spatial_memory"
                ),
                "writer": writer.last_trace,
                "store_apply": applied,
                "bank_created": True,
                "spatial_records": spatial_count,
            }
        )
        return summary, 0
    except Exception as exc:
        summary.update({"phase": "replay_error", "error": connection_diagnostics(exc)})
        return summary, 1
    finally:
        save_summary(destination, summary)
        if client is not None:
            client.close()


def read_json(path: Path) -> dict:
    """Read a source artifact, accepting a UTF-8 BOM from transferred files."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_input(scene_expert_dir: Path) -> dict:
    """Load exact input or transparently reconstruct the old evidence contract."""
    exact = scene_expert_dir / "memory/memory_writer_input.json"
    if exact.is_file():
        payload = read_json(exact)
        if payload.get("schema_version") != "memory-writer-input.v1":
            raise ValueError("Unsupported writer input schema")
        return {**payload, "replay_source": "exact_writer_input"}
    debug = read_json(scene_expert_dir / "memory/memory_writer_debug.json")
    paths = sorted(
        p
        for p in (scene_expert_dir / "trace").glob("trace_*.json")
        if not p.name.endswith("_partial.json")
    )
    if len(paths) != 1:
        raise ValueError(
            "Old replay requires exactly one final trace and original writer debug"
        )
    trace = read_json(paths[0])
    activity = read_json(scene_expert_dir / "memory_activity.json")
    entries = list((activity.get("stages") or {}).values())
    entries += [r["entry"] for r in activity.get("stage_attempt_history", [])]
    episodes = [
        e
        for entry in entries
        for e in (entry.get("placement_episodes") or {}).get("episodes", [])
    ]
    if not any("placement_episodes" in entry for entry in entries):
        raise ValueError(
            "No archived placement catalog; refusing to reconstruct speculative evidence"
        )
    evidence = {
        k: trace.get(k)
        for k in (
            "trace_id",
            "scene_id",
            "experiment_name",
            "config_hash",
            "experiment_signature",
            "control_signature",
            "prompt",
            "task_spec",
            "component_flags",
            "code_provenance",
            "memory_identity",
            "evaluation_contract",
        )
    }
    evidence["stages"] = [
        {
            k: entry.get(k)
            for k in (
                "stage",
                "scene_state_path",
                "stage_brief",
                "relation_context",
                "verify_report",
                "repair_actions",
                "execution_evidence",
                "retrieved_memory_ids",
                "memory_pack",
            )
        }
        for entry in trace.get("stages", [])
    ]
    # Keep original run provenance even when the package has moved machines.
    states = [str(s.get("scene_state_path") or "") for s in evidence["stages"]]
    evidence["run_id"] = trace.get("run_id") or next((s for s in states if s), "")
    if not evidence["run_id"] or not evidence.get("trace_id"):
        raise ValueError("Missing original run provenance")
    evidence["full_report"] = debug["full_report"]
    evidence["placement_experience_catalog"] = {"episodes": episodes}
    return {
        "schema_version": "memory-writer-input.v1",
        "replay_source": "reconstructed_final_trace_and_activity_not_exact_original_request",
        "trace_summary": debug.get("trace_summary_excerpt", ""),
        "related_old_memory": "",
        "evidence": evidence,
        "full_report": debug["full_report"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-expert-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must not exist; a separate audit/ and bank/ will be created here",
    )
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1"),
    )
    parser.add_argument("--service-wait-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.service_wait_seconds <= 7200:
        parser.error("service-wait-seconds must be between 0 and 7200")
    source, destination = args.scene_expert_dir.resolve(), args.output_dir.resolve()
    if destination == source or source in destination.parents:
        parser.error("Output must be outside the source scene_expert directory")
    payload = load_input(source)
    FullVerifyReport.model_validate(payload["full_report"])
    destination.mkdir(parents=True, exist_ok=False)
    print(
        f"Writer replay endpoint={safe_endpoint(args.api_base_url)} model={args.model}",
        flush=True,
    )
    summary, exit_code = execute_replay(
        payload,
        destination,
        args.model,
        args.api_base_url,
        dry_run=args.dry_run,
        service_wait_seconds=args.service_wait_seconds,
    )
    summary["source"] = str(source)
    save_summary(destination, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
