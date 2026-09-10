"""Connection and lifecycle tests; no external model, GPU, or scene generation."""

import errno
import json
import os
import subprocess
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.service_diagnostics import (
    connection_diagnostics,
    safe_endpoint,
)
from scripts import replay_sceneexpert_memory_writer as replay
from tests.unit.test_memory_writer_resilience import _evidence
from tests.unit.test_scene_expert_structured_llm import _FakeOpenAI, _response


def refused():
    error = type("APIConnectionError", (Exception,), {})("Connection error.")
    cause = httpx.ConnectError("All connection attempts failed")
    cause.__cause__ = ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
    error.__cause__ = cause
    return error


def client(outcomes=()):
    fake = _FakeOpenAI(list(outcomes))
    fake.models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="qwen")])
    )
    fake.close = Mock()
    return fake


def payload():
    return {
        "replay_source": "exact_writer_input",
        "trace_summary": "sample",
        "full_report": FullVerifyReport().model_dump(),
        "related_old_memory": "",
        "evidence": _evidence(),
    }


def test_nested_connection_error_retains_root_errno():
    result = connection_diagnostics(refused())
    assert result["kind"] == "connection_refused"
    assert result["exception_chain"][-1]["errno"] == errno.ECONNREFUSED


def test_connection_diagnostics_never_record_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")
    error = Exception(
        "private-test-value https://alice:password@proxy.local Bearer bearer-secret ?token=query-secret"
    )
    text = json.dumps(connection_diagnostics(error))
    for secret in (
        "private-test-value",
        "alice",
        "password@",
        "bearer-secret",
        "query-secret",
    ):
        assert secret not in text
    assert (
        safe_endpoint("https://alice:pass@host:8002/v1?token=private#secret")
        == "https://host:8002/v1"
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:8002/v1", False),
        ("http://localhost:8002/v1", False),
        ("https://remote.example/v1", True),
    ],
)
def test_only_loopback_bypasses_proxy(monkeypatch, url, expected):
    real = httpx.Client
    seen = []

    class RecordingClient(real):
        def __init__(self, **kwargs):
            seen.append(kwargs["trust_env"])
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "Client", RecordingClient)
    api, _ = replay.build_service_client(url)
    api.close()
    assert seen == [expected]


def test_rejects_secrets_in_base_url():
    with pytest.raises(ValueError):
        replay.build_service_client("http://user:password@localhost:8002/v1")


def test_service_preflight_checks_exact_model_without_chat():
    fake = client()
    assert replay.wait_for_service(fake, "qwen", 0)["ready"]
    assert replay.wait_for_service(fake, "other", 0)["failure_kind"] == "model_mismatch"
    assert fake.calls == []


def test_preflight_refused_is_diagnostic_and_does_not_create_bank(
    tmp_path, monkeypatch
):
    fake = client()
    fake.models.list = Mock(side_effect=refused())
    monkeypatch.setattr(
        replay, "build_service_client", lambda _: (fake, "direct_loopback")
    )
    writer = Mock(side_effect=AssertionError("Writer must not be called"))
    monkeypatch.setattr(replay, "MemoryWriter", writer)
    summary, code = replay.execute_replay(
        payload(), tmp_path, "qwen", "http://127.0.0.1:8002/v1", service_wait_seconds=0
    )
    assert code == 2
    assert summary["phase"] == "preflight_failed"
    assert summary["service_preflight"]["failure_kind"] == "connection_refused"
    assert not (tmp_path / "bank").exists()
    assert not (tmp_path / "audit").exists()
    assert (tmp_path / "replay_summary.json").is_file()
    fake.close.assert_called_once()


def test_connection_lost_after_preflight_creates_no_bootstrap_bank(
    tmp_path, monkeypatch
):
    fake = client([refused(), refused()])
    monkeypatch.setattr(
        replay, "build_service_client", lambda _: (fake, "direct_loopback")
    )
    monkeypatch.setenv("SCENEEXPERT_SKILL_BOOTSTRAP_ENABLED", "true")
    summary, code = replay.execute_replay(
        payload(), tmp_path, "qwen", "http://127.0.0.1:8002/v1", service_wait_seconds=0
    )
    assert code == 1
    assert summary["phase"] == "writer_failed"
    assert summary["writer"]["generated_candidate_count"] == 0
    assert summary["writer"]["bootstrap_skill_candidate_count"] == 0
    assert (
        summary["writer"]["attempts"][0]["connection_diagnostics"]["kind"]
        == "connection_refused"
    )
    assert not (tmp_path / "bank").exists()
    assert (tmp_path / "audit/memory_writer_input.json").is_file()


def test_live_noop_is_not_reported_as_spatial_extraction(tmp_path, monkeypatch):
    fake = client(
        [
            _response(
                content=json.dumps(
                    {
                        "success_cases": [],
                        "failure_cases": [],
                        "skills": [],
                        "noop_reason": "No reusable method",
                    }
                )
            )
        ]
    )
    monkeypatch.setattr(
        replay, "build_service_client", lambda _: (fake, "direct_loopback")
    )
    summary, code = replay.execute_replay(
        payload(), tmp_path, "qwen", "http://127.0.0.1:8002/v1", service_wait_seconds=0
    )
    assert code == 0
    assert summary["phase"] == "completed_no_spatial_memory"
    assert summary["spatial_records"] == 0
    assert summary["writer"]["success"]


def test_dry_run_never_constructs_network_client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replay,
        "build_service_client",
        Mock(side_effect=AssertionError("No service access")),
    )
    summary, code = replay.execute_replay(
        payload(), tmp_path, "qwen", "http://127.0.0.1:8002/v1", dry_run=True
    )
    assert code == 0
    assert summary["dry_run"]
    assert not (tmp_path / "bank").exists()


def test_loading_then_ready_retries_only_readiness(monkeypatch):
    fake = client()
    error = type("APIStatusError", (Exception,), {"status_code": 503})("Loading model")
    fake.models.list = Mock(
        side_effect=[error, SimpleNamespace(data=[SimpleNamespace(id="qwen")])]
    )
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    summary = replay.wait_for_service(fake, "qwen", 10)
    assert summary["ready"]
    assert len(summary["attempts"]) == 1
    assert fake.calls == []


def test_auth_error_does_not_retry_until_deadline():
    fake = client()
    error = type("AuthenticationError", (Exception,), {"status_code": 401})(
        "Invalid key"
    )
    fake.models.list = Mock(side_effect=error)
    result = replay.wait_for_service(fake, "qwen", 60)
    assert result["failure_kind"] == "http_401"
    fake.models.list.assert_called_once()


def test_real_http_sdk_replay_bypasses_broken_proxy(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(self.path)
            self.send_json(
                {"object": "list", "data": [{"id": "qwen", "object": "model"}]}
            )

        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_json(
                {
                    "id": "test-local",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "success_cases": [],
                                        "failure_cases": [],
                                        "skills": [],
                                        "noop_reason": "No methods",
                                    }
                                ),
                            },
                        }
                    ],
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        summary, code = replay.execute_replay(
            payload(),
            tmp_path,
            "qwen",
            f"http://127.0.0.1:{server.server_port}/v1",
            service_wait_seconds=0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert code == 0
    assert summary["writer"]["success"]
    assert requests == ["/v1/models", "/v1/chat/completions"]
    assert summary["proxy_policy"] == "direct_loopback"


def test_acp_existing_service_smoke_does_not_launch_or_stop_service(tmp_path):
    import shutil

    bash = (
        "C:/Program Files/Git/bin/bash.exe"
        if os.name == "nt"
        else (shutil.which("bash") or "")
    )
    if not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    project = tmp_path / "project"
    source = project / "source_scene_expert"
    source.mkdir(parents=True)
    bin_dir = project / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python-stub"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$PROJECT_ROOT/python_args.txt"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    curl.chmod(0o755)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_qwen38_memory_writer_replay.sh"
    )
    env = {
        **os.environ,
        "PROJECT_ROOT": project.as_posix(),
        "SCENE_EXPERT_DIR": source.as_posix(),
        "PYTHON_BIN": fake_python.as_posix(),
        "REUSE_EXISTING_MODEL_SERVICES": "true",
        "RUN_ID": "service_smoke",
        "WAIT_TIMEOUT": "1",
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
    }
    path = bin_dir.as_posix()
    if os.name == "nt":
        path = "/" + path[0].lower() + path[2:]
    result = subprocess.run(
        [
            bash,
            "-c",
            'export PATH="$1:$PATH"; exec bash "$2"',
            "test",
            path,
            script.as_posix(),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = (project / "python_args.txt").read_text()
    assert "--api-base-url\nhttp://127.0.0.1:8002/v1" in args
    log = project / "tmp/acp_logs/service_smoke"
    assert "exit_code=0" in (log / "exit_status.env").read_text()
    assert "owned_llm_pid" not in (log / "service.env").read_text()
    assert not (log / "llama_qwen38_27b.log").exists()
