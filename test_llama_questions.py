#!/usr/bin/env python3
"""Ask a local llama-server several smoke-test questions and save the results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS: list[dict[str, str]] = [
    {
        "title": "模型身份",
        "prompt": (
            "你是什么模型？请区分说明：模型家族、你能从当前对话中可靠知道的版本信息，"
            "以及你不能自行确认的部署细节。不要编造参数量、量化格式或推理框架。"
        ),
    },
    {
        "title": "伯努利猜想",
        "prompt": (
            "请介绍一下数学史上常被称为“伯努利猜想”的命题。若这个中文名称可能指向"
            "多个不同命题，请先消歧；然后介绍最常见含义的历史背景、准确表述、关键人物、"
            "证明或未解决状态。请明确区分已知事实与不确定信息。"
        ),
    },
    {
        "title": "伯努利概率概念辨析",
        "prompt": (
            "请用一个统一的抛硬币例子，解释伯努利试验、伯努利分布、二项分布和伯努利大数定律"
            "之间的区别与联系。给出必要公式，并指出最容易混淆的地方。"
        ),
    },
    {
        "title": "逻辑推理",
        "prompt": (
            "有三个盒子，标签分别是“苹果”“橙子”“苹果和橙子”，但三个标签全部贴错了。"
            "你只能从一个盒子里取出一个水果看一次。如何确定三个盒子的正确标签？请逐步解释。"
        ),
    },
    {
        "title": "室内场景规划",
        "prompt": (
            "为一个4米×5米的卧室制定简洁布局方案：必须包含双人床、两个床头柜、衣柜和书桌，"
            "同时保留从门到床和衣柜的通行路径。请说明功能分区、摆放原则、关键尺寸约束和"
            "最需要检查的碰撞/可达性问题。"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Qwen3.6 llama-server and save Markdown + JSONL results."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "LLAMA_TEST_BASE_URL", "http://127.0.0.1:8002/v1"
        ),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLAMA_TEST_MODEL", ""),
        help="Model ID. When omitted, use the first ID returned by /v1/models.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "sk-123"),
        help="Bearer token sent to the local server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs" / "llama_question_tests",
    )
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen thinking for every request.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Append an additional question. May be supplied multiple times.",
    )
    return parser.parse_args()


def normalize_base_url(value: str) -> tuple[str, str]:
    api_base = value.rstrip("/")
    if api_base.endswith("/v1"):
        server_base = api_base[:-3]
    else:
        server_base = api_base
        api_base = f"{api_base}/v1"
    return api_base, server_base.rstrip("/")


def request_json(
    url: str,
    *,
    api_key: str,
    timeout: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {body[:2000]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected JSON response from {url}: {type(result).__name__}")
    return result


def wait_until_ready(server_base: str, api_key: str, timeout: int) -> None:
    health_url = f"{server_base}/health"
    deadline = time.monotonic() + timeout
    last_error = "server not ready"
    next_report = 0.0
    while time.monotonic() < deadline:
        try:
            request_json(health_url, api_key=api_key, timeout=5)
            print(f"[OK] llama-server is ready: {health_url}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - report the latest readiness error
            last_error = str(exc)
        now = time.monotonic()
        if now >= next_report:
            remaining = max(0, int(deadline - now))
            print(
                f"[WAIT] llama-server is loading; remaining timeout={remaining}s; "
                f"last error={last_error}",
                flush=True,
            )
            next_report = now + 15
        time.sleep(5)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s: {last_error}")


def resolve_model(api_base: str, api_key: str, requested_model: str) -> str:
    models = request_json(
        f"{api_base}/models", api_key=api_key, timeout=30
    ).get("data", [])
    available = [item.get("id", "") for item in models if isinstance(item, dict)]
    available = [model for model in available if model]
    if requested_model:
        if available and requested_model not in available:
            raise RuntimeError(
                f"requested model {requested_model!r} is not served; available={available}"
            )
        return requested_model
    if not available:
        raise RuntimeError("/v1/models returned no model IDs")
    return available[0]


def extract_message(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", ""
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content), str(reasoning)


def main() -> int:
    args = parse_args()
    if args.wait_timeout < 1 or args.request_timeout < 1 or args.max_tokens < 1:
        raise SystemExit("timeouts and max-tokens must be positive integers")

    api_base, server_base = normalize_base_url(args.base_url)
    wait_until_ready(server_base, args.api_key, args.wait_timeout)
    model = resolve_model(api_base, args.api_key, args.model)

    questions = list(DEFAULT_QUESTIONS)
    questions.extend(
        {"title": f"自定义问题 {index}", "prompt": question}
        for index, question in enumerate(args.question, start=1)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    markdown_path = args.output_dir / f"llama_questions_{timestamp}.md"
    jsonl_path = args.output_dir / f"llama_questions_{timestamp}.jsonl"

    system_prompt = (
        "你是一个严谨、诚实的中文助手。优先保证事实准确；遇到名称歧义或信息不足时明确说明，"
        "不要用猜测填补事实。"
    )
    failures = 0

    with markdown_path.open("w", encoding="utf-8") as markdown, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl:
        markdown.write("# Qwen3.6-27B-MTP-GGUF 问答测试\n\n")
        markdown.write(f"- 时间：{dt.datetime.now().astimezone().isoformat()}\n")
        markdown.write(f"- API：`{api_base}`\n")
        markdown.write(f"- 模型：`{model}`\n")
        markdown.write(f"- Thinking：`{not args.no_thinking}`\n")
        markdown.write(f"- temperature：`{args.temperature}`\n\n")
        markdown.flush()

        for index, question in enumerate(questions, start=1):
            title = question["title"]
            prompt = question["prompt"]
            print(f"[{index}/{len(questions)}] {title}", flush=True)
            request_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": args.temperature,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": args.max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
            }
            started = time.monotonic()
            error = ""
            response: dict[str, Any] = {}
            content = ""
            reasoning = ""
            try:
                response = request_json(
                    f"{api_base}/chat/completions",
                    api_key=args.api_key,
                    timeout=args.request_timeout,
                    payload=request_payload,
                )
                content, reasoning = extract_message(response)
                if not content and not reasoning:
                    error = "response contained neither content nor reasoning_content"
                    failures += 1
            except Exception as exc:  # noqa: BLE001 - save each failed request
                error = str(exc)
                failures += 1
            elapsed = time.monotonic() - started

            record = {
                "index": index,
                "title": title,
                "prompt": prompt,
                "model": model,
                "elapsed_seconds": round(elapsed, 3),
                "error": error,
                "content": content,
                "reasoning_content": reasoning,
                "request": request_payload,
                "response": response,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()

            markdown.write(f"## {index}. {title}\n\n")
            markdown.write(f"**问题**\n\n{prompt}\n\n")
            markdown.write(f"**耗时**：{elapsed:.3f} 秒\n\n")
            if error:
                markdown.write(f"**错误**\n\n```text\n{error}\n```\n\n")
            if reasoning:
                markdown.write("<details>\n<summary>推理内容</summary>\n\n")
                markdown.write(reasoning + "\n\n</details>\n\n")
            if content:
                markdown.write(f"**回答**\n\n{content}\n\n")
            markdown.flush()

            status = "ERROR" if error else "OK"
            print(f"[{status}] {title}: {elapsed:.3f}s", flush=True)

    print(f"[DONE] Markdown: {markdown_path}")
    print(f"[DONE] JSONL:   {jsonl_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - concise CLI failure message
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)
