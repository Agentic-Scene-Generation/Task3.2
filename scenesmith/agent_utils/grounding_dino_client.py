"""HTTP client and deterministic batching for the GroundingDINO sidecar."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


TokenCountMany = Callable[[list[str]], list[int]]


@dataclass(frozen=True)
class GroundingDinoClientConfig:
    """Validated connection and inference settings for the sidecar."""

    base_url: str = "http://127.0.0.1:18030"
    timeout_seconds: float = 45.0
    connect_timeout_seconds: float = 3.0
    max_retries: int = 0
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    token_budget: int = 240
    max_category_batches: int = 8

    @classmethod
    def from_config(cls, cfg: Any) -> "GroundingDinoClientConfig":
        """Build settings from a mapping or OmegaConf node."""
        return cls(
            base_url=str(cfg.get("base_url", cls.base_url)).rstrip("/"),
            timeout_seconds=float(cfg.get("timeout_seconds", 45)),
            connect_timeout_seconds=float(cfg.get("connect_timeout_seconds", 3)),
            max_retries=max(0, int(cfg.get("max_retries", 0))),
            box_threshold=_threshold(cfg.get("box_threshold", 0.25), "box"),
            text_threshold=_threshold(cfg.get("text_threshold", 0.20), "text"),
            token_budget=_positive_int(cfg.get("token_budget", 240), "token_budget"),
            max_category_batches=_positive_int(
                cfg.get("max_category_batches", 8), "max_category_batches"
            ),
        )


def build_grounding_prompt(categories: Iterable[str]) -> str:
    """Return the exact stable category prompt used by the sidecar."""
    normalized = [str(category).strip().lower() for category in categories]
    normalized = [category for category in normalized if category]
    if not normalized:
        raise ValueError("GroundingDINO categories must not be empty")
    return ". ".join(normalized) + "."


def batch_categories(
    categories: Iterable[str],
    *,
    token_budget: int,
    max_batches: int,
    token_count_many: TokenCountMany,
) -> list[dict[str, Any]]:
    """Split categories using exact tokenizer counts without dropping any phrase."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_category in categories:
        category = str(raw_category).strip().lower()
        if category and category not in seen:
            ordered.append(category)
            seen.add(category)
    if not ordered:
        raise ValueError("GroundingDINO categories must not be empty")
    if token_budget <= 0 or max_batches <= 0:
        raise ValueError("token_budget and max_batches must be positive")

    batches: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(ordered):
        candidate_texts = [
            build_grounding_prompt(ordered[cursor : end + 1])
            for end in range(cursor, len(ordered))
        ]
        token_counts = token_count_many(candidate_texts)
        if len(token_counts) != len(candidate_texts):
            raise RuntimeError("Sidecar returned an invalid token-count response")
        take = 0
        for index, raw_count in enumerate(token_counts, start=1):
            count = int(raw_count)
            if count <= token_budget:
                take = index
            else:
                break
        if take == 0:
            raise ValueError(
                f"Category {ordered[cursor]!r} exceeds the {token_budget}-token budget"
            )
        batch_items = ordered[cursor : cursor + take]
        batch_token_count = int(token_counts[take - 1])
        batches.append(
            {
                "categories": batch_items,
                "token_count": batch_token_count,
                "prompt": build_grounding_prompt(batch_items),
            }
        )
        cursor += take
        if len(batches) > max_batches:
            raise ValueError(
                f"Grounding vocabulary requires more than {max_batches} batches"
            )

    flattened = [item for batch in batches for item in batch["categories"]]
    if flattened != ordered:
        raise RuntimeError("Grounding category batching changed or dropped phrases")
    return batches


class GroundingDinoClient:
    """Small JSON client for a persistent GroundingDINO process."""

    def __init__(self, config: GroundingDinoClientConfig) -> None:
        self.config = config

    def health(self) -> dict[str, Any]:
        """Return ready-state and model identity from the sidecar."""
        return self._request_json("GET", "/health")

    def token_count_many(self, texts: list[str]) -> list[int]:
        """Count tokens with the exact tokenizer owned by the sidecar."""
        payload = self._request_json("POST", "/v1/token-count", {"texts": texts})
        counts = payload.get("token_counts")
        if not isinstance(counts, list):
            raise RuntimeError("GroundingDINO token-count response is malformed")
        return [int(count) for count in counts]

    def ground_image(
        self,
        image_path: Path,
        categories: Iterable[str],
    ) -> dict[str, Any]:
        """Run all stable category batches and merge their raw detections."""
        image_path = Path(image_path)
        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise ValueError(f"Grounding image is empty: {image_path}")
        batches = batch_categories(
            categories,
            token_budget=self.config.token_budget,
            max_batches=self.config.max_category_batches,
            token_count_many=self.token_count_many,
        )
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        detections: list[dict[str, Any]] = []
        audit_batches: list[dict[str, Any]] = []
        image_width: int | None = None
        image_height: int | None = None
        model: str | None = None
        started_at = time.monotonic()

        for index, batch in enumerate(batches, start=1):
            batch_started_at = time.monotonic()
            response = self._request_json(
                "POST",
                "/v1/ground",
                {
                    "image_base64": encoded_image,
                    "categories": batch["categories"],
                    "box_threshold": self.config.box_threshold,
                    "text_threshold": self.config.text_threshold,
                },
            )
            raw_detections = response.get("detections")
            if not isinstance(raw_detections, list):
                raise RuntimeError("GroundingDINO detection response is malformed")
            detections.extend(raw_detections)
            image_width = _consistent_dimension(
                image_width, response.get("image_width"), "width"
            )
            image_height = _consistent_dimension(
                image_height, response.get("image_height"), "height"
            )
            model = str(response.get("model") or model or "unknown")
            audit_batches.append(
                {
                    "batch_index": index,
                    "categories": list(batch["categories"]),
                    "token_count": batch["token_count"],
                    "detection_count": len(raw_detections),
                    "inference_ms": response.get("inference_ms"),
                    "request_ms": round(
                        (time.monotonic() - batch_started_at) * 1000.0, 3
                    ),
                }
            )

        return {
            "model": model or "unknown",
            "image_width": image_width,
            "image_height": image_height,
            "detections": detections,
            "batches": audit_batches,
            "request_ms": round((time.monotonic() - started_at) * 1000.0, 3),
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        timeout = max(self.config.connect_timeout_seconds, self.config.timeout_seconds)
        for _attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("GroundingDINO response must be a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(
                    f"GroundingDINO HTTP {exc.code} for {path}: {detail}"
                )
            except (OSError, ValueError, RuntimeError) as exc:
                last_error = exc
        raise RuntimeError(f"GroundingDINO request failed for {path}: {last_error}")


def _threshold(value: Any, name: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name}_threshold must be between 0 and 1")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _consistent_dimension(previous: int | None, value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"GroundingDINO returned an invalid image {name}")
    if previous is not None and previous != parsed:
        raise RuntimeError(f"GroundingDINO returned inconsistent image {name}s")
    return parsed
