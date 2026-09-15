#!/usr/bin/env python3
"""Validate GPU and cgroup memory headroom from a concurrency metrics CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized in {"unknown", "max", "n/a"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def evaluate_rows(rows: list[dict[str, str]], min_available_fraction: float) -> dict:
    """Calculate peak utilization and whether the requested headroom remains."""
    if not rows:
        raise ValueError("metrics CSV has no samples")
    gpu_fractions: list[float] = []
    cgroup_fractions: list[float] = []
    for row in rows:
        gpu_used = parse_number(row.get("gpu_memory_used_mib"))
        gpu_total = parse_number(row.get("gpu_memory_total_mib"))
        memory_current = parse_number(row.get("cgroup_memory_current"))
        memory_limit = parse_number(row.get("cgroup_memory_limit"))
        if gpu_used is not None and gpu_total and gpu_total > 0:
            gpu_fractions.append(gpu_used / gpu_total)
        if memory_current is not None and memory_limit and memory_limit > 0:
            cgroup_fractions.append(memory_current / memory_limit)
    if not gpu_fractions or not cgroup_fractions:
        raise ValueError("metrics CSV lacks usable GPU or cgroup memory samples")

    peak_gpu = max(gpu_fractions)
    peak_cgroup = max(cgroup_fractions)
    maximum_used = 1.0 - min_available_fraction
    return {
        "samples": len(rows),
        "peak_gpu_used_fraction": round(peak_gpu, 6),
        "peak_cgroup_used_fraction": round(peak_cgroup, 6),
        "minimum_available_fraction": min_available_fraction,
        "passed": peak_gpu <= maximum_used and peak_cgroup <= maximum_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--min-available-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.min_available_fraction < 1.0:
        raise SystemExit("--min-available-fraction must be between 0 and 1")

    with args.metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        report = evaluate_rows(rows, args.min_available_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
