from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ProgressFn = Callable[[str], None]
RuleEvaluator = Callable[
    [dict[str, Any], dict[str, Any], Any], dict[str, Any] | None
]
CheckBuilder = Callable[
    [dict[str, Any], tuple[str, ...] | list[str] | None], list[dict[str, Any]]
]
CheckAugmenter = Callable[
    [dict[str, Any], Any, list[str] | None, ProgressFn | None], bool
]
ExtensionEvaluator = Callable[[dict[str, Any]], list[dict[str, Any]]]
SummaryPolicy = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class MetricPlugin:
    name: str
    display_label_zh: str
    check_builder: CheckBuilder | None = None
    rule_evaluator: RuleEvaluator | None = None
    check_augmenter: CheckAugmenter | None = None
    extension_evaluators: tuple[ExtensionEvaluator, ...] = ()
    counts_toward_summary: SummaryPolicy | None = None
