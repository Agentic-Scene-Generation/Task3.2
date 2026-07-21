"""Single registry for the critic metric plugins."""

from __future__ import annotations

from typing import Iterable

from scenesmith.scenebenchmark_critic.metrics.base import MetricPlugin
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.builder import (
    build_functional_dependency_checks,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.evaluator import (
    evaluate_functional_dependency,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.bedside_group import (
    evaluate_bedside_group_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_place_setting import (
    evaluate_dining_place_setting_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.dining_seat import (
    evaluate_dining_seat_distribution,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.manipuland_completeness import (
    evaluate_manipuland_completeness,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.media_support import (
    evaluate_media_support_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.room_center import (
    evaluate_room_center_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.workstation_alignment import (
    evaluate_workstation_focal_alignment,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.proposer import (
    augment_functional_dependency_checks,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.builder import (
    build_spatial_accessibility_checks,
)
from scenesmith.scenebenchmark_critic.metrics.spatial_accessibility.evaluator import (
    evaluate_spatial_accessibility as _evaluate_spatial_accessibility,
)
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.builder import (
    build_visual_clearance_checks,
)
from scenesmith.scenebenchmark_critic.metrics.visual_clearance.evaluator import (
    evaluate_visual_clearance,
)
from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.builder import (
    build_interaction_clearance_checks,
)
from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    evaluate_clearance,
)


def _interaction_evaluator(
    _case_pack: dict, check: dict, _config: object
) -> dict | None:
    return evaluate_clearance(check)


def _spatial_evaluator(
    case_pack: dict, check: dict, config: object
) -> dict | None:
    from scenesmith.scenebenchmark_critic.core.geometry import load_geometry

    store = load_geometry(case_pack)
    if store is None:
        return None
    return _evaluate_spatial_accessibility(store, check, config)


METRIC_REGISTRY: dict[str, MetricPlugin] = {
    "functional_dependency": MetricPlugin(
        name="functional_dependency",
        display_label_zh="功能依赖",
        check_builder=build_functional_dependency_checks,
        rule_evaluator=evaluate_functional_dependency,
        check_augmenter=augment_functional_dependency_checks,
        extension_evaluators=(
            evaluate_media_support_alignment,
            evaluate_bedside_group_alignment,
            evaluate_room_center_alignment,
            evaluate_dining_seat_distribution,
            evaluate_manipuland_completeness,
            evaluate_dining_place_setting_alignment,
            evaluate_workstation_focal_alignment,
        ),
    ),
    "spatial_accessibility": MetricPlugin(
        name="spatial_accessibility",
        display_label_zh="空间可达",
        check_builder=build_spatial_accessibility_checks,
        rule_evaluator=_spatial_evaluator,
    ),
    "interaction_clearance": MetricPlugin(
        name="interaction_clearance",
        display_label_zh="交互净空",
        check_builder=build_interaction_clearance_checks,
        rule_evaluator=_interaction_evaluator,
    ),
    "visual_clearance": MetricPlugin(
        name="visual_clearance",
        display_label_zh="视觉净空",
        check_builder=build_visual_clearance_checks,
        extension_evaluators=(evaluate_visual_clearance,),
    ),
}


def get_metric_plugin(name: str) -> MetricPlugin:
    try:
        return METRIC_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SceneBenchmark critic metric {name!r}; "
            f"choose from {sorted(METRIC_REGISTRY)}"
        ) from exc


def get_metric_plugins(metrics: Iterable[str]) -> tuple[MetricPlugin, ...]:
    names = tuple(dict.fromkeys(str(metric) for metric in metrics))
    return tuple(get_metric_plugin(name) for name in names)
