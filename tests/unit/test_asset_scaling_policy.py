import json
import unittest

from types import SimpleNamespace

from scenesmith.agent_utils.asset_scaling_policy import (
    AssetScalingPolicy,
    agent_rescale_tools_enabled,
    filter_agent_rescale_tools,
)
from scenesmith.ceiling_agents.tools.ceiling_tools import CeilingTools
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.prompts import prompt_manager
from scenesmith.prompts.registry import (
    CeilingAgentPrompts,
    FurnitureAgentPrompts,
    ManipulandAgentPrompts,
    WallAgentPrompts,
)
from scenesmith.wall_agents.tools.wall_tools import WallTools


class TestAssetScalingPolicy(unittest.TestCase):
    def test_defaults_keep_existing_behavior(self):
        policy = AssetScalingPolicy.from_experiment_config({})

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.hssd_dimension_fit_enabled)
        self.assertTrue(policy.agent_rescale_tools_enabled)

    def test_master_switch_disables_both_layers(self):
        policy = AssetScalingPolicy.from_experiment_config(
            {
                "hssd_scale_to_requested_dimensions": True,
                "asset_scaling": {
                    "enabled": False,
                    "hssd_dimension_fit_enabled": True,
                    "agent_rescale_tools_enabled": True,
                },
            }
        )

        self.assertFalse(policy.enabled)
        self.assertFalse(policy.hssd_dimension_fit_enabled)
        self.assertFalse(policy.agent_rescale_tools_enabled)

    def test_subswitches_support_independent_ablations(self):
        policy = AssetScalingPolicy.from_experiment_config(
            {
                "asset_scaling": {
                    "enabled": True,
                    "hssd_dimension_fit_enabled": False,
                    "agent_rescale_tools_enabled": True,
                }
            }
        )

        self.assertFalse(policy.hssd_dimension_fit_enabled)
        self.assertTrue(policy.agent_rescale_tools_enabled)

    def test_legacy_switch_remains_hssd_only(self):
        policy = AssetScalingPolicy.from_experiment_config(
            {"hssd_scale_to_requested_dimensions": False}
        )

        self.assertFalse(policy.hssd_dimension_fit_enabled)
        self.assertTrue(policy.agent_rescale_tools_enabled)

    def test_policy_is_propagated_without_mutating_source(self):
        source = {"asset_manager": {"hssd": {"data_path": "hssd"}}}
        configured = BaseExperiment._with_experiment_hssd_scaling_config(
            source,
            {"asset_scaling": {"enabled": False}},
        )

        self.assertNotIn("asset_scaling", source)
        self.assertFalse(configured["asset_scaling"]["enabled"])
        self.assertFalse(
            configured["asset_manager"]["hssd"]["scale_to_requested_dimensions"]
        )
        self.assertFalse(agent_rescale_tools_enabled(configured))

    def test_filter_removes_every_stage_rescale_name(self):
        tools = {
            "move": object(),
            "rescale_furniture_tool": object(),
            "rescale_manipuland": object(),
            "rescale_wall_object": object(),
            "rescale_ceiling_object": object(),
        }
        disabled_config = {"asset_scaling": {"enabled": False}}

        filtered = filter_agent_rescale_tools(
            tools,
            disabled_config,
            tool_names={
                "rescale_furniture_tool",
                "rescale_manipuland",
                "rescale_wall_object",
                "rescale_ceiling_object",
            },
        )

        self.assertEqual(set(filtered), {"move"})

    def test_all_stage_tool_schemas_omit_rescale_when_disabled(self):
        disabled_config = {"asset_scaling": {"enabled": False}}
        stages = (
            (FurnitureTools, "rescale_furniture_tool"),
            (ManipulandTools, "rescale_manipuland"),
            (WallTools, "rescale_wall_object"),
            (CeilingTools, "rescale_ceiling_object"),
        )

        for tools_class, rescale_name in stages:
            with self.subTest(stage=tools_class.__name__):
                tools_instance = object.__new__(tools_class)
                tools_instance.cfg = disabled_config
                tool_schema = tools_instance._create_tool_closures()
                self.assertNotIn(rescale_name, tool_schema)

    def test_all_direct_rescale_entries_hard_reject_when_disabled(self):
        disabled_config = {"asset_scaling": {"enabled": False}}
        stages = (
            (FurnitureTools, "_rescale_furniture_impl"),
            (ManipulandTools, "_rescale_manipuland_impl"),
            (WallTools, "_rescale_wall_object_impl"),
            (CeilingTools, "_rescale_ceiling_object_impl"),
        )

        for tools_class, method_name in stages:
            with self.subTest(stage=tools_class.__name__):
                tools_instance = object.__new__(tools_class)
                tools_instance.cfg = disabled_config
                tools_instance.scene = SimpleNamespace(action_log_path=None)
                result = json.loads(
                    getattr(tools_instance, method_name)(
                        object_id="asset_0",
                        scale_factor=0.5,
                    )
                )
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "rescaling_disabled")


class TestAssetScalingPrompts(unittest.TestCase):
    CASES = (
        (
            FurnitureAgentPrompts.DESIGNER_AGENT,
            {"has_reference_image": False},
            "rescale_furniture",
        ),
        (
            FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
            {"scene_description": "A room"},
            "rescale_furniture",
        ),
        (
            ManipulandAgentPrompts.MANIPULAND_DESIGNER_AGENT,
            {
                "furniture_description": "table",
                "suggested_items": [],
                "prompt_constraints": "",
                "style_notes": "",
                "has_reference_image": False,
            },
            "rescale_manipuland",
        ),
        (
            ManipulandAgentPrompts.MANIPULAND_CRITIC_AGENT,
            {
                "furniture_description": "table",
                "suggested_items": [],
                "prompt_constraints": "",
                "style_notes": "",
            },
            "rescale_manipuland",
        ),
        (
            WallAgentPrompts.DESIGNER_AGENT,
            {
                "room_description": "room",
                "wall_count": 4,
                "required_wall_objects": "",
            },
            "rescale_wall_object",
        ),
        (
            WallAgentPrompts.STATEFUL_CRITIC_AGENT,
            {
                "room_description": "room",
                "wall_count": 4,
                "required_wall_objects": "",
            },
            "rescale_wall_object",
        ),
        (
            CeilingAgentPrompts.DESIGNER_AGENT,
            {
                "room_description": "room",
                "room_width": 4,
                "room_depth": 3,
                "ceiling_height": 2.8,
            },
            "rescale_ceiling_object",
        ),
        (
            CeilingAgentPrompts.STATEFUL_CRITIC_AGENT,
            {
                "room_description": "room",
                "room_width": 4,
                "room_depth": 3,
                "ceiling_height": 2.8,
            },
            "rescale_ceiling_object",
        ),
    )

    def test_disabled_prompts_do_not_recommend_rescale_tools(self):
        for prompt_name, kwargs, tool_name in self.CASES:
            with self.subTest(prompt=prompt_name):
                rendered = prompt_manager.get_prompt(
                    prompt_name=prompt_name,
                    asset_rescaling_enabled=False,
                    **kwargs,
                )
                self.assertNotIn(tool_name, rendered)
                self.assertIn("rescal", rendered.lower())

    def test_enabled_prompts_keep_existing_rescale_guidance(self):
        for prompt_name, kwargs, tool_name in self.CASES:
            with self.subTest(prompt=prompt_name):
                rendered = prompt_manager.get_prompt(
                    prompt_name=prompt_name,
                    asset_rescaling_enabled=True,
                    **kwargs,
                )
                self.assertIn(tool_name, rendered)


if __name__ == "__main__":
    unittest.main()
