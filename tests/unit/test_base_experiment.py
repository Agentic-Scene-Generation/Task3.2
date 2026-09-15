import unittest

from scenesmith.experiments.base_experiment import BaseExperiment


class TestExperimentHssdScalePolicy(unittest.TestCase):
    def test_injects_experiment_policy_without_mutating_agent_config(self):
        agent_config = {
            "asset_manager": {
                "hssd": {"data_path": "data/hssd-models"},
            }
        }

        configured = BaseExperiment._with_experiment_hssd_scaling_config(
            agent_config,
            {"hssd_scale_to_requested_dimensions": False},
        )

        self.assertFalse(
            configured["asset_manager"]["hssd"]["scale_to_requested_dimensions"]
        )
        self.assertNotIn(
            "scale_to_requested_dimensions", agent_config["asset_manager"]["hssd"]
        )

    def test_defaults_experiment_policy_to_enabled(self):
        configured = BaseExperiment._with_experiment_hssd_scaling_config(
            {"asset_manager": {"hssd": {}}},
            {},
        )

        self.assertTrue(
            configured["asset_manager"]["hssd"]["scale_to_requested_dimensions"]
        )
