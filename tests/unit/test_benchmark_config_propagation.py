import unittest

from unittest.mock import Mock

from omegaconf import OmegaConf

from scenesmith.experiments.base_experiment import BaseExperiment


class _FakeAgent:
    def __init__(self, *, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs


class TestBenchmarkConfigPropagation(unittest.TestCase):
    """Experiment-level benchmark settings reach isolated agent configs."""

    def setUp(self):
        self.critic_config = {
            "enabled": True,
            "inject_into_llm_critic": True,
            "metrics": ["functional_dependency"],
        }
        self.cfg = OmegaConf.create(
            {
                "furniture_agent": {"_name": "fake_furniture"},
                "manipuland_agent": {"_name": "fake_manipuland"},
                "wall_agent": {"_name": "fake_wall"},
                "floor_plan_agent": {},
                "experiment": {
                    "num_workers": 1,
                    "scenebenchmark_critic": self.critic_config,
                    "geometry_generation_server": {"host": "127.0.0.1", "port": 1},
                    "hssd_retrieval_server": {"host": "127.0.0.1", "port": 2},
                    "articulated_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 3,
                    },
                    "materials_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 4,
                    },
                },
            }
        )
        self.logger = Mock()

    def test_furniture_agent_receives_experiment_critic_config(self):
        agent = BaseExperiment.build_furniture_agent(
            cfg_dict=self.cfg,
            compatible_agents={"fake_furniture": _FakeAgent},
            logger=self.logger,
        )

        self.assertEqual(
            dict(agent.cfg.scenebenchmark_critic), self.critic_config
        )

    def test_manipuland_agent_receives_experiment_critic_config(self):
        agent = BaseExperiment.build_manipuland_agent(
            cfg_dict=self.cfg,
            compatible_agents={"fake_manipuland": _FakeAgent},
            logger=self.logger,
        )

        self.assertEqual(
            dict(agent.cfg.scenebenchmark_critic), self.critic_config
        )

    def test_wall_agent_receives_experiment_critic_config(self):
        agent = BaseExperiment.build_wall_agent(
            cfg_dict=self.cfg,
            compatible_agents={"fake_wall": _FakeAgent},
            logger=self.logger,
            house_layout=Mock(),
            ceiling_height=2.7,
        )

        self.assertEqual(
            dict(agent.cfg.scenebenchmark_critic), self.critic_config
        )


if __name__ == "__main__":
    unittest.main()
