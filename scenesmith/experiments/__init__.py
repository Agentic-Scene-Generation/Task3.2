from omegaconf import DictConfig

from .base_experiment import BaseExperiment
from .indoor_scene_generation import IndoorSceneGenerationExperiment

# Each key has to be a yaml file under '[project_root]/configurations/experiment'
# without .yaml suffix.
exp_registry = dict(
    indoor_scene_generation=IndoorSceneGenerationExperiment,
    ablation_1_scenesmith_original=IndoorSceneGenerationExperiment,
    ablation_2_qwen3_naive=IndoorSceneGenerationExperiment,
    ablation_3_qwen3_harness=IndoorSceneGenerationExperiment,
    ablation_4_qwen3_harness_memory=IndoorSceneGenerationExperiment,
    ablation_4a_qwen3_lexical_memory=IndoorSceneGenerationExperiment,
    ablation_4b_qwen3_vector_memory=IndoorSceneGenerationExperiment,
    ablation_4c_qwen3_hybrid_memory=IndoorSceneGenerationExperiment,
    ablation_5_qwen3_full=IndoorSceneGenerationExperiment,
)


def build_experiment(cfg: DictConfig) -> BaseExperiment:
    """
    Build an experiment instance based on registry

    Args:
        cfg (DictConfig): The experiment configuration.

    Returns:
        BaseExperiment: The experiment instance.
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry "
            f"{list(exp_registry.keys())}. Make sure you register it correctly in "
            "'experiment/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg)
