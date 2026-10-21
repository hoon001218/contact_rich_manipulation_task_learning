"""Sweeping-Policy random-task PPO settings with the research actor encoder."""

from dataclasses import asdict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .model import ModelCfg


@configclass
class ContactActorCfg(RslRlMLPModelCfg):
    class_name: str = "contact_manipulation.config.ur5e.agents.model:NativeContactActor"
    model_cfg: dict = asdict(ModelCfg())
    hidden_dims = [256, 128]
    activation = "relu"
    max_forward_batch_size: int = 8192
    # Inductor fuses the unchanged FP32 model/mask math; no CUDA graph replay.
    compile_model: bool = True
    obs_normalization: bool = False
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution", "std_type": "scalar", "init_std": 1.0,
    }
    # Compatibility with Isaac Lab's native RSL-RL configuration migration.
    stochastic: bool = False


@configclass
class ContactManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Match examples/Sweeping-Policy's registered random-task rsl_rl_ppo_cfg_02."""

    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = "contact_manipulation"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = None  # Match the min baseline's Gaussian output without an extra action clip.
    actor = ContactActorCfg()
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
