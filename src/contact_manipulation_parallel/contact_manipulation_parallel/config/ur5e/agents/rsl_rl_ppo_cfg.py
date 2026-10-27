"""Sweeping-Policy PPO settings with independent native MLP models."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ContactManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Keep the example PPO settings; use separate official MLPs for both roles."""

    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = "contact_manipulation_parallel"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None  # Match the min baseline's Gaussian output without an extra action clip.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="relu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="relu",
        obs_normalization=False,
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
