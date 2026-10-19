"""Native RSL-RL runner settings for the shared GNN/attention research policy."""

from dataclasses import asdict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .model import ModelCfg


@configclass
class ContactActorCfg:
    class_name: str = "contact_manipulation.config.ur5e.agents.ppo:NativeContactActor"
    model_cfg: dict = asdict(ModelCfg())
    max_forward_batch_size: int = 8192
    # Inductor fuses the unchanged FP32 model/mask math; no CUDA graph replay.
    compile_model: bool = True
    obs_normalization: bool = False
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution", "std_type": "log", "init_std": ModelCfg().initial_std,
    }
    # Compatibility with Isaac Lab's native RSL-RL configuration migration.
    stochastic: bool = False


@configclass
class ContactCriticCfg:
    class_name: str = "contact_manipulation.config.ur5e.agents.ppo:NativeContactCritic"
    max_forward_batch_size: int = 8192
    obs_normalization: bool = False
    distribution_cfg: dict | None = None
    stochastic: bool = False


@configclass
class ContactPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "contact_manipulation.config.ur5e.agents.ppo:ContactManipulationPPO"
    max_microbatch_size: int = 8192
    num_learning_epochs = 4
    num_mini_batches = 4
    learning_rate = 3.0e-4
    schedule = "fixed"
    gamma = 0.99
    lam = 0.95
    clip_param = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.25
    use_clipped_value_loss = True
    max_grad_norm = 0.5
    desired_kl = None


@configclass
class ContactManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Keep native rollout/GAE/logging; bound graph activations in the models/PPO."""

    seed = 0
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "contact_manipulation"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = None  # Match the min baseline's Gaussian output without an extra action clip.
    actor = ContactActorCfg()
    critic = ContactCriticCfg()
    algorithm = ContactPpoAlgorithmCfg()
