"""Sweeping-Policy PPO settings with independent research actor/critic encoders."""

from dataclasses import asdict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .model import CRITIC_OBSERVATION_CONTRACT, ModelCfg


@configclass
class ContactActorCfg(RslRlMLPModelCfg):
    class_name: str = "contact_manipulation_symmetric.config.ur5e.agents.model:NativeContactActor"
    model_cfg: dict = asdict(ModelCfg())
    hidden_dims = [256, 128]
    # Native actor/critic MLP heads follow the examples; scene GNN stays ReLU.
    activation = "elu"
    max_forward_batch_size: int = 8192
    # Compile tensor features; keep Triton CUDA graph replay disabled in the model.
    compile_model: bool = True
    obs_normalization: bool = False
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution", "std_type": "scalar", "init_std": 1.0,
    }
    # Compatibility with Isaac Lab's native RSL-RL configuration migration.
    stochastic: bool = False


@configclass
class ContactCriticCfg(ContactActorCfg):
    class_name: str = "contact_manipulation_symmetric.config.ur5e.agents.model:NativeContactCritic"
    observation_contract: str = CRITIC_OBSERVATION_CONTRACT
    distribution_cfg: dict | None = None


@configclass
class ContactManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Keep the example PPO settings; use separate graph models for both roles."""

    num_steps_per_env = 36
    max_iterations = 90000
    save_interval = 50
    experiment_name = "contact_manipulation_symmetric"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None  # Match the min baseline's Gaussian output without an extra action clip.
    actor = ContactActorCfg()
    critic = ContactCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
