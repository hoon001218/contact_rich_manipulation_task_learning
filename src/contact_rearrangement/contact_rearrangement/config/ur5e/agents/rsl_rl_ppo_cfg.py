"""Native Isaac Lab runner settings, following the sweeping policy example."""

from dataclasses import asdict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .ppo_cfg import ModelCfg


@configclass
class ContactActorCfg:
    class_name: str = "contact_rearrangement.config.ur5e.agents.model:NativeContactActor"
    model_cfg: dict = asdict(ModelCfg())
    max_forward_batch_size: int = 32


@configclass
class ContactCriticCfg:
    class_name: str = "contact_rearrangement.config.ur5e.agents.model:NativeContactCritic"


@configclass
class ContactPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "contact_rearrangement.config.ur5e.agents.ppo:ContactRearrangementPPO"
    max_microbatch_size: int = 32


@configclass
class ContactRearrangementPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "contact_rearrangement"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    empirical_normalization = False
    policy = None  # RSL-RL 5 uses actor/critic model configuration.
    actor: ContactActorCfg = ContactActorCfg()
    critic: ContactCriticCfg = ContactCriticCfg()
    algorithm = ContactPpoAlgorithmCfg(
        value_loss_coef=0.25,  # Native loss omits the old trainer's extra factor 0.5.
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.5,
    )
