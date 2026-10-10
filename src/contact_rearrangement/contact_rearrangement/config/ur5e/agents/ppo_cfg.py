"""Policy, PPO, and runner settings for the UR5e rearrangement task.

The layout follows the sweeping example's ``config/ur5e/agents`` package.
These dataclasses stay independent of Isaac Sim so CPU tools can instantiate
models and resume checkpoints without launching the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class ModelCfg:
    max_objects: int = 40
    # Newest first: the current sample and the immediately preceding sample.
    history_steps: int = 2
    action_dim: int = 18
    node_features: int = 10
    step_features: int = 89
    node_dim: int = 64
    graph_layers: int = 2
    temporal_dim: int = 64
    latent_dim: int = 128
    initial_std: float = 0.3

    def __post_init__(self):
        dimensions = (
            self.max_objects, self.history_steps, self.action_dim, self.node_features,
            self.step_features, self.node_dim, self.graph_layers, self.temporal_dim, self.latent_dim,
        )
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive integers.")
        if self.node_features != 10:
            raise ValueError("Initial geometry must contain position(3), quaternion(4), and size(3).")
        if not math.isfinite(self.initial_std) or self.initial_std <= 0:
            raise ValueError("initial_std must be finite and positive.")

    @property
    def observation_dim(self) -> int:
        return self.max_objects * (self.node_features + 3) + self.history_steps * (self.step_features + 1)



@dataclass
class PPOCfg:
    rollout_steps: int = 32
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    def __post_init__(self):
        if any(not isinstance(x, int) or x <= 0 for x in (self.rollout_steps, self.epochs, self.minibatches)):
            raise ValueError("PPO rollout_steps, epochs, and minibatches must be positive integers.")
        for name in ("learning_rate", "clip_ratio", "value_clip", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("value_coefficient", "entropy_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be finite and positive when supplied.")


@dataclass
class ContactRearrangementPPORunnerCfg:
    """Task-level training defaults, parallel to the sweeping runner config."""

    policy: ModelCfg = field(default_factory=ModelCfg)
    algorithm: PPOCfg = field(default_factory=PPOCfg)
    max_iterations: int = 1000
    save_interval: int = 50
    experiment_name: str = "contact_rearrangement"

    def __post_init__(self):
        if not isinstance(self.policy, ModelCfg) or not isinstance(self.algorithm, PPOCfg):
            raise TypeError("Runner policy and algorithm must be ModelCfg and PPOCfg instances.")
        if any(not isinstance(value, int) or value <= 0 for value in (self.max_iterations, self.save_interval)):
            raise ValueError("Runner max_iterations and save_interval must be positive integers.")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("Runner experiment_name must be a nonempty string.")

