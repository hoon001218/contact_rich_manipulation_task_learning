"""Research model dimensions; native runner settings live in rsl_rl_ppo_cfg.py."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ModelCfg:
    # Serialized checkpoint contract: target slot 0, blocker slot 1, no role-ID channels.
    observation_layout: str = "fixed_roles_v1"
    max_objects: int = 40
    # Newest first: the current sample and the immediately preceding sample.
    history_steps: int = 2
    action_dim: int = 18
    node_features: int = 10
    step_features: int = 72
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
        if self.max_objects < 2:
            raise ValueError("At least two object slots are required for target and blocker.")
        if self.observation_layout != "fixed_roles_v1":
            raise ValueError("observation_layout must use fixed_roles_v1 with target/blocker slots 0/1.")
        if self.node_features != 10:
            raise ValueError("Initial geometry must contain position(3), quaternion(4), and size(3).")
        if self.step_features != 72:
            raise ValueError("A history sample must contain 72 physical observation values; mask channels are not observations.")
        if not math.isfinite(self.initial_std) or self.initial_std <= 0:
            raise ValueError("initial_std must be finite and positive.")

    @property
    def observation_dim(self) -> int:
        return self.max_objects * self.node_features + self.history_steps * self.step_features

