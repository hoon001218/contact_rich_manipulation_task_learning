"""Jointly trained initial-scene graph and finite sensor-history policy.

This module deliberately has no Isaac Sim imports. The observation contains
initial camera geometry, with target and blocker in reserved slots 0 and 1,
and the current/previous robot and sensor samples. Callers derive masks from
that observation before passing observation and masks separately to the model.
Masks control padding, sensor filtering, and role selection; they are not learned features.
Both actor and critic use the same observable information.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ppo_cfg import ModelCfg


@dataclass(frozen=True)
class PolicyMasks:
    """Boolean selectors derived from the public observation, outside the model."""

    object_valid: Tensor
    roles: Tensor
    history_valid: Tensor
    tactile_valid: Tensor


def _split_observation(observation: Tensor, cfg: ModelCfg) -> tuple[Tensor, Tensor]:
    if observation.ndim < 1 or observation.shape[-1] != cfg.observation_dim:
        width = observation.shape[-1] if observation.ndim else 0
        raise ValueError(f"Expected observation width {cfg.observation_dim}, got {width}.")
    geometry, history = observation.split(
        (cfg.max_objects * 10, cfg.history_steps * cfg.step_features), dim=-1,
    )
    leading = observation.shape[:-1]
    return (
        geometry.reshape(*leading, cfg.max_objects, 10),
        history.reshape(*leading, cfg.history_steps, cfg.step_features),
    )


def build_policy_masks(observation: Tensor, cfg: ModelCfg) -> PolicyMasks:
    """Derive every selector from observation values without simulator metadata.

    An observed OBB has positive extents and a nonzero quaternion. A history
    sample has a finite robot/force/action state and a nonzero TCP quaternion;
    all-zero reset padding fails that check. Tactile 0/1 are measured values,
    while -1 (or nonfinite input) means unavailable. Slots 0 and 1 always
    represent target and blocker; their selectors remain false if the reserved
    slots are unobserved. Later slots represent ordinary observed objects.
    """
    geometry, history = _split_observation(observation, cfg)
    object_valid = (
        torch.isfinite(geometry).all(-1)
        & (geometry[..., 7:10] > 0).all(-1)
        & (geometry[..., 3:7].square().sum(-1) > 1e-12)
    )
    roles = torch.zeros((*object_valid.shape, 2), dtype=torch.bool, device=observation.device)
    roles[..., 0, 0] = object_valid[..., 0]
    roles[..., 1, 1] = object_valid[..., 1]
    history_valid = (
        torch.isfinite(history[..., :37]).all(-1)
        & torch.isfinite(history[..., 54:]).all(-1)
        & (history[..., 27:31].square().sum(-1) > 1e-12)
    )
    tactile = history[..., 37:54]
    tactile_valid = history_valid[..., None] & ((tactile == 0) | (tactile == 1))
    return PolicyMasks(object_valid, roles, history_valid, tactile_valid)




def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(-2) / weights.sum(-2).clamp_min(1.0)


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Normalize valid entries and return zeros for a completely empty set."""
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).eps)


class SceneGraphEncoder(nn.Module):
    """Dense geometric message passing with target/blocker-conditioned readouts."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        width = cfg.node_dim
        self.node_encoder = nn.Sequential(nn.Linear(cfg.node_features, width), nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width))
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 6, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(cfg.graph_layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width))
            for _ in range(cfg.graph_layers)
        ])
        self.role_queries = nn.Linear(2 * width, 2 * width)
        self.keys = nn.Linear(width, width, bias=False)
        self.values = nn.Linear(width, width, bias=False)
        self.width = width

    def forward(self, geometry: Tensor, valid: Tensor, roles: Tensor) -> Tensor:
        geometry = torch.where(valid[..., None], geometry, torch.zeros_like(geometry))
        roles = torch.where(valid[..., None], roles, torch.zeros_like(roles))
        hidden = self.node_encoder(geometry) * valid[..., None]
        count = geometry.shape[-2]
        edges = valid[..., :, None] & valid[..., None, :]
        edges &= ~torch.eye(count, dtype=torch.bool, device=geometry.device)
        # Receiver i and sender j; these relations come solely from the initial
        # camera geometry, and never from oracle contact or occlusion labels.
        relative_position = geometry[..., None, :, :3] - geometry[..., :, None, :3]
        relative_size = geometry[..., None, :, 7:10] - geometry[..., :, None, 7:10]
        relations = torch.cat((relative_position, relative_size), dim=-1)
        for message, update in zip(self.messages, self.updates):
            receivers = hidden[..., :, None, :].expand(*hidden.shape[:-2], count, count, self.width)
            senders = hidden[..., None, :, :].expand_as(receivers)
            messages = message(torch.cat((receivers, senders, relations), dim=-1))
            messages = messages * edges[..., None]
            aggregated = messages.sum(-2) / edges.sum(-1, keepdim=True).clamp_min(1)
            hidden = (hidden + update(torch.cat((hidden, aggregated), dim=-1))) * valid[..., None]
        target = _masked_mean(hidden, valid & roles[..., 0])
        blocker = _masked_mean(hidden, valid & roles[..., 1])
        queries = self.role_queries(torch.cat((target, blocker), dim=-1)).reshape(*target.shape[:-1], 2, self.width)
        scores = torch.matmul(queries, self.keys(hidden).transpose(-2, -1)) / math.sqrt(self.width)
        attention = _masked_softmax(scores, valid[..., None, :].expand_as(scores))
        contextual = torch.matmul(attention, self.values(hidden)).flatten(-2)
        return torch.cat((target, blocker, contextual, _masked_mean(hidden, valid)), dim=-1)


class HistoryEncoder(nn.Module):
    """Encode the current/previous pair, with explicit previous-sample reset padding."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.project = nn.Conv1d(cfg.step_features, cfg.temporal_dim, kernel_size=1)
        # The receptive field covers the complete requested window.  No hidden
        # state survives an episode, and invalid reset padding is zero at each layer.
        self.dilations = []
        receptive_field, dilation = 1, 1
        while receptive_field < cfg.history_steps:
            self.dilations.append(dilation)
            receptive_field += 2 * dilation
            dilation *= 2
        self.convolutions = nn.ModuleList([
            nn.Conv1d(cfg.temporal_dim, cfg.temporal_dim, kernel_size=3, dilation=dilation)
            for dilation in self.dilations
        ])

    def forward(self, history: Tensor, valid: Tensor, tactile_valid: Tensor) -> Tensor:
        # Filter unavailable tactile values before learned projection. Boolean
        # validity selectors are never concatenated to the sample features.
        history = torch.cat((history[..., :37],
                             torch.where(tactile_valid, history[..., 37:54],
                                         torch.zeros_like(history[..., 37:54])),
                             history[..., 54:]), dim=-1)
        history = torch.where(valid[..., None], history, torch.zeros_like(history))
        # Public histories are newest first; convolutions run oldest to newest.
        valid = valid.flip(-1)
        history = history.flip(-2)
        inputs = history.transpose(-2, -1)
        padding_mask = valid[:, None, :]
        hidden = F.silu(self.project(inputs)) * padding_mask
        for convolution, dilation in zip(self.convolutions, self.dilations):
            hidden = F.silu(hidden + convolution(F.pad(hidden, (2 * dilation, 0)))) * padding_mask
        hidden = hidden.transpose(-2, -1)
        positions = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
        newest_index = torch.where(valid, positions, torch.zeros_like(positions)).max(-1).values
        latest = hidden[torch.arange(hidden.shape[0], device=hidden.device), newest_index]
        latest = latest * valid.any(-1, keepdim=True)
        return torch.cat((latest, _masked_mean(hidden, valid)), dim=-1)


class ContactRearrangementPolicy(nn.Module):
    """Tanh-Gaussian 18D actor and a critic sharing graph/history encoders."""

    def __init__(self, cfg: ModelCfg | None = None):
        super().__init__()
        self.cfg = cfg or ModelCfg()
        self.scene_encoder = SceneGraphEncoder(self.cfg)
        self.history_encoder = HistoryEncoder(self.cfg)
        self.fusion = nn.Sequential(
            nn.Linear(5 * self.cfg.node_dim + 2 * self.cfg.temporal_dim, 2 * self.cfg.latent_dim), nn.SiLU(),
            nn.Linear(2 * self.cfg.latent_dim, self.cfg.latent_dim), nn.SiLU(),
        )
        self.actor = nn.Linear(self.cfg.latent_dim, self.cfg.action_dim)
        self.critic = nn.Linear(self.cfg.latent_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.cfg.action_dim,), math.log(self.cfg.initial_std)))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def _split(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        geometry, history = _split_observation(observation, self.cfg)
        return (
            geometry.reshape(-1, self.cfg.max_objects, 10),
            history.reshape(-1, self.cfg.history_steps, self.cfg.step_features),
        )

    def encode(self, observation: Tensor, *, masks: PolicyMasks) -> Tensor:
        leading = observation.shape[:-1]
        geometry, history = self._split(observation)
        expected_shapes = (
            (*leading, self.cfg.max_objects), (*leading, self.cfg.max_objects, 2),
            (*leading, self.cfg.history_steps), (*leading, self.cfg.history_steps, 17),
        )
        values = (masks.object_valid, masks.roles, masks.history_valid, masks.tactile_valid)
        for value, shape in zip(values, expected_shapes):
            if value.dtype != torch.bool or tuple(value.shape) != shape or value.device != observation.device:
                raise ValueError(f"Policy masks must be Boolean tensors on the observation device with shape {shape}.")
        scene = self.scene_encoder(geometry, masks.object_valid.reshape(-1, self.cfg.max_objects),
                                   masks.roles.reshape(-1, self.cfg.max_objects, 2))
        temporal = self.history_encoder(history, masks.history_valid.reshape(-1, self.cfg.history_steps),
                                        masks.tactile_valid.reshape(-1, self.cfg.history_steps, 17))
        return self.fusion(torch.cat((scene, temporal), dim=-1)).reshape(*leading, self.cfg.latent_dim)

    def forward(self, observation: Tensor, *, masks: PolicyMasks) -> tuple[Tensor, Tensor]:
        latent = self.encode(observation, masks=masks)
        return self.actor(latent), self.critic(latent).squeeze(-1)

    def value(self, observation: Tensor, *, masks: PolicyMasks) -> Tensor:
        return self.forward(observation, masks=masks)[1]

    def _distribution(self, mean: Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(mean, self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean))

    @staticmethod
    def _bounded_action(unconstrained: Tensor) -> Tensor:
        # Leave a representable inverse even when float32 tanh rounds to +/-1.
        return unconstrained.tanh().clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _action_log_prob(distribution: torch.distributions.Normal, actions: Tensor) -> Tensor:
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        unconstrained = torch.atanh(bounded)
        # Stable exact log|d tanh(u)/du| for interior actions.
        log_jacobian = 2.0 * (math.log(2.0) - unconstrained - F.softplus(-2.0 * unconstrained))
        elementwise = distribution.log_prob(unconstrained) - log_jacobian
        # Numerical clipping puts the tail probability at each boundary.  Use
        # that actual probability mass instead of attributing a distant latent
        # sample to the edge density; PPO ratios then remain consistent even
        # when a large mean or variance saturates tanh.
        limit = torch.atanh(actions.new_tensor(1.0 - 1e-6))
        upper = torch.special.log_ndtr((distribution.loc - limit) / distribution.scale)
        lower = torch.special.log_ndtr((-limit - distribution.loc) / distribution.scale)
        elementwise = torch.where(actions >= 1.0 - 1e-6, upper, elementwise)
        elementwise = torch.where(actions <= -1.0 + 1e-6, lower, elementwise)
        return elementwise.sum(-1)

    def act(self, observation: Tensor, deterministic: bool = False, *, masks: PolicyMasks) -> tuple[Tensor, Tensor, Tensor]:
        mean, value = self.forward(observation, masks=masks)
        distribution = self._distribution(mean)
        unconstrained = mean if deterministic else distribution.rsample()
        actions = self._bounded_action(unconstrained)
        return actions, self._action_log_prob(distribution, actions), value

    def evaluate_actions(
        self, observation: Tensor, actions: Tensor, *, masks: PolicyMasks, entropy_noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, value = self.forward(observation, masks=masks)
        if actions.shape != mean.shape:
            raise ValueError(f"Actions must have shape {tuple(mean.shape)}, got {tuple(actions.shape)}.")
        distribution = self._distribution(mean)
        log_prob = self._action_log_prob(distribution, actions)
        # Squashed-Gaussian entropy has no analytic expression.  Use one
        # differentiable Monte Carlo sample of the actual bounded distribution.
        # A caller splitting a PPO minibatch can supply slices of one shared
        # noise draw so tensor chunk sizes do not change that sampled objective.
        if entropy_noise is None:
            sample = distribution.rsample()
        else:
            if entropy_noise.shape != mean.shape or entropy_noise.device != mean.device or entropy_noise.dtype != mean.dtype:
                raise ValueError("Entropy noise must match the action mean shape, device, and dtype.")
            sample = mean + distribution.scale * entropy_noise
        entropy = -self._action_log_prob(distribution, self._bounded_action(sample))
        return log_prob, entropy, value


class NativeContactActor(nn.Module):
    """Expose the research policy through the RSL-RL model interface.

    The actor owns the complete shared policy, including its value head. The
    companion critic keeps an unregistered reference to this same policy so
    the native PPO optimizer sees every parameter exactly once. Observation
    masks are derived before each research-model call, never added as features.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: Mapping[str, Tensor],
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        model_cfg: dict | ModelCfg | None = None,
        max_forward_batch_size: int = 32,
        obs_normalization: bool = False,
    ):
        super().__init__()
        self.cfg = model_cfg if isinstance(model_cfg, ModelCfg) else ModelCfg(**(model_cfg or {}))
        if output_dim != self.cfg.action_dim:
            raise ValueError(f"Actor output must contain {self.cfg.action_dim} research actions.")
        if obs_normalization:
            raise ValueError("Research observations retain physical values for observation-derived masks.")
        if not isinstance(max_forward_batch_size, int) or isinstance(max_forward_batch_size, bool) or max_forward_batch_size <= 0:
            raise ValueError("max_forward_batch_size must be a positive integer.")
        self.obs_groups = list(obs_groups[obs_set])
        if self.obs_groups != ["policy"]:
            raise ValueError("The research model requires the single public 'policy' observation group.")
        self.obs_dim = self.cfg.observation_dim
        self.output_dim = output_dim
        self.max_forward_batch_size = max_forward_batch_size
        self.policy = ContactRearrangementPolicy(self.cfg)
        self._normal: torch.distributions.Normal | None = None
        self._entropy: Tensor | None = None
        self._entropy_noise: Tensor | None = None
        self._cached_values: tuple | None = None
        self.get_policy_observation(obs)

    def get_policy_observation(self, obs: Mapping[str, Tensor]) -> Tensor:
        """Return the public observation without adding masks or oracle values."""
        observation = obs["policy"]
        if observation.ndim < 2 or observation.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected a batched policy observation with width {self.obs_dim}.")
        return observation

    def _forward_policy(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        leading = observation.shape[:-1]
        flattened = observation.reshape(-1, self.obs_dim)
        if not flattened.shape[0]:
            raise ValueError("The research model requires a nonempty observation batch.")
        results = [
            self.policy(chunk, masks=build_policy_masks(chunk, self.cfg))
            for chunk in flattened.split(self.max_forward_batch_size)
        ]
        return (
            torch.cat([result[0] for result in results]).reshape(*leading, self.output_dim),
            torch.cat([result[1] for result in results]).reshape(*leading, 1),
        )

    def _cache_token(self, obs: Mapping[str, Tensor], observation: Tensor) -> tuple | None:
        try:
            observation_version = observation._version
        except RuntimeError:
            # Inference tensors have no mutation counter. Recompute their
            # values rather than accepting potentially mutated cached inputs.
            return None
        return id(obs), id(observation), observation_version, tuple(parameter._version for parameter in self.policy.parameters())

    def consume_cached_value(self, obs: Mapping[str, Tensor]) -> Tensor | None:
        """Consume an immediately preceding actor value at most once."""
        cached, self._cached_values = self._cached_values, None
        if cached is None:
            return None
        observation = self.get_policy_observation(obs)
        token = self._cache_token(obs, observation)
        return cached[1] if token is not None and token == cached[0] else None

    def forward(
        self,
        obs: Mapping[str, Tensor],
        masks: Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
        entropy_noise: Tensor | None = None,
    ) -> Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("The finite observation history does not use recurrent state or trajectory masks.")
        observation = self.get_policy_observation(obs)
        mean, values = self._forward_policy(observation)
        self._normal = self.policy._distribution(mean)
        if entropy_noise is not None and (
            entropy_noise.shape != mean.shape or entropy_noise.device != mean.device or entropy_noise.dtype != mean.dtype
        ):
            raise ValueError("Entropy noise must match the action mean shape, device, and dtype.")
        self._entropy, self._entropy_noise = None, entropy_noise
        token = self._cache_token(obs, observation)
        self._cached_values = None if token is None else (token, values)
        return self.policy._bounded_action(self._normal.rsample() if stochastic_output else mean)

    def _current_distribution(self) -> torch.distributions.Normal:
        if self._normal is None:
            raise RuntimeError("Call the actor before querying its current action distribution.")
        return self._normal

    @property
    def output_mean(self) -> Tensor:
        """Mean of the underlying Gaussian, before the action tanh transform."""
        return self._current_distribution().loc

    @property
    def output_std(self) -> Tensor:
        """Standard deviation of the underlying Gaussian, matching PPO state."""
        return self._current_distribution().scale

    @property
    def output_distribution_params(self) -> tuple[Tensor, Tensor]:
        return self.output_mean, self.output_std

    @property
    def output_entropy(self) -> Tensor:
        if self._entropy is None:
            distribution = self._current_distribution()
            sample = distribution.rsample() if self._entropy_noise is None else distribution.loc + distribution.scale * self._entropy_noise
            self._entropy = -self.policy._action_log_prob(distribution, self.policy._bounded_action(sample))
        return self._entropy

    def get_output_log_prob(self, outputs: Tensor) -> Tensor:
        distribution = self._current_distribution()
        if outputs.shape != distribution.loc.shape:
            raise ValueError(f"Actions must have shape {tuple(distribution.loc.shape)}, got {tuple(outputs.shape)}.")
        return self.policy._action_log_prob(distribution, outputs)

    def get_kl_divergence(
        self, old_params: tuple[Tensor, Tensor], new_params: tuple[Tensor, Tensor],
    ) -> Tensor:
        return torch.distributions.kl_divergence(
            torch.distributions.Normal(*old_params), torch.distributions.Normal(*new_params),
        ).sum(-1)

    def reset(self, dones: Tensor | None = None, hidden_state=None) -> None:
        self._cached_values = None

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: Mapping[str, Tensor]) -> None:
        pass

    def as_jit(self):
        raise NotImplementedError("Graph/history policy export is not implemented.")

    def as_onnx(self, verbose: bool = False):
        raise NotImplementedError("Graph/history policy export is not implemented.")


class NativeContactCritic(nn.Module):
    """Native critic view of the value head owned by the shared actor policy."""

    is_recurrent = False

    def __init__(
        self,
        obs: Mapping[str, Tensor],
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        shared_policy: ContactRearrangementPolicy,
        actor: NativeContactActor | None = None,
        model_cfg: dict | ModelCfg | None = None,
        max_forward_batch_size: int = 32,
        obs_normalization: bool = False,
    ):
        super().__init__()
        self.cfg = shared_policy.cfg if model_cfg is None else (
            model_cfg if isinstance(model_cfg, ModelCfg) else ModelCfg(**model_cfg)
        )
        if asdict(self.cfg) != asdict(shared_policy.cfg):
            raise ValueError("Actor and critic must use the same shared research model configuration.")
        if output_dim != 1 or obs_normalization:
            raise ValueError("The shared research critic requires one value output and unnormalized observations.")
        if not isinstance(max_forward_batch_size, int) or isinstance(max_forward_batch_size, bool) or max_forward_batch_size <= 0:
            raise ValueError("max_forward_batch_size must be a positive integer.")
        self.obs_groups = list(obs_groups[obs_set])
        if self.obs_groups != ["policy"]:
            raise ValueError("Actor and critic must use the same single public 'policy' observation group.")
        self.obs_dim = self.cfg.observation_dim
        self.output_dim = output_dim
        self.max_forward_batch_size = max_forward_batch_size
        # nn.Module registration would add these parameters to native PPO a
        # second time. The actor owns and moves the entire shared policy.
        object.__setattr__(self, "policy", shared_policy)
        object.__setattr__(self, "_actor", actor)
        if obs["policy"].shape[-1] != self.obs_dim:
            raise ValueError(f"Expected a policy observation with width {self.obs_dim}.")
        if actor is not None and actor.policy is not shared_policy:
            raise ValueError("The critic must reference the same policy owned by its actor.")

    def forward(self, obs: Mapping[str, Tensor], masks: Tensor | None = None, hidden_state=None) -> Tensor:
        if masks is not None or hidden_state is not None:
            raise ValueError("The finite observation history does not use recurrent state or trajectory masks.")
        if self._actor is not None:
            cached = self._actor.consume_cached_value(obs)
            if cached is not None:
                return cached
        observation = obs["policy"]
        if observation.ndim < 2 or observation.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected a batched policy observation with width {self.obs_dim}.")
        flattened = observation.reshape(-1, self.obs_dim)
        if not flattened.shape[0]:
            raise ValueError("The research model requires a nonempty observation batch.")
        values = [
            self.policy.value(chunk, masks=build_policy_masks(chunk, self.cfg))
            for chunk in flattened.split(self.max_forward_batch_size)
        ]
        return torch.cat(values).reshape(*observation.shape[:-1], 1)

    def reset(self, dones: Tensor | None = None, hidden_state=None) -> None:
        if self._actor is not None:
            self._actor._cached_values = None

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: Mapping[str, Tensor]) -> None:
        pass
