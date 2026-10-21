"""Initial-scene graph and finite sensor-history actor.

This module deliberately has no Isaac Sim imports. The observation contains
initial camera geometry, with target and blocker in reserved slots 0 and 1,
and the current/previous robot and sensor samples. Callers derive masks from
that observation before passing observation and masks separately to the model.
Masks control padding, sensor filtering, and role selection; they are not learned features.
The independent native MLP critic is configured in the runner settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from rsl_rl.modules import GaussianDistribution
import triton
import triton.language as tl


@triton.jit
def _edge_mean_forward(receiver, sender, edges, degree, output,
                       M: tl.constexpr, D: tl.constexpr, K: tl.constexpr, C: tl.constexpr):
    node = tl.program_id(0)
    batch, i = node // M, node % M
    j, d = tl.arange(0, K), tl.arange(0, C)
    r = tl.load(receiver + node * D + d, d < D, other=0)
    s = tl.load(sender + (batch * M + j[:, None]) * D + d[None, :],
                (j[:, None] < M) & (d[None, :] < D), other=0)
    valid = tl.load(edges + node * M + j, j < M, other=0)
    count = tl.maximum(tl.load(degree + node), 1)
    value = tl.sum(tl.where(valid[:, None], tl.maximum(r[None, :] + s, 0.), 0.), axis=0) / count
    tl.store(output + node * D + d, value, d < D)


@triton.jit
def _edge_mean_backward(receiver, sender, edges, degree, gradient, grad_r, grad_s,
                        M: tl.constexpr, D: tl.constexpr, K: tl.constexpr, C: tl.constexpr):
    node = tl.program_id(0)
    batch, i = node // M, node % M
    j, d = tl.arange(0, K), tl.arange(0, C)
    mask = (j[:, None] < M) & (d[None, :] < D)
    address = (batch * M + j[:, None]) * D + d[None, :]
    r = tl.load(receiver + node * D + d, d < D, other=0)
    s = tl.load(sender + address, mask, other=0)
    valid = tl.load(edges + node * M + j, j < M, other=0)
    own_g = tl.load(gradient + node * D + d, d < D, other=0)
    own_count = tl.maximum(tl.load(degree + node), 1)
    dr = tl.sum(((r[None, :] + s > 0.) & valid[:, None]).to(tl.float32), axis=0) * own_g / own_count
    r = tl.load(receiver + address, mask, other=0)
    s = tl.load(sender + node * D + d, d < D, other=0)
    valid = tl.load(edges + (batch * M + j) * M + i, j < M, other=0)
    g = tl.load(gradient + address, mask, other=0)
    count = tl.maximum(tl.load(degree + batch * M + j, j < M, other=1), 1)
    ds = tl.sum(tl.where((r + s[None, :] > 0.) & valid[:, None], g / count[:, None], 0.), axis=0)
    tl.store(grad_r + node * D + d, dr, d < D)
    tl.store(grad_s + node * D + d, ds, d < D)


class _EdgeMean(torch.autograd.Function):
    """Fuse edge ReLU/masking/reduction; recompute signs in backward, never save pairs."""

    @staticmethod
    def forward(ctx, receiver, sender, edges, degree):
        receiver, sender, edges, degree = (t.contiguous() for t in (receiver, sender, edges, degree))
        b, m, d = receiver.shape
        output = torch.empty_like(receiver)
        _edge_mean_forward[(b * m,)](receiver, sender, edges, degree, output,
                                     m, d, triton.next_power_of_2(m), triton.next_power_of_2(d))
        ctx.save_for_backward(receiver, sender, edges, degree)
        return output

    @staticmethod
    def backward(ctx, gradient):
        receiver, sender, edges, degree = ctx.saved_tensors
        b, m, d = receiver.shape
        grad_r, grad_s = torch.empty_like(receiver), torch.empty_like(sender)
        _edge_mean_backward[(b * m,)](receiver, sender, edges, degree, gradient.contiguous(), grad_r, grad_s,
                                      m, d, triton.next_power_of_2(m), triton.next_power_of_2(d))
        return grad_r, grad_s, None, None


def _edge_mean(receiver, sender, edges, degree):
    if receiver.is_cuda and receiver.dtype == torch.float32:
        return _EdgeMean.apply(receiver, sender, edges, degree)
    return (F.relu(receiver[:, :, None, :] + sender[:, None, :, :]) * edges[..., None]).sum(-2) / degree.clamp_min(1)

@dataclass
class ModelCfg:
    # Serialized checkpoint contract: target slot 0, blocker slot 1, no role-ID channels.
    observation_layout: str = "fixed_roles_v1"
    activation: str = "relu"  # Checkpoint metadata includes parameter-free activations.
    max_objects: int = 10
    # Newest first: the current sample and the immediately preceding sample.
    history_steps: int = 2
    action_dim: int = 18
    node_features: int = 10
    step_features: int = 72
    node_dim: int = 64
    graph_layers: int = 2
    latent_dim: int = 128
    initial_std: float = 0.3

    def __post_init__(self):
        if self.activation != "relu":
            raise ValueError("This baseline uses relu activation.")
        dimensions = (
            self.max_objects, self.history_steps, self.action_dim, self.node_features,
            self.step_features, self.node_dim, self.graph_layers, self.latent_dim,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive integers.")
        if self.history_steps != 2 or self.action_dim != 18:
            raise ValueError("The MDP contract requires current/previous history (2) and 18 action channels.")
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
        self.node_encoder = nn.Sequential(nn.Linear(cfg.node_features, width), nn.ReLU(), nn.Linear(width, width), nn.LayerNorm(width))
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 6, width), nn.ReLU(), nn.Linear(width, width))
            for _ in range(cfg.graph_layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width), nn.LayerNorm(width))
            for _ in range(cfg.graph_layers)
        ])
        self.role_queries = nn.Linear(2 * width, 2 * width)
        self.keys = nn.Linear(width, width, bias=False)
        self.values = nn.Linear(width, width, bias=False)
        self.width = width
        self.register_buffer("nonself_edges", ~torch.eye(cfg.max_objects, dtype=torch.bool), persistent=False)

    def forward(self, geometry: Tensor, valid: Tensor, roles: Tensor) -> Tensor:
        geometry = torch.where(valid[..., None], geometry, torch.zeros_like(geometry))
        roles = torch.where(valid[..., None], roles, torch.zeros_like(roles))
        hidden = self.node_encoder(geometry) * valid[..., None]
        edges = valid[..., :, None] & valid[..., None, :] & self.nonself_edges
        degree = edges.sum(-1, keepdim=True)
        # The edge MLP is exactly Linear([h_i, h_j, p_j-p_i, size_j-size_i]).
        # Distribute its first affine map before broadcasting: this avoids a
        # [B, M, M, 2D+6] concatenation and repeating node projections M times.
        # Native rollout/PPO split B into bounded chunks outside this encoder.
        spatial = torch.cat((geometry[..., :3], geometry[..., 7:10]), dim=-1)
        for message, update in zip(self.messages, self.updates):
            weight, bias = message[0].weight, message[0].bias
            relation = F.linear(spatial, weight[:, 2 * self.width:])
            receivers = F.linear(hidden, weight[:, :self.width], bias) - relation
            senders = F.linear(hidden, weight[:, self.width:2 * self.width]) + relation
            aggregated = _edge_mean(receivers, senders, edges, degree)
            # Linear(mean(messages)) == mean(Linear(messages)): apply the
            # second affine map once per node, instead of once per edge.
            # Empty neighborhoods must not acquire the affine bias.
            aggregated = message[2](aggregated) * (degree > 0)
            hidden = (hidden + update(torch.cat((hidden, aggregated), dim=-1))) * valid[..., None]
        target = _masked_mean(hidden, valid & roles[..., 0])
        blocker = _masked_mean(hidden, valid & roles[..., 1])
        queries = self.role_queries(torch.cat((target, blocker), dim=-1)).reshape(*target.shape[:-1], 2, self.width)
        scores = torch.matmul(queries, self.keys(hidden).transpose(-2, -1)) / math.sqrt(self.width)
        attention = _masked_softmax(scores, valid[..., None, :].expand_as(scores))
        contextual = torch.matmul(attention, self.values(hidden)).flatten(-2)
        return torch.cat((target, blocker, contextual, _masked_mean(hidden, valid)), dim=-1)


class ContactManipulationPolicy(nn.Module):
    """Gaussian 18D actor with the scene encoder and final MLP."""

    def __init__(self, cfg: ModelCfg | None = None):
        super().__init__()
        self.cfg = cfg or ModelCfg()
        self.scene_encoder = SceneGraphEncoder(self.cfg)
        # Final feed-forward MLP after concat; Sequential only orders its layers.
        self.fusion = nn.Sequential(
            nn.Linear(5 * self.cfg.node_dim + self.cfg.history_steps * self.cfg.step_features, 2 * self.cfg.latent_dim), nn.ReLU(),
            nn.Linear(2 * self.cfg.latent_dim, self.cfg.latent_dim), nn.ReLU(),
        )
        self.actor = nn.Linear(self.cfg.latent_dim, self.cfg.action_dim)
        self.distribution = GaussianDistribution(
            self.cfg.action_dim, init_std=self.cfg.initial_std, std_type="log",
        )
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
        # No learned history encoder: preserve current/previous order and all
        # 144 physical features for the final MLP. Masks only remove invalid data.
        tactile_valid = masks.tactile_valid.reshape(-1, self.cfg.history_steps, 17)
        history = torch.cat((history[..., :37],
                             torch.where(tactile_valid, history[..., 37:54], 0),
                             history[..., 54:]), dim=-1)
        history = torch.where(masks.history_valid.reshape(-1, self.cfg.history_steps, 1), history, 0)
        return self.fusion(torch.cat((scene, history.flatten(1)), dim=-1)).reshape(*leading, self.cfg.latent_dim)

    def forward(self, observation: Tensor, *, masks: PolicyMasks) -> Tensor:
        return self.actor(self.encode(observation, masks=masks))

    def act(self, observation: Tensor, deterministic: bool = False, *, masks: PolicyMasks) -> tuple[Tensor, Tensor]:
        mean = self.forward(observation, masks=masks)
        self.distribution.update(mean)
        actions = self.distribution.deterministic_output(mean) if deterministic else self.distribution.sample()
        return actions, self.distribution.log_prob(actions)

    def evaluate_actions(
        self, observation: Tensor, actions: Tensor, *, masks: PolicyMasks,
    ) -> tuple[Tensor, Tensor]:
        mean = self.forward(observation, masks=masks)
        if actions.shape != mean.shape:
            raise ValueError(f"Actions must have shape {tuple(mean.shape)}, got {tuple(actions.shape)}.")
        self.distribution.update(mean)
        return self.distribution.log_prob(actions), self.distribution.entropy

