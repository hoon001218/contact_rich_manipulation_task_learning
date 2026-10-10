"""PPO for the shared graph/history model, independent of simulator APIs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model import ContactRearrangementPolicy
from .ppo_cfg import ModelCfg, PPOCfg




def generalized_advantage_estimate(
    rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor,
    gamma: float = 0.99, gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """GAE with final-observation bootstrapping and episode-boundary isolation.

    ``next_values[t]`` must evaluate the true next state before auto-reset.  A
    physical terminal state has zero bootstrap; a time limit bootstraps that
    final state but never carries the following episode's advantages backward.
    """
    if rewards.ndim < 1 or any(x.shape != rewards.shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE tensors must have identical [time, ...environment] shapes.")
    if rewards.shape[0] == 0:
        raise ValueError("GAE requires at least one transition.")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE gamma and lambda must be in [0, 1].")
    terminated, truncated = terminated.bool(), truncated.bool()
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_values[step] * ~terminated[step] - values[step]
        carry = delta + gamma * gae_lambda * ~(terminated[step] | truncated[step]) * carry
        advantages[step] = carry
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, model: ContactRearrangementPolicy, cfg: PPOCfg | None = None):
        self.model = model
        self.cfg = cfg or PPOCfg()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.update_count = 0

    def update(
        self, observations: Tensor, actions: Tensor, old_log_probs: Tensor, rewards: Tensor,
        old_values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor,
    ) -> dict[str, float]:
        if rewards.ndim != 2:
            raise ValueError("PPO rewards must have shape [rollout time, environment].")
        batch_shape = rewards.shape
        if observations.shape != (*batch_shape, self.model.cfg.observation_dim):
            raise ValueError("PPO observations do not match the rollout or model dimensions.")
        if actions.shape != (*batch_shape, self.model.cfg.action_dim):
            raise ValueError("PPO actions do not match the rollout or model dimensions.")
        if any(x.shape != batch_shape for x in (old_log_probs, old_values, next_values, terminated, truncated)):
            raise ValueError("PPO scalar rollout fields must have shape [time, environment].")
        # Rollouts may have been captured under inference_mode.  Copies made
        # outside it can safely be saved by autograd during the model forward.
        with torch.inference_mode(False):
            observations, actions, old_log_probs, rewards, old_values, next_values, terminated, truncated = (
                x.detach().clone() for x in (
                    observations, actions, old_log_probs, rewards, old_values, next_values, terminated, truncated,
                )
            )
        for name, data in (("actions", actions), ("old_log_probs", old_log_probs), ("rewards", rewards),
                           ("old_values", old_values), ("next_values", next_values)):
            if not torch.isfinite(data).all():
                raise ValueError(f"PPO {name} must be finite.")
        if (actions.abs() > 1.0).any():
            raise ValueError("PPO actions must be in [-1, 1].")
        with torch.no_grad():
            advantages, returns = generalized_advantage_estimate(
                rewards, old_values, next_values, terminated, truncated, self.cfg.gamma, self.cfg.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        total = rewards.numel()
        if total == 0:
            raise ValueError("PPO cannot update an empty rollout.")
        observations, actions = observations.reshape(total, -1), actions.reshape(total, -1)
        old_log_probs, old_values, advantages, returns = (
            x.reshape(total) for x in (old_log_probs, old_values, advantages, returns)
        )
        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approximate_kl=0.0,
                      clip_fraction=0.0, grad_norm=0.0)
        updates, stopped = 0, False
        self.model.train()
        for _ in range(self.cfg.epochs):
            order = torch.randperm(total, device=observations.device)
            for indices in order.tensor_split(min(total, self.cfg.minibatches)):
                log_probs, entropy, values = self.model.evaluate_actions(observations[indices], actions[indices])
                log_ratio = log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[indices], ratio.clamp(1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * advantages[indices],
                ).mean()
                clipped_values = old_values[indices] + (values - old_values[indices]).clamp(
                    -self.cfg.value_clip, self.cfg.value_clip,
                )
                value_loss = 0.5 * torch.maximum(
                    (values - returns[indices]).square(), (clipped_values - returns[indices]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                approximate_kl = (ratio - 1.0 - log_ratio).mean()
                if self.cfg.target_kl is not None and approximate_kl > self.cfg.target_kl:
                    stopped = True
                    break
                loss = policy_loss + self.cfg.value_coefficient * value_loss - self.cfg.entropy_coefficient * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is not finite.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("PPO gradients are not finite.")
                self.optimizer.step()
                metrics = dict(
                    policy_loss=policy_loss, value_loss=value_loss, entropy=entropy_mean,
                    approximate_kl=approximate_kl.clamp_min(0.0),
                    clip_fraction=((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean(), grad_norm=grad_norm,
                )
                for key, value in metrics.items():
                    totals[key] += float(value.detach())
                updates += 1
            if stopped:
                break
        self.update_count += 1
        totals = {key: value / max(updates, 1) for key, value in totals.items()}
        totals.update(minibatch_updates=float(updates), update=float(self.update_count),
                      learning_rate=float(self.optimizer.param_groups[0]["lr"]), kl_early_stop=float(stopped))
        return totals

    def save(self, path: str | Path, **extra: Any) -> None:
        """Save configuration, learned weights, optimizer, and caller metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_cfg=asdict(self.model.cfg), ppo_cfg=asdict(self.cfg),
                        model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        update_count=self.update_count, extra=extra), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "PPOTrainer":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = ContactRearrangementPolicy(ModelCfg(**checkpoint["model_cfg"])).to(device)
        trainer = cls(model, PPOCfg(**checkpoint["ppo_cfg"]))
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.update_count = checkpoint["update_count"]
        return trainer
