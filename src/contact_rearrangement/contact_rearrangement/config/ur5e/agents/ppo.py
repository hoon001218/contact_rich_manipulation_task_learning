"""RSL-RL PPO extensions required by the shared research policy.

The native runner owns collection, rollout storage, GAE, logging, and saving.
This subclass changes timeout values, shared-model construction, and the size
of encoder forward/backward batches; it does not implement another runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .model import build_policy_masks


class ContactRearrangementPPO(PPO):
    """Native PPO with final-state bootstrapping and shared-encoder microbatches."""

    def __init__(
        self, actor, critic, storage, *, max_microbatch_size: int | None = 32,
        checkpoint_metadata: dict | None = None, **kwargs,
    ):
        if max_microbatch_size is not None and (
            not isinstance(max_microbatch_size, int) or isinstance(max_microbatch_size, bool)
            or max_microbatch_size <= 0
        ):
            raise ValueError("max_microbatch_size must be a positive integer or None.")
        # These extensions have not been defined for the shared research model.
        # Fail explicitly rather than silently changing their training semantics.
        for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if kwargs.get(name) is not None:
                raise ValueError(f"{name} is not supported by the contact rearrangement PPO extension.")
        if critic.policy is not actor.policy or list(critic.parameters()):
            raise ValueError("The critic must reference the actor-owned shared policy without duplicate parameters.")
        super().__init__(actor, critic, storage, **kwargs)
        self.max_microbatch_size = max_microbatch_size
        self.checkpoint_metadata = deepcopy(checkpoint_metadata)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        """Use native model resolution/storage with one shared actor/critic core."""
        algorithm_cfg = deepcopy(cfg["algorithm"])
        algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
        actor_cfg, critic_cfg = deepcopy(cfg["actor"]), deepcopy(cfg["critic"])
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        critic_class = resolve_callable(critic_cfg.pop("class_name"))
        if algorithm_cfg.pop("share_cnn_encoders", False):
            raise ValueError("share_cnn_encoders is not applicable to the shared scene/history encoder.")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic = critic_class(
            obs, cfg["obs_groups"], "critic", 1, shared_policy=actor.policy, actor=actor, **critic_cfg,
        ).to(device)
        print(f"Actor Model: {actor}")
        print(f"Critic Model: {critic}")
        sample_count = cfg["num_steps_per_env"] * env.num_envs
        if sample_count < 2 or sample_count < algorithm_cfg.get("num_mini_batches", 4):
            raise ValueError("The rollout must contain at least two samples and one sample per PPO minibatch.")
        if sample_count % algorithm_cfg.get("num_mini_batches", 4):
            raise ValueError("Rollout sample count must be divisible by num_mini_batches for native storage.")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        calibration = getattr(env.cfg, "force_torque", None)
        if calibration is None:
            raise ValueError("The environment must specify its F/T calibration for checkpoint compatibility.")
        calibration = asdict(calibration) if is_dataclass(calibration) else deepcopy(calibration)
        metadata = {"model_cfg": asdict(actor.policy.cfg), "force_torque": calibration}
        return algorithm_class(
            actor, critic, storage, device=device, checkpoint_metadata=metadata,
            multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg,
        )

    def process_env_step(self, obs: TensorDict, rewards, dones, extras):
        """Bootstrap time limits from the preserved final observation before reset."""
        native_extras = dict(extras)
        time_outs = native_extras.pop("time_outs", None)
        if time_outs is not None:
            time_outs = time_outs.to(device=self.device, dtype=torch.bool)
            terminated = extras.get("terminated")
            if terminated is not None:
                time_outs = time_outs & ~terminated.to(device=self.device, dtype=torch.bool)
            if time_outs.any():
                if "terminal_observation" not in extras:
                    raise ValueError("Timeout bootstrapping requires the pre-reset terminal_observation.")
                final = extras["terminal_observation"]
                if isinstance(final, TensorDict):
                    final = final[time_outs].to(self.device)
                else:
                    final = TensorDict(
                        {"policy": final.to(self.device)[time_outs]}, batch_size=[int(time_outs.sum())],
                    )
                with torch.no_grad():
                    final_values = self.critic(final).squeeze(-1)
                rewards = rewards.clone()
                rewards[time_outs] += self.gamma * final_values
        # Native storage and reset processing remain unchanged. Removing the
        # timeout key prevents native PPO from additionally bootstrapping V(s_t).
        super().process_env_step(obs, rewards, dones, native_extras)

    def _adapt_learning_rate(self, batch, chunk_size):
        """Compute native Normal KL over a complete minibatch without large graphs."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.no_grad():
            kl_sum = torch.zeros((), device=self.device)
            total = batch.observations.batch_size[0]
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                inputs = self.actor.get_policy_observation(batch.observations[start:stop])
                means, _ = self.actor.policy(inputs, masks=build_policy_masks(inputs, self.actor.policy.cfg))
                std = self.actor.policy.log_std.clamp(-5.0, 2.0).exp().expand_as(means)
                old_params = tuple(parameter[start:stop] for parameter in batch.old_distribution_params)
                kl_sum += self.actor.get_kl_divergence(old_params, (means, std)).sum()
            kl_mean = float(kl_sum / total)
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self):
        """Accumulate native PPO losses, then clip/step once per original minibatch.

        Splitting changes memory use only. Unequal final chunks contribute their
        sample share, and one entropy-noise draw is shared across the chunks.
        The native storage controls minibatch order and epoch count.
        """
        totals = dict(value=0.0, surrogate=0.0, entropy=0.0, grad_norm=0.0,
                      approximate_kl=0.0, clip_fraction=0.0)
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            count = batch.observations.batch_size[0]
            chunk_size = self.max_microbatch_size or count
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self._adapt_learning_rate(batch, chunk_size)
            entropy_noise = torch.randn_like(batch.actions)
            self.optimizer.zero_grad(set_to_none=True)
            metrics = {name: torch.zeros((), device=self.device) for name in totals if name != "grad_norm"}
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                inputs = self.actor.get_policy_observation(batch.observations[start:stop])
                log_prob, entropy, values = self.actor.policy.evaluate_actions(
                    inputs, batch.actions[start:stop], masks=build_policy_masks(inputs, self.actor.policy.cfg),
                    entropy_noise=entropy_noise[start:stop],
                )
                old_log_prob = batch.old_actions_log_prob[start:stop].squeeze(-1)
                advantage = batch.advantages[start:stop].squeeze(-1)
                ratio = (log_prob - old_log_prob).exp()
                surrogate = -advantage * ratio
                surrogate_clipped = -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
                old_values = batch.values[start:stop].squeeze(-1)
                returns = batch.returns[start:stop].squeeze(-1)
                value_loss = (values - returns).square()
                if self.use_clipped_value_loss:
                    clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
                    value_loss = torch.maximum(value_loss, (clipped - returns).square())
                value_loss = value_loss.mean()
                entropy_mean = entropy.mean()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("PPO loss is not finite.")
                share = (stop - start) / count
                (loss * share).backward()
                log_ratio = log_prob - old_log_prob
                chunk_metrics = dict(
                    value=value_loss, surrogate=surrogate_loss, entropy=entropy_mean,
                    approximate_kl=(ratio - 1.0 - log_ratio).mean(),
                    clip_fraction=((ratio - 1.0).abs() > self.clip_param).float().mean(),
                )
                for name, value in chunk_metrics.items():
                    metrics[name] += value.detach() * share
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.policy.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("PPO gradients are not finite.")
            self.optimizer.step()
            metrics["grad_norm"] = grad_norm
            for name, value in metrics.items():
                totals[name] += float(value)
            updates += 1
        self.storage.clear()
        return {name: value / max(updates, 1) for name, value in totals.items()}

    def save(self):
        """Add observation and physical sensor semantics to native checkpoint fields."""
        saved = super().save()
        saved["research_metadata"] = deepcopy(self.checkpoint_metadata)
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        """Reject incompatible observation/F/T profiles before changing any weights."""
        metadata = loaded_dict.get("research_metadata")
        if metadata is None or metadata != self.checkpoint_metadata:
            raise ValueError(
                "Checkpoint observation layout/model configuration or F/T sensor profile is missing or differs "
                "from this environment. Legacy custom-PPO checkpoints require a new native RSL-RL run."
            )
        return super().load(loaded_dict, load_cfg, strict)
