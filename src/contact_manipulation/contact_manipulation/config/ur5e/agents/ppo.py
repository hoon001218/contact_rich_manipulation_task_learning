"""Native RSL-RL interfaces for a shared GNN/attention policy.

The native runner owns rollout storage, GAE, logging, and checkpoint files.
Only shared network construction, final-state timeout values, and PPO gradient
accumulation differ. Rollout and update batches are bounded independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import Tensor, nn
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from ....mdp.observations import SI_200_8_RANGES, SI_200_8_RESOLUTION
from .model import ContactManipulationPolicy, ModelCfg, build_policy_masks


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
        max_forward_batch_size: int = 512,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stochastic: bool = False,
    ):
        super().__init__()
        self.cfg = model_cfg if isinstance(model_cfg, ModelCfg) else ModelCfg(**(model_cfg or {}))
        if output_dim != self.cfg.action_dim:
            raise ValueError(f"Actor output must contain {self.cfg.action_dim} research actions.")
        if obs_normalization:
            raise ValueError("Research observations retain physical values for observation-derived masks.")
        if distribution_cfg is not None and distribution_cfg != {
            "class_name": "tanh_gaussian", "init_std": self.cfg.initial_std,
        }:
            raise ValueError("The actor requires the model's tanh_gaussian output distribution and initial_std.")
        if not isinstance(max_forward_batch_size, int) or isinstance(max_forward_batch_size, bool) or max_forward_batch_size <= 0:
            raise ValueError("max_forward_batch_size must be a positive integer.")
        self.obs_groups = list(obs_groups[obs_set])
        if self.obs_groups != ["policy"]:
            raise ValueError("The research model requires the single public 'policy' observation group.")
        self.obs_dim = self.cfg.observation_dim
        self.output_dim = output_dim
        self.max_forward_batch_size = max_forward_batch_size
        self.policy = ContactManipulationPolicy(self.cfg)
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

    def _cache_token(self, obs: Mapping[str, Tensor], observation: Tensor) -> tuple:
        try:
            observation_version = observation._version
        except RuntimeError:
            # Native PPO calls actor then critic immediately, without changing
            # the observation. Inference tensors omit a version counter; a
            # one-use identity cache still avoids running the GNN twice.
            observation_version = None
        return id(obs), id(observation), observation_version, tuple(parameter._version for parameter in self.policy.parameters())

    def consume_cached_value(self, obs: Mapping[str, Tensor]) -> Tensor | None:
        """Consume an immediately preceding actor value at most once."""
        cached, self._cached_values = self._cached_values, None
        if cached is None:
            return None
        observation = self.get_policy_observation(obs)
        token = self._cache_token(obs, observation)
        return cached[1] if token == cached[0] else None

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
        self._cached_values = (token, values)
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
            torch.distributions.Normal(*old_params, validate_args=False),
            torch.distributions.Normal(*new_params, validate_args=False),
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
        shared_policy: ContactManipulationPolicy,
        actor: NativeContactActor | None = None,
        model_cfg: dict | ModelCfg | None = None,
        max_forward_batch_size: int = 512,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stochastic: bool = False,
    ):
        super().__init__()
        self.cfg = shared_policy.cfg if model_cfg is None else (
            model_cfg if isinstance(model_cfg, ModelCfg) else ModelCfg(**model_cfg)
        )
        if asdict(self.cfg) != asdict(shared_policy.cfg):
            raise ValueError("Actor and critic must use the same shared research model configuration.")
        if output_dim != 1 or obs_normalization or distribution_cfg is not None or stochastic:
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


class ContactManipulationPPO(PPO):
    """Native PPO with final-state bootstrapping and shared-encoder microbatches."""

    def __init__(
        self, actor, critic, storage, *, max_microbatch_size: int | None = 1024,
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
                raise ValueError(f"{name} is not supported by the contact manipulation PPO extension.")
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
            raise ValueError("share_cnn_encoders is not applicable to the shared scene encoder and final MLP.")
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
        if (algorithm_cfg.get("normalize_advantage_per_mini_batch", False)
                and sample_count // algorithm_cfg.get("num_mini_batches", 4) < 2):
            raise ValueError("Per-minibatch advantage normalization requires at least two samples per minibatch.")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        metadata = {
            "model_cfg": asdict(actor.policy.cfg),
            "force_torque": {"name": "ATI Axia80 SI-200-8", "ranges": SI_200_8_RANGES,
                             "resolution": SI_200_8_RESOLUTION},
        }
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
            if "terminal_observation" not in extras:
                raise ValueError("Timeout bootstrapping requires the pre-reset terminal_observation each step.")
            indices = extras.get("terminal_env_ids")
            if indices is not None and indices.numel() == 0:
                # Native reset already produced these IDs; their shape is
                # known without reading a CUDA tensor scalar.
                super().process_env_step(obs, rewards, dones, native_extras)
                return
            final = extras["terminal_observation"]
            if isinstance(final, TensorDict):
                final = final.to(self.device)
            else:
                final = TensorDict({"policy": final.to(self.device)}, batch_size=obs.batch_size)
            if indices is not None:
                indices = indices.to(self.device)
                final = final[indices]
                time_outs = time_outs[indices]
            # Reuse native reset indices when supplied. Otherwise evaluate the
            # full batch: neither route adds a nonzero()/scalar GPU read.
            with torch.no_grad():
                final_values = self.critic(final).squeeze(-1)
            bootstrap = self.gamma * torch.where(time_outs, final_values, 0.0)
            if indices is None:
                rewards = rewards + bootstrap
            else:
                rewards = rewards.clone().index_add_(0, indices, bootstrap)
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
        names = ("value", "surrogate", "entropy", "grad_norm", "approximate_kl", "clip_fraction")
        totals = torch.zeros(len(names), device=self.device)
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            count = batch.observations.batch_size[0]
            chunk_size = self.max_microbatch_size or count
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self._adapt_learning_rate(batch, chunk_size)
            entropy_noise = torch.randn_like(batch.actions)
            self.optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros_like(totals)
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
                share = (stop - start) / count
                (loss * share).backward()
                log_ratio = log_prob - old_log_prob
                metrics += torch.stack((value_loss, surrogate_loss, entropy_mean,
                                        torch.zeros_like(value_loss), (ratio - 1.0 - log_ratio).mean(),
                                        ((ratio - 1.0).abs() > self.clip_param).float().mean())).detach() * share
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            metrics[3] = grad_norm
            totals += metrics
            updates += 1
        self.storage.clear()
        return dict(zip(names, (totals / max(updates, 1)).tolist()))

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
                "from this model."
            )
        return super().load(loaded_dict, load_cfg, strict)
