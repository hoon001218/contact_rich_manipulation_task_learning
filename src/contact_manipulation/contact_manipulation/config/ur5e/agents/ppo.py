"""Checkpoint compatibility only; training uses the native RSL-RL PPO class.

Older checkpoints keep their original actor/distribution semantics for playback.
Their custom optimizer state is deliberately not resumed under the new baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ....mdp.observations import SI_200_8_RANGES, SI_200_8_RESOLUTION
# Keep old actor import paths usable without retaining a custom PPO algorithm.
from .model import ACTION_CONTRACT, ModelCfg, NativeContactActor


def _actor_contract(checkpoint):
    """Read a native actor contract or translate the two supported legacy layouts."""
    actor = checkpoint.get("actor_state_dict", {})
    contract = actor.get("_extra_state")
    if contract is not None:
        if (not isinstance(contract, Mapping)
                or contract.get("architecture") != "native_graph_actor_v3"
                or contract.get("action_contract") != ACTION_CONTRACT):
            raise ValueError("Checkpoint actor architecture or action contract differs from this task.")
        return deepcopy(dict(contract))
    metadata = checkpoint.get("research_metadata")
    if (not isinstance(metadata, Mapping) or metadata.get("action_contract") != ACTION_CONTRACT):
        raise ValueError("Checkpoint action contract is missing or differs from this task.")
    architecture = metadata.get("actor_critic_architecture")
    if architecture not in ("separate_actor_mlp_critic_v1", "separate_native_graph_actor_mlp_critic_v2"):
        raise ValueError("Unsupported legacy actor/critic architecture.")
    sensor = {"name": "ATI Axia80 SI-200-8", "ranges": SI_200_8_RANGES, "resolution": SI_200_8_RESOLUTION}
    if metadata.get("force_torque") != sensor:
        raise ValueError("Checkpoint F/T sensor profile differs from this task.")
    model = deepcopy(metadata.get("model_cfg", {}))
    initial_std = model.pop("initial_std", 0.3)
    model = asdict(ModelCfg(**model))
    if architecture == "separate_actor_mlp_critic_v1":
        actor_model = {
            "hidden_dims": [2 * model["latent_dim"], model["latent_dim"]],
            "activation": "relu", "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "std_type": "log", "init_std": initial_std},
        }
    else:
        actor_model = deepcopy(metadata.get("actor_model"))
    if (not isinstance(actor_model, Mapping)
            or actor_model.get("distribution_cfg") != metadata.get("output_distribution")):
        raise ValueError("Checkpoint output distribution is missing or inconsistent with its actor.")
    return {"architecture": "native_graph_actor_v3", "action_contract": ACTION_CONTRACT,
            "model_cfg": model, "actor_model": dict(actor_model)}


def prepare_inference_checkpoint(agent_cfg, path):
    """Restore checkpoint model settings before constructing the playback runner.

    PPO hyperparameters stay at the current baseline: only the actor and legacy
    critic architecture are restored. In particular, a log-Gaussian checkpoint
    must never be interpreted as a scalar-Gaussian policy.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = _actor_contract(checkpoint)
    agent_cfg.actor.model_cfg = deepcopy(contract["model_cfg"])
    for name in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
        if name not in contract["actor_model"]:
            raise ValueError(f"Checkpoint actor setting is missing: {name}.")
        setattr(agent_cfg.actor, name, deepcopy(contract["actor_model"][name]))
    critic = checkpoint.get("research_metadata", {}).get("critic_model")
    if critic is not None:
        if not isinstance(critic, Mapping) or critic.get("class_name") != "MLPModel":
            raise ValueError("Legacy playback requires an independent native MLP critic.")
        for name in ("class_name", "hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
            if name not in critic:
                raise ValueError(f"Checkpoint critic setting is missing: {name}.")
            setattr(agent_cfg.critic, name, deepcopy(critic[name]))
    return checkpoint


def load_checkpoint(runner, path, *, inference=False, checkpoint=None):
    """Use native loading; convert legacy weights only for explicit playback."""
    checkpoint = (torch.load(path, map_location="cpu", weights_only=False)
                  if checkpoint is None else checkpoint)
    actor = checkpoint.get("actor_state_dict", {})
    legacy = "_extra_state" not in actor
    if legacy and not inference:
        raise ValueError("Legacy optimizer resume is unsupported by the native PPO baseline; start a new training run.")
    contract = _actor_contract(checkpoint)
    if contract != runner.alg.actor.get_extra_state():
        raise ValueError("Checkpoint observation/model configuration or action distribution differs from this actor.")
    load_cfg = ({"actor": True, "critic": True, "iteration": True, "optimizer": False, "rnd": False}
                if inference else None)
    device = str(runner.device)
    if not legacy:
        return runner.load(str(path), load_cfg=load_cfg, map_location=device)
    converted = dict(checkpoint)
    converted_actor = {}
    prefixes = (("policy.scene_encoder.", "encoder.scene_encoder."),
                ("policy.fusion.0.", "mlp.0."), ("policy.fusion.2.", "mlp.2."),
                ("policy.actor.", "mlp.4."), ("policy.distribution.", "distribution."))
    v1 = checkpoint["research_metadata"]["actor_critic_architecture"] == "separate_actor_mlp_critic_v1"
    for name, value in actor.items():
        mapped = name
        if v1:
            for previous, current in prefixes:
                if name.startswith(previous):
                    mapped = current + name[len(previous):]
                    break
            else:
                raise ValueError(f"Unexpected legacy actor parameter: {name}.")
        converted_actor[mapped] = value
    converted_actor["_extra_state"] = runner.alg.actor.get_extra_state()
    converted["actor_state_dict"] = converted_actor
    with TemporaryDirectory(prefix="contact-policy-") as directory:
        native_path = Path(directory) / "checkpoint.pt"
        torch.save(converted, native_path)
        return runner.load(str(native_path), load_cfg=load_cfg, map_location=device)
