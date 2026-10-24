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
from .model import ACTION_CONTRACT, CRITIC_OBSERVATION_CONTRACT, ModelCfg, NativeContactActor


_MODEL_SETTINGS = ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
_GRAPH_CRITIC = "contact_manipulation.config.ur5e.agents.model:NativeContactCritic"


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


def _critic_contract(checkpoint):
    """Older independent MLP critics have no extra state; graph critics do."""
    state = checkpoint.get("critic_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint critic weights are missing.")
    contract = state.get("_extra_state")
    if contract is None:
        return None
    if (not isinstance(contract, Mapping)
            or contract.get("architecture") not in ("native_graph_critic_v1", "native_graph_critic_v2")
            or contract.get("action_contract") != ACTION_CONTRACT):
        raise ValueError("Checkpoint critic architecture or action contract differs from this task.")
    if (contract["architecture"] == "native_graph_critic_v2"
            and contract.get("observation_contract") != CRITIC_OBSERVATION_CONTRACT):
        raise ValueError("Checkpoint critic tracked-object observation contract differs from this task.")
    return deepcopy(dict(contract))


def _critic_settings(checkpoint):
    contract = _critic_contract(checkpoint)
    if contract is not None:
        settings = contract.get("critic_model")
        if not isinstance(settings, Mapping) or not isinstance(contract.get("model_cfg"), Mapping):
            raise ValueError("Checkpoint graph critic configuration is missing.")
        settings = dict(settings)
        settings.update(class_name=_GRAPH_CRITIC, model_cfg=deepcopy(contract["model_cfg"]))
        settings["observation_contract"] = (
            CRITIC_OBSERVATION_CONTRACT if contract["architecture"] == "native_graph_critic_v2"
            else "initial_vision_v1")
    else:
        metadata = checkpoint.get("research_metadata")
        settings = metadata.get("critic_model") if isinstance(metadata, Mapping) else None
        if settings is None:
            # Native Actor v3 predates graph critics and used this fixed native MLP.
            settings = dict(class_name="MLPModel", hidden_dims=[256, 128, 64], activation="elu",
                            obs_normalization=False, distribution_cfg=None)
        if not isinstance(settings, Mapping) or settings.get("class_name") != "MLPModel":
            raise ValueError("Legacy playback requires an independent native MLP critic.")
        settings = dict(settings)
    for name in _MODEL_SETTINGS:
        if name not in settings:
            raise ValueError(f"Checkpoint critic setting is missing: {name}.")
    if settings["distribution_cfg"] is not None:
        raise ValueError("The value critic must not contain an action distribution.")
    return deepcopy(settings)


def _check_critic_before_load(checkpoint, critic, *, inference):
    """Reject incompatible critics before native PPO loads any actor weights."""
    saved_contract = _critic_contract(checkpoint)
    expected = critic.state_dict()
    if saved_contract != expected.get("_extra_state"):
        suffix = ("Prepare playback with prepare_inference_checkpoint first." if inference
                  else "The critic architecture changed; start a new training run.")
        raise ValueError(f"Checkpoint critic model configuration differs from this critic. {suffix}")
    saved = checkpoint["critic_state_dict"]
    if (saved.keys() != expected.keys()
            or any(not isinstance(saved[name], torch.Tensor) or saved[name].shape != value.shape
                   for name, value in expected.items() if isinstance(value, torch.Tensor))):
        raise ValueError("Checkpoint critic weight layout differs from this critic; start a new training run.")


def prepare_inference_checkpoint(agent_cfg, path):
    """Restore checkpoint model settings before constructing the playback runner.

    PPO hyperparameters stay at the current baseline: only the actor and critic
    architectures are restored. In particular, a log-Gaussian checkpoint
    must never be interpreted as a scalar-Gaussian policy.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = _actor_contract(checkpoint)
    agent_cfg.actor.model_cfg = deepcopy(contract["model_cfg"])
    for name in _MODEL_SETTINGS:
        if name not in contract["actor_model"]:
            raise ValueError(f"Checkpoint actor setting is missing: {name}.")
        setattr(agent_cfg.actor, name, deepcopy(contract["actor_model"][name]))
    critic = _critic_settings(checkpoint)
    if critic["class_name"] == "MLPModel":
        # Configclass serialization uses instance fields. Remove graph-specific
        # kwargs rather than passing them into the old native MLP constructor.
        for name in tuple(vars(agent_cfg.critic)):
            if name not in critic:
                delattr(agent_cfg.critic, name)
    for name, value in critic.items():
        setattr(agent_cfg.critic, name, deepcopy(value))
    agent_cfg.obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"] if critic.get("observation_contract") == CRITIC_OBSERVATION_CONTRACT else ["policy"],
    }
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
    _check_critic_before_load(checkpoint, runner.alg.critic, inference=inference)
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
