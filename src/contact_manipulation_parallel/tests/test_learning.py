"""CPU checks for the production native MLP configuration, without Isaac Sim."""

import ast
from copy import deepcopy
import io
from pathlib import Path
import unittest

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from tensordict import TensorDict


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_SUFFIX = Path("config/ur5e/agents/rsl_rl_ppo_cfg.py")
CONFIG = PROJECT / "contact_manipulation_parallel" / CONFIG_SUFFIX


def declarations(path):
    """Read config expressions without importing its Isaac Lab dependencies."""
    tree = ast.parse(path.read_text(), filename=str(path))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ContactManipulationPPORunnerCfg")
    return {node.targets[0].id: node.value for node in runner.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}


def model_kwargs(node):
    """Decode public model/distribution config calls into native constructor kwargs."""
    if not isinstance(node, ast.Call):
        return ast.literal_eval(node)
    name = ast.unparse(node.func)
    defaults = {
        "RslRlMLPModelCfg": {"class_name": "MLPModel"},
        "RslRlMLPModelCfg.GaussianDistributionCfg": {"class_name": "GaussianDistribution"},
    }
    if name not in defaults or node.args or any(item.arg is None for item in node.keywords):
        raise ValueError(f"Unexpected model configuration: {name}")
    return defaults[name] | {item.arg: model_kwargs(item.value) for item in node.keywords}


class NativeMlpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = declarations(CONFIG)
        cls.groups = ast.literal_eval(cls.cfg["obs_groups"])

    def setUp(self):
        torch.manual_seed(81)
        self.obs = TensorDict({"policy": torch.randn(4, 244), "critic": torch.randn(4, 244)}, batch_size=[4])
        self.actor, self.critic = self.models()

    def models(self):
        models = []
        for role, output_dim in (("actor", 18), ("critic", 1)):
            kwargs = deepcopy(model_kwargs(self.cfg[role]))
            self.assertEqual(kwargs.pop("class_name"), "MLPModel")
            models.append(MLPModel(self.obs, self.groups, role, output_dim, **kwargs))
        return models

    def test_native_forward_and_separate_observation_groups(self):
        self.assertIs(type(self.actor), MLPModel)
        self.assertIs(type(self.critic), MLPModel)
        self.assertEqual(self.actor(self.obs, stochastic_output=True).shape, (4, 18))
        self.assertEqual(self.critic(self.obs).shape, (4, 1))
        for model in (self.actor, self.critic):
            self.assertEqual(model.mlp[0].in_features, 244)
            self.assertFalse(hasattr(model, "encoder"))
            self.assertTrue(torch.isfinite(model(self.obs)).all())
        changed = self.obs.clone()
        changed["critic"] = changed["critic"] + 3
        torch.testing.assert_close(self.actor(changed), self.actor(self.obs))
        self.assertFalse(torch.allclose(self.critic(changed), self.critic(self.obs)))

    def test_gaussian_scalar_std_and_deterministic_critic(self):
        self.assertIsInstance(self.actor.distribution, GaussianDistribution)
        self.assertEqual(self.actor.distribution.std_type, "scalar")
        sampled = self.actor(self.obs, stochastic_output=True)
        torch.testing.assert_close(self.actor.output_std, torch.ones(4, 18))
        self.assertTrue(torch.isfinite(self.actor.get_output_log_prob(sampled)).all())
        self.assertIsNone(self.critic.distribution)
        torch.testing.assert_close(self.critic(self.obs), self.critic(self.obs, stochastic_output=True))

    def test_critic_update_cannot_change_actor_parameters(self):
        self.assertTrue({id(p) for p in self.actor.parameters()}.isdisjoint(id(p) for p in self.critic.parameters()))
        actor_before = deepcopy(self.actor.state_dict())
        critic_before = self.critic(self.obs).detach().clone()
        optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        (self.critic(self.obs) - 2).square().mean().backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in self.actor.parameters()))
        for name, value in self.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)
        self.assertFalse(torch.allclose(self.critic(self.obs), critic_before))

    def test_native_checkpoint_weights_roundtrip(self):
        buffer = io.BytesIO()
        torch.save({"actor_state_dict": self.actor.state_dict(), "critic_state_dict": self.critic.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
        actor, critic = self.models()
        for role, restored, original in (("actor", actor, self.actor), ("critic", critic, self.critic)):
            self.assertNotIn("_extra_state", checkpoint[f"{role}_state_dict"])
            restored.load_state_dict(checkpoint[f"{role}_state_dict"], strict=True)
            torch.testing.assert_close(restored(self.obs), original(self.obs), rtol=0, atol=0)

    def test_native_script_loads(self):
        for script in ("train.py", "play.py"):
            tree = ast.parse((PROJECT / "scripts" / script).read_text())
            self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("agents.ppo")
                                 for n in ast.walk(tree)))
            loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "runner" and n.func.attr == "load"]
            self.assertEqual(len(loads), 1)
            if script == "play.py":
                load_cfg = ast.literal_eval(next(kw.value for kw in loads[0].keywords if kw.arg == "load_cfg"))
                self.assertTrue(load_cfg["actor"] and load_cfg["critic"])
                self.assertFalse(load_cfg["optimizer"])

    def test_ppo_settings_match_original_project(self):
        original = PROJECT.parent / "contact_manipulation" / "contact_manipulation" / CONFIG_SUFFIX
        if not original.is_file():
            self.skipTest("Original project is absent in this standalone copy.")
        baseline = declarations(original)
        for name in ("algorithm", "num_steps_per_env", "max_iterations", "save_interval", "obs_groups", "clip_actions"):
            self.assertEqual(ast.dump(self.cfg[name]), ast.dump(baseline[name]), name)


if __name__ == "__main__":
    unittest.main()
