"""Resolve registered task and runner configuration without launching Isaac Sim."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def fresh_process(self, script):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_package_registers_task_and_keeps_cpu_mdp_import_free_of_simulator(self):
        result = self.fresh_process("""
            import json
            import sys
            import gymnasium as gym
            import contact_rearrangement
            from contact_rearrangement import mdp

            spec = gym.spec("Isaac-Contact-Rearrangement-UR5e-v0")
            simulator_modules = [name for name in sys.modules
                                 if name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"}]
            print(json.dumps(dict(entry_point=spec.entry_point, kwargs=spec.kwargs,
                                  checker_disabled=spec.disable_env_checker,
                                  history_length=mdp.MdpCfg().history_length,
                                  simulator_modules=simulator_modules)))
        """)
        self.assertEqual(result["entry_point"], "contact_rearrangement.env:ContactRearrangementRLEnv")
        self.assertEqual(result["kwargs"]["env_cfg_entry_point"],
                         "contact_rearrangement.config.ur5e.osc_env_cfg:ContactRearrangementRLEnvCfg")
        self.assertEqual(result["kwargs"]["ppo_cfg_entry_point"],
                         "contact_rearrangement.config.ur5e.agents.ppo_cfg:ContactRearrangementPPORunnerCfg")
        self.assertTrue(result["checker_disabled"])
        self.assertEqual(result["history_length"], 2)
        self.assertEqual(result["simulator_modules"], [])

    def test_registered_runner_resolves_to_model_and_optimizer_with_matching_mdp_interface(self):
        result = self.fresh_process("""
            import importlib
            import json
            import sys
            import gymnasium as gym
            import torch
            import contact_rearrangement
            from contact_rearrangement import mdp
            from contact_rearrangement.config.ur5e.agents.model import ContactRearrangementPolicy
            from contact_rearrangement.config.ur5e.agents.ppo import PPOTrainer

            entry = gym.spec("Isaac-Contact-Rearrangement-UR5e-v0").kwargs["ppo_cfg_entry_point"]
            module_name, class_name = entry.split(":")
            runner_class = getattr(importlib.import_module(module_name), class_name)
            first, second = runner_class(), runner_class()
            first.policy.max_objects = 7
            first.algorithm.rollout_steps = 3
            model = ContactRearrangementPolicy(second.policy)
            trainer = PPOTrainer(model, second.algorithm)
            cfg = mdp.MdpCfg()
            actor_input = mdp.pack_policy_observation(
                torch.zeros((2, cfg.max_objects, 10)),
                torch.zeros((2, cfg.max_objects), dtype=torch.bool),
                torch.zeros((2, cfg.max_objects, 2)),
                torch.zeros((2, cfg.history_length, 89)),
                torch.zeros((2, cfg.history_length), dtype=torch.bool),
            )
            with torch.no_grad():
                action, log_probability, value = model.act(actor_input, deterministic=True)
            simulator_modules = [name for name in sys.modules
                                 if name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"}]
            print(json.dumps(dict(observation_dim=model.cfg.observation_dim,
                                  actor_input_shape=list(actor_input.shape),
                                  action_shape=list(action.shape),
                                  finite=bool(torch.isfinite(action).all()
                                              and torch.isfinite(log_probability).all()
                                              and torch.isfinite(value).all()),
                                  max_objects=second.policy.max_objects,
                                  history_steps=second.policy.history_steps,
                                  rollout_steps=trainer.cfg.rollout_steps,
                                  gamma=trainer.cfg.gamma,
                                  optimizer_groups=len(trainer.optimizer.param_groups),
                                  simulator_modules=simulator_modules)))
        """)
        self.assertEqual(result["observation_dim"], 700)
        self.assertEqual(result["actor_input_shape"], [2, 700])
        self.assertEqual(result["action_shape"], [2, 18])
        self.assertTrue(result["finite"])
        self.assertEqual(result["max_objects"], 40)
        self.assertEqual(result["history_steps"], 2)
        self.assertEqual(result["rollout_steps"], 32)
        self.assertEqual(result["gamma"], .99)
        self.assertEqual(result["optimizer_groups"], 1)
        self.assertEqual(result["simulator_modules"], [])


if __name__ == "__main__":
    unittest.main()
