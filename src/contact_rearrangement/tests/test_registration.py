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
        self.assertEqual(result["kwargs"]["rsl_rl_cfg_entry_point"],
                         "contact_rearrangement.config.ur5e.agents.rsl_rl_ppo_cfg:ContactRearrangementPPORunnerCfg")
        self.assertTrue(result["checker_disabled"])
        self.assertEqual(result["history_length"], 2)
        self.assertEqual(result["simulator_modules"], [])

    def test_research_model_and_mdp_remain_usable_without_launching_simulator(self):
        result = self.fresh_process("""
            import json
            import sys
            import torch
            from contact_rearrangement import mdp
            from contact_rearrangement.config.ur5e.agents.model import ContactRearrangementPolicy, build_policy_masks
            from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg

            cfg = mdp.MdpCfg()
            model = ContactRearrangementPolicy(ModelCfg())
            actor_input = mdp.pack_policy_observation(
                torch.zeros((2, cfg.max_objects, 10)),
                torch.zeros((2, cfg.max_objects), dtype=torch.bool),
                torch.zeros((2, cfg.max_objects, 2)),
                torch.zeros((2, cfg.history_length, 72)),
                torch.zeros((2, cfg.history_length), dtype=torch.bool),
            )
            with torch.no_grad():
                masks = build_policy_masks(actor_input, model.cfg)
                action, log_probability, value = model.act(actor_input, deterministic=True, masks=masks)
            print(json.dumps(dict(observation_dim=model.cfg.observation_dim,
                                  observation_layout=model.cfg.observation_layout,
                                  action_shape=list(action.shape),
                                  finite=bool(torch.isfinite(action).all()
                                              and torch.isfinite(log_probability).all()
                                              and torch.isfinite(value).all()),
                                  simulator_modules=[name for name in sys.modules
                                      if name.split(".")[0] in {"isaaclab", "isaacsim", "omni", "pxr"}])))
        """)
        self.assertEqual(result["observation_dim"], 544)
        self.assertEqual(result["observation_layout"], "fixed_roles_v1")
        self.assertEqual(result["action_shape"], [2, 18])
        self.assertTrue(result["finite"])
        self.assertEqual(result["simulator_modules"], [])


if __name__ == "__main__":
    unittest.main()
