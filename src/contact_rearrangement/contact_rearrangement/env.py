"""RL environment preserving final observations across selective auto-resets."""

from isaaclab.envs import ManagerBasedRLEnv

from . import mdp as task_mdp
from .mdp.actions import forbidden_contacts


class ContactRearrangementRLEnv(ManagerBasedRLEnv):
    """Preserve pre-reset terminal observations for correct PPO timeout values."""

    def __init__(self, cfg, **kwargs):
        if cfg.spawn_rules.max_objects != cfg.mdp.max_objects:
            raise ValueError("Spawn capacity must match the MDP's 40 object slots.")
        self._collecting_step = False
        self._terminal_capture = None
        super().__init__(cfg, **kwargs)

    @property
    def forbidden_contact(self):
        return forbidden_contacts(self, threshold=self.cfg.forbidden_contact_threshold)

    def _reset_idx(self, env_ids):
        if self._collecting_step:
            state = task_mdp.update_mdp(self)
            self._terminal_capture = (
                env_ids.clone(), task_mdp.policy_observation(self)[env_ids].clone(),
                state.components[env_ids].clone(), state.failure_reasons[env_ids].clone(),
                task_mdp.corrected_wrench(self)[env_ids].clone(),
            )
        super()._reset_idx(env_ids)

    def step(self, action):
        self._collecting_step = True
        self._terminal_capture = None
        try:
            observations, rewards, terminated, truncated, extras = super().step(action)
        finally:
            self._collecting_step = False
        terminal_observation = observations["policy"].clone()
        components = self.mdp_state.components.clone()
        reasons = self.mdp_state.failure_reasons.clone()
        wrench = task_mdp.corrected_wrench(self)
        if self._terminal_capture is not None:
            ids, final, final_components, final_reasons, final_wrench = self._terminal_capture
            terminal_observation[ids] = final
            components[ids] = final_components
            reasons[ids] = final_reasons
            wrench[ids] = final_wrench
        extras["terminal_observation"] = terminal_observation
        extras["reward_components"] = components
        extras["failure_reasons"] = reasons
        extras["compensated_wrench"] = wrench
        return observations, rewards, terminated, truncated, extras
