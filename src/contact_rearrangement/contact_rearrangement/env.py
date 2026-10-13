"""RL environment preserving final observations across selective auto-resets."""

from isaaclab.envs import ManagerBasedRLEnv

from . import mdp as task_mdp
from .mdp.actions import forbidden_contacts


class ContactRearrangementRLEnv(ManagerBasedRLEnv):
    """Preserve pre-reset terminal observations for correct PPO timeout values."""

    def __init__(self, cfg, **kwargs):
        if cfg.spawn_rules.max_objects != cfg.mdp.max_objects:
            raise ValueError("Spawn capacity must match the MDP's 40 object slots.")
        task_mdp.sync_reward_weights(cfg)
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
                state.motion_costs[env_ids].clone(),
                state.previous_wrench[env_ids].clone(),
                state.elapsed_time[env_ids].clone(),
                state.previous_corners[env_ids].clone(),
                state.rearrangement_increment[env_ids].clone(),
                state.path_length[env_ids].clone(),
                self.tool_force_torque_sensor.read(env_ids),
                self.tool_force_torque_sensor.read_ideal(env_ids),
                self.tool_force_torque_sensor.range_exceeded[env_ids].clone(),
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
        motion_costs = self.mdp_state.motion_costs.clone()
        reward_wrench = self.mdp_state.previous_wrench.clone()
        elapsed_time = self.mdp_state.elapsed_time.clone()
        corners = self.mdp_state.previous_corners.clone()
        rearrangement_increment = self.mdp_state.rearrangement_increment.clone()
        rearrangement_path = self.mdp_state.path_length.clone()
        reasons = self.mdp_state.failure_reasons.clone()
        wrench = task_mdp.corrected_wrench(self)
        raw_wrench = self.tool_force_torque_sensor.read()
        ideal_wrench = self.tool_force_torque_sensor.read_ideal()
        range_exceeded = self.tool_force_torque_sensor.range_exceeded.clone()
        if self._terminal_capture is not None:
            (ids, final, final_components, final_reasons, final_wrench,
             final_motion_costs, final_reward_wrench, final_elapsed_time,
             final_corners, final_increment, final_path,
             final_raw_wrench, final_ideal_wrench, final_range_exceeded) = self._terminal_capture
            terminal_observation[ids] = final
            components[ids] = final_components
            reasons[ids] = final_reasons
            wrench[ids] = final_wrench
            motion_costs[ids] = final_motion_costs
            reward_wrench[ids] = final_reward_wrench
            elapsed_time[ids] = final_elapsed_time
            corners[ids] = final_corners
            rearrangement_increment[ids] = final_increment
            rearrangement_path[ids] = final_path
            raw_wrench[ids] = final_raw_wrench
            ideal_wrench[ids] = final_ideal_wrench
            range_exceeded[ids] = final_range_exceeded
        extras["terminal_observation"] = terminal_observation
        extras["terminated"] = terminated
        extras["reward_components"] = components
        extras["motion_costs"] = motion_costs
        extras["reward_wrench"] = reward_wrench
        extras["episode_elapsed_time"] = elapsed_time
        extras["rearrangement_corners"] = corners
        extras["rearrangement_increment"] = rearrangement_increment
        extras["rearrangement_path"] = rearrangement_path
        extras["failure_reasons"] = reasons
        extras["compensated_wrench"] = wrench
        extras["sensor_raw_wrench"] = raw_wrench
        extras["sensor_ideal_wrench"] = ideal_wrench
        extras["sensor_range_exceeded"] = range_exceeded
        visualizer = getattr(self, "role_obb_visualizer", None)
        if visualizer is not None:
            visualizer.update()
        return observations, rewards, terminated, truncated, extras
