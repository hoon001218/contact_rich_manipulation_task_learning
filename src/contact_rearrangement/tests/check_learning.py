"""GPU checks for the research MDP; imported only with training --validate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, quat_mul

from contact_rearrangement.src.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from contact_rearrangement.config.ur5e.agents.model import build_policy_masks
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg

def _initial_scene_validation_reference(env):
    """Build an independent base-frame reference for the fixed policy slots.

    This simulator metadata is a validation oracle, never a model input. Each
    observation producer must place Target/Blocker first and retain the order
    of the remaining visible objects while excluding unobserved padding.
    """
    geometry = torch.zeros_like(env.initial_object_geometry)
    valid = torch.zeros_like(env.initial_object_valid)
    roles = torch.zeros_like(env.initial_object_roles, dtype=torch.bool)
    for env_id in range(env.num_envs):
        source_valid = env.initial_object_valid[env_id]
        source_roles = env.initial_object_roles[env_id].bool()
        for role in range(2):
            indices = torch.nonzero(source_valid & source_roles[:, role], as_tuple=False).flatten()
            if indices.numel() > 1:
                raise AssertionError("Initial perception contains duplicate role assignments.")
            if indices.numel():
                geometry[env_id, role] = env.initial_object_geometry[env_id, indices[0]]
                valid[env_id, role] = True
                roles[env_id, role, role] = True
        distractors = torch.nonzero(source_valid & ~source_roles.any(-1), as_tuple=False).flatten()
        if distractors.numel() > geometry.shape[1] - 2:
            raise AssertionError("Initial perception exceeds the available distractor slots.")
        end = 2 + distractors.numel()
        geometry[env_id, 2:end] = env.initial_object_geometry[env_id, distractors]
        valid[env_id, 2:end] = True
    return geometry, valid, roles


def _physical_corner_validation_reference(env):
    """Reconstruct fixed physical OBB corners with SDK pose transforms.

    This privileged oracle is used only by --validate, never as policy input.
    """
    from contact_rearrangement.src.catalog import OBJECT_SPECS

    ids = env.active_object_ids.clamp_min(0)
    pose = env.scene["objects"].data.object_link_pose_w.gather(1, ids[..., None].expand(-1, -1, 7))
    lower = pose.new_tensor([geometry.lower for geometry in env.spawn_geometries])[ids]
    upper = pose.new_tensor([geometry.upper for geometry in env.spawn_geometries])[ids]
    upright = pose.new_tensor([spec.upright_quat for spec in OBJECT_SPECS])[ids]
    bits = (torch.arange(8, device=env.device)[:, None] >> torch.tensor([2, 1, 0], device=env.device)) & 1
    signs = bits.to(pose) * 2 - 1
    local = (lower + upper)[..., None, :] * .5 + (upper - lower)[..., None, :] * .5 * signs
    rotation = quat_mul(pose[..., 3:7].reshape(-1, 4), quat_conjugate(upright.reshape(-1, 4)))
    rotation = rotation.reshape(*ids.shape, 4)[..., None, :].expand(*ids.shape, 8, 4)
    rotated = quat_apply(rotation.reshape(-1, 4), local.reshape(-1, 3)).reshape_as(local)
    return pose[..., None, :3] + rotated - env.scene.env_origins[:, None, None, :]


def validate_transition(env, before_geometry, observation, reward, terminated, truncated, extra, action,
                        before_history, before_wrench, before_episode_steps,
                        before_corners, before_path, before_active):
    """Check completed transitions before/after the SDK's selective auto-reset."""
    state = env.mdp_state
    object_width = env.cfg.mdp.max_objects * 10
    history_steps = env.cfg.mdp.history_length
    policy_cfg = ModelCfg(max_objects=env.cfg.mdp.max_objects, history_steps=history_steps)
    expected_width = policy_cfg.observation_dim
    if observation.shape != (env.num_envs, expected_width) or not torch.isfinite(observation).all():
        raise AssertionError(f"Policy observation must be finite [N,{expected_width}].")
    if history_steps != 2:
        raise AssertionError("Policy sensing must contain exactly current and previous samples.")
    done = terminated | truncated
    continuing = ~done
    masks = build_policy_masks(observation, policy_cfg)
    terminal_masks = build_policy_masks(extra["terminal_observation"], policy_cfg)
    # Simulator metadata is an independent validation oracle only. The policy
    # gets the masks derived above solely from its observation values.
    reference, reference_valid, reference_roles = _initial_scene_validation_reference(env)
    torch.testing.assert_close(masks.object_valid, reference_valid)
    torch.testing.assert_close(masks.roles, reference_roles)
    torch.testing.assert_close(masks.history_valid, state.history_valid)
    torch.testing.assert_close(env.initial_object_geometry[continuing], before_geometry[continuing], rtol=0, atol=0)
    torch.testing.assert_close(extra["terminal_observation"][continuing], observation[continuing], rtol=0, atol=0)
    weights = reward.new_tensor([getattr(env.cfg.rewards, name).weight
                                 if getattr(env.cfg.rewards, name) is not None else 0.0
                                 for name in ("clearance", "rearrangement", "motion", "time", "terminal")])
    expected = extra["reward_components"] @ weights
    torch.testing.assert_close(reward, expected, rtol=1e-5, atol=1e-5)
    # Use the SDK counter as an independent clock, including the completed
    # terminal interval captured before auto-reset. This diagnostic is not
    # part of the actor observation.
    time_before = before_episode_steps.to(torch.float64) * env.step_dt
    time_after = time_before + env.step_dt
    torch.testing.assert_close(extra["episode_elapsed_time"], time_after, rtol=0, atol=1e-10)
    torch.testing.assert_close(state.elapsed_time[continuing], time_after[continuing], rtol=0, atol=1e-10)
    torch.testing.assert_close(state.elapsed_time[done], torch.zeros_like(state.elapsed_time[done]), rtol=0, atol=0)
    cfg = env.cfg.mdp
    growth = cfg.time_penalty_growth_rate
    # Independent antiderivative check of the exponential cost curve.
    expected_time = -cfg.time_penalty_scale * (
        (torch.exp(growth * time_after) - torch.exp(growth * time_before)) / growth - env.step_dt)
    torch.testing.assert_close(extra["reward_components"][:, 3], expected_time.to(reward), rtol=1e-5, atol=1e-8)
    displacement = (extra["rearrangement_corners"] - before_corners).norm(dim=-1).mean(-1)
    increment = (displacement * before_active).sum(-1)
    path = before_path + increment
    torch.testing.assert_close(extra["rearrangement_increment"], increment, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(extra["rearrangement_path"], path, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(extra["reward_components"][:, 1], torch.exp(-cfg.rearrangement_decay * path),
                               rtol=1e-5, atol=1e-6)
    native_corners = _physical_corner_validation_reference(env)
    torch.testing.assert_close(extra["rearrangement_corners"][continuing], native_corners[continuing],
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(state.previous_corners, native_corners, rtol=0, atol=1e-6)
    torch.testing.assert_close(state.path_length[done], torch.zeros_like(state.path_length[done]), rtol=0, atol=0)
    torch.testing.assert_close(state.rearrangement_increment[done], torch.zeros_like(state.rearrangement_increment[done]),
                               rtol=0, atol=0)
    if done.any():
        if not (state.history_valid[done].sum(-1) == 1).all():
            raise AssertionError("Reset must leave exactly one fresh history sample.")
        torch.testing.assert_close(state.history[done, 0, -18:], torch.zeros_like(action[done]), rtol=0, atol=0)
        torch.testing.assert_close(state.history[done, 1], torch.zeros_like(state.history[done, 1]), rtol=0, atol=0)
        if state.history_valid[done, 1].any():
            raise AssertionError("Reset previous sample must be invalid.")
        if masks.tactile_valid[done].any():
            raise AssertionError("Reset tactile samples must remain unavailable until the next physics step.")
        if not (terminal_masks.history_valid[done].sum(-1) == 2).all():
            raise AssertionError("PPO requires the completed episode's final history, before reset.")
    if continuing.any():
        torch.testing.assert_close(state.history[continuing, 0, -18:], action[continuing], rtol=0, atol=0)
    final_history = extra["terminal_observation"][:, object_width:].reshape(env.num_envs, 2, policy_cfg.step_features)
    torch.testing.assert_close(final_history[:, 0, -18:], action, rtol=0, atol=0)
    torch.testing.assert_close(final_history[:, 1], before_history[:, 0], rtol=0, atol=0)
    robot = env.scene["robot"]
    # Independently reconstruct joint Motion costs from the sensor history,
    # including terminal transitions whose articulation has already reset.
    mdp_cfg = env.cfg.mdp
    joint_velocity = final_history[:, 0, 12:24] * mdp_cfg.joint_velocity_scale
    previous_joint_velocity = before_history[:, 0, 12:24] * mdp_cfg.joint_velocity_scale
    acceleration_ref = joint_velocity.new_tensor([mdp_cfg.arm_joint_acceleration_ref] * 6
                                                 + [mdp_cfg.hand_joint_acceleration_ref] * 6)
    acceleration_cost = (((joint_velocity - previous_joint_velocity) / env.step_dt)
                         / acceleration_ref).square().sum(-1)
    wrench = extra["reward_wrench"]
    force_increase = (wrench[:, :3].norm(dim=-1) - before_wrench[:, :3].norm(dim=-1)).clamp_min(0)
    torque_increase = (wrench[:, 3:].norm(dim=-1) - before_wrench[:, 3:].norm(dim=-1)).clamp_min(0)
    increase_cost = ((force_increase / (env.step_dt * mdp_cfg.force_rate_ref)).square()
                     + (torque_increase / (env.step_dt * mdp_cfg.torque_rate_ref)).square())
    if extra["motion_costs"].shape != (env.num_envs, 2):
        raise AssertionError("Motion must contain only joint acceleration and wrench increase costs.")
    # Hardware limits apply to the total raw sensor-frame load before any
    # gravity/inertia subtraction. Check the measurement and policy scaling
    # independently, including transitions captured before automatic reset.
    ranges = reward.new_tensor(env.cfg.force_torque.ranges)
    resolution = reward.new_tensor(env.cfg.force_torque.resolution)
    ideal = extra["sensor_ideal_wrench"]
    expected_range_exceeded = ideal.abs() > ranges
    expected_raw = torch.round(ideal.clamp(-ranges, ranges) / resolution) * resolution
    expected_raw = expected_raw.clamp(-ranges, ranges)
    torch.testing.assert_close(extra["sensor_range_exceeded"], expected_range_exceeded)
    torch.testing.assert_close(extra["sensor_raw_wrench"], expected_raw, rtol=0, atol=1e-6)
    normalized_wrench = (extra["compensated_wrench"] / ranges).clamp(-5, 5)
    torch.testing.assert_close(final_history[:, 0, 31:37], normalized_wrench, rtol=1e-5, atol=1e-6)
    if (expected_range_exceeded.any(-1) & ~extra["failure_reasons"][:, 4]).any():
        raise AssertionError("A raw sensor-range violation must trigger the hard-load failure.")
    if env.tool_force_torque_sensor.range_exceeded[done].any():
        raise AssertionError("Auto-reset must clear the previous episode's range-exceeded flags.")
    torch.testing.assert_close(extra["motion_costs"][:, 0], acceleration_cost, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(extra["motion_costs"][:, 1], increase_cost, rtol=1e-5, atol=1e-5)
    decay = extra["motion_costs"].new_tensor((mdp_cfg.motion_acceleration_decay, mdp_cfg.motion_wrench_increase_decay))
    torch.testing.assert_close(extra["reward_components"][:, 2], (-extra["motion_costs"] * decay).exp().mean(-1),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(joint_velocity[continuing], robot.data.joint_vel[continuing][:, state.joint_ids],
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state.previous_joint_velocity, robot.data.joint_vel[:, state.joint_ids],
                               rtol=1e-6, atol=1e-6)
    arm = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
    hand = [robot.joint_names.index(name) for name in HAND_JOINT_NAMES]
    mimic = [i for i, name in enumerate(robot.joint_names) if i not in arm + hand]
    for indices in (arm, mimic):
        torch.testing.assert_close(robot.data.joint_stiffness[:, indices], torch.zeros_like(robot.data.joint_stiffness[:, indices]))
        torch.testing.assert_close(robot.data.joint_damping[:, indices], torch.zeros_like(robot.data.joint_damping[:, indices]))
    if not torch.isfinite(robot.data.joint_pos).all():
        raise AssertionError("Native articulation state is nonfinite.")
    term = env.action_manager.get_term("manipulate")
    if not torch.isfinite(term.joint_efforts).all():
        raise AssertionError("OSC effort is nonfinite.")
    # Independently reconstruct the base-frame initial scene from actor EEF
    # geometry using the SDK transform utility used in examples.
    geometry = observation[:, :400].reshape(env.num_envs, 40, 10)
    tcp = term.measured_tcp_pose_b()[:, None].expand(-1, 40, -1)
    position, rotation = combine_frame_transforms(tcp[..., :3].reshape(-1, 3), tcp[..., 3:].reshape(-1, 4),
                                                 geometry[..., :3].reshape(-1, 3), geometry[..., 3:7].reshape(-1, 4))
    valid = reference_valid
    expected = reference
    torch.testing.assert_close(position.reshape(env.num_envs, 40, 3)[valid], expected[..., :3][valid], atol=1e-6, rtol=0)
    alignment = (rotation.reshape(env.num_envs, 40, 4)[valid] * expected[..., 3:7][valid]).sum(-1).abs()
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-6, rtol=0)
    torch.testing.assert_close(geometry[..., 7:][valid], expected[..., 7:][valid], atol=0, rtol=0)


def validate_forbidden_contact(env):
    """Place a real pool object into the proximal spacer and verify its reporter.

    This physics fixture uses simulator geometry only to set up a collision.
    The scene's frozen initial perception is left untouched. The normal SDK
    termination/reset path must retain the physical contact failure in extras.
    """
    from contact_rearrangement.src.assets import FORCE_TORQUE_PROXIMAL_BODY_NAME
    from contact_rearrangement.mdp import quat_apply, quat_mul, quat_conjugate
    from contact_rearrangement.src.catalog import OBJECT_SPECS

    robot, objects = env.scene["robot"], env.scene["objects"]
    body = robot.body_names.index(FORCE_TORQUE_PROXIMAL_BODY_NAME)
    object_id = int(env.blocker_object_id[0])
    geometry = env.spawn_geometries[object_id]
    pose = objects.data.object_link_pose_w[0, object_id].clone()
    relative = quat_mul(pose[3:7], quat_conjugate(pose.new_tensor(OBJECT_SPECS[object_id].upright_quat)))
    center_offset = pose.new_tensor(geometry.lower) + pose.new_tensor(geometry.upper)
    pose[:3] = robot.data.body_com_pos_w[0, body] - quat_apply(relative, center_offset * 0.5)
    objects.write_object_link_pose_to_sim(pose[None, None], object_ids=[object_id],
                                         env_ids=torch.tensor([0], device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    term = env.action_manager.get_term("manipulate")
    action = torch.zeros((env.num_envs, 18), device=env.device)
    limits = robot.data.soft_joint_pos_limits[:, term._hand_ids]
    action[:, 12:] = (2 * robot.data.joint_pos[:, term._hand_ids] - limits[..., 0] - limits[..., 1]) / (
        limits[..., 1] - limits[..., 0])
    with torch.no_grad():
        _, _, terminated, _, extra = env.step(action)
    if not terminated[0] or not extra["failure_reasons"][0, 5]:
        raise AssertionError("Physical spacer/object collision did not trigger the filtered contact failure.")
    print("FORBIDDEN_CONTACT_VALIDATED " + json.dumps(dict(reasons=extra["failure_reasons"][0].tolist())), flush=True)
    # Start fresh after the deliberately invalid fixture for the PPO smoke.
    return env.reset()[0]


def validate_control_hold(env):
    """An unloaded open-hand hold must stay stable without hard-load failures."""
    term, robot = env.action_manager.get_term("manipulate"), env.scene["robot"]
    initial = term.measured_tcp_pose_b().clone()
    limits = robot.data.soft_joint_pos_limits[:, term._hand_ids]
    action = torch.zeros((env.num_envs, 18), device=env.device)
    action[:, 12:] = (2 * robot.data.joint_pos[:, term._hand_ids] - limits[..., 0] - limits[..., 1]) / (
        limits[..., 1] - limits[..., 0])
    steps = min(16, env.max_episode_length - 1)
    if steps < 1:
        raise ValueError("Control validation needs an episode longer than one policy step.")
    force_torque_peak = torch.zeros(2, device=env.device)
    for _ in range(steps):
        before = env.initial_object_geometry.clone()
        before_history = env.mdp_state.history.clone()
        before_wrench = env.mdp_state.previous_wrench.clone()
        before_episode_steps = env.episode_length_buf.clone()
        before_corners = _physical_corner_validation_reference(env)
        before_path = env.mdp_state.path_length.clone()
        before_active = env.active_object_ids >= 0
        with torch.no_grad():
            observations, reward, terminated, truncated, extra = env.step(action)
        validate_transition(env, before, observations["policy"], reward, terminated, truncated, extra, action,
                            before_history, before_wrench, before_episode_steps,
                            before_corners, before_path, before_active)
        if terminated.any() or truncated.any():
            raise AssertionError(f"Unloaded hold terminated: {extra['failure_reasons'].tolist()}")
        wrench = extra["compensated_wrench"]
        force_torque_peak = torch.maximum(force_torque_peak, torch.stack((wrench[:, :3].norm(dim=-1).max(),
                                                                        wrench[:, 3:].norm(dim=-1).max())))
    final = term.measured_tcp_pose_b()
    position_drift = (final[:, :3] - initial[:, :3]).norm(dim=-1)
    angle_drift = 2 * torch.acos((final[:, 3:] * initial[:, 3:]).sum(-1).abs().clamp(0, 1))
    if position_drift.max() > 0.01 or angle_drift.max() > 0.05:
        raise AssertionError("Unloaded OSC hold drift exceeded 1 cm / 0.05 rad.")
    print("CONTROL_HOLD_VALIDATED " + json.dumps(dict(steps=steps, position_drift=position_drift.tolist(),
                                                     angle_drift=angle_drift.tolist(),
                                                     force_torque_peak=force_torque_peak.tolist())), flush=True)
    return env.reset()[0]


class LearningValidation:
    """Check research invariants around the unmodified native training loop."""

    def __init__(self, env):
        self.env = env
        self.transitions = self.terminated = self.truncated = 0
        env.reset()
        validate_control_hold(env)
        validate_forbidden_contact(env)
        original_step = env.step

        def audited_step(action):
            before_geometry = env.initial_object_geometry.clone()
            before_history = env.mdp_state.history.clone()
            before_wrench = env.mdp_state.previous_wrench.clone()
            before_steps = env.episode_length_buf.clone()
            before_corners = _physical_corner_validation_reference(env)
            before_path = env.mdp_state.path_length.clone()
            before_active = env.active_object_ids >= 0
            result = original_step(action)
            observation, reward, terminated, truncated, extras = result
            validate_transition(env, before_geometry, observation["policy"], reward, terminated, truncated,
                                extras, action, before_history, before_wrench, before_steps,
                                before_corners, before_path, before_active)
            self.transitions += env.num_envs
            self.terminated += int(terminated.sum())
            self.truncated += int(truncated.sum())
            return result

        env.step = audited_step

    def before_training(self, runner):
        # Deliberate contact fixtures are outside the training episodes.
        self.env.extras["log"] = {
            key: torch.zeros_like(value) if isinstance(value, torch.Tensor) else 0.0
            for key, value in self.env.extras.get("log", {}).items()
        }
        self.parameters = [p.detach().clone() for p in runner.alg.actor.parameters()]

    def after_training(self, runner, train_cfg):
        parameters = list(runner.alg.actor.parameters())
        if not all(torch.isfinite(p).all() for p in parameters):
            raise AssertionError("Native PPO produced nonfinite parameters.")
        if not any(not torch.equal(before, after) for before, after in zip(self.parameters, parameters)):
            raise AssertionError("Native PPO did not update the research network.")
        checkpoint = Path(runner.logger.log_dir) / f"model_{runner.current_learning_iteration}.pt"
        restored = OnPolicyRunner(runner.env, train_cfg, log_dir=None, device=runner.device)
        restored.load(str(checkpoint), map_location=runner.device)
        observation = runner.env.get_observations()
        with torch.no_grad():
            torch.testing.assert_close(runner.alg.actor(observation), restored.alg.actor(observation))
            torch.testing.assert_close(runner.alg.critic(observation), restored.alg.critic(observation))
        print("LEARNING_VALIDATED " + json.dumps(dict(transitions=self.transitions,
              terminated=self.terminated, truncated=self.truncated, checkpoint=str(checkpoint))), flush=True)

