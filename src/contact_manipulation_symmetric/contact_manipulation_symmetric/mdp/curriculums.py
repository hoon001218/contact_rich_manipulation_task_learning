"""Shared-scene layout and legacy object-count curriculum schedules."""

from dataclasses import replace
import math
import random


SCENE_LAYOUTS_BY_DIFFICULTY = (
    (("direct", "direct"), ("direct", "front"), ("direct", "blocked")),
    (("front", "direct"), ("front", "front"), ("front", "blocked"), ("blocked", "direct")),
    (("blocked", "front"),),
)
INITIAL_LAYOUT_WEIGHTS = (.85, .10, .05)
EQUAL_LAYOUT_WEIGHTS = (1.0 / 3.0,) * 3


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def checkpoint_completed_iterations(checkpoint):
    """Convert the native checkpoint's zero-based iteration to completed rollouts."""
    if checkpoint is None:
        return 0
    return _nonnegative_integer(checkpoint.get("iter"), "checkpoint 'iter'") + 1


def layout_difficulty(near_status, far_status):
    """Count the fewest extra requirements across the layout's available routes."""
    for difficulty, layouts in enumerate(SCENE_LAYOUTS_BY_DIFFICULTY):
        if (near_status, far_status) in layouts:
            return difficulty
    raise ValueError(f"Invalid or unsolvable scene layout: {(near_status, far_status)!r}.")


def scene_layout_probabilities(step, *, start_step, end_step,
                               start_weights=INITIAL_LAYOUT_WEIGHTS, end_weights=EQUAL_LAYOUT_WEIGHTS):
    """Interpolate the easy, medium, and hard probabilities on the training clock."""
    step = _nonnegative_integer(step, "step")
    start_step = _nonnegative_integer(start_step, "start_step")
    end_step = _nonnegative_integer(end_step, "end_step")
    if end_step <= start_step:
        raise ValueError("Layout curriculum end_step must exceed start_step.")
    for weights in (start_weights, end_weights):
        if (len(weights) != 3 or any(not math.isfinite(weight) or weight < 0 for weight in weights)
                or not math.isclose(math.fsum(weights), 1.0, rel_tol=0., abs_tol=1.e-9)):
            raise ValueError("Layout curriculum needs three nonnegative weights summing to one.")
    if step <= start_step:
        return tuple(start_weights)
    if step >= end_step:
        return tuple(end_weights)
    fraction = (step - start_step) / (end_step - start_step)
    return tuple(start + fraction * (end - start) for start, end in zip(start_weights, end_weights))


def choose_scene_layout(seed, scene_generation, step, *, start_step, end_step,
                        start_weights=INITIAL_LAYOUT_WEIGHTS, end_weights=EQUAL_LAYOUT_WEIGHTS):
    """Choose a near side and near/far conditions for one common scene replacement.

    Each replacement draws difficulty independently from the current weights.
    The seed and scene number make the draw reproducible and independent of the
    random numbers consumed by asset selection and placement.
    """
    scene_generation = _nonnegative_integer(scene_generation, "scene_generation")
    weights = scene_layout_probabilities(
        step, start_step=start_step, end_step=end_step, start_weights=start_weights, end_weights=end_weights,
    )
    rng = random.Random(f"scene-layout/{seed}/{scene_generation}")
    difficulty = rng.choices(range(len(SCENE_LAYOUTS_BY_DIFFICULTY)), weights=weights, k=1)[0]
    near_status, far_status = rng.choice(SCENE_LAYOUTS_BY_DIFFICULTY[difficulty])
    return rng.choice((-1, 1)), near_status, far_status


def object_count_range(step, *, start_step, end_step, final_min, final_max):
    """Interpolate from Target/Blocker only to the configured full count range.

    Counts advance by integer floors. ``step`` counts policy steps rather than
    individual environments or physics substeps, keeping the schedule independent
    of the number of parallel environments.
    """
    step = _nonnegative_integer(step, "step")
    start_step = _nonnegative_integer(start_step, "start_step")
    end_step = _nonnegative_integer(end_step, "end_step")
    final_min = _nonnegative_integer(final_min, "final_min")
    final_max = _nonnegative_integer(final_max, "final_max")
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step.")
    if not 2 <= final_min <= final_max:
        raise ValueError("Final object limits must satisfy 2 <= min <= max.")
    duration = end_step - start_step
    elapsed = min(max(step - start_step, 0), duration)
    return (2 + (final_min - 2) * elapsed // duration,
            2 + (final_max - 2) * elapsed // duration)


def scene_object_count_curriculum(
    env, env_ids, *, start_step, end_step, final_min, final_max, resume_steps=0,
):
    """Set the count range used by the next shared scene, without resetting it.

    Native curriculum evaluation precedes reset events. Partial resets still
    reuse the current shared scene; its active bodies change only when the
    existing scene-refresh event constructs a replacement layout.
    """
    del env_ids
    resume_steps = _nonnegative_integer(resume_steps, "resume_steps")
    step = _nonnegative_integer(env.common_step_counter, "common_step_counter") + resume_steps
    bounds = object_count_range(step, start_step=start_step, end_step=end_step,
                                final_min=final_min, final_max=final_max)
    env.cfg.spawn_object_count_range = bounds
    progress = min(max((step - start_step) / (end_step - start_step), 0.0), 1.0)
    return {"min_objects": bounds[0], "max_objects": bounds[1], "progress": progress}


def prepare_scene_curriculum(env_cfg, agent_cfg, *, completed_iterations, advance=True):
    """Select the shared spawn schedule before the first reset."""
    env_cfg.layout_curriculum_start_step = None
    env_cfg.layout_curriculum_end_step = None
    env_cfg.layout_curriculum_resume_steps = 0
    env_cfg.curriculum.scene_object_count = None
    env_cfg.spawn_object_count_range = None
    if env_cfg.layout_scene_override is not None:
        return None, None

    start = env_cfg.layout_curriculum_start_iteration
    if start is not None:
        end = env_cfg.layout_curriculum_end_iteration
        start_step, end_step, resume_steps = _scene_schedule_steps(
            env_cfg, agent_cfg, start, end, completed_iterations, "layout_curriculum",
        )
        weights = scene_layout_probabilities(
            resume_steps, start_step=start_step, end_step=end_step,
            start_weights=env_cfg.layout_curriculum_start_weights,
            end_weights=env_cfg.layout_curriculum_end_weights,
        )
        env_cfg.layout_curriculum_start_step = start_step
        env_cfg.layout_curriculum_end_step = end_step
        env_cfg.layout_curriculum_resume_steps = resume_steps
        if not advance:
            # Keep the checkpoint's probabilities while still drawing every new scene.
            env_cfg.layout_curriculum_start_weights = env_cfg.layout_curriculum_end_weights = weights
        return (start, end), None

    start = env_cfg.scene_object_curriculum_start_iteration
    if start is None:
        return None, None
    end = env_cfg.scene_object_curriculum_end_iteration
    start_step, end_step, resume_steps = _scene_schedule_steps(
        env_cfg, agent_cfg, start, end, completed_iterations, "scene_object_curriculum",
    )
    params = dict(start_step=start_step, end_step=end_step, final_min=env_cfg.spawn_rules.min_objects,
                  final_max=env_cfg.spawn_rules.max_objects)
    env_cfg.spawn_object_count_range = object_count_range(resume_steps, **params)
    if advance:
        from isaaclab.managers import CurriculumTermCfg

        env_cfg.curriculum.scene_object_count = CurriculumTermCfg(
            func=scene_object_count_curriculum, params={**params, "resume_steps": resume_steps},
        )
    return None, (start, end)


def _scene_schedule_steps(env_cfg, agent_cfg, start, end, completed_iterations, schedule):
    _nonnegative_integer(start, f"{schedule}_start_iteration")
    _nonnegative_integer(end, f"{schedule}_end_iteration")
    rollout = _nonnegative_integer(agent_cfg.num_steps_per_env, "num_steps_per_env")
    _nonnegative_integer(completed_iterations, "completed_iterations")
    if rollout == 0 or end <= start:
        raise ValueError(f"{schedule} needs a positive rollout length and an end after its start.")
    if env_cfg.spawn_rules.generation_mode != "sequential":
        raise ValueError(f"{schedule} requires sequential generation.")
    return start * rollout, end * rollout, completed_iterations * rollout


def curriculum_spawn_rules(env):
    """Return temporary sampling rules while preserving physical pool capacity.

    ``env.spawn_rules.max_objects`` also defines tensor capacity and captured
    graph shapes, so it must remain unchanged throughout training and playback.
    """
    rules = env.spawn_rules
    bounds = getattr(env.cfg, "spawn_object_count_range", None)
    if bounds is None:
        return rules
    if getattr(rules, "generation_mode", None) != "sequential":
        raise ValueError("Object-count curriculum requires the sequential scene generator.")
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("spawn_object_count_range must contain two integer count bounds.")
    low = _nonnegative_integer(bounds[0], "spawn_object_count_range minimum")
    high = _nonnegative_integer(bounds[1], "spawn_object_count_range maximum")
    if not 2 <= low <= high <= rules.max_objects:
        raise ValueError("Object-count curriculum must satisfy 2 <= min <= max <= pool capacity.")
    return replace(rules, min_objects=low, max_objects=high)
