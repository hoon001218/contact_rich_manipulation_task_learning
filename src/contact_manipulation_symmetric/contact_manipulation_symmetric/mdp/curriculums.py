"""Object-count schedules for the native curriculum manager and scene builder."""

from dataclasses import replace


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


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


def curriculum_spawn_rules(env):
    """Return temporary sampling rules while preserving physical pool capacity.

    ``env.spawn_rules.max_objects`` also defines tensor capacity and captured
    graph shapes, so it must remain unchanged throughout training. Playback
    leaves the independent count override unset and receives the original rules.
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
