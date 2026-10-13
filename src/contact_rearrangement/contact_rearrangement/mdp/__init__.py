"""Manager terms for contact-rich shelf rearrangement.

Explicit exports keep task configuration readable without importing Isaac Sim
or expanding Isaac Lab's entire generic MDP namespace during CPU tests.
"""

from .state import (
    MdpCfg, MdpState, ensure_mdp, reset_mdp, update_mdp,
    quat_apply, quat_conjugate, quat_mul,
    convex_hull_xy, polygon_area, rectangle_overlap_area, object_geometry,
    _physical_objects, _fixed_wrench,
)
from .observations import (
    policy_observation, pack_policy_observation, geometry_in_eef_frame,
    corrected_wrench, tare_compensated_wrench,
)
from .rewards import (
    reward_components, reward_rate, clearance, rearrangement, motion,
    time_penalty, terminal, sync_reward_weights,
)
from .terminations import terminal_transition, success, failure

__all__ = [
    "MdpCfg", "MdpState", "ensure_mdp", "reset_mdp", "update_mdp",
    "quat_apply", "quat_conjugate", "quat_mul", "convex_hull_xy",
    "polygon_area", "rectangle_overlap_area", "object_geometry",
    "policy_observation", "pack_policy_observation", "geometry_in_eef_frame",
    "corrected_wrench", "tare_compensated_wrench", "reward_components",
    "reward_rate", "clearance", "rearrangement", "motion", "time_penalty",
    "terminal", "sync_reward_weights", "terminal_transition", "success", "failure",
]


def __getattr__(name):
    # Scene reset imports USD/Isaac Sim. Keep numerical MDP and model imports
    # usable before AppLauncher starts, as required by the CPU test suite.
    if name == "reset_spawn_scene":
        from .events import reset_spawn_scene

        return reset_spawn_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
