"""Viewport-only outlines for the episode's Target and Blocker."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .catalog import OBJECT_SPECS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# product((0, 1), repeat=3) orders corners by the three coordinate bits.
# An edge joins corners differing in exactly one bit.
_EDGES = tuple((index, index ^ bit) for index in range(8) for bit in (1, 2, 4) if index < (index ^ bit))
_ROLE_COLORS = ((0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0))


class RoleObbVisualizer:
    """Draw scaled object bounds using lines, without authoring USD or physics.

    This script owns the debug-draw line buffer. Bounds are fixed in each
    object's canonical upright frame; current rigid-body link poses orient
    them in the world, including any motion or tilt after spawning.
    """

    def __init__(self, env: ManagerBasedEnv):
        # GUI experiences already load this extension. Explicit enabling also
        # lets --headless --validate exercise the same line drawing API.
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._env = env
        self._env_ids = torch.arange(env.num_envs, device=env.device)[:, None]
        self._corners = torch.tensor(
            [list(product(*zip(geometry.lower, geometry.upper))) for geometry in env.spawn_geometries],
            dtype=torch.float32, device=env.device,
        )
        self._upright_inverse = quat_conjugate(
            torch.tensor([spec.upright_quat for spec in OBJECT_SPECS], dtype=torch.float32, device=env.device)
        )
        self._edge_start = torch.tensor([edge[0] for edge in _EDGES], device=env.device)
        self._edge_end = torch.tensor([edge[1] for edge in _EDGES], device=env.device)
        self._colors = [color for _ in range(env.num_envs) for color in _ROLE_COLORS for _ in _EDGES]
        self._widths = [3.0] * (24 * env.num_envs)

    def world_corners(self) -> torch.Tensor:
        """Return (env, Target/Blocker, corner, xyz) from current role IDs/poses."""
        env = self._env
        role_ids = torch.stack((env.target_object_id, env.blocker_object_id), dim=-1)
        objects = env.scene["objects"]
        position = objects.data.object_link_pos_w[self._env_ids, role_ids]
        orientation = objects.data.object_link_quat_w[self._env_ids, role_ids]
        # Inspected bounds already include catalog upright rotation and scale.
        # Undo only that rotation before applying the current body-link pose.
        rotation = quat_mul(orientation, self._upright_inverse[role_ids])
        corners = quat_apply(rotation[..., None, :].expand(-1, -1, 8, -1), self._corners[role_ids])
        return corners + position[..., None, :]

    def update(self) -> None:
        """Replace the two wire boxes per environment; never write body state."""
        corners = self.world_corners()
        starts = corners[:, :, self._edge_start].reshape(-1, 3).cpu().tolist()
        ends = corners[:, :, self._edge_end].reshape(-1, 3).cpu().tolist()
        self._draw.clear_lines()
        self._draw.draw_lines(starts, ends, self._colors, self._widths)

    @property
    def line_count(self) -> int:
        return self._draw.get_num_lines()

    def close(self) -> None:
        self._draw.clear_lines()
