"""Keep unused pool bodies stationary without changing live GPU body flags."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObjectCollection


class TaskObjectPool(RigidObjectCollection):
    """Apply countergravity at parked bodies' COM in every physics substep.

    Task objects retain native gravity and get zero countergravity. A live GPU
    simulation/switch is avoided, and world COM application prevents torque on
    assets with off-center pivots or COMs. Other external wrenches stay separate.
    """

    def __init__(self, cfg):
        self.parked_mask: torch.Tensor | None = None
        self.gravity: torch.Tensor | None = None
        super().__init__(cfg)

    def write_data_to_sim(self) -> None:
        super().write_data_to_sim()
        if self.parked_mask is None:
            return
        forces = -self.data.default_mass.to(self.device) * self.gravity * self.parked_mask[..., None]
        self.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.reshape_data_to_view(forces).contiguous(),
            torque_data=None,
            position_data=self.reshape_data_to_view(self.data.object_com_pos_w).contiguous(),
            indices=torch.arange(self.root_physx_view.count, device=self.device, dtype=torch.int32),
            is_global=True,
        )
