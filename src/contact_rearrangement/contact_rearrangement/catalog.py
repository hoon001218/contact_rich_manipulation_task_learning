"""Simulator-independent asset catalog, upright poses, and reusable pool limits."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


NUCLEUS_ROOT = os.environ.get("CONTACT_REARRANGEMENT_NUCLEUS_ROOT", "omniverse://192.168.0.13").rstrip("/")
DEFAULT_SHELF_USD_PATH = f"{NUCLEUS_ROOT}/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd"
SHELF_POSITION = (-0.7, 0.0, 0.0)


@dataclass(frozen=True)
class ObjectSpec:
    """An authored physics asset and its canonical upright orientation (wxyz)."""

    name: str
    relative_usd_path: str
    upright_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    asset_name: str = ""

    @property
    def usd_path(self) -> str:
        return os.environ.get(
            f"CONTACT_REARRANGEMENT_{(self.asset_name or self.name).upper()}_USD_PATH",
            f"{NUCLEUS_ROOT}/{self.relative_usd_path}",
        )


# Axis_Aligned_Physics currently contains these four authored physics assets.
# Rendered labels and mustard's cap show that local -Y is the top of all four.
# Rotate -Y to world +Z before applying a sampled world-Z yaw.
_YCB_UPRIGHT = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
_YCB_FOLDER = "NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics"
ASSET_SPECS = (
    ObjectSpec("bottle_1", "Library/Shelf/Objects/Bottle_6/Bottle_6.usd"),
    ObjectSpec("cup_1", "Library/Shelf/Objects/Collected_Cup_1/Cup_1.usd"),
    ObjectSpec("cup_2", "Library/Shelf/Objects/Collected_Cup_4/Cup_4.usd"),
    ObjectSpec("mug_1", "Library/Shelf/Objects/Collected_Mug_2/Mug_2.usd"),
    ObjectSpec("mug_2", "Library/Shelf/Objects/Collected_Mug_3/Mug_3.usd"),
    ObjectSpec("can_1", "Library/Shelf/Objects/Can_6/Can_6.usd"),
    ObjectSpec("ycb_003_cracker_box", f"{_YCB_FOLDER}/003_cracker_box.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_004_sugar_box", f"{_YCB_FOLDER}/004_sugar_box.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_005_tomato_soup_can", f"{_YCB_FOLDER}/005_tomato_soup_can.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_006_mustard_bottle", f"{_YCB_FOLDER}/006_mustard_bottle.usd", _YCB_UPRIGHT),
)
# Load a reusable pool so non-Target assets can occupy every surrounding slot,
# identities can change on reset without rebuilding USD or PhysX topology.
MAX_ACTIVE_OBJECTS = 40
OBJECT_COPIES_PER_ASSET = 10
OBJECT_SPECS = tuple(
    ObjectSpec(
        name=spec.name if copy_index == 0 else f"{spec.name}_copy_{copy_index + 1:02d}",
        relative_usd_path=spec.relative_usd_path,
        upright_quat=spec.upright_quat,
        asset_name=spec.name,
    )
    for spec in ASSET_SPECS
    for copy_index in range(OBJECT_COPIES_PER_ASSET)
)
OBJECT_NAMES = tuple(spec.name for spec in OBJECT_SPECS)


def parking_position(index: int) -> tuple[float, float, float]:
    """Isolate unused bodies above the scene, clear of floor/shelf and each other."""
    return (-0.64 + 0.32 * (index % 5), -0.64 + 0.32 * ((index // 5) % 5), 4.0 + 0.4 * (index // 25))
