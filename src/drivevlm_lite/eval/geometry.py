"""Geometry helpers for Stage 6 (off-road / drivable-area rate).

The model predicts waypoints in the **current ego frame** (Impromptu
convention: +x forward, +y left, metres relative to the ego at t=0). To check
them against the nuScenes HD map we must lift them into the **global** map
frame, which needs the ego's global pose at the keyframe: a translation and a
yaw.

nuScenes stores ego pose as ``translation`` ``[x, y, z]`` and ``rotation`` as a
quaternion ``[w, x, y, z]``. Only the yaw (rotation about the vertical z axis)
matters for a planar BEV map query.

This module is pure-python (no numpy, no nuscenes) so it can be unit-tested in
isolation; the nuScenes I/O lives in ``scripts/eval/eval_offroad.py``.
"""

from __future__ import annotations

import math
from typing import Sequence


def yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Return the yaw (rotation about the global z axis), in radians, of a
    nuScenes ego-pose quaternion ``[w, x, y, z]``.

    Uses the standard ZYX-yaw extraction; for a near-planar driving pose this is
    the heading of the vehicle's +x (forward) axis in the global frame.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def ego_to_global(
    point: Sequence[float],
    translation_xy: Sequence[float],
    yaw: float,
) -> tuple[float, float]:
    """Map a single ego-frame ``(x_forward, y_left)`` point to global ``(X, Y)``.

    The ego frame is rotated by ``yaw`` and offset by ``translation_xy`` in the
    global frame::

        X = Tx + x * cos(yaw) - y * sin(yaw)
        Y = Ty + x * sin(yaw) + y * cos(yaw)
    """
    px, py = float(point[0]), float(point[1])
    tx, ty = float(translation_xy[0]), float(translation_xy[1])
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    gx = tx + px * cos_y - py * sin_y
    gy = ty + px * sin_y + py * cos_y
    return gx, gy


def ego_to_global_path(
    points: Sequence[Sequence[float]],
    translation_xy: Sequence[float],
    yaw: float,
) -> list[tuple[float, float]]:
    """Vectorised :func:`ego_to_global` over a list of waypoints."""
    return [ego_to_global(p, translation_xy, yaw) for p in points]
