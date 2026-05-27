"""2D rotated bounding-box geometry for Stage 7 open-loop collision rate.

nuScenes ``sample_annotation`` records give each agent (other vehicles,
pedestrians) a 3-D oriented bounding box: ``translation`` ``[x, y, z]``,
``size`` ``[w, l, h]`` (width across, length along forward, height), and
``rotation`` as a quaternion. For an open-loop collision check we only need
the BEV (top-down) projection: a 2-D rectangle of size ``length × width``
centred at ``(cx, cy)``, rotated by ``yaw`` (the heading of the box's +x
axis in the global frame).

The convention here is **length along the box's local +x, width along +y**,
matching the nuScenes box convention used by the devkit's ``Box`` class.

This module is pure-python so it can be unit-tested without nuscenes-devkit.
"""

from __future__ import annotations

import math
from typing import Sequence


def point_in_rotated_bbox(
    point: Sequence[float],
    center: Sequence[float],
    length: float,
    width: float,
    yaw: float,
) -> bool:
    """True if the 2-D point falls inside the rotated rectangle.

    - ``point``  ``(px, py)`` in the global frame.
    - ``center`` ``(cx, cy)`` of the box, global frame.
    - ``length`` size along the box's local +x axis (the heading direction).
    - ``width``  size along the box's local +y axis.
    - ``yaw``    radians; angle from global +x to the box's local +x.

    Implementation: translate the point so the box centre is the origin, then
    rotate by ``-yaw`` to bring it into the box-local frame; the box is then
    axis-aligned and the inside test is two abs comparisons.
    """
    px, py = float(point[0]), float(point[1])
    cx, cy = float(center[0]), float(center[1])
    dx, dy = px - cx, py - cy
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    # Rotation by -yaw, applied as 2-D rotation matrix to (dx, dy).
    rx = dx * cos_y + dy * sin_y
    ry = -dx * sin_y + dy * cos_y
    return abs(rx) <= length / 2.0 and abs(ry) <= width / 2.0


def quick_radius_overlap(
    point: Sequence[float],
    center: Sequence[float],
    length: float,
    width: float,
) -> bool:
    """Cheap circumscribed-circle early-out before the rotated-bbox check.

    A point inside the rotated rectangle is always inside the circle of
    radius ``sqrt((l/2)^2 + (w/2)^2)`` around the centre. Caller can skip the
    rotation/abs test for points far away.
    """
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    r2 = (length * length + width * width) / 4.0
    return dx * dx + dy * dy <= r2
