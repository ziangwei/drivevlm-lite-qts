"""Small camera-path utility used by VLA ablations.

The only thing v1 needs from the old QTS code is the ability to recognise
which nuScenes camera a given image path corresponds to. Anything more
complex (query-conditioned camera selection for DriveLM VQA) is inlined in
the few VQA scripts that still need it.
"""

from __future__ import annotations

import re
from pathlib import Path


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_RE = re.compile(r"CAM_(?:FRONT_RIGHT|FRONT_LEFT|BACK_RIGHT|BACK_LEFT|FRONT|BACK)")


def camera_name_from_path(path: str | Path) -> str | None:
    """Return ``CAM_FRONT`` / ``CAM_BACK_LEFT`` / ... if the path contains a
    nuScenes camera tag, otherwise ``None``."""
    match = CAMERA_RE.search(str(path).upper())
    return match.group(0) if match else None
