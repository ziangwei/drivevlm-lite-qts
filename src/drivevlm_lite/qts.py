"""Deprecated shim.

This module was split during the v1 cleanup:
- Rule-based camera selection lives in :mod:`drivevlm_lite.camera_selection`.
- The unused learned token selector is parked in
  :mod:`drivevlm_lite.experimental.qts_neural`.

Imports from ``drivevlm_lite.qts`` continue to work via this shim, but new
code should import from the new locations. This file can be removed once
no caller still references it.
"""

from __future__ import annotations

import warnings

from drivevlm_lite.camera_selection import (
    CAMERA_NAMES,
    CAMERA_RE,
    ImageSelection,
    camera_name_from_path,
    infer_query_cameras,
    select_images_by_query,
)

warnings.warn(
    "drivevlm_lite.qts is deprecated; import from drivevlm_lite.camera_selection instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CAMERA_NAMES",
    "CAMERA_RE",
    "ImageSelection",
    "camera_name_from_path",
    "infer_query_cameras",
    "select_images_by_query",
]
