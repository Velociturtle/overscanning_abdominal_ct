"""Mask validation helpers shared by abdominal pipeline modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nibabel as nib  # type: ignore
import numpy as np  # type: ignore


def mask_is_valid(
    mask_path: Path,
    reference_shape: tuple[int, ...] | None = None,
    min_voxels: int = 1,
) -> bool:
    """Return True when a mask exists, loads, matches shape, and has voxels."""
    try:
        data = nib.load(mask_path).get_fdata()
    except Exception:
        return False

    if reference_shape and data.shape != reference_shape:
        return False

    return bool(np.count_nonzero(data) >= min_voxels)


def remove_if_exists(paths: Iterable[Path]) -> None:
    """Best-effort cleanup for stale generated masks."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
