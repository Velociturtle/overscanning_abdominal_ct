"""Caudal overscan detection using YOLO and femur segmentation."""
from __future__ import annotations

import contextlib
import gc
import io
import os
import time
import traceback
import warnings
from pathlib import Path

from . import config

try:  # optional dependencies
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import nibabel as nib  # type: ignore
    import cv2  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    pd = None  # type: ignore
    nib = None  # type: ignore
    cv2 = None  # type: ignore
    torch = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------

def _require(dep, name: str) -> None:
    if dep is None:
        raise ImportError(f"{name} package is required")


# ---------------------------------------------------------------------------
# core functions
# ---------------------------------------------------------------------------

def run_ts_silent(*args, **kwargs):
    """Run TotalSegmentator with output suppressed and warnings filtered."""
    _require(nib, "nibabel")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 8))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*IProgress not found.*")
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
        try:
            from totalsegmentator.python_api import totalsegmentator  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("totalsegmentator package is required") from exc
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return totalsegmentator(*args, **kwargs)


def preprocess_slice(arr):
    """Normalise a slice and convert to BGR."""
    _require(np, "numpy")
    _require(cv2, "opencv-python")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype("uint8")
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def is_ct_vol(p: Path) -> bool:
    """Return ``True`` if ``p`` looks like a CT volume."""
    if p.parent.name.startswith("ts_"):
        return False
    if p.name.endswith("_combined.nii.gz"):
        return False
    if p.name.startswith(("femur_", "liver_", "spleen_")):
        return False
    return True


def ensure_femur_mask(ct_path: Path) -> Path | None:
    """Ensure femur segmentation exists and return its path."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    out_dir = ct_path.parent / "ts_femur"
    fem_l_path = out_dir / "femur_left.nii.gz"
    fem_r_path = out_dir / "femur_right.nii.gz"
    merged_path = ct_path.parent / "femur_combined.nii.gz"

    if merged_path.exists():
        return merged_path

    if not (fem_l_path.exists() and fem_r_path.exists()):
        out_dir.mkdir(exist_ok=True)
        for dev in ("gpu", "cpu"):
            try:
                run_ts_silent(
                    ct_path,
                    out_dir,
                    roi_subset=["femur_left", "femur_right"],
                    task="total",
                    fast=config.FAST_MODEL,
                    device=dev,
                )
                break
            except Exception:
                continue
        else:
            return None

    try:
        fem_l = nib.load(fem_l_path).get_fdata() > 0
        fem_r = nib.load(fem_r_path).get_fdata() > 0
    except FileNotFoundError:
        return None

    merged = (fem_l | fem_r).astype("uint8")
    if not merged.any():
        return None

    ref = nib.load(fem_l_path if fem_l_path.exists() else fem_r_path)
    nib.save(nib.Nifti1Image(merged, ref.affine, ref.header), merged_path)

    for p in (fem_l_path, fem_r_path):
        if p.exists():
            p.unlink()
    if out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()
    return merged_path


def femur_top_info(ct_path: Path) -> tuple[int, float] | None:
    """Return index and z-coordinate of the highest femur voxel."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    m = ensure_femur_mask(ct_path)
    if m is None:
        return None
    mask = nib.load(str(m))
    mask_np = mask.get_fdata() > 0
    slices = np.where(mask_np.any(axis=(0, 1)))[0]
    if slices.size == 0:
        return None
    affine = mask.affine
    z_coords = [(k, float((affine @ [0, 0, k, 1])[2])) for k in slices]
    return max(z_coords, key=lambda t: t[1])


DEVICE = 0 if torch and torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def find_valid_pubic_slice(ct_path: Path, z_cutoff_mm: float) -> int | None:
    """Return slice index containing the pubic symphysis."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    _require(cv2, "opencv-python")
    model = config.get_yolo_model()

    ct = nib.load(str(ct_path))
    affine = ct.affine
    vol = ct.get_fdata()
    _, W, Z = vol.shape

    best_conf, best_slice = -1.0, None
    for z in range(Z):
        if float((affine @ [0, 0, z, 1])[2]) > z_cutoff_mm:
            continue
        img = preprocess_slice(vol[:, :, z])
        res = model.predict(img, conf=config.FINAL_CONF, device=DEVICE, save=False, verbose=False)[0]
        for b in sorted(res.boxes, key=lambda bb: float(bb.conf), reverse=True):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if vol[int((y1+y2)/2), int((x1+x2)/2), z] <= config.BACKGROUND_HU:
                continue
            cx = int((x1+x2)/2)
            if abs(cx - W // 2) > 0.20 * W:
                continue
            win = vol[
                max(0, int((y1+y2)/2) - 10):min(vol.shape[0], int((y1+y2)/2) + 10),
                max(0, cx - 10):min(W, cx + 10),
                z,
            ]
            if win.mean() < 150:
                continue
            conf = float(b.conf)
            if conf > best_conf:
                best_conf, best_slice = conf, z
            break
    return best_slice


def process_single_case(ct_path: Path) -> dict | None:
    """Process a single CT volume and return result dictionary."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    try:
        fem_data = femur_top_info(ct_path)
        if fem_data:
            fem_slice, fem_top_z = fem_data
            z_cut = fem_top_z
        else:
            fem_slice, fem_top_z = None, float("nan")
            z_cut = float("inf")

        pubic_slice = find_valid_pubic_slice(ct_path, z_cut)
        if pubic_slice is None and fem_slice is not None:
            pubic_slice, source_label = fem_slice, "FemurFallback"
        elif pubic_slice is None:
            return None
        else:
            source_label = "YOLO" if not np.isnan(fem_top_z) else "YOLO_NoFemur"

        ct_img = nib.load(str(ct_path))
        affine = ct_img.affine
        Z = ct_img.shape[2]
        pubic_z = float((affine @ [0, 0, pubic_slice, 1])[2])
        end_z = min(float((affine @ [0, 0, k, 1])[2]) for k in range(Z))
        caudal = abs(end_z - pubic_z)

        return {
            "file_name": ct_path.name,
            "pubic_z_mm": int(round(pubic_z)),
            "scan_end_z_mm": int(round(end_z)),
            "caudal_overscan_mm": int(round(caudal)),
            "femur_top_z_mm": int(round(fem_top_z)) if not np.isnan(fem_top_z) else float("nan"),
            "pubic_source": source_label,
        }
    except Exception:  # pragma: no cover
        traceback.print_exc(limit=1)
        return None
    finally:
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_batch() -> None:
    """Run caudal overscan detection over available CT volumes."""
    _require(pd, "pandas")
    patterns = ("*.nii.gz", "*.nii")
    ct_files = sorted({
        p
        for pat in patterns
        for p in config.NIFTI_DIR.rglob(pat)
        if p.is_file() and is_ct_vol(p)
    })

    if not ct_files:
        print(f"No matching NIfTI files in {config.NIFTI_DIR}")
        return

    caudal_cols = [
        "pubic_z_mm",
        "scan_end_z_mm",
        "caudal_overscan_mm",
        "femur_top_z_mm",
        "pubic_source",
    ]

    if config.CSV_PATH.exists():
        df_prev = pd.read_csv(config.CSV_PATH, encoding="utf-8-sig")
        for c in caudal_cols:
            if c not in df_prev.columns:
                df_prev[c] = np.nan
    else:
        df_prev = pd.DataFrame(columns=["file_name"] + caudal_cols)

    done_mask = (~df_prev[caudal_cols].isna()).all(axis=1)
    done_set = set(df_prev.loc[done_mask, "file_name"])

    results = []
    t0 = time.time()

    iterator = tqdm(ct_files, desc="Processing caudal overscan", unit="vol") if tqdm else ct_files
    for ct_path in iterator:
        if ct_path.name in done_set:
            continue
        row = process_single_case(ct_path)
        if row:
            results.append(row)

    if not results:
        print("No new successful cases.")
        return

    df_new = pd.DataFrame(results)
    df_out = df_prev.merge(df_new, on="file_name", how="outer", suffixes=("", "_new"))

    for col in caudal_cols:
        new_col = col + "_new"
        if new_col in df_out.columns:
            mask = df_out[col].isna()
            df_out.loc[mask, col] = df_out.loc[mask, new_col]
            df_out.drop(columns=[new_col], inplace=True)

    df_out.sort_values("file_name").to_csv(config.CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Finished. CSV now contains {len(df_out)} rows")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    run_batch()
