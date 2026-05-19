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
from .failure_reporting import write_failed_cases
from .mask_utils import mask_is_valid, remove_if_exists

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

from concurrent.futures import ProcessPoolExecutor


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
    # Ensure numexpr can utilise enough threads.  Some systems set
    # ``NUMEXPR_MAX_THREADS`` to a small value which leads to warnings when
    # TotalSegmentator internally requests more threads.  Setting a high upper
    # bound avoids these messages while still respecting the system's CPU count.
    max_threads = max(os.cpu_count() or 8, 64)
    os.environ["NUMEXPR_MAX_THREADS"] = str(max_threads)
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


def nifti_basename(p: Path) -> str:
    """Return basename without ``.nii`` or ``.nii.gz``."""
    n = p.name
    low = n.lower()
    if low.endswith(".nii.gz"):
        return n[:-7]
    if low.endswith(".nii"):
        return n[:-4]
    return p.stem


def file_matches_parent(p: Path) -> bool:
    """Return ``True`` if basename matches parent or file resides in ``NIFTI_DIR``."""
    if p.parent == config.NIFTI_DIR:
        return True
    return nifti_basename(p).lower() == p.parent.name.lower()


def is_ct_vol(p: Path) -> bool:
    """Return ``True`` if ``p`` looks like a CT volume."""
    if p.parent.name.startswith("ts_"):
        return False
    if p.name.endswith("_combined.nii.gz"):
        return False
    if p.name.startswith(("femur_", "liver_", "spleen_")):
        return False
    return True


def ensure_patient_dirs() -> None:
    """Move CT volumes into per-patient folders under :data:`config.NIFTI_DIR`."""
    patterns = ("*.nii.gz", "*.nii")
    ct_files = [
        p
        for pat in patterns
        for p in config.NIFTI_DIR.rglob(pat)
        if p.is_file() and is_ct_vol(p)
    ]

    for p in ct_files:
        folder = config.NIFTI_DIR / nifti_basename(p)
        if p.parent != folder:
            folder.mkdir(exist_ok=True)
            target = folder / p.name
            if not target.exists():
                p.replace(target)
            try:
                if p.parent != config.NIFTI_DIR and not any(p.parent.iterdir()):
                    p.parent.rmdir()
            except Exception:
                pass


def ensure_femur_mask(ct_path: Path) -> Path | None:
    """Ensure femur segmentation exists and return its path."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    out_dir = ct_path.parent / "ts_femur"
    fem_l_path = out_dir / "femur_left.nii.gz"
    fem_r_path = out_dir / "femur_right.nii.gz"
    merged_path = ct_path.parent / "femur_combined.nii.gz"

    ct_shape = nib.load(ct_path).shape
    if mask_is_valid(merged_path, ct_shape, min_voxels=10):
        return merged_path

    remove_if_exists([merged_path])

    if not (mask_is_valid(fem_l_path, ct_shape) and mask_is_valid(fem_r_path, ct_shape)):
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

    masks = []
    for path in (fem_l_path, fem_r_path):
        if mask_is_valid(path, ct_shape):
            masks.append(nib.load(path).get_fdata() > 0)
    if not masks:
        return None

    merged = np.logical_or.reduce(masks).astype("uint8")
    if not merged.any():
        return None

    ref = nib.load(fem_l_path if mask_is_valid(fem_l_path, ct_shape) else fem_r_path)
    nib.save(nib.Nifti1Image(merged, ref.affine, ref.header), merged_path)

    remove_if_exists([fem_l_path, fem_r_path])
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


def _to_numpy(value):
    """Convert NumPy or torch-like values to a host NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _box_conf(box) -> float:
    return float(_to_numpy(box.conf).reshape(-1)[0])


def _box_xyxy(box) -> list[float]:
    return _to_numpy(box.xyxy[0]).reshape(-1).tolist()


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
    target_z_val = float("inf")

    def _to_world_z(z_idx: int) -> float:
        return float((affine @ [0, 0, z_idx, 1])[2])

    def _inferior_neighbor(z_idx: int) -> int | None:
        """Return the neighbouring slice that lies more caudally (smaller world z)."""
        opts = []
        if z_idx - 1 >= 0:
            opts.append((z_idx - 1, _to_world_z(z_idx - 1)))
        if z_idx + 1 < Z:
            opts.append((z_idx + 1, _to_world_z(z_idx + 1)))
        if not opts:
            return None
        inferior = min(opts, key=lambda t: t[1])
        if inferior[1] < _to_world_z(z_idx):
            return inferior[0]
        return None

    for z in range(Z):
        z_mm = _to_world_z(z)
        if z_mm > z_cutoff_mm:
            continue
        img = preprocess_slice(vol[:, :, z])
        res = model.predict(img, conf=config.FINAL_CONF, device=DEVICE, save=False, verbose=False)[0]
        for b in sorted(res.boxes, key=_box_conf, reverse=True):
            x1, y1, x2, y2 = _box_xyxy(b)
            cy = int((y1 + y2) / 2)
            cx = int((x1 + x2) / 2)
            if vol[cy, cx, z] <= config.BACKGROUND_HU:
                continue
            if abs(cx - W // 2) > 0.20 * W:
                continue
            win = vol[
                max(0, cy - 10):min(vol.shape[0], cy + 10),
                max(0, cx - 10):min(W, cx + 10),
                z,
            ]
            if win.mean() < 150:
                continue
            bone_frac = (win > config.BONE_HU_THRESHOLD).mean()
            if bone_frac < config.BONE_MIN_FRACTION:
                continue

            # walk caudally while bone remains to align with inferior margin
            refined_z = z
            for _ in range(config.PUBIC_MISS_TOLERANCE):
                next_z = _inferior_neighbor(refined_z)
                if next_z is None:
                    break
                z_slice = vol[:, :, next_z]
                local = z_slice[
                    max(0, cy - 6):min(z_slice.shape[0], cy + 6),
                    max(0, cx - 6):min(z_slice.shape[1], cx + 6),
                ]
                if (local > config.BONE_HU_THRESHOLD).mean() >= config.BONE_MIN_FRACTION / 2:
                    refined_z = next_z
                else:
                    break

            conf = _box_conf(b)
            refined_mm = _to_world_z(refined_z)
            if refined_mm < target_z_val or (refined_mm == target_z_val and conf > best_conf):
                target_z_val = refined_mm
                best_conf, best_slice = conf, refined_z
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


def run_batch(num_workers: int = 1) -> None:
    """Run caudal overscan detection over available CT volumes."""
    _require(pd, "pandas")
    ensure_patient_dirs()
    patterns = ("*.nii.gz", "*.nii")
    ct_files = sorted({
        p
        for pat in patterns
        for p in config.NIFTI_DIR.rglob(pat)
        if p.is_file() and file_matches_parent(p) and is_ct_vol(p)
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
    failures = []
    t0 = time.time()

    to_process = [p for p in ct_files if p.name not in done_set]
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for ct_path, row in zip(to_process, tqdm(ex.map(process_single_case, to_process),
                           total=len(to_process),
                           desc="Processing caudal overscan", unit="vol")):
                if row:
                    results.append(row)
                else:
                    failures.append({
                        "file_name": ct_path.name,
                        "reason": "no valid pubic symphysis or femur fallback",
                    })
    else:
        for ct_path in tqdm(to_process, desc="Processing caudal overscan", unit="vol"):
            row = process_single_case(ct_path)
            if row:
                results.append(row)
            else:
                failures.append({
                    "file_name": ct_path.name,
                    "reason": "no valid pubic symphysis or femur fallback",
                })

    failed_report = write_failed_cases(config.CSV_PATH, "caudal", failures)
    if failures:
        print(f"{len(failures)} caudal case(s) written to {failed_report}")

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
    import argparse

    parser = argparse.ArgumentParser(description="Compute caudal overscan")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="number of parallel workers")
    args = parser.parse_args()

    run_batch(num_workers=max(1, args.workers))