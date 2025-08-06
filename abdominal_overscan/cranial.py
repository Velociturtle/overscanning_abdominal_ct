"""Cranial overscan detection via liver and spleen segmentation."""
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
    import torch  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    pd = None  # type: ignore
    nib = None  # type: ignore
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


def run_ts_silent(*args, **kwargs):
    """Run TotalSegmentator silently with warnings filtered."""
    _require(nib, "nibabel")
    # numexpr can raise an error if a library internally requests more threads
    # than allowed by ``NUMEXPR_MAX_THREADS``.  Some environments set this
    # variable to a low value (e.g. 8) which then triggers warnings when
    # TotalSegmentator uses more workers.  To avoid spurious warnings we set a
    # reasonable upper bound that should exceed any threads used here.
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
    """Return ``True`` if basename matches parent folder or file sits in ``NIFTI_DIR``."""
    if p.parent == config.NIFTI_DIR:
        return True
    return nifti_basename(p).lower() == p.parent.name.lower()


def is_ct_vol(p: Path) -> bool:
    """Check if ``p`` is a CT volume for processing."""
    if p.parent.name.startswith("ts_"):
        return False
    if p.name.startswith("ts_"):
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


def ensure_liver_spleen_mask(ct_path: Path) -> Path | None:
    """Generate or return the combined liver/spleen mask."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    out_dir = ct_path.parent / "ts_liver_spleen"
    liver_mask = out_dir / "liver.nii.gz"
    spleen_mask = out_dir / "spleen.nii.gz"
    merged_path = ct_path.parent / "liver_spleen_combined.nii.gz"

    if merged_path.exists():
        return merged_path

    if not (liver_mask.exists() and spleen_mask.exists()):
        out_dir.mkdir(exist_ok=True)
        for dev in ("gpu", "cpu"):
            try:
                run_ts_silent(
                    ct_path,
                    out_dir,
                    roi_subset=["liver", "spleen"],
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
        liver_data = nib.load(liver_mask).get_fdata() > 0
        spleen_data = nib.load(spleen_mask).get_fdata() > 0
    except FileNotFoundError:
        return None

    if config.MULTI_LABEL_MASK:
        combined = np.zeros(liver_data.shape, dtype="uint8")
        combined[liver_data] = 1
        combined[spleen_data] = 2
    else:
        combined = (liver_data | spleen_data).astype("uint8")

    ref_img = nib.load(liver_mask if liver_mask.exists() else spleen_mask)
    nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), merged_path)

    for p in (liver_mask, spleen_mask):
        if p.exists():
            p.unlink()
    if out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()

    return merged_path


def cranial_overscan(ct_path: Path, mask_path: Path) -> tuple[int, int, int, str]:
    """Compute cranial overscan metrics."""
    _require(nib, "nibabel")
    _require(np, "numpy")
    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))
    affine = ct_img.affine
    mask_np = mask_img.get_fdata()

    seg_slices = np.where(mask_np.any(axis=(0, 1)))[0]
    if seg_slices.size == 0:
        raise RuntimeError("combined mask empty")

    z_coords = [(k, float((affine @ [0, 0, k, 1])[2])) for k in seg_slices]

    Z = ct_img.shape[2]
    z_edge0 = float((affine @ [0, 0, 0, 1])[2])
    z_edgeN = float((affine @ [0, 0, Z - 1, 1])[2])
    cranial_edge_z = max(z_edge0, z_edgeN)

    highest_slice, highest_z = min(z_coords, key=lambda t: abs(t[1] - cranial_edge_z))

    labels = mask_np[:, :, highest_slice][mask_np[:, :, highest_slice] > 0].astype(int)
    organ_map = {1: "Liver", 2: "Spleen"}
    organ_top = organ_map.get(int(np.bincount(labels).argmax()), "Unknown")

    cranial_mm = int(round(abs(cranial_edge_z - highest_z)))
    scan_start_mm = int(round(cranial_edge_z))
    organ_z_mm = int(round(highest_z))
    return cranial_mm, organ_z_mm, scan_start_mm, organ_top


def process_single_case(ct_path: Path) -> dict | None:
    """Process a single CT for cranial overscan."""
    _require(nib, "nibabel")
    try:
        mask_path = ct_path.parent / "liver_spleen_combined.nii.gz"
        if not mask_path.exists():
            mask_path = ensure_liver_spleen_mask(ct_path)
            if mask_path is None or not mask_path.exists():
                return None

        cranial_mm, organ_z_mm, scan_start_mm, organ_top = cranial_overscan(ct_path, mask_path)

        return {
            "file_name": ct_path.name,
            "liver_spleen_z_mm": organ_z_mm,
            "scan_start_z_mm": scan_start_mm,
            "cranial_overscan_mm": cranial_mm,
            "top_organ": organ_top,
        }
    except Exception:  # pragma: no cover
        traceback.print_exc(limit=1)
        return None
    finally:
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_batch(num_workers: int = 1) -> None:
    """Run cranial overscan detection over CT volumes."""
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

    cranial_cols = [
        "liver_spleen_z_mm",
        "scan_start_z_mm",
        "cranial_overscan_mm",
        "top_organ",
    ]

    if config.CSV_PATH.exists():
        df_prev = pd.read_csv(config.CSV_PATH, encoding="utf-8-sig")
        for c in cranial_cols:
            if c not in df_prev.columns:
                df_prev[c] = np.nan
    else:
        df_prev = pd.DataFrame(columns=["file_name"] + cranial_cols)

    done_mask = (~df_prev[cranial_cols].isna()).all(axis=1)
    done_set = set(df_prev.loc[done_mask, "file_name"])

    results = []
    t0 = time.time()

    to_process = [p for p in ct_files if p.name not in done_set]
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for row in tqdm(ex.map(process_single_case, to_process),
                           total=len(to_process),
                           desc="Processing cranial overscan", unit="vol"):
                if row:
                    results.append(row)
    else:
        for ct_path in tqdm(to_process, desc="Processing cranial overscan", unit="vol"):
            row = process_single_case(ct_path)
            if row:
                results.append(row)

    if not results:
        print("No new successful cases.")
        return

    df_new = pd.DataFrame(results)
    df_out = df_prev.merge(df_new, on="file_name", how="outer", suffixes=("", "_new"))

    if "top_organ" in df_out.columns:
        df_out["top_organ"] = df_out["top_organ"].astype(object)

    for col in cranial_cols:
        new_col = col + "_new"
        if new_col in df_out.columns:
            if df_out[col].dtype != df_out[new_col].dtype:
                df_out[col] = df_out[col].astype(object)
            mask = df_out[col].isna()
            df_out.loc[mask, col] = df_out.loc[mask, new_col]
            df_out.drop(columns=[new_col], inplace=True)

    df_out.sort_values("file_name").to_csv(config.CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Finished. CSV now contains {len(df_out)} rows")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import argparse

    parser = argparse.ArgumentParser(description="Compute cranial overscan")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="number of parallel workers")
    args = parser.parse_args()

    run_batch(num_workers=max(1, args.workers))
