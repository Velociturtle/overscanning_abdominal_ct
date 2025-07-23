# BLOCK 3 – Liver & spleen segmentation → cranial overscan

def run_ts_silent(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return totalsegmentator(*args, **kwargs)

def nifti_basename(p: Path) -> str:
    n = p.name
    low = n.lower()
    if low.endswith(".nii.gz"):
        return n[:-7]
    if low.endswith(".nii"):
        return n[:-4]
    return p.stem

def file_matches_parent(p: Path) -> bool:
    return nifti_basename(p).lower() == p.parent.name.lower()

def is_ct_vol(p: Path) -> bool:
    if p.parent.name.startswith("ts_"):
        return False
    if p.name.startswith("ts_"):
        return False
    if p.name.endswith("_combined.nii.gz"):
        return False
    if p.name.startswith(("femur_", "liver_", "spleen_")):
        return False
    return True

def ensure_liver_spleen_mask(ct_path: Path) -> Path | None:
    out_dir      = ct_path.parent / "ts_liver_spleen"
    liver_mask   = out_dir / "liver.nii.gz"
    spleen_mask  = out_dir / "spleen.nii.gz"
    merged_path  = ct_path.parent / "liver_spleen_combined.nii.gz"

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
                    fast=FAST_MODEL,
                    device=dev,
                )
                break
            except Exception as e:
                print(f"TS({dev}) {ct_path.name}: {e}")
        else:
            return None

    try:
        liver_data  = nib.load(liver_mask ).get_fdata() > 0
        spleen_data = nib.load(spleen_mask).get_fdata() > 0
    except FileNotFoundError:
        return None

    if MULTI_LABEL_MASK:
        combined = np.zeros(liver_data.shape, np.uint8)
        combined[liver_data]  = 1
        combined[spleen_data] = 2
    else:
        combined = (liver_data | spleen_data).astype(np.uint8)

    ref_img = nib.load(liver_mask if liver_mask.exists() else spleen_mask)
    nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), merged_path)

    for p in (liver_mask, spleen_mask):
        if p.exists():
            p.unlink()
    if out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()

    return merged_path

def cranial_overscan(ct_path: Path, mask_path: Path) -> tuple[int, int, int, str]:
    ct_img   = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))
    affine   = ct_img.affine
    mask_np  = mask_img.get_fdata()

    seg_slices = np.where(mask_np.any(axis=(0, 1)))[0]
    if seg_slices.size == 0:
        raise RuntimeError("combined mask empty")

    z_coords = [(k, float((affine @ [0, 0, k, 1])[2])) for k in seg_slices]

    Z = ct_img.shape[2]
    z_edge0 = float((affine @ [0, 0,      0, 1])[2])
    z_edgeN = float((affine @ [0, 0, Z - 1, 1])[2])
    cranial_edge_z = max(z_edge0, z_edgeN)

    highest_slice, highest_z = min(z_coords, key=lambda t: abs(t[1] - cranial_edge_z))

    labels     = mask_np[:, :, highest_slice][mask_np[:, :, highest_slice] > 0].astype(int)
    organ_map  = {1: "Liver", 2: "Spleen"}
    organ_top  = organ_map.get(int(np.bincount(labels).argmax()), "Unknown")

    cranial_mm    = int(round(abs(cranial_edge_z - highest_z)))
    scan_start_mm = int(round(cranial_edge_z))
    organ_z_mm    = int(round(highest_z))
    return cranial_mm, organ_z_mm, scan_start_mm, organ_top

def process_single_case(ct_path: Path) -> dict | None:
    try:
        mask_path = ct_path.parent / "liver_spleen_combined.nii.gz"
        if not mask_path.exists():
            mask_path = ensure_liver_spleen_mask(ct_path)
            if mask_path is None or not mask_path.exists():
                return None

        cranial_mm, organ_z_mm, scan_start_mm, organ_top = cranial_overscan(ct_path, mask_path)

        return {
            "file_name"          : ct_path.name,
            "liver_spleen_z_mm"  : organ_z_mm,
            "scan_start_z_mm"    : scan_start_mm,
            "cranial_overscan_mm": cranial_mm,
            "top_organ"          : organ_top,
        }

    except Exception:
        traceback.print_exc(limit=1)
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_batch():
    patterns = ("*.nii.gz", "*.nii")
    ct_files = sorted({
        p for pat in patterns
        for p in NIFTI_DIR.rglob(pat)
        if p.is_file() and file_matches_parent(p) and is_ct_vol(p)
    })

    if not ct_files:
        print(f"No matching NIfTI files in {NIFTI_DIR}")
        return

    cranial_cols = ["liver_spleen_z_mm", "scan_start_z_mm", "cranial_overscan_mm", "top_organ"]

    if CSV_PATH.exists():
        df_prev = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        for c in cranial_cols:
            if c not in df_prev.columns:
                df_prev[c] = np.nan
    else:
        df_prev = pd.DataFrame(columns=["file_name"] + cranial_cols)

    done_mask = (~df_prev[cranial_cols].isna()).all(axis=1)
    done_set  = set(df_prev.loc[done_mask, "file_name"])

    results = []
    t0 = time.time()

    for ct_path in tqdm(ct_files, desc="Processing cranial overscan", unit="vol"):
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

    df_out.sort_values("file_name").to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Finished. CSV now contains {len(df_out)} rows")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    run_batch()