# BLOCK 4 – Generate preview MP4s of mid-coronal slices (ABDOMEN)

OUT_DIR = NIFTI_DIR.parent / "trauma_overscan_videos_test"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

print(f"{len(df)} CT volumes found")


def build_mp4_abdomen(scan_id: str,
                      pubic_z_mm: float | None,
                      organ_z_mm: float | None,
                      organ_label: str | None,
                      pubic_source: str | None,
                      fps: int = 48,
                      slice_span: int = 100):
    folder = NIFTI_DIR / scan_id
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found")

    candidates = [f for f in folder.glob("*.nii*")
                  if f.name.startswith(scan_id)
                  and "_combined" not in f.name.lower()
                  and not f.name.startswith("ts_")]
    if not candidates:
        raise FileNotFoundError("CT volume not found")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple CT volumes: {[c.name for c in candidates]}")

    ct_path = candidates[0]

    fem_path = None
    try: fem_path = ensure_femur_mask(ct_path)
    except Exception: pass
    org_path = None
    try: org_path = ensure_liver_spleen_mask(ct_path)
    except Exception: pass

    mp4_path = OUT_DIR / f"{scan_id}.mp4"
    if mp4_path.exists():
        mp4_path.unlink()

    ct_img  = nib.load(str(ct_path))
    vol     = ct_img.get_fdata()
    fem_msk = nib.load(str(fem_path)).get_fdata() > 0 if fem_path and fem_path.exists() else None
    org_msk = nib.load(str(org_path)).get_fdata() > 0 if org_path and org_path.exists() else None
    vx, _, vz = ct_img.header.get_zooms()[:3]

    _, Y, Z = vol.shape
    z_world = np.flip((ct_img.affine @ np.vstack([
        np.zeros(Z), np.zeros(Z), np.arange(Z), np.ones(Z)
    ]))[2])

    pubic_row = int(np.argmin(np.abs(z_world - pubic_z_mm))) if (pubic_z_mm is not None and np.isfinite(pubic_z_mm)) else None
    organ_row = int(np.argmin(np.abs(z_world - organ_z_mm))) if (organ_z_mm is not None and np.isfinite(organ_z_mm)) else None

    mid_y, half = Y // 2, slice_span // 2
    start_y, end_y = max(0, mid_y - half), min(Y - 1, mid_y + half)
    y_stretch = vz / vx

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    red, green = (0, 0, 255), (0, 255, 0)
    femur_color = (255, 0, 0)
    organ_color = (0, 255, 255)
    ALPHA = 0.35
    landmark_name = "Femur" if (pubic_source == "FemurFallback") else "Pubic Symphysis"

    def render(y_idx: int):
        base = np.clip((np.flipud(vol[:, y_idx, :].T) + 200) / 500, 0, 1) * 255
        base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        overlay = np.zeros_like(base, dtype=np.uint8)
        if fem_msk is not None:
            mask_fem = np.flipud(fem_msk[:, y_idx, :].T)
            overlay[mask_fem] = femur_color
        if org_msk is not None:
            mask_org = np.flipud(org_msk[:, y_idx, :].T)
            overlay[mask_org] = organ_color

        img = cv2.addWeighted(base, 1.0, overlay, ALPHA, 0)

        if y_stretch != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w, int(h * y_stretch)), interpolation=cv2.INTER_CUBIC)

        h, w = img.shape[:2]
        if pubic_row is not None:
            y_line = int(pubic_row * y_stretch)
            cv2.line(img, (0, y_line), (w - 1, y_line), red, 2)
            cv2.putText(img, f"{landmark_name} z={pubic_z_mm:.0f} mm",
                        (10, max(20, y_line - 6)), font, fs, red, th, cv2.LINE_AA)

        if organ_row is not None and organ_label:
            y_line = int(organ_row * y_stretch)
            cv2.line(img, (0, y_line), (w - 1, y_line), green, 2)
            cv2.putText(img, f"{organ_label} z={organ_z_mm:.0f} mm",
                        (10, min(h - 10, y_line + 20)), font, fs, green, th, cv2.LINE_AA)

        cv2.putText(img, f"{scan_id} | y={y_idx}",
                    (10, h - 10), font, fs, (255, 255, 0), th, cv2.LINE_AA)
        return img

    first = render(start_y)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vw.write(first)
    for y in range(start_y + 1, end_y + 1):
        vw.write(render(y))
    vw.release()

ok = failed = 0
for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing (abdomen MP4s)", unit="scan"):
    sid = r["file_name"].split(".nii")[0]
    try:
        build_mp4_abdomen(
            sid,
            float(r["pubic_z_mm"])        if "pubic_z_mm"        in r and pd.notna(r["pubic_z_mm"])        else None,
            float(r["liver_spleen_z_mm"]) if "liver_spleen_z_mm" in r and pd.notna(r["liver_spleen_z_mm"]) else None,
            str(r["top_organ"]).strip()   if "top_organ"         in r and pd.notna(r["top_organ"])         else None,
            str(r["pubic_source"]).strip()if "pubic_source"      in r and pd.notna(r["pubic_source"])      else None,
        )
        ok += 1
    except Exception as e:
        failed += 1
        tqdm.write(f"✗ {sid}: {e}")
        traceback.print_exc()

print(f"Finished. {ok} MP4s created and saved to {OUT_DIR}")