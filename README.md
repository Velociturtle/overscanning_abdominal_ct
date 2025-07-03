# Overscanning Abdominal CT

This repository contains notebooks and sample data for calculating cranial and caudal overscanning on abdominal CT volumes. Overscanning is measured relative to the pubic symphysis (caudal) and to liver/spleen segmentation (cranial). Detection uses YOLOv11 and segmentation uses TotalSegmentator.

The project is organized around a main notebook, **`overscanning_calculator.ipynb`**, which processes a folder of NIfTI volumes and incrementally updates `overscanning_results.csv` with overscanning metrics. Additional notebooks demonstrate data preparation, model training and result validation.

## Contents

- `Patient_CTs/` – Example NIfTI volumes with ground‑truth pubic symphysis masks.
- `overscanning_calculator.ipynb` – Computes overscanning metrics for all scans in a folder.
- `final_infer_pubic_symphysis.ipynb` – Single‑scan inference demo using YOLO.
- `yolo.ipynb` – Generates a YOLO training dataset and shows training/validation routines.
- `finder.ipynb`, `reorient_niftis.ipynb` – Utilities for converting DICOM to NIfTI.
- `mp4_validator.ipynb` – Renders scrolling MP4s to visually validate CSV results.

## Installation

Create a Python environment (3.9+ recommended) and install the dependencies listed in `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` should be installed with CUDA support if GPU processing is desired for YOLO and TotalSegmentator.

## Usage

1. **Prepare your data**
   - Organise CT scans as NIfTI files under a single directory. Each patient folder should contain one `.nii` or `.nii.gz` volume.
   - For YOLO training you will also need `pubic_mask.nii.gz` masks in each folder (see `yolo.ipynb`).

2. **Edit the paths in `overscanning_calculator.ipynb`**
   - `MODEL_PATH` – path to your trained YOLO model weights (`.pt`).
   - `NIFTI_DIR` – root directory containing patient subfolders with NIfTI volumes.
   - `CSV_PATH` – output CSV path (defaults to `NIFTI_DIR/overscanning_results.csv`).

   Flags controlling behaviour:
   - `DISPLAY_DETECTION` – show the best pubic‑symphysis detection slice.
   - `FAST_MODEL` – pass `--fast` to TotalSegmentator for quicker but less accurate masks.
   - `MULTI_LABEL_MASK` – store liver and spleen in a combined mask as labels `1` and `2`.

3. **Run the notebook**
   Execute all cells in `overscanning_calculator.ipynb`. Processing is incremental; existing rows in the CSV will be updated or appended. Femur, liver and spleen masks are generated on the fly using TotalSegmentator (GPU is tried first with CPU fallback).

4. **Validate results** (optional)
   - `mp4_validator.ipynb` builds scrolling MP4s showing the detected landmarks and segmentations for manual review.
   - `final_infer_pubic_symphysis.ipynb` demonstrates running YOLO on a single volume and visualising the highest‑confidence detection.

## File Requirements

- Input CTs must be NIfTI files (`.nii` or `.nii.gz`). Ensure volumes are correctly oriented head‑to‑feet. Utilities for DICOM conversion are provided in `finder.ipynb` and `reorient_niftis.ipynb` (requires `dcm2niix`).
- YOLO expects images in BGR format; the notebooks handle normalisation automatically.

## Methodology Notes

Cranial overscanning is computed using the highest axial slice containing liver or spleen voxels. Caudal overscanning uses a YOLO detection of the pubic symphysis with femur segmentation as a fallback when YOLO fails. Distances are reported in millimetres, derived from the NIfTI affine.

The CSV columns are:

- `file_name` – scan filename
- `pubic_z_mm` – world z‑coordinate of detected pubic symphysis
- `scan_end_z_mm` – caudal edge of scan
- `caudal_overscan_mm` – distance from pubic symphysis to scan end
- `femur_top_z_mm` – cranial‑most femur coordinate
- `pubic_source` – `YOLO`, `YOLO_NoFemur`, or `FemurFallback`
- `liver_spleen_z_mm` – highest liver/spleen voxel
- `scan_start_z_mm` – cranial edge of scan
- `cranial_overscan_mm` – distance from cranial edge to top organ
- `top_organ` – `Liver`, `Spleen`, or `Unknown`

## License

This project is released under the MIT License – see `LICENSE` for details.

