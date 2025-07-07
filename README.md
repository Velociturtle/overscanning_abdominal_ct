# Overscanning Abdominal CT

This repository provides a single Jupyter notebook for calculating cranial and caudal overscanning on abdominal CT volumes. Overscanning is measured relative to the pubic symphysis (caudal) and to liver/spleen segmentation (cranial). Detection uses YOLOv11 and segmentation uses TotalSegmentator.

The main workflow lives in **`overscanning_calculator.ipynb`** which processes a folder of NIfTI volumes and incrementally updates `overscanning_results.csv`. Later sections of the notebook can optionally render MP4 previews, write a summary statistics CSV and save scatter, box and bar plots of overscanning metrics.

## Contents

- `overscanning_calculator.ipynb` – full pipeline for overscanning metrics, MP4 previews and CSV summaries.
- `YOLO/` – example training outputs and model weights for the pubic symphysis detector.

## Installation

Create a Python environment (Python 3.9+ recommended) and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` should be installed with CUDA support if GPU processing is desired for YOLO and TotalSegmentator.

## Usage

1. **Prepare your data** – organise CT scans as NIfTI files under a single directory. Each patient folder should contain one `.nii` or `.nii.gz` volume.
2. **Edit the paths in `overscanning_calculator.ipynb`**
   - `MODEL_PATH` – path to your trained YOLO model weights (`.pt`).
   - `NIFTI_DIR` – directory containing patient subfolders with NIfTI volumes.
   - `CSV_PATH` – output CSV path (defaults to `NIFTI_DIR/overscanning_results.csv`).
   - Flags controlling behaviour:
     - `DISPLAY_DETECTION` – show the best pubic-symphysis detection slice.
     - `FAST_MODEL` – pass `--fast` to TotalSegmentator for quicker but less accurate masks.
     - `MULTI_LABEL_MASK` – store liver and spleen in a combined mask as labels `1` and `2`.
3. **Run the notebook** – execute the cells. Processing is incremental; existing rows in the CSV are updated or appended. Femur, liver and spleen masks are generated on the fly (GPU is tried first with CPU fallback). Optional cells near the end can generate MP4 preview videos, write a summary statistics CSV and export PNG plots summarising overscanning.

## File Requirements

- Input CTs must be NIfTI files (`.nii` or `.nii.gz`) oriented head-to-feet.
- YOLO expects images in BGR format; the notebook handles normalisation automatically.

## Methodology Notes

Cranial overscanning is computed using the highest axial slice containing liver or spleen voxels. Caudal overscanning uses a YOLO detection of the pubic symphysis with femur segmentation as a fallback when YOLO fails. Distances are reported in millimetres, derived from the NIfTI affine.

The CSV columns are:

- `file_name` – scan filename
- `pubic_z_mm` – world z-coordinate of detected pubic symphysis
- `scan_end_z_mm` – caudal edge of scan
- `caudal_overscan_mm` – distance from pubic symphysis to scan end
- `femur_top_z_mm` – cranial-most femur coordinate
- `pubic_source` – `YOLO`, `YOLO_NoFemur`, or `FemurFallback`
- `liver_spleen_z_mm` – highest liver/spleen voxel
- `scan_start_z_mm` – cranial edge of scan
- `cranial_overscan_mm` – distance from cranial edge to top organ
- `top_organ` – `Liver`, `Spleen`, or `Unknown`

## Outputs

- `overscanning_results.csv` – main results updated after each run.
- `summary_statistics.csv` – vertical table of mean/SD overscan metrics and overscan frequencies.
- `scatter_cranial_caudal.png`, `box_cranial_caudal_total.png`, `bar_cranial_caudal.png` – visualisations of overscan distribution.

## License

This project is released under the MIT License – see `LICENSE` for details.
