# Overscanning Abdominal CT

This repository provides tools to quantify cranial and caudal overscanning on abdominal CT volumes.  Overscanning is measured relative to the pubic symphysis (caudal) and to combined liver/spleen segmentation (cranial).  Detection relies on a YOLOv11 model and segmentation uses TotalSegmentator.

Two workflows are available:

- **`overscanning_calculator.ipynb`** – the original reference notebook containing the complete pipeline.
- **Modular pipeline** – a set of Python modules with a small helper notebook (`abdomen_overscanning_helper.ipynb`) that executes them in sequence.

## Contents

- `overscanning_calculator.ipynb` – reference notebook with the full workflow
- `abdomen_overscanning_helper.ipynb` – example notebook calling the modules
- `abdominal_overscan/config.py` – edit paths and flags here before running
- `abdominal_overscan/caudal.py` – caudal overscan detection and CSV update
- `abdominal_overscan/cranial.py` – cranial overscan detection and CSV update
- `abdominal_overscan/mp4_preview.py` – optional video generation
- `abdominal_overscan/metrics.py` – optional summary statistics table
- `abdominal_overscan/figures.py` – optional figure generation
- `tests/` – pytest suite covering the modules
- `YOLO/` – example training outputs and model weights for the pubic symphysis detector

## Installation

Create a Python environment (Python 3.9+ recommended) and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` should be installed with CUDA support if GPU processing is desired for YOLO and TotalSegmentator.

## Default paths

The module `abdominal_overscan.config` defines the locations used by the package:

```python
BASE_DIR = Path(__file__).resolve().parent.parent
NIFTI_DIR = BASE_DIR / "data"
CSV_PATH = NIFTI_DIR / "overscanning_results.csv"
```

By default NIfTI files are read from the `data/` directory next to the
repository and results are written there as well.  These paths can be modified
either by editing `abdominal_overscan/config.py` or at runtime.

## Usage

1. **Download the model weights** – retrieve `best.pt` from `YOLO/model_and_training/yolo11_pubic_symphysis_m_hardtrain/weights`.
2. **Prepare your data** – organise CT scans as NIfTI files under a single directory. Each patient folder should contain one `.nii` or `.nii.gz` volume.
3. **Configure paths** – edit `abdominal_overscan/config.py` to point `MODEL_PATH`, `NIFTI_DIR` and `CSV_PATH` to your locations.  Optional flags controlling detection and segmentation can also be adjusted.
4. **Run the pipeline** – either execute the modules directly:

```bash
python -m abdominal_overscan.caudal   # caudal overscan
python -m abdominal_overscan.cranial  # cranial overscan
python -m abdominal_overscan.mp4_preview   # optional video previews
python -m abdominal_overscan.metrics  # optional summary CSV
python -m abdominal_overscan.figures  # optional figures
```

   or open `abdomen_overscanning_helper.ipynb` and run the cells.

Results accumulate in `overscanning_results.csv` under `NIFTI_DIR`.

## File Requirements

- Input CTs must be NIfTI files (`.nii` or `.nii.gz`) oriented head-to-feet.
- YOLO expects images in BGR format; the modules handle normalisation automatically.

## Methodology Notes

Cranial overscanning is computed using the highest axial slice containing liver or spleen voxels.  Caudal overscanning uses a YOLO detection of the pubic symphysis with femur segmentation as a fallback when YOLO fails.  Distances are reported in millimetres, derived from the NIfTI affine.

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

- `overscanning_results.csv` – main results updated after each run
- `summary_statistics.csv` – vertical table of mean/SD overscan metrics and overscan frequencies
- `scatter_cranial_caudal.png`, `box_cranial_caudal_total.png`, `bar_cranial_caudal.png` – visualisations of overscan distribution

## Testing

Run the tests with:

```bash
pytest -q
```

## License

This project is released under the MIT License – see `LICENSE` for details.
