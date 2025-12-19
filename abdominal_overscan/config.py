"""Configuration and shared constants for the overscanning pipeline."""
from pathlib import Path

# --- Paths ---
# By default the repository expects a ``data`` directory alongside the
# :mod:`abdominal_overscan` package.  The model weights are assumed to live
# in the ``YOLO/model_and_training`` subdirectory.  All paths can be
# overridden at runtime or by editing this file.

BASE_DIR = Path(__file__).resolve().parent.parent
NIFTI_DIR = BASE_DIR / "data"
CSV_PATH = NIFTI_DIR / "overscanning_results.csv"
MODEL_PATH = (
    BASE_DIR
    / "YOLO/model_and_training/yolo11_pubic_symphysis_m_hardtrain/weights/best.pt"
)

# --- Flags ---
DISPLAY_DETECTION = True
FAST_MODEL = False
MULTI_LABEL_MASK = True

# --- Constants ---
FINAL_CONF = 0.20
BACKGROUND_HU = -300
BONE_HU_THRESHOLD = 200
BONE_MIN_FRACTION = 0.02
PUBIC_MISS_TOLERANCE = 1


def get_yolo_model():
    """Return a YOLO model loaded from ``MODEL_PATH``.

    Raises
    ------
    ImportError
        If the ``ultralytics`` package is not available.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - handled via ImportError
        raise ImportError(
            "ultralytics package is required for YOLO functionality"
        ) from exc
    return YOLO(str(MODEL_PATH))
