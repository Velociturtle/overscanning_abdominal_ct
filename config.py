"""Configuration and shared constants for the overscanning pipeline."""
from pathlib import Path

# --- Paths (edit these) ---
MODEL_PATH = Path("PATH_TO_MODEL_WEIGHTS.pt")
NIFTI_DIR = Path("PATH_TO_NIFTI_DIRECTORY")
CSV_PATH = NIFTI_DIR / "overscanning_results.csv"

# --- Flags ---
DISPLAY_DETECTION = True
FAST_MODEL = False
MULTI_LABEL_MASK = True

# --- Constants ---
FINAL_CONF = 0.20
BACKGROUND_HU = -300


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
