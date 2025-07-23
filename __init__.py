# BLOCK 1 – Setup imports, paths, flags, and constants

import gc, io, time, traceback, contextlib
import traceback
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from totalsegmentator.python_api import totalsegmentator
from ultralytics import YOLO
from tqdm import tqdm
import shutil, time
import matplotlib.pyplot as plt


# ── PATHS (EDIT HERE) ────────────────────────────────────────
MODEL_PATH = Path("PATH_TO_MODEL_WEIGHTS.pt")
NIFTI_DIR  = Path("PATH_TO_NIFTI_DIRECTORY")
CSV_PATH = NIFTI_DIR / "overscanning_results.csv"

# ── FLAGS ─────────────────────────────────────────────────────
DISPLAY_DETECTION = True
FAST_MODEL = False
MULTI_LABEL_MASK = True

# ── CONSTANTS ─────────────────────────────────────────────────
FINAL_CONF = 0.20
BACKGROUND_HU = -300
model = YOLO(str(MODEL_PATH))