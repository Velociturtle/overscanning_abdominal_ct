import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import importlib
from pathlib import Path
import pytest

import config


def test_paths_are_path_objects():
    assert isinstance(config.MODEL_PATH, Path)
    assert isinstance(config.NIFTI_DIR, Path)
    assert isinstance(config.CSV_PATH, Path)


def test_get_yolo_model_import_error(monkeypatch):
    monkeypatch.setitem(importlib.sys.modules, 'ultralytics', None)
    with pytest.raises(ImportError):
        config.get_yolo_model()
