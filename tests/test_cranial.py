import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

cranial = pytest.importorskip('abdominal_overscan.cranial')
from pathlib import Path
pytest.importorskip("scipy")


def test_nifti_basename():
    assert cranial.nifti_basename(Path('scan.nii.gz')) == 'scan'
    assert cranial.nifti_basename(Path('scan.nii')) == 'scan'


def test_cranial_overscan_import_error(monkeypatch):
    monkeypatch.setattr(cranial, 'nib', None)
    with pytest.raises(ImportError):
        cranial.cranial_overscan('ct', 'mask')


def test_ensure_patient_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cranial.config, 'NIFTI_DIR', tmp_path)
    f = tmp_path / 'case.nii.gz'
    f.write_text('x')
    cranial.ensure_patient_dirs()
    assert not f.exists()
    moved = tmp_path / 'case' / 'case.nii.gz'
    assert moved.exists()


def test_clean_mask_fills_holes(monkeypatch):
    import numpy as np
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, :] = True
    structure[:, 1, 1] = True
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    mask[1, 1, 0] = True
    mask[1, 1, 2] = True
    monkeypatch.setattr(cranial, "ndimage", __import__("scipy").ndimage)
    filled = cranial._clean_mask(mask)
    assert filled.any()
    assert filled[1, 1, :].all()
