import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

cranial = pytest.importorskip('abdominal_overscan.cranial')
from pathlib import Path


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
