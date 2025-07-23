import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

cranial = pytest.importorskip('cranial')
from pathlib import Path


def test_nifti_basename():
    assert cranial.nifti_basename(Path('scan.nii.gz')) == 'scan'
    assert cranial.nifti_basename(Path('scan.nii')) == 'scan'


def test_cranial_overscan_import_error(monkeypatch):
    monkeypatch.setattr(cranial, 'nib', None)
    with pytest.raises(ImportError):
        cranial.cranial_overscan('ct', 'mask')
