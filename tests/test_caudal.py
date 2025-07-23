import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

caudal = pytest.importorskip('abdominal_overscan.caudal')


def test_preprocess_slice_import_error(monkeypatch):
    monkeypatch.setattr(caudal, 'np', None)
    with pytest.raises(ImportError):
        caudal.preprocess_slice(None)


def test_find_valid_pubic_slice_import_error(monkeypatch):
    monkeypatch.setattr(caudal, 'nib', None)
    with pytest.raises(ImportError):
        caudal.find_valid_pubic_slice('dummy', 0)
