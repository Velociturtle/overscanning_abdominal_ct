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


def test_find_valid_pubic_slice_prefers_inferior(monkeypatch):
    import types
    import numpy as np

    # synthetic volume with bone on slices 1 and 2
    vol = np.zeros((4, 4, 3), dtype=float)
    vol[:, :, 1:] = 300
    affine = np.diag([1, 1, -1, 1])

    class DummyNifti:
        def __init__(self, data, aff):
            self._data = data
            self.affine = aff
            self.shape = data.shape
        def get_fdata(self):
            return self._data

    def fake_load(_):
        return DummyNifti(vol, affine)

    monkeypatch.setattr(caudal, "nib", types.SimpleNamespace(load=fake_load))
    monkeypatch.setattr(caudal, "preprocess_slice", lambda arr: arr)
    monkeypatch.setattr(caudal, "np", __import__("numpy"))
    monkeypatch.setattr(caudal, "cv2", object())

    class TensorLike:
        def __init__(self, arr, fail_without_cpu=True):
            self.arr = np.asarray(arr)
            self.fail_without_cpu = fail_without_cpu

        def detach(self):
            return self

        def cpu(self):
            return TensorLike(self.arr, fail_without_cpu=False)

        def __getitem__(self, item):
            return TensorLike(self.arr[item], self.fail_without_cpu)

        def __array__(self, dtype=None):
            if self.fail_without_cpu:
                raise TypeError("can't convert cuda:0 device type tensor to numpy")
            return np.asarray(self.arr, dtype=dtype)

    class DummyBox:
        def __init__(self):
            self.xyxy = TensorLike([[1, 1, 3, 3]])
            self.conf = TensorLike([0.9])

    class DummyRes:
        def __init__(self):
            self.boxes = [DummyBox()]

    class DummyModel:
        def predict(self, *_, **__):
            return [DummyRes()]

    monkeypatch.setattr(caudal.config, "get_yolo_model", lambda: DummyModel())

    assert caudal.find_valid_pubic_slice(pathlib.Path("scan.nii.gz"), float("inf")) == 2
