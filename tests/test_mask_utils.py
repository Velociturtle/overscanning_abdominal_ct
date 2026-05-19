import pathlib
import sys

import nibabel as nib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from abdominal_overscan import mask_utils


def test_mask_is_valid_checks_content_and_shape(tmp_path):
    mask_path = tmp_path / "mask.nii.gz"
    empty = np.zeros((2, 2, 2))
    nib.save(nib.Nifti1Image(empty, np.eye(4)), mask_path)

    assert mask_utils.mask_is_valid(mask_path) is False

    filled = np.ones((2, 2, 2))
    nib.save(nib.Nifti1Image(filled, np.eye(4)), mask_path)

    assert mask_utils.mask_is_valid(mask_path, reference_shape=(2, 2, 2)) is True
    assert mask_utils.mask_is_valid(mask_path, reference_shape=(3, 2, 2)) is False


def test_remove_if_exists(tmp_path):
    mask_path = tmp_path / "mask.nii.gz"
    mask_path.write_text("placeholder")

    mask_utils.remove_if_exists([mask_path])

    assert not mask_path.exists()
