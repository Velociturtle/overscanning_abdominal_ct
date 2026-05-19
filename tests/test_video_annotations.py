import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from abdominal_overscan import video_annotations as annotations


def test_annotation_style_scales_with_frame_size():
    small = annotations.annotation_style((80, 120, 3))
    large = annotations.annotation_style((512, 512, 3))

    assert small.font_scale < large.font_scale
    assert small.line_thickness <= large.line_thickness
    assert small.text_thickness <= large.text_thickness


def test_draw_labeled_line_handles_small_frame():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    annotations.draw_labeled_line(
        frame,
        4,
        "Pubic Symphysis z=1234 mm",
        (0, 0, 255),
        prefer_above=True,
    )

    assert frame.any()
    assert annotations.annotation_style(frame.shape).line_thickness == 1


def test_small_frames_prefer_compact_landmark_labels():
    style = annotations.annotation_style((512, 512, 3))

    assert annotations._fit_text("Pubic Symphysis z=123 mm", 512, style).startswith("Pubic z=")


def test_put_corner_label_handles_small_frame():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    annotations.put_corner_label(frame, "LONG_SCAN_ID_FOR_SMALL_FRAME | y=12", (255, 255, 0))

    assert frame.any()
