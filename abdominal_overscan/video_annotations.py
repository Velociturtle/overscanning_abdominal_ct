"""Small helpers for readable MP4 preview annotations."""
from __future__ import annotations

from dataclasses import dataclass

import cv2  # type: ignore


@dataclass(frozen=True)
class AnnotationStyle:
    font: int
    font_scale: float
    text_thickness: int
    line_thickness: int
    margin: int
    text_gap: int


def annotation_style(frame_shape: tuple[int, ...]) -> AnnotationStyle:
    """Return text/line sizes scaled to the rendered frame."""
    height, width = frame_shape[:2]
    short_side = max(1, min(height, width))
    font_scale = max(0.28, min(0.50, short_side / 1200))
    text_thickness = max(1, min(2, round(short_side / 360)))
    line_thickness = max(1, min(2, round(short_side / 420)))
    margin = max(4, min(10, round(short_side * 0.02)))
    text_gap = max(6, min(16, round(short_side * 0.035)))
    return AnnotationStyle(
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=font_scale,
        text_thickness=text_thickness,
        line_thickness=line_thickness,
        margin=margin,
        text_gap=text_gap,
    )


def compact_label(text: str) -> str:
    """Shorten anatomical labels for small frames."""
    replacements = {
        "Pubic Symphysis": "Pubic",
        "Kidneys": "Kidney",
    }
    for original, compact in replacements.items():
        text = text.replace(original, compact)
    return text


def _fit_text(text: str, max_width: int, style: AnnotationStyle) -> str:
    text_width = cv2.getTextSize(text, style.font, style.font_scale, style.text_thickness)[0][0]
    if max_width > 640 and text_width <= max_width:
        return text
    compact = compact_label(text)
    if cv2.getTextSize(compact, style.font, style.font_scale, style.text_thickness)[0][0] <= max_width:
        return compact
    if " z=" in compact:
        label, value = compact.split(" z=", 1)
        shorter = f"{label[:6].rstrip()} z={value}"
        if cv2.getTextSize(shorter, style.font, style.font_scale, style.text_thickness)[0][0] <= max_width:
            return shorter
    return compact


def draw_labeled_line(
    image,
    y_line: int,
    text: str,
    color: tuple[int, int, int],
    prefer_above: bool = True,
) -> None:
    """Draw a thin landmark line with an in-bounds, frame-scaled label."""
    height, width = image.shape[:2]
    style = annotation_style(image.shape)
    y_line = max(0, min(height - 1, int(y_line)))
    cv2.line(image, (0, y_line), (width - 1, y_line), color, style.line_thickness)

    max_text_width = max(1, width - 2 * style.margin)
    label = _fit_text(text, max_text_width, style)
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        style.font,
        style.font_scale,
        style.text_thickness,
    )

    above_y = y_line - style.text_gap
    below_y = y_line + style.text_gap + text_height
    if prefer_above and above_y - text_height - baseline >= style.margin:
        y_text = above_y
    elif below_y + baseline <= height - style.margin:
        y_text = below_y
    else:
        y_text = max(style.margin + text_height, min(height - style.margin - baseline, above_y))

    x_text = max(style.margin, min(width - style.margin - text_width, style.margin))
    cv2.putText(
        image,
        label,
        (x_text, y_text),
        style.font,
        style.font_scale,
        color,
        style.text_thickness,
        cv2.LINE_AA,
    )


def put_corner_label(image, text: str, color: tuple[int, int, int]) -> None:
    """Draw a compact bottom-left frame label."""
    height, width = image.shape[:2]
    style = annotation_style(image.shape)
    label = _fit_text(text, max(1, width - 2 * style.margin), style)
    cv2.putText(
        image,
        label,
        (style.margin, height - style.margin),
        style.font,
        style.font_scale,
        color,
        style.text_thickness,
        cv2.LINE_AA,
    )