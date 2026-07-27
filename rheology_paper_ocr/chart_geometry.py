from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChartGeometry:
    image_width: int
    image_height: int
    plot_bbox: tuple[int, int, int, int] | None
    horizontal_line_count: int
    vertical_line_count: int
    ink_fraction: float
    x_tick_pixels: list[int]
    y_tick_pixels: list[int]

    @property
    def has_plot_frame(self) -> bool:
        return self.plot_bbox is not None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_plot_frame"] = self.has_plot_frame
        return data


def analyze_chart_geometry(image_path: Path) -> ChartGeometry:
    """Detect a chart frame from long raster line segments without OCR or ML."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("chart geometry analysis requires the optional '.[raster]' dependencies") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read chart image: {image_path}")
    height, width = image.shape
    binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]
    ink_fraction = float(np.count_nonzero(binary)) / float(binary.size)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 6), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height // 6))))
    horizontal_lines = _long_segments(horizontal, horizontal=True)
    vertical_lines = _long_segments(vertical, horizontal=False)
    plot_bbox = _plot_bbox(horizontal_lines, vertical_lines)
    return ChartGeometry(
        image_width=width,
        image_height=height,
        plot_bbox=plot_bbox,
        horizontal_line_count=len(horizontal_lines),
        vertical_line_count=len(vertical_lines),
        ink_fraction=ink_fraction,
        x_tick_pixels=_axis_ticks(binary, plot_bbox, x_axis=True),
        y_tick_pixels=_axis_ticks(binary, plot_bbox, x_axis=False),
    )


def _long_segments(mask, horizontal: bool) -> list[tuple[int, int, int, int]]:
    import cv2
    segments = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, width, height = cv2.boundingRect(contour)
        if horizontal and width >= max(12, mask.shape[1] // 4):
            segments.append((x, y, x + width, y + height))
        if not horizontal and height >= max(12, mask.shape[0] // 4):
            segments.append((x, y, x + width, y + height))
    return segments


def _plot_bbox(horizontal: list[tuple[int, int, int, int]], vertical: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    left = min(segment[0] for segment in vertical)
    right = max(segment[2] for segment in vertical)
    top = min(segment[1] for segment in horizontal)
    bottom = max(segment[3] for segment in horizontal)
    return (left, top, right, bottom) if left < right and top < bottom else None


def _axis_ticks(binary, plot_bbox: tuple[int, int, int, int] | None, x_axis: bool) -> list[int]:
    """Find short marks crossing an axis, expressed in source-image pixels."""
    import numpy as np

    if plot_bbox is None:
        return []
    left, top, right, bottom = plot_bbox
    if x_axis:
        region = binary[max(0, bottom - 2) : min(binary.shape[0], bottom + 24), left:right]
        counts = np.count_nonzero(region, axis=0)
        coordinates = _run_centres(counts, minimum=4)
        return [left + coordinate for coordinate in coordinates if coordinate > 2 and coordinate < right - left - 2]
    region = binary[top:bottom, max(0, left - 24) : min(binary.shape[1], left + 3)]
    counts = np.count_nonzero(region, axis=1)
    coordinates = _run_centres(counts, minimum=4)
    return [top + coordinate for coordinate in coordinates if coordinate > 2 and coordinate < bottom - top - 2]


def _run_centres(counts, minimum: int) -> list[int]:
    centres = []
    start = None
    for index, value in enumerate(counts):
        if value >= minimum and start is None:
            start = index
        if start is not None and (value < minimum or index == len(counts) - 1):
            end = index if value < minimum else index + 1
            if end - start <= 12:
                centres.append((start + end - 1) // 2)
            start = None
    return centres
