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
    color_traces: list["ColorTrace"]

    @property
    def has_plot_frame(self) -> bool:
        return self.plot_bbox is not None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["has_plot_frame"] = self.has_plot_frame
        return data


@dataclass(frozen=True)
class ColorTrace:
    """An uncalibrated coloured line candidate, tracked in image pixels."""

    rgb: tuple[int, int, int]
    sampled_pixels: list[tuple[int, int]]
    visual_direction: str


def analyze_chart_geometry(image_path: Path) -> ChartGeometry:
    """Detect chart structure and conservative coloured-line traces without OCR."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("chart geometry analysis requires the optional '.[raster]' dependencies") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read chart image: {image_path}")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale.shape
    binary = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY_INV)[1]
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
        color_traces=_color_traces(image, plot_bbox),
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


def _color_traces(image, plot_bbox: tuple[int, int, int, int] | None) -> list[ColorTrace]:
    """Track saturated curve lines that originate at the left plot axis.

    This intentionally does not try to read tick labels, infer data values, or
    associate a trace with its legend. Those tasks need OCR/vision evidence.
    Starting at the left axis excludes most coloured legend swatches and makes
    same-colour series independently traceable when their starts are separated.
    """
    import cv2
    import numpy as np

    if plot_bbox is None:
        return []
    left, top, right, bottom = plot_bbox
    if right - left < 40 or bottom - top < 40:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    crop = hsv[top:bottom, left:right]
    saturated = crop[:, :, 1] >= 90
    hue_counts = np.bincount(crop[:, :, 0][saturated], minlength=180)
    hue_centres = _dominant_hues(hue_counts, max(12, int(saturated.sum() * 0.008)))
    traces: list[ColorTrace] = []
    for hue in hue_centres:
        hue_distance = np.minimum(np.abs(hsv[:, :, 0].astype(int) - hue), 180 - np.abs(hsv[:, :, 0].astype(int) - hue))
        mask = ((hue_distance <= 7) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
        seed_region = mask[top:bottom, left : min(right, left + 16)]
        seed_counts = np.count_nonzero(seed_region, axis=1)
        seed_rows = _separated_centres(
            _run_centres(seed_counts, minimum=2, maximum_width=32),
            minimum_distance=18,
        )
        for row in seed_rows:
            trace = _trace_from_seed(mask, (left, top + row), (left, top, right, bottom))
            if len(trace) < 3 or trace[-1][0] - trace[0][0] < (right - left) * 0.35:
                continue
            rgb = _trace_rgb(image, mask, left, top + row, right, bottom)
            traces.append(
                ColorTrace(
                    rgb=rgb,
                    sampled_pixels=trace,
                    visual_direction=_visual_direction(trace),
                )
            )
    return _deduplicate_traces(traces)


def _dominant_hues(counts, minimum: int) -> list[int]:
    """Return centres of broad hue bands, including the circular red boundary."""
    active = [int(value >= minimum) for value in counts]
    if not any(active):
        return []
    doubled = active + active
    groups: list[list[int]] = []
    start = None
    for index, value in enumerate(doubled):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(doubled) - 1):
            end = index if not value else index + 1
            if end - start <= 20:
                groups.append(list(range(start, end)))
            start = None
    unique: list[int] = []
    for group in groups:
        centre = int(round(sum(index % 180 for index in group) / len(group))) % 180
        if not any(min(abs(centre - existing), 180 - abs(centre - existing)) <= 10 for existing in unique):
            unique.append(centre)
    return unique


def _trace_from_seed(mask, seed: tuple[int, int], plot_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = plot_bbox
    width = right - left
    sample_xs = [left + round(width * ratio / 5) for ratio in range(6)]
    previous_y = seed[1]
    trace: list[tuple[int, int]] = []
    for x in sample_xs:
        candidates = _trace_candidates(mask, x, top, bottom)
        if not candidates:
            continue
        y = min(candidates, key=lambda candidate: abs(candidate - previous_y))
        if abs(y - previous_y) > max(50, int((bottom - top) * 0.6)):
            continue
        trace.append((x, y))
        previous_y = y
    return trace


def _trace_rgb(image, mask, left: int, seed_y: int, right: int, bottom: int) -> tuple[int, int, int]:
    import numpy as np

    top = max(0, seed_y - 18)
    end = min(bottom, seed_y + 19)
    pixels = image[top:end, left : min(right, left + 16)]
    coloured = pixels[mask[top:end, left : min(right, left + 16)].astype(bool)]
    if len(coloured):
        return tuple(int(value) for value in np.median(coloured, axis=0)[::-1])
    return tuple(int(value) for value in image[seed_y, min(right - 1, left + 3)][::-1])


def _trace_candidates(mask, x: int, top: int, bottom: int) -> list[int]:
    import numpy as np

    window = mask[top:bottom, max(0, x - 5) : min(mask.shape[1], x + 6)]
    return [
        top + row
        for row in _run_centres(
            np.count_nonzero(window, axis=1),
            minimum=1,
            maximum_width=32,
        )
    ]


def _separated_centres(centres: list[int], minimum_distance: int) -> list[int]:
    result: list[int] = []
    for centre in centres:
        if not result or centre - result[-1] >= minimum_distance:
            result.append(centre)
    return result


def _visual_direction(trace: list[tuple[int, int]]) -> str:
    start_y = trace[0][1]
    end_y = trace[-1][1]
    threshold = 8
    if end_y > start_y + threshold:
        return "downward left-to-right"
    if end_y < start_y - threshold:
        return "upward left-to-right"
    return "approximately level"


def _deduplicate_traces(traces: list[ColorTrace]) -> list[ColorTrace]:
    """Suppress duplicate seed fragments while retaining separate same-hue lines."""
    result: list[ColorTrace] = []
    for trace in traces:
        if any(
            trace.rgb == existing.rgb
            and abs(trace.sampled_pixels[0][1] - existing.sampled_pixels[0][1]) < 10
            for existing in result
        ):
            continue
        result.append(trace)
    return result


def _run_centres(counts, minimum: int, maximum_width: int = 12) -> list[int]:
    centres = []
    start = None
    for index, value in enumerate(counts):
        if value >= minimum and start is None:
            start = index
        if start is not None and (value < minimum or index == len(counts) - 1):
            end = index if value < minimum else index + 1
            if end - start <= maximum_width:
                centres.append((start + end - 1) // 2)
            start = None
    return centres
