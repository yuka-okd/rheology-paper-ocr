from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from rheology_paper_ocr.chart_geometry import analyze_chart_geometry


def test_detects_plot_frame_from_axis_lines(tmp_path: Path):
    image = np.full((300, 400), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 30), (350, 250), 0, 2)
    for x in (100, 200, 300):
        cv2.line(image, (x, 250), (x, 262), 0, 2)
    for y in (80, 140, 200):
        cv2.line(image, (38, y), (50, y), 0, 2)
    cv2.line(image, (50, 220), (350, 80), 0, 2)
    path = tmp_path / "chart.png"
    cv2.imwrite(str(path), image)

    geometry = analyze_chart_geometry(path)

    assert geometry.has_plot_frame is True
    assert geometry.plot_bbox is not None
    left, top, right, bottom = geometry.plot_bbox
    assert left <= 55 < right
    assert top <= 35 < bottom
    assert right >= 345
    assert bottom >= 245
    assert any(abs(tick - 100) <= 3 for tick in geometry.x_tick_pixels)
    assert any(abs(tick - 200) <= 3 for tick in geometry.x_tick_pixels)
    assert any(abs(tick - 140) <= 3 for tick in geometry.y_tick_pixels)
