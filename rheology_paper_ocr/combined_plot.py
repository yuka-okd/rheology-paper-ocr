from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pint
from matplotlib.lines import Line2D

from rheology_paper_ocr.schemas import JoinedResult


FibreCategory = Literal["can", "cannot", "unclear"]

CAN_FORM_FIBRE = {"formed fibres", "formed beaded fibres"}
CANNOT_FORM_FIBRE = {"failed or no fibres"}

CATEGORY_COLORS: dict[FibreCategory, str] = {
    "can": "#2ca02c",
    "cannot": "#d62728",
    "unclear": "#7f7f7f",
}
CATEGORY_LABELS: dict[FibreCategory, str] = {
    "can": "Can form fibre",
    "cannot": "Cannot form fibre",
    "unclear": "Unclear",
}

# Preferred display unit per pint dimensionality, for the rheology quantities
# this pipeline reports. Anything else falls back to the first curve's unit.
_PREFERRED_UNITS: dict[str, str] = {
    "[mass] / [length] / [time]": "Pa*s",  # viscosity
    "1 / [time]": "1/s",  # shear rate / frequency
    "[mass] / [length] / [time] ** 2": "Pa",  # stress / modulus
}


def _normalize_label(label: str | None) -> str | None:
    if not label:
        return None
    label = re.sub(r"\([^)]*\)", "", label)
    label = re.sub(r"\s+", " ", label).strip().lower()
    return label or None


def _fibre_category(fibre_outcome: str) -> FibreCategory:
    if fibre_outcome in CAN_FORM_FIBRE:
        return "can"
    if fibre_outcome in CANNOT_FORM_FIBRE:
        return "cannot"
    return "unclear"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "group"


def _group_key(result: JoinedResult) -> tuple[str, str] | None:
    x_label = _normalize_label(result.x_axis_label)
    y_label = _normalize_label(result.y_axis_label)
    if x_label is None or y_label is None:
        return None
    return (x_label, y_label)


def _axis_scale_is_log(results: list[JoinedResult], attr: str) -> bool:
    votes = Counter((getattr(result, attr) or "").strip().lower() for result in results)
    return votes["log"] > sum(votes.values()) / 2


def _canonical_unit(ureg: pint.UnitRegistry, unit: pint.Unit) -> pint.Unit:
    preferred = _PREFERRED_UNITS.get(str(unit.dimensionality))
    return ureg.Unit(preferred) if preferred else unit


def _convert_group(
    ureg: pint.UnitRegistry, results: list[JoinedResult]
) -> tuple[list[dict], str | None, str | None, list[dict]]:
    included: list[dict] = []
    excluded: list[dict] = []
    x_canonical: pint.Unit | None = None
    y_canonical: pint.Unit | None = None

    for result in results:
        identity = {"paper_id": result.paper_id, "curve_id": result.curve_id}
        if not result.points:
            excluded.append({**identity, "reason": "no digitized points"})
            continue
        if not result.x_axis_unit or not result.y_axis_unit:
            excluded.append({**identity, "reason": "missing axis unit"})
            continue
        try:
            x_unit = ureg.Unit(result.x_axis_unit)
            y_unit = ureg.Unit(result.y_axis_unit)
        except Exception as exc:
            excluded.append({**identity, "reason": f"unparseable unit: {exc}"})
            continue

        if x_canonical is None:
            x_canonical = _canonical_unit(ureg, x_unit)
            y_canonical = _canonical_unit(ureg, y_unit)

        try:
            converted_x = [ureg.Quantity(point.x, x_unit).to(x_canonical).magnitude for point in result.points]
            converted_y = [ureg.Quantity(point.y, y_unit).to(y_canonical).magnitude for point in result.points]
        except pint.errors.DimensionalityError as exc:
            excluded.append({**identity, "reason": f"incompatible unit: {exc}"})
            continue

        included.append(
            {
                "paper_id": result.paper_id,
                "curve_id": result.curve_id,
                "figure_id": result.figure_id,
                "category": _fibre_category(result.fibre_outcome),
                "x": converted_x,
                "y": converted_y,
            }
        )

    x_unit_str = str(x_canonical) if x_canonical is not None else None
    y_unit_str = str(y_canonical) if y_canonical is not None else None
    return included, x_unit_str, y_unit_str, excluded


def write_combined_plots(out_dir: Path, results: list[JoinedResult]) -> None:
    """Render one PNG per (x-axis, y-axis) quantity pair, lines colored by fibre outcome."""
    ureg = pint.UnitRegistry()
    groups: dict[tuple[str, str], list[JoinedResult]] = defaultdict(list)
    excluded: list[dict] = []

    for result in results:
        key = _group_key(result)
        if key is None:
            excluded.append({"paper_id": result.paper_id, "curve_id": result.curve_id, "reason": "missing axis label"})
            continue
        groups[key].append(result)

    plot_dir = out_dir / "combined_plots"
    manifest_groups = []

    for (x_label, y_label), group_results in groups.items():
        converted, x_unit, y_unit, group_excluded = _convert_group(ureg, group_results)
        excluded.extend(group_excluded)
        if not converted:
            continue

        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        present_categories: set[FibreCategory] = set()
        for curve in converted:
            ax.plot(
                curve["x"],
                curve["y"],
                color=CATEGORY_COLORS[curve["category"]],
                marker="o",
                linewidth=1.5,
                markersize=3,
            )
            present_categories.add(curve["category"])

        if _axis_scale_is_log(group_results, "x_axis_scale"):
            ax.set_xscale("log")
        if _axis_scale_is_log(group_results, "y_axis_scale"):
            ax.set_yscale("log")

        original_x_label = next((r.x_axis_label for r in group_results if r.x_axis_label), x_label)
        original_y_label = next((r.y_axis_label for r in group_results if r.y_axis_label), y_label)
        ax.set_xlabel(f"{original_x_label} ({x_unit})")
        ax.set_ylabel(f"{original_y_label} ({y_unit})")
        ax.set_title(f"{original_y_label} vs {original_x_label}")

        handles = [
            Line2D([], [], color=CATEGORY_COLORS[category], marker="o", label=CATEGORY_LABELS[category])
            for category in ("can", "cannot", "unclear")
            if category in present_categories
        ]
        ax.legend(handles=handles)

        slug = _slug(f"{y_label}_vs_{x_label}")
        plot_path = plot_dir / f"{slug}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        manifest_groups.append(
            {
                "x_label": x_label,
                "y_label": y_label,
                "x_unit": x_unit,
                "y_unit": y_unit,
                "plot_path": str(plot_path.relative_to(out_dir)),
                "curves": converted,
            }
        )

    manifest = {"groups": manifest_groups, "excluded": excluded}
    (out_dir / "combined_plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
