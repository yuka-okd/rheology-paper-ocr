from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low", "unclear"]
Decision = Literal["accepted", "needs_review"]


class SourcePaper(BaseModel):
    paper_id: str
    path: Path


class DataPoint(BaseModel):
    x: float
    y: float


class DigitizedSeries(BaseModel):
    curve_id: str
    visual_label: str | None = None
    legend_text: str | None = None
    points: list[DataPoint] = Field(default_factory=list)
    confidence: Confidence = "unclear"
    warnings: list[str] = Field(default_factory=list)

class RheologyChart(BaseModel):
    figure_id: str | None = None
    page: int | None = None
    chart_crop_path: str | None = None
    x_axis_label: str | None = None
    x_axis_unit: str | None = None
    x_axis_scale: str | None = None
    y_axis_label: str | None = None
    y_axis_unit: str | None = None
    y_axis_scale: str | None = None
    series: list[DigitizedSeries] = Field(default_factory=list)
    confidence: Confidence = "unclear"
    warnings: list[str] = Field(default_factory=list)


class SeriesSummary(BaseModel):
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    fold_change: float | None = None
    loglog_slope: float | None = None
    rheology_class: str = "unclear"


class SampleLink(BaseModel):
    sample_id: str | None = None
    sample_display_name: str | None = None
    sample_composition: str | None = None
    evidence_text: str | None = None
    evidence_source: str | None = None
    confidence: Confidence = "unclear"


class FibreOutcome(BaseModel):
    outcome: Literal[
        "formed fibres",
        "formed beaded fibres",
        "failed or no fibres",
        "not tested",
        "unclear",
    ] = "unclear"
    evidence_text: str | None = None
    evidence_source: str | None = None
    confidence: Confidence = "unclear"


class ExtractedFinding(BaseModel):
    figure_id: str | None = None
    page: int | None = None
    chart_crop_path: str | None = None
    curve_id: str
    curve_visual_label: str | None = None
    curve_legend_text: str | None = None
    x_axis_label: str | None = None
    x_axis_unit: str | None = None
    x_axis_scale: str | None = None
    y_axis_label: str | None = None
    y_axis_unit: str | None = None
    y_axis_scale: str | None = None
    points: list[DataPoint] = Field(default_factory=list)
    sample: SampleLink = Field(default_factory=SampleLink)
    fibre_outcome: FibreOutcome = Field(default_factory=FibreOutcome)
    confidence: Confidence = "unclear"
    warnings: list[str] = Field(default_factory=list)


class PaperLLMExtraction(BaseModel):
    paper_id: str
    source_pdf: str
    has_rheology_chart: bool
    findings: list[ExtractedFinding] = Field(default_factory=list)
    paper_warnings: list[str] = Field(default_factory=list)


class JoinedResult(BaseModel):
    paper_id: str
    source_pdf: str
    figure_id: str | None = None
    page: int | None = None
    chart_crop_path: str | None = None
    curve_id: str
    curve_visual_label: str | None = None
    curve_legend_text: str | None = None
    sample_id: str | None = None
    sample_display_name: str | None = None
    sample_composition: str | None = None
    x_axis_label: str | None = None
    x_axis_unit: str | None = None
    x_axis_scale: str | None = None
    y_axis_label: str | None = None
    y_axis_unit: str | None = None
    y_axis_scale: str | None = None
    points: list[DataPoint] = Field(default_factory=list)
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    fold_change: float | None = None
    loglog_slope: float | None = None
    rheology_class: str = "unclear"
    fibre_outcome: str = "unclear"
    fibre_evidence_text: str | None = None
    fibre_evidence_source: str | None = None
    confidence: Confidence = "unclear"
    warnings: list[str] = Field(default_factory=list)
    decision: Decision = "needs_review"
    review_reasons: list[str] = Field(default_factory=list)


class ReviewCandidate(BaseModel):
    paper_id: str
    source_pdf: str
    figure_id: str | None = None
    page: int | None = None
    chart_crop_path: str | None = None
    curve_id: str | None = None
    sample_display_name: str | None = None
    fibre_outcome: str | None = None
    confidence: Confidence = "unclear"
    reasons: list[str] = Field(default_factory=list)
