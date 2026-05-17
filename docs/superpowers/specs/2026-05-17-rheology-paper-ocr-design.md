# Rheology Paper OCR Design

Date: 2026-05-17

## Goal

Build a command-line workflow that processes a directory of chemistry paper PDFs and produces an evidence-backed table connecting:

- each detected rheology chart,
- each chart line or marker series,
- the sample/formulation represented by that series,
- the extracted rheology behavior,
- and whether the paper reports that sample/solution as having produced fibres.

The first version optimizes for an 80/20 workflow: fewer brittle computer-vision components, higher reliance on a strong vision-capable LLM through OpenRouter, structured outputs, and auditable evidence. Exact publication-grade curve reconstruction is out of scope for v1; sparse or moderate-density digitization is enough if it supports start/end values, trend shape, and shear classification.

## User Interface

The primary interface is a CLI. The user points the tool at a folder of PDFs:

```bash
rheology-paper-ocr run ~/Desktop/chemistry-papers
```

Useful flags:

```bash
rheology-paper-ocr run <pdf_dir> \
  --out ./outputs/run-001 \
  --model anthropic/claude-sonnet-4.6 \
  --max-papers 50 \
  --resume
```

Secondary commands:

```bash
rheology-paper-ocr inspect <pdf_path> --out ./debug/paper-001
rheology-paper-ocr report <run_dir>
rheology-paper-ocr resume <run_dir>
```

`run` executes the full pipeline. `inspect` runs one paper with verbose intermediate artifacts. `report` rebuilds CSV/JSON/HTML outputs from saved intermediate JSON. `resume` continues an interrupted run without reprocessing completed papers.

## Configuration

Configuration comes from CLI flags and environment variables.

Required:

```bash
OPENROUTER_API_KEY=...
```

Optional:

```bash
OPENROUTER_MODEL=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

The OpenRouter adapter uses the OpenAI-compatible chat completions endpoint. For extraction calls, the adapter requests structured JSON outputs with `response_format: { "type": "json_schema" }` when the selected model supports it. The model must support image inputs for chart and page/figure analysis.

References:

- OpenRouter chat completions: https://openrouter.ai/docs/api-reference/chat-completion
- OpenRouter structured outputs: https://openrouter.ai/docs/features/structured-outputs
- PlotPick summary: https://www.catalyzex.com/paper/plotpick-ai-powered-batch-extraction-of
- PDFFigures2: https://github.com/allenai/pdffigures2
- PicAxe: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.574

## Pipeline

### 1. PDF Ingestion

The ingestion stage recursively scans the input directory for `.pdf` files and assigns stable paper IDs. It records file path, file hash, size, and processing status in a run manifest.

The manifest allows interrupted runs to resume and makes results traceable to source files.

### 2. Text, Caption, and Figure Extraction

The extraction stage produces:

- full-paper text where available,
- page images rendered at high DPI,
- candidate figure crops,
- captions and nearby text around each figure,
- page-level metadata for evidence links.

For born-digital PDFs, the first implementation can use lightweight Python PDF tooling plus page rendering. PDFFigures2 is a candidate optional backend for figure/caption extraction. PicAxe is a candidate fallback for scanned or messy PDFs.

The local output for each paper should include:

```text
paper_0001/
  text.md
  pages/
  figures/
  extraction.json
```

### 3. Rheology Chart Detection

Each figure crop, and optionally each full page when figure extraction is weak, is classified as:

- rheology chart,
- possibly relevant chart,
- not relevant.

The detector looks for axis labels and chart semantics such as:

- viscosity,
- shear rate,
- shear stress,
- storage modulus,
- loss modulus,
- frequency sweep,
- flow curve,
- oscillatory rheology.

For v1, this can be a vision LLM call through OpenRouter with a strict schema. The result must include a confidence score and a reason. Low-confidence positives are kept for review rather than discarded.

### 4. Chart Digitization

The chart digitization stage extracts structured data from each rheology chart:

- chart type,
- x-axis label, units, and scale,
- y-axis label, units, and scale,
- series count,
- visual series identifiers,
- legend text where visible,
- extracted points per series,
- extraction confidence and warnings.

PlotPick is the preferred v1 inspiration or adapter because it is designed for batch VLM-based extraction of numerical data from scientific figures. The repository should keep this behind a `ChartDigitizer` interface so the implementation can switch between:

- PlotPick-backed extraction,
- direct OpenRouter vision extraction,
- or a future deterministic digitizer.

The v1 digitizer should return enough points per curve to classify shape and approximate start/end values. Dense pixel-perfect reconstruction is not required.

### 5. Rheology Analysis

Rheology classification is deterministic and operates on extracted points. The LLM reads charts; local code classifies the trend.

For each series, compute:

- first valid x/y point,
- last valid x/y point,
- fold change in viscosity or modulus,
- monotonicity,
- slope on log-log axes when applicable,
- plateau behavior,
- non-monotonic or U-shaped behavior.

Initial classes:

- shear-thinning,
- shear-thickening,
- near-Newtonian or plateau,
- non-monotonic,
- unclear.

The code should avoid interpreting a linear y-intercept when the x-axis is logarithmic. For log rheology charts, report the lowest measured shear-rate value instead of a mathematical intercept at zero.

### 6. Chart Series to Sample Linking

The sample linker maps each chart series to a formulation/sample/solution.

Inputs:

- chart image,
- chart legend text,
- figure caption,
- nearby paragraphs mentioning the figure,
- methods/sample preparation text,
- tables containing sample formulations,
- previously extracted sample aliases.

The linker produces normalized sample records:

```json
{
  "sample_id": "G10H1",
  "display_name": "GelMA 10% + HA 1%",
  "aliases": ["G10H1", "10% GelMA / 1% HA", "GelMA10-HA1"],
  "composition": "10 wt% GelMA + 1 wt% hyaluronic acid",
  "solvent": "PBS",
  "evidence": [
    {
      "source": "figure_caption",
      "text": "..."
    }
  ],
  "confidence": "high"
}
```

If the legend only says `S1`, `S2`, or similar, the linker resolves those aliases from the paper text and tables. If the mapping is not explicitly supported by evidence, the result is marked `unclear`.

### 7. Fibre Outcome Extraction

Fibre outcome extraction is text-only in v1. The system must not infer fibre formation from SEM images or morphology images.

The extractor searches the paper text, captions, and tables for terms including:

- fiber,
- fibre,
- fibrous,
- nanofiber,
- nanofibre,
- electrospun,
- electrospinning,
- filament,
- jet,
- bead-free,
- beaded,
- morphology,
- scaffold.

For each normalized sample, classify:

- formed fibres,
- formed beaded fibres,
- failed or no fibres,
- not tested,
- unclear.

Each positive or negative outcome must include evidence text and source location. If the paper reports fibre formation generally but not per sample, the sample-level outcome should be `unclear` unless the formulation link is explicit.

### 8. Joining and Reporting

The final join connects:

- paper,
- figure,
- chart series,
- normalized sample,
- rheology metrics,
- fibre outcome,
- evidence,
- confidence.

Primary outputs:

```text
outputs/run-001/
  manifest.json
  results.csv
  results.json
  report.html
  papers/
    paper_0001/
      text.md
      pages/
      figures/
      charts/
      llm/
      extraction.json
```

Recommended CSV columns:

```text
paper_id
source_pdf
figure_id
page
chart_crop_path
curve_id
curve_visual_label
curve_legend_text
sample_id
sample_display_name
sample_composition
x_axis_label
x_axis_unit
x_axis_scale
y_axis_label
y_axis_unit
y_axis_scale
start_x
start_y
end_x
end_y
fold_change
loglog_slope
rheology_class
fibre_outcome
fibre_evidence_text
fibre_evidence_source
confidence
warnings
```

The HTML report should show the crop image next to extracted rows so low-confidence outputs can be reviewed quickly.

## Data Models

Use Pydantic models for every LLM output and internal artifact:

- `PaperManifest`
- `ExtractedPaper`
- `FigureCandidate`
- `RheologyChartDetection`
- `DigitizedChart`
- `DigitizedSeries`
- `SampleRecord`
- `FibreOutcome`
- `JoinedResult`

Each model should include `confidence`, `warnings`, and evidence fields where appropriate.

## Error Handling

The CLI should continue processing other PDFs when one paper fails. Failures are recorded in `manifest.json`.

Common failure modes:

- unreadable PDF,
- no text layer,
- figure extraction failed,
- no rheology charts found,
- model returned invalid JSON,
- selected OpenRouter model does not support vision or structured outputs,
- chart series cannot be mapped to sample,
- fibre outcome cannot be linked to sample.

For invalid LLM responses, retry with a repair prompt once. If still invalid, save raw response under `llm/` and mark the item failed or uncertain.

## Testing Strategy

Start with a small fixture set:

- one born-digital paper with clear rheology plots,
- one scanned or low-quality PDF,
- one paper with no rheology charts,
- one paper where series labels are sample codes resolved in a table,
- one paper where fibre formation is ambiguous or not per-sample.

Test layers:

- unit tests for deterministic rheology classification,
- schema validation tests for saved JSON artifacts,
- CLI smoke test over a tiny fixture directory,
- golden-output tests for hand-curated papers where expected sample/fibre outcomes are known.

Do not judge v1 only by exact numeric digitization accuracy. Track practical workflow metrics:

- chart detection recall,
- correct curve count,
- sample-link correctness,
- fibre-outcome correctness,
- percentage of rows marked `unclear`,
- manual review time per paper.

## Implementation Boundaries

In scope for v1:

- CLI workflow over a directory of PDFs,
- OpenRouter-backed structured extraction,
- figure/page rendering,
- rheology chart detection,
- chart digitization sufficient for curve shape,
- text-only fibre outcome extraction,
- sample/series linking with evidence,
- CSV/JSON/HTML reports,
- resumable run artifacts.

Out of scope for v1:

- training custom CV models,
- publication-grade dense digitization,
- inferring fibre formation from SEM images,
- building a web app,
- fully automated citation metadata enrichment,
- claiming sample-level outcomes without explicit textual evidence.

## Acceptance Criteria

The first implementation is acceptable when:

1. A user can run one CLI command against a folder of PDFs.
2. The tool produces `results.csv`, `results.json`, and `report.html`.
3. Every fibre outcome is text-evidence-backed or marked `unclear`.
4. Every chart row links to a crop image or page image.
5. Rheology classification is computed from extracted points, not free-form model prose.
6. The run can resume without repeating completed papers.
7. Low-confidence mappings are visible in the report.
