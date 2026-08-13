# rheology-paper-ocr

An evidence-first local browser application for connecting rheology chart series in chemistry papers to the represented formulation and its reported fibre outcome.

This is an evidence-first automation workflow. It extracts sparse points to classify broad rheology behaviour; it does not claim publication-grade curve reconstruction.

## What a result means

Each row represents one chart series and preserves:

- source paper, page, and chart image;
- axis metadata, sparse points, approximate endpoints, and deterministic rheology class;
- the linked sample/formulation and its supporting text;
- a text-evidence-backed fibre outcome; and
- confidence and warnings for review.

Rows without enough curve evidence, an explicit sample link, or fibre evidence remain `unclear` or carry a warning. The tool must not infer fibre formation from microscopy images. Every row is also assigned `accepted` or `needs_review`; relevant charts without a supported row are preserved in the review queue.

## Local Browser App

Install the browser dependencies and start the local-only service. It opens a
workspace where PDFs or ZIP archives of PDFs can be uploaded, runs can be reviewed later, evidence can
be adjudicated, and CSV or printable PDF reports can be exported. Uploaded
papers and outputs stay on the computer in the selected data directory.

macOS and Linux:

```bash
python3 -m pip install -e '.[web]'
python3 -m rheology_paper_ocr.cli serve
```

Windows PowerShell:

```powershell
py -m pip install -e ".[web]"
py -m rheology_paper_ocr.cli serve
```

The browser opens at `http://127.0.0.1:8787`. Use `--no-open` to suppress the
automatic browser launch, `--port 8790` to select another port, or `--data-dir
path/to/folder` to place local sessions somewhere specific. The default data
directory is `.rheology-paper-ocr` in the current user's home directory. Enter
the OpenRouter key in the upload dialog. It is supplied only to that run in
memory. Selecting **Remember on this device** stores it in that browser's local
storage; use this only in a trusted personal browser profile. Leaving the box
unchecked removes any previously saved browser key after the next submission.
Setting `OPENROUTER_API_KEY` before launching remains supported for unattended
batch runs. The browser workflow always requires a key entered in its upload
dialog. The service rejects non-loopback host bindings so papers and
browser-supplied keys cannot be exposed on a LAN.

## Batch CLI

```bash
python3 -m pip install -e .
export OPENROUTER_API_KEY="..."
python3 -m rheology_paper_ocr.cli run /path/to/papers --out outputs/run-001 --max-papers 5
```

For the recommended hybrid mode, install the optional local figure locator. It
uses Docling to find figures, render an exact high-resolution crop, and recover
the caption before sending the crop plus page context to the vision model.
Without it, the pipeline falls back to full candidate pages.
The local classifier also records chart type and native vector text in the
per-paper figure metadata, while conservatively skipping captioned morphology
and fibre-size figures.

```bash
python3 -m pip install -e '.[docling]'
```

For deterministic raster chart-frame detection, install the optional image-processing dependencies:

```bash
python3 -m pip install -e '.[raster]'
```

Use `--text-only` to skip chart images, or `--resume` to reuse a completed paper only when its source hash is unchanged.
Set `OPENROUTER_MAX_TOKENS` to change the per-page extraction response budget; it defaults to `3000` to keep first-pass runs bounded. If a response is malformed or cut off, the JSON-only retry automatically uses at least `6000` tokens; set `OPENROUTER_FALLBACK_MAX_TOKENS` to change that retry budget.

The default primary model is `google/gemini-3.6-flash`. Set `OPENROUTER_FALLBACK_MODEL` to choose the model used after a transport or provider failure; it defaults to `openai/gpt-4o`. `OPENROUTER_READ_TIMEOUT_SECONDS` defaults to `180`, preventing a stalled provider route from blocking a paper indefinitely while avoiding a short foreground timeout.

```bash
python3 -m rheology_paper_ocr.cli inspect /path/to/paper.pdf --out outputs/inspect-paper
python3 -m rheology_paper_ocr.cli report outputs/run-001
python3 -m rheology_paper_ocr.cli resume outputs/run-001
```

## Output

```text
outputs/run-001/
  manifest.json
  results.csv
  results.json
  review_queue.csv         # candidates that need human adjudication
  review_queue.json
  report.html
  papers/<paper_id>/
    text.md
    pages/
    figures/             # Docling-localized chart crops when available
    native_graphics/     # high-resolution embedded chart assets when available
    native_graphics.json # PDF vector/raster inventory for candidate pages
    chart_geometry/      # optional plot frame, tick candidates, and uncalibrated colour traces
    docling_figures.json # figure bounding boxes and recovered captions
    llm/
    results.json
```

Open `report.html` to review each extracted row alongside the page image and its decision. Use `review_queue.csv` for the deliberately small human adjudication queue.

## Validation boundary

The pipeline is ready for a small, manually reviewed evaluation set. Before using the output for scientific conclusions, measure chart-detection recall, correct curve count, sample-link correctness, fibre-outcome correctness, and the rate of `unclear` rows against hand-labelled papers.
