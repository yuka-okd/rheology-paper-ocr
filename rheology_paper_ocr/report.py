from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from rheology_paper_ocr.schemas import JoinedResult, ReviewCandidate


CSV_FIELDS = list(JoinedResult.model_fields.keys())


def write_reports(out_dir: Path, results: list[JoinedResult], review_candidates: list[ReviewCandidate] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump([result.model_dump(mode="json") for result in results], handle, indent=2)

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            row = result.model_dump(mode="json")
            row["warnings"] = "; ".join(row.get("warnings") or [])
            row["points"] = json.dumps(row.get("points") or [])
            writer.writerow(row)

    reviews = review_candidates or []
    with (out_dir / "review_queue.json").open("w", encoding="utf-8") as handle:
        json.dump([candidate.model_dump(mode="json") for candidate in reviews], handle, indent=2)
    with (out_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReviewCandidate.model_fields.keys()))
        writer.writeheader()
        for candidate in reviews:
            row = candidate.model_dump(mode="json")
            row["reasons"] = "; ".join(row.get("reasons") or [])
            writer.writerow(row)

    rows = []
    for result in results:
        crop = ""
        if result.chart_crop_path:
            path = html.escape(result.chart_crop_path, quote=True)
            crop = f'<a href="{path}"><img src="{path}" alt="Chart crop" loading="lazy"></a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.paper_id)}</td>"
            f"<td>{html.escape(result.figure_id or '')}</td>"
            f"<td>{crop}</td>"
            f"<td>{html.escape(result.curve_visual_label or result.curve_id)}</td>"
            f"<td>{html.escape(result.sample_display_name or '')}</td>"
            f"<td>{html.escape(result.rheology_class)}</td>"
            f"<td>{html.escape(result.fibre_outcome)}</td>"
            f"<td>{html.escape(result.fibre_evidence_text or '')}</td>"
            f"<td>{html.escape(result.confidence)}</td>"
            f"<td>{html.escape(result.decision)}</td>"
            f"<td>{html.escape('; '.join(result.warnings))}</td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rheology Paper OCR Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f4f4f4; text-align: left; }}
    img {{ display: block; max-width: 240px; max-height: 180px; }}
  </style>
</head>
<body>
  <h1>Rheology Paper OCR Report</h1>
  <table>
    <thead>
      <tr><th>Paper</th><th>Figure</th><th>Chart</th><th>Curve</th><th>Sample</th><th>Rheology</th><th>Fibre Outcome</th><th>Evidence</th><th>Confidence</th><th>Decision</th><th>Warnings</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (out_dir / "report.html").write_text(html_doc, encoding="utf-8")
