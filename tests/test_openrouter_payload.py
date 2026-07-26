from pathlib import Path

import httpx

from rheology_paper_ocr.openrouter_client import build_chat_payload
from rheology_paper_ocr.extract_with_llm import build_extraction_prompt, normalize_extraction_payload
from rheology_paper_ocr.openrouter_client import OpenRouterClient


def test_builds_json_schema_vision_payload(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"fake image bytes")

    payload = build_chat_payload(
        model="anthropic/claude-sonnet-4.6",
        prompt="Extract.",
        image_paths=[image],
    )

    assert payload["model"] == "anthropic/claude-sonnet-4.6"
    assert payload["max_tokens"] == 3000
    assert payload["response_format"]["type"] == "json_schema"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_normalizes_chart_level_llm_payload():
    payload = {
        "paper_id": "paper_0001",
        "has_rheology_chart": True,
        "findings": [
            {
                "chart_id": "chart_1",
                "x_axis": "concentration",
                "y_axis": "viscosity",
                "series": [{"sample_id": "sample_a", "label": "Sample A", "points": []}],
                "fibre_outcomes": [{"sample_id": "sample_a", "outcome": "fibre", "evidence": "Sample A formed fibres."}],
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0001", source_pdf="paper.pdf")

    assert normalized["source_pdf"] == "paper.pdf"
    assert normalized["findings"][0]["curve_id"] == "sample_a"
    assert normalized["findings"][0]["fibre_outcome"]["outcome"] == "formed fibres"


def test_normalizes_series_level_fibre_outcome_fields():
    payload = {
        "paper_id": "paper_0001",
        "has_rheology_chart": True,
        "findings": [
            {
                "chart_id": "chart_1",
                "series": [
                    {
                        "sample_id": "sample_a",
                        "label": "Sample A",
                        "points": [],
                        "fibre_outcome": "fibre_formed",
                        "fibre_evidence": "Sample A was readily electrospun into fibres.",
                    }
                ],
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0001", source_pdf="paper.pdf")

    assert normalized["findings"][0]["fibre_outcome"]["outcome"] == "formed fibres"
    assert normalized["findings"][0]["fibre_outcome"]["evidence_text"] == "Sample A was readily electrospun into fibres."


def test_normalizes_points_by_dropping_incomplete_coordinates():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": True,
        "findings": [
            {
                "chart_id": "chart_1",
                "series": [
                    {
                        "sample_id": "sample_a",
                        "label": "Sample A",
                        "points": [{"x": 1, "y": None}, {"x": 2, "y": 3}],
                    }
                ],
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0003", source_pdf="paper.pdf")

    assert normalized["findings"][0]["points"] == [{"x": 2, "y": 3}]


def test_normalizes_flat_chart_payload_with_data_points_and_sample_description():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": True,
        "findings": [
            {
                "chart_id": "figure_2_sample_a",
                "sample_id": "sample_a",
                "sample_description": "Sample A in water",
                "data_points": [{"x": 0.1, "y": 1}, {"x": 1, "y": 10}, {"x": 10, "y": 100}],
                "fibre_outcome": "fibres",
                "fibre_evidence": "Sample A formed fibres.",
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0003", source_pdf="paper.pdf")

    finding = normalized["findings"][0]
    assert finding["points"] == [{"x": 0.1, "y": 1}, {"x": 1, "y": 10}, {"x": 10, "y": 100}]
    assert finding["sample"]["sample_composition"] == "Sample A in water"


def test_assigns_the_attached_page_and_crop_to_unannotated_findings():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": True,
        "findings": [{"curve_id": "curve_a", "points": []}],
    }

    normalized = normalize_extraction_payload(
        payload,
        paper_id="paper_0003",
        source_pdf="paper.pdf",
        default_page=7,
        default_chart_crop_path="papers/paper_0003/pages/page_007.png",
    )

    assert normalized["findings"][0]["page"] == 7
    assert normalized["findings"][0]["chart_crop_path"] == "papers/paper_0003/pages/page_007.png"


def test_downgrades_unsupported_fibre_outcomes_to_unclear():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": True,
        "findings": [{"curve_id": "curve_a", "fibre_outcome": {"outcome": "formed fibres"}}],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0003", source_pdf="paper.pdf")

    finding = normalized["findings"][0]
    assert finding["fibre_outcome"]["outcome"] == "unclear"
    assert "fibre outcome lacks text evidence" in finding["warnings"]


def test_drops_explicit_fibre_diameter_plot_from_rheology_findings():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": True,
        "findings": [
            {
                "curve_id": "curve_a",
                "x_axis_label": "C/Ce",
                "y_axis_label": "Fibre diameter",
                "points": [{"x": 1, "y": 100}],
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0003", source_pdf="paper.pdf")

    assert normalized["findings"] == []
    assert "dropped 1 explicitly non-rheology finding(s)" in normalized["paper_warnings"]


def test_client_requests_schema_response_format(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeResponse:
        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"paper_id":"paper_0001","source_pdf":"paper.pdf","has_rheology_chart":false,"findings":[]}'
                        }
                    }
                ]
            }

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    client.extract("Extract.", [], raw_response_path=tmp_path / "raw.json")

    assert captured["payload"]["response_format"]["type"] == "json_schema"


def test_client_retries_without_schema_when_provider_rejects_response_format(monkeypatch):
    payloads = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        payloads.append(kwargs["json"])
        if len(payloads) == 1:
            return FakeResponse({"error": {"message": "response_format json_schema is not supported by this provider"}})
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"paper_id":"paper_0001","source_pdf":"paper.pdf","has_rheology_chart":false,"findings":[]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    client.extract("Extract.", [])

    assert [payload["response_format"]["type"] for payload in payloads] == ["json_schema", "json_object"]


def test_client_retries_without_schema_on_generic_provider_400(monkeypatch):
    payloads = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        payloads.append(kwargs["json"])
        if len(payloads) == 1:
            return FakeResponse({"error": {"message": "Provider returned error", "code": 400}})
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"paper_id":"paper_0001","source_pdf":"paper.pdf","has_rheology_chart":false,"findings":[]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    client.extract("Extract.", [])

    assert [payload["response_format"]["type"] for payload in payloads] == ["json_schema", "json_object"]


def test_client_retries_without_schema_after_malformed_schema_response(monkeypatch, tmp_path: Path):
    payloads = []

    class FakeResponse:
        def __init__(self, content):
            self.content = content

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        payloads.append(kwargs["json"])
        if len(payloads) == 1:
            return FakeResponse('{"findings": [}')
        return FakeResponse('{"paper_id":"paper_0001","source_pdf":"paper.pdf","has_rheology_chart":false,"findings":[]}')

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    result = client.extract("Extract.", [], raw_response_path=tmp_path / "raw.json")

    assert result["findings"] == []
    assert [payload["response_format"]["type"] for payload in payloads] == ["json_schema", "json_object"]
    assert (tmp_path / "raw_malformed_schema.json").exists()


def test_client_retries_transient_read_timeout(monkeypatch):
    attempts = []

    class FakeResponse:
        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"paper_id":"paper_0001","source_pdf":"paper.pdf","has_rheology_chart":false,"findings":[]}'
                        }
                    }
                ]
            }

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        attempts.append(kwargs["json"])
        if len(attempts) == 1:
            raise httpx.ReadTimeout("timed out")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    client.extract("Extract.", [])

    assert len(attempts) == 2


def test_prompt_includes_relevant_late_rheology_context():
    text = "Intro only.\n" + ("background filler\n" * 700) + (
        "Fig. 2 shows viscosity as a function of shear rate for electrospinning solutions. "
        "Sample A formed fibres."
    )

    prompt = build_extraction_prompt("paper_0001", "paper.pdf", text, text_only=True)

    assert "Intro only." in prompt
    assert "Fig. 2 shows viscosity" in prompt
    assert len(prompt) < len(text) + 2000


def test_normalizes_top_level_fibre_outcomes_without_dropping_evidence():
    payload = {
        "paper_id": "paper_0003",
        "has_rheology_chart": False,
        "findings": [],
        "fibre_outcomes": [
            {
                "sample_id": "Lignin/PAN/GRP 85/15/1 wt%",
                "outcome": "fibres formed",
                "evidence": "deformity-free nanoﬁbers",
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0003", source_pdf="paper.pdf")

    assert len(normalized["findings"]) == 1
    finding = normalized["findings"][0]
    assert finding["curve_id"] == "fibre_outcome_1"
    assert finding["sample"]["sample_display_name"] == "Lignin/PAN/GRP 85/15/1 wt%"
    assert finding["fibre_outcome"]["outcome"] == "formed fibres"
    assert "unjoined fibre outcome" in finding["warnings"]


def test_normalizes_flat_finding_shape_from_text_only_run():
    payload = {
        "paper_id": "paper_0005",
        "has_rheology_chart": False,
        "findings": [
            {
                "chart_id": None,
                "series_id": None,
                "sample_id": "PA6/FeCl3 solutions",
                "formulation_alias": "PA6-FeCl3 in formic acid",
                "rheology_points": [],
                "trend": "other",
                "fibre_outcome": "beaded/defective",
                "evidence_text": "the electrospun fibres show severely inhomogeneous structures",
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0005", source_pdf="paper.pdf")

    assert len(normalized["findings"]) == 1
    finding = normalized["findings"][0]
    assert finding["curve_id"] == "PA6/FeCl3 solutions"
    assert finding["sample"]["sample_composition"] == "PA6-FeCl3 in formic acid"
    assert finding["fibre_outcome"]["outcome"] == "formed beaded fibres"
    assert "normalized from flat LLM output" in finding["warnings"]


def test_flat_finding_fibre_outcome_prefers_fibre_specific_evidence():
    payload = {
        "paper_id": "paper_0005",
        "has_rheology_chart": True,
        "findings": [
            {
                "sample_id": "sample_a",
                "evidence_text": "Rheology evidence.",
                "fibre_outcome": "fibres",
                "fibre_evidence": "Fibre evidence.",
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0005", source_pdf="paper.pdf")

    assert normalized["findings"][0]["sample"]["evidence_text"] == "Rheology evidence."
    assert normalized["findings"][0]["fibre_outcome"]["evidence_text"] == "Fibre evidence."


def test_beads_without_nanofibres_normalizes_as_failed_outcome():
    payload = {
        "paper_id": "paper_0002",
        "has_rheology_chart": False,
        "findings": [],
        "fibre_outcomes": [
            {
                "sample_id": "high lignin fraction",
                "outcome": "beaded_fibres",
                "evidence": "couldn't be electrospun into NFs. The materials form beads.",
            }
        ],
    }

    normalized = normalize_extraction_payload(payload, paper_id="paper_0002", source_pdf="paper.pdf")

    assert normalized["findings"][0]["fibre_outcome"]["outcome"] == "failed or no fibres"
