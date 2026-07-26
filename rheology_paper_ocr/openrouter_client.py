from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx

from rheology_paper_ocr.schemas import PaperLLMExtraction



def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extraction_schema() -> dict[str, Any]:
    return PaperLLMExtraction.model_json_schema()


def build_chat_payload(
    model: str,
    prompt: str,
    image_paths: list[Path],
    use_schema: bool = True,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "paper_rheology_extraction",
                "strict": False,
                "schema": extraction_schema(),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    return payload


def parse_json_response(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model or os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-sonnet-4.6"
        self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "3000"))

    def extract(self, prompt: str, image_paths: list[Path], raw_response_path: Path | None = None) -> dict[str, Any]:
        payload = build_chat_payload(self.model, prompt, image_paths, use_schema=True, max_tokens=self.max_tokens)
        response = self._post_chat_with_retries(payload)
        data = response.json()
        if "choices" not in data and _is_schema_rejection(data):
            if raw_response_path:
                _write_json(raw_response_path.with_name(f"{raw_response_path.stem}_schema_rejected.json"), data)
            payload = build_chat_payload(self.model, prompt, image_paths, use_schema=False, max_tokens=self.max_tokens)
            response = self._post_chat_with_retries(payload)
            data = response.json()
        if raw_response_path:
            _write_json(raw_response_path, data)
        if "choices" not in data:
            message = data.get("error", {}).get("message") or json.dumps(data)[:1000]
            raise RuntimeError(f"OpenRouter response did not include choices: {message}")
        response.raise_for_status()
        content = data["choices"][0]["message"]["content"]
        try:
            return parse_json_response(content)
        except json.JSONDecodeError:
            if raw_response_path:
                _write_json(raw_response_path.with_name(f"{raw_response_path.stem}_malformed_schema.json"), data)
            fallback_payload = build_chat_payload(self.model, prompt, image_paths, use_schema=False, max_tokens=self.max_tokens)
            fallback_response = self._post_chat_with_retries(fallback_payload)
            fallback_data = fallback_response.json()
            if raw_response_path:
                _write_json(raw_response_path, fallback_data)
            if "choices" not in fallback_data:
                message = fallback_data.get("error", {}).get("message") or json.dumps(fallback_data)[:1000]
                raise RuntimeError(f"OpenRouter fallback response did not include choices: {message}")
            fallback_response.raise_for_status()
            return parse_json_response(fallback_data["choices"][0]["message"]["content"])

    def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/zilun-lin/rheology-paper-ocr",
                "X-Title": "rheology-paper-ocr",
            },
            json=payload,
            timeout=httpx.Timeout(connect=30.0, read=None, write=60.0, pool=30.0),
        )

    def _post_chat_with_retries(self, payload: dict[str, Any], attempts: int = 2) -> httpx.Response:
        last_exc: httpx.ReadTimeout | None = None
        for _ in range(attempts):
            try:
                return self._post_chat(payload)
            except httpx.ReadTimeout as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc


def _is_schema_rejection(data: dict[str, Any]) -> bool:
    error = data.get("error", {})
    if error.get("code") == 400:
        return True
    message = (error.get("message") or "").lower()
    return "response_format" in message and ("schema" in message or "json_schema" in message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
