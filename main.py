import json
import os
import re
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

app = FastAPI(title="Invoice Extractor API", version="0.1.0")


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw invoice or receipt text")


class ExtractedInvoice(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    needs_review: bool = False
    review_reason: Optional[str] = None


def _parse_json_from_llm(raw_content: str) -> dict:
    cleaned = raw_content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": OLLAMA_MODEL, "ollama_url": OLLAMA_BASE_URL}


@app.post("/extract", response_model=ExtractedInvoice)
async def extract_invoice(payload: ExtractRequest) -> ExtractedInvoice:
    system_prompt = (
        "You are a financial document parser. Extract fields from the receipt or invoice text "
        "and return ONLY valid JSON with these exact keys: "
        '{"vendor": string|null, "date": string|null, "total": number|null, '
        '"currency": string|null, "category": string|null}. '
        "Use YYYY-MM-DD for date when possible. Use a 3-letter code like USD or EUR for currency. "
        "Choose category from: software, travel, meals, office_supplies, utilities, other. "
        "Use null if a field is not present."
    )

    request_body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.text},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=request_body)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    raw_text = resp.json().get("message", {}).get("content", "")

    try:
        parsed = _parse_json_from_llm(raw_text)
    except json.JSONDecodeError as exc:
        return ExtractedInvoice(
            needs_review=True,
            review_reason=f"Model returned non-JSON output: {exc}",
        )

    invoice = ExtractedInvoice(
        vendor=parsed.get("vendor"),
        date=parsed.get("date"),
        total=parsed.get("total"),
        currency=parsed.get("currency"),
        category=parsed.get("category"),
    )

    missing_fields = [
        name
        for name, val in [
            ("vendor", invoice.vendor),
            ("total", invoice.total),
            ("currency", invoice.currency),
        ]
        if val is None
    ]
    if missing_fields:
        invoice.needs_review = True
        invoice.review_reason = f"Missing required fields: {', '.join(missing_fields)}"

    return invoice
