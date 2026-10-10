"""JSON repair of document outputs that ``json`` cannot decode.

1. strict  -- ``json.loads`` (non-strict control characters), then again with
   stray backslashes (invalid JSON escapes, e.g. LaTeX ``\\alpha``) escaped;
   elements are reordered by ``reading_order``.
2. tolerant -- regex scan of ``"category"`` / content-field pairs, for outputs
   that are truncated or contain unescaped quotes.
"""

from __future__ import annotations

import json
import re
from typing import Any

CAT_RE = re.compile(r'"category"\s*:\s*"([A-Za-z_]+)"')
CONTENT_KEY_RE = re.compile(r'"(text|latex|otsl|markdown|caption|html)"\s*:\s*"')
STR_END_RE = re.compile(r"\s*[,}\]]")

VALID_ESCAPES = set('"\\/bfnrtu')


def _escape_stray_backslashes(s: str) -> str:
    """Replace backslashes that don't form a valid JSON escape with ``\\\\``."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            if s[i + 1] in VALID_ESCAPES:
                out.append(c)
                out.append(s[i + 1])
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _read_json_string(s: str, i: int) -> str:
    """Read a JSON string body starting at *i*, tolerating unescaped quotes."""
    buf: list[str] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            buf.append(c)
            buf.append(s[i + 1])
            i += 2
            continue
        if c == '"':
            if STR_END_RE.match(s, i + 1):
                return "".join(buf)
            buf.append('\\"')
            i += 1
            continue
        buf.append(c)
        i += 1
    return "".join(buf)


def _unescape(s: str) -> str:
    """Unescape a JSON string value; falls back to stray-backslash fix."""
    for candidate in (s, _escape_stray_backslashes(s)):
        try:
            return json.loads('"' + candidate + '"', strict=False)
        except ValueError:
            continue
    return s


def _parse_strict(raw: str) -> Any:
    """Try ``json.loads`` on *raw*, then on an escaped-backslash version."""
    for candidate in (raw, _escape_stray_backslashes(raw)):
        try:
            return json.loads(candidate, strict=False)
        except ValueError:
            continue
    return None


def _parse_tolerant(raw: str) -> list[dict[str, Any]]:
    """Recover ``{"category", "content"}`` elements (no id / bbox) from malformed JSON."""
    marks: list[tuple[int, str]] = [(m.start(), m.group(1)) for m in CAT_RE.finditer(raw)]
    elements: list[dict[str, Any]] = []
    for idx, (pos, category) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(raw)
        segment = raw[pos:end]
        content: dict[str, Any] = {}
        for km in CONTENT_KEY_RE.finditer(segment):
            key = km.group(1)
            if key in content:
                continue
            content[key] = _unescape(_read_json_string(segment, km.end()))
        elements.append({"category": category, "content": content})
    return elements


def _ordered_elements(inner: dict) -> list[dict]:
    """Elements referenced by ``reading_order`` first (in that order), then the others."""
    elements: list[dict] = [e for e in inner.get("elements", []) if isinstance(e, dict)]
    order = inner.get("reading_order")
    if not isinstance(order, list) or not order:
        return elements
    by_id = {e.get("id"): e for e in elements if e.get("id")}
    ordered: list[dict] = [by_id[i] for i in order if i in by_id]
    seen = set(order)
    ordered.extend(e for e in elements if e.get("id") not in seen)
    return ordered


def repair_document_json(raw_str: str) -> tuple[dict | None, str | None]:
    """Return ``({"elements", "reading_order"}, method)``, or ``(None, None)`` if both layers fail.

    *raw_str* must already be stripped of the code fence and trailing special tokens.
    """
    inner = _parse_strict(raw_str)
    if isinstance(inner, dict):
        elements = _ordered_elements(inner)
        reading_order = inner.get("reading_order")
        reading_order = reading_order if isinstance(reading_order, list) else []
        return {"elements": elements, "reading_order": reading_order}, "strict"
    if isinstance(inner, list):
        return {"elements": inner, "reading_order": []}, "strict"

    elements = _parse_tolerant(raw_str)
    if elements:
        return {"elements": elements, "reading_order": []}, "tolerant"

    return None, None
