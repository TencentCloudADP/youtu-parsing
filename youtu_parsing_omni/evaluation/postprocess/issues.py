"""Issues recorded while post-processing one page (stored in ``parse_status["issues"]``)."""

from __future__ import annotations

import json
from typing import Any

# levels
SAMPLE = "sample"   # fatal: the page Markdown is empty
ENTRY = "entry"     # one element is dropped / degraded
ATTR = "attr"       # one field is repaired / filled with a default

# codes
SAMPLE_JSON_SYNTAX = "sample.json_syntax"
SAMPLE_NOT_ARRAY = "sample.not_array"
SAMPLE_EMPTY_OUTPUT = "sample.empty_output"
ENTRY_NOT_DICT = "entry.not_dict"
ENTRY_UNKNOWN_CATEGORY = "entry.unknown_category"
ATTR_MISSING_ATTR = "attr.missing_attr"
ATTR_INVALID_BBOX = "attr.invalid_bbox"
ATTR_READING_ORDER_BROKEN_REF = "attr.reading_order_broken_ref"
ATTR_READING_ORDER_MISSING_ELEM = "attr.reading_order_missing_elem"
ATTR_CONVERSION_FAILED = "attr.conversion_failed"
ATTR_JSON_REPAIRED = "attr.json_repaired"
ATTR_TAIL_REPETITION_REPAIRED = "attr.tail_repetition_repaired"
ATTR_PATHOLOGICAL_REPETITION = "attr.pathological_repetition"


def clip_issue_value(v: Any, max_len: int = 200) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        raw = v
    else:
        try:
            raw = json.dumps(v, ensure_ascii=False)
        except Exception:
            raw = repr(v)
    if isinstance(raw, str) and len(raw) > max_len:
        return raw[:max_len] + "...(truncated)"
    return raw


def add_issue(
    issues: list[dict[str, Any]],
    *,
    level: str,
    code: str,
    message: str,
    entry_index: int | None = None,
    field: str | None = None,
    value: Any = None,
) -> None:
    issues.append({
        "level": level,
        "code": code,
        "message": message,
        "entry_index": entry_index,
        "field": field,
        "value": clip_issue_value(value),
    })
