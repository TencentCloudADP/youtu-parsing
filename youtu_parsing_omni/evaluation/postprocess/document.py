"""Document JSON -> per-page Markdown (the OmniDocBench prediction).

Elements are mapped to text blocks (formula -> LaTeX, table -> HTML, text-like
-> text), ordered by ``reading_order`` (elements outside it are appended) and
joined with blank lines. Schema problems are recorded as issues.
"""

from __future__ import annotations

import re
from typing import Any

from .issues import (
    ATTR,
    ATTR_CONVERSION_FAILED,
    ATTR_INVALID_BBOX,
    ATTR_MISSING_ATTR,
    ATTR_READING_ORDER_BROKEN_REF,
    ATTR_READING_ORDER_MISSING_ELEM,
    ENTRY,
    ENTRY_NOT_DICT,
    ENTRY_UNKNOWN_CATEGORY,
    SAMPLE,
    SAMPLE_NOT_ARRAY,
    add_issue,
)
from .table import is_valid_html_table, markdown_to_html_table, otsl_to_html

# "<box><x_N><y_N><x_N><y_N></box>", N in [0, 1000]
_BOX_RE = re.compile(r"<box><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></box>")

TEXT_CATEGORIES = {"text", "title", "header", "footer", "caption", "code"}
VISUAL_CATEGORIES = {"figure", "chart", "flowchart", "geometric"}

# (element id, block text)
Block = tuple[str, str]


def bbox_error(raw: Any) -> str | None:
    if raw is None:
        return "bbox missing; filled with default"
    if not isinstance(raw, str):
        return "bbox not a string; filled with default"
    m = _BOX_RE.search(raw)
    if not m:
        return "bbox string unparseable; filled with default"
    x1, y1, x2, y2 = (float(g) for g in m.groups())
    if x1 > x2 or y1 > y2:
        return "bbox ordering invalid; filled with default"
    return None


def _first_text(content: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def element_block(elem: dict[str, Any], *, issues: list, entry_index: int) -> Block | None:
    """Text block of one element (aligned with ``omni2layout.element_text``)."""
    err = bbox_error(elem.get("bbox"))
    if err:
        add_issue(issues, level=ATTR, code=ATTR_INVALID_BBOX, field="bbox",
                  entry_index=entry_index, value=elem.get("bbox"), message=err)

    category = str(elem.get("category", "")).strip().lower()
    content = elem.get("content")
    if not isinstance(content, dict):
        content = {}
    elem_id = str(elem.get("id", "") or "")

    if category == "formula":
        latex = content.get("latex")
        if isinstance(latex, str) and latex.strip():
            return elem_id, latex.strip()
        text = _first_text(content, ("text", "markdown", "latex", "html", "caption"))
        if text is not None:
            return elem_id, text
        add_issue(issues, level=ATTR, code=ATTR_MISSING_ATTR, field="content.latex",
                  entry_index=entry_index, value=None, message="filled with default")
        return elem_id, ""

    if category == "table":
        # html -> markdown -> otsl
        html_raw = content.get("html")
        if html_raw is not None:
            html_str = str(html_raw).strip()
            if is_valid_html_table(html_str):
                return elem_id, html_str
            add_issue(issues, level=ATTR, code=ATTR_CONVERSION_FAILED, field="content.html",
                      entry_index=entry_index, value=str(html_raw),
                      message="content.html is not a valid HTML table; filled with empty")
            return elem_id, ""

        md_raw = content.get("markdown")
        if md_raw is not None:
            html = markdown_to_html_table(str(md_raw))
            if is_valid_html_table(html):
                return elem_id, html
            add_issue(issues, level=ATTR, code=ATTR_CONVERSION_FAILED, field="content.markdown",
                      entry_index=entry_index, value=str(md_raw),
                      message="markdown could not be converted to an HTML table; filled with empty")
            return elem_id, ""

        otsl_raw = content.get("otsl")
        if otsl_raw is not None:
            html = otsl_to_html(str(otsl_raw))
            if not is_valid_html_table(html):
                add_issue(issues, level=ATTR, code=ATTR_CONVERSION_FAILED, field="content.otsl",
                          entry_index=entry_index, value=str(otsl_raw),
                          message="otsl could not be converted to a valid HTML table; kept raw output")
            return elem_id, html

        add_issue(issues, level=ATTR, code=ATTR_MISSING_ATTR, field="content.otsl/html/markdown",
                  entry_index=entry_index, value=None,
                  message="no table content field found; filled with default")
        return elem_id, ""

    if category in TEXT_CATEGORIES:
        text = _first_text(content, ("text", "markdown", "latex", "html", "caption"))
        if text is not None:
            return elem_id, text
        add_issue(issues, level=ATTR, code=ATTR_MISSING_ATTR, field="content.text",
                  entry_index=entry_index, value=None, message="filled with default")
        return elem_id, ""

    add_issue(issues, level=ENTRY, code=ENTRY_UNKNOWN_CATEGORY, entry_index=entry_index,
              field="category", value=category,
              message="unsupported category for document task; falling back to text extraction")
    if category in VISUAL_CATEGORIES:
        return None
    text = _first_text(content, ("text", "markdown", "latex", "html", "caption"))
    return (elem_id, text) if text is not None else None


def blocks_to_markdown(blocks: list[Block], reading_order: list[str]) -> str:
    if reading_order:
        ro_set = set(reading_order)
        by_id: dict[str, Block] = {}
        not_in_ro: list[Block] = []
        for block in blocks:
            if block[0] and block[0] in ro_set:
                by_id[block[0]] = block
            else:
                not_in_ro.append(block)
        ordered = [by_id[rid] for rid in reading_order if rid in by_id]
        ordered.extend(not_in_ro)
    else:
        ordered = blocks
    parts = [text.strip() for _, text in ordered]
    return "\n\n".join(part for part in parts if part)


def document_to_markdown(parsed: dict[str, Any], issues: list) -> tuple[str, int, int]:
    """Return ``(markdown, n_entries_in, n_entries_kept)``; issues are appended.

    A missing / non-list ``elements`` is fatal (``sample.not_array``, empty Markdown).
    """
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        add_issue(issues, level=SAMPLE, code=SAMPLE_NOT_ARRAY, value=elements,
                  message="field `elements` missing or not array")
        return "", 0, 0

    reading_order = parsed.get("reading_order")
    if isinstance(reading_order, list):
        reading_order = [str(x) for x in reading_order]
    else:
        add_issue(issues, level=ATTR, code=ATTR_MISSING_ATTR, field="reading_order",
                  value=reading_order, message="missing or not a list, filled with default")
        reading_order = []

    blocks: list[Block] = []
    for idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            add_issue(issues, level=ENTRY, code=ENTRY_NOT_DICT, entry_index=idx, value=elem,
                      message=f"element is not dict: {type(elem).__name__}")
            continue
        block = element_block(elem, issues=issues, entry_index=idx)
        if block is not None:
            blocks.append(block)

    if reading_order:
        item_ids = {block_id for block_id, _ in blocks if block_id}
        ro_set = set(reading_order)
        broken_refs = ro_set - item_ids
        if broken_refs:
            add_issue(issues, level=ATTR, code=ATTR_READING_ORDER_BROKEN_REF, field="reading_order",
                      value=sorted(broken_refs),
                      message=f"reading_order references {len(broken_refs)} element(s) not found in elements")
        missing_from_ro = item_ids - ro_set
        if missing_from_ro:
            add_issue(issues, level=ATTR, code=ATTR_READING_ORDER_MISSING_ELEM, field="reading_order",
                      value=sorted(missing_from_ro),
                      message=f"{len(missing_from_ro)} element(s) not in reading_order")

    return blocks_to_markdown(blocks, reading_order), len(elements), len(blocks)
