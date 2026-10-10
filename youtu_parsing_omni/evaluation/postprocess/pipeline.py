"""Post-processing of one raw OmniDocBench output into the page Markdown.

Steps: restore the code fence -> strip -> drop trailing special tokens -> drop the
```json code fence -> trim tail repetition -> decode the first JSON value,
repairing it on failure -> JSON -> Markdown -> empty a pathological repetition page.
"""

from __future__ import annotations

import copy
import json
import os
import re
import zlib
from pathlib import Path
from typing import Any

from .document import document_to_markdown
from .issues import (
    ATTR,
    ATTR_JSON_REPAIRED,
    ATTR_PATHOLOGICAL_REPETITION,
    ATTR_TAIL_REPETITION_REPAIRED,
    SAMPLE,
    SAMPLE_EMPTY_OUTPUT,
    SAMPLE_JSON_SYNTAX,
    SAMPLE_NOT_ARRAY,
    add_issue,
)
from .repair import repair_document_json
from .tail_repetition import detect_tail_repetition, trim_tail_repetition

VERSION = 1

#: Trim degenerate repetition at the tail of the output before decoding.
TAIL_REPETITION = {
    "min_reps": 3.0,   # minimum number of complete repetitions at the tail
    "min_span": 80,    # minimum number of characters of a repeated span
    "max_p": 512,      # maximum candidate period length
    "keep_reps": 1,    # number of periods kept when trimming
    "tail_n": 4096,    # only search the last N characters
}

#: A page Markdown of at least ``min_bytes`` UTF-8 bytes whose zlib (level 1)
#: compression ratio is at most ``max_ratio`` is a generation loop that survived the
#: tail trimming (e.g. a repeated table row); it is written as an empty page, which
#: the official evaluator scores as an empty prediction.
PATHOLOGICAL_REPETITION = {"min_bytes": 20000, "max_ratio": 0.05}

#: Per-page Markdown: ``<output-dir>/predictions/document_markdown/<page>.md``
#: (``<page>`` = image file name without extension), the official ``prediction.data_path``.
DOCUMENT_MARKDOWN_DIR = "document_markdown"


def markdown_path(markdown_dir: str | Path, sample_id: str) -> Path:
    """``<markdown_dir>/<sample id without extension>.md``; rejects ids that are not plain file names."""
    if not sample_id or os.path.basename(sample_id) != sample_id or sample_id in (".", ".."):
        raise ValueError(f"invalid document sample id {sample_id!r}")
    stem = os.path.splitext(sample_id)[0]
    if not stem:
        raise ValueError(f"invalid document sample id {sample_id!r}")
    return Path(markdown_dir) / f"{stem}.md"


def write_markdown(markdown_dir: str | Path, sample_id: str, markdown: str) -> Path:
    """Write one page atomically (tmp file + rename); text is written verbatim (no newline translation)."""
    path = markdown_path(markdown_dir, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(markdown)
    os.replace(tmp, path)
    return path


def restore_document_fence(raw_output: str) -> str:
    """Restore the code fence of outputs that start with ``json\\n``."""
    if raw_output.startswith("json\n"):
        return "```" + raw_output
    return raw_output


def decode(raw_output: str, issues: list) -> tuple[dict | None, bool]:
    """Decode the JSON object of *raw_output*; returns ``(object | None, tail_repetition_detected)``.

    ``None`` means the page is fatal (a ``sample.*`` issue is appended).
    """
    raw_str = str(raw_output).strip()
    if not raw_str:
        add_issue(issues, level=SAMPLE, code=SAMPLE_EMPTY_OUTPUT, message="empty raw_output", value=raw_str)
        return None, False

    # Trailing special tokens first, so that a closing ``` lands at the end.
    raw_str = re.sub(r"(?:<\|[^|]*\|>\s*)+$", "", raw_str).strip()
    if raw_str.startswith("```"):
        raw_str = re.sub(r"^```(?:json)?\s*\n?", "", raw_str)
        raw_str = re.sub(r"\n?```\s*$", "", raw_str)

    cfg = TAIL_REPETITION
    tail_rep = detect_tail_repetition(raw_str, max_p=int(cfg["max_p"]), min_reps=float(cfg["min_reps"]),
                                      min_span=int(cfg["min_span"]), tail_n=int(cfg["tail_n"]))
    if tail_rep is not None:
        raw_str = trim_tail_repetition(raw_str, tail_rep, keep_reps=int(cfg["keep_reps"]))
        add_issue(issues, level=ATTR, code=ATTR_TAIL_REPETITION_REPAIRED,
                  message=(f"tail repetition trimmed: period={tail_rep.period} "
                           f"reps={tail_rep.reps:.1f} span={tail_rep.span}"),
                  value=tail_rep.unit[:80])
    tail_detected = tail_rep is not None

    # Only the first top-level value is decoded; trailing residue is ignored.
    parsed: Any = None
    try:
        parsed, _end = json.JSONDecoder().raw_decode(raw_str)
    except json.JSONDecodeError as exc:
        parsed, method = repair_document_json(raw_str)
        if parsed is None:
            add_issue(issues, level=SAMPLE, code=SAMPLE_JSON_SYNTAX, message=str(exc), value=raw_str)
            return None, tail_detected
        add_issue(issues, level=ATTR, code=ATTR_JSON_REPAIRED,
                  message=f"raw_decode failed, recovered via {method}", value=str(exc))

    if not isinstance(parsed, dict):
        add_issue(issues, level=SAMPLE, code=SAMPLE_NOT_ARRAY, value=parsed,
                  message=f"top-level type is {type(parsed).__name__}, expected object with elements")
        return None, tail_detected
    return parsed, tail_detected


def pathological_repetition(markdown: str) -> str | None:
    """Reason string if *markdown* is a pathological repetition loop, else ``None``."""
    raw = markdown.encode("utf-8")
    if len(raw) < PATHOLOGICAL_REPETITION["min_bytes"]:
        return None
    ratio = len(zlib.compress(raw, level=1)) / max(1, len(raw))
    if ratio > PATHOLOGICAL_REPETITION["max_ratio"]:
        return None
    return (f"markdown bytes={len(raw)} compression_ratio={ratio:.6f} "
            f"threshold={PATHOLOGICAL_REPETITION['max_ratio']:.6f}")


def postprocess(raw_output: str) -> tuple[dict[str, Any], str]:
    """Return ``(fields, markdown)`` of one OmniDocBench page.

    *fields* are stored next to ``raw_output`` in ``predictions/document.jsonl``:

    * ``parsed_json`` -- decoded / repaired JSON object (``null`` if undecodable)
    * ``parse_status`` -- flags + issues of decoding and Markdown conversion

    *markdown* is the page prediction (``""`` when the page is fatal or a
    pathological repetition); save it with :func:`write_markdown`.
    """
    issues: list[dict[str, Any]] = []
    parsed, tail_detected = decode(restore_document_fence(raw_output), issues)
    out: dict[str, Any] = {"parsed_json": copy.deepcopy(parsed)}
    markdown, n_in, n_kept = "", 0, 0
    if parsed is not None:
        markdown, n_in, n_kept = document_to_markdown(parsed, issues)

    pathological = pathological_repetition(markdown)
    if pathological is not None:
        add_issue(issues, level=ATTR, code=ATTR_PATHOLOGICAL_REPETITION,
                  message=f"page written empty: {pathological}", value=markdown[:80])
        markdown = ""

    codes = {issue["code"] for issue in issues}
    fatal = next((issue["code"] for issue in issues if issue["level"] == SAMPLE), None)
    out["parse_status"] = {
        "version": VERSION,
        "is_fatal": fatal is not None,
        "fatal_code": fatal,
        "is_repaired": ATTR_JSON_REPAIRED in codes,
        "is_tail_repetition": tail_detected,
        "is_tail_repetition_trimmed": ATTR_TAIL_REPETITION_REPAIRED in codes,
        "is_pathological_repetition": pathological is not None,
        "n_entries_in": n_in,
        "n_entries_kept": n_kept,
        "issues": issues,
    }
    return out, markdown
