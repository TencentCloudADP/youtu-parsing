"""Re-run the post-processing on an existing ``predictions/document.jsonl``.

    python evaluation/postprocess --input outputs/omnidocbench/predictions/document.jsonl \
        --output outputs/omnidocbench_repost/predictions/document.jsonl

Only ``id`` and ``raw_output`` of the input rows are used. The per-page Markdown goes
to ``--markdown-dir`` (default: ``document_markdown/`` next to ``--output``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:  # run as ``python evaluation/postprocess``
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from postprocess import DOCUMENT_MARKDOWN_DIR, postprocess, write_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="prediction JSONL with id / raw_output")
    parser.add_argument("--output", type=Path, required=True, help="output JSONL (must differ from --input)")
    parser.add_argument("--markdown-dir", type=Path, default=None,
                        help=f"per-page .md directory (default: <output dir>/{DOCUMENT_MARKDOWN_DIR})")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    markdown_dir = args.markdown_dir or args.output.parent / DOCUMENT_MARKDOWN_DIR
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            fields, markdown = postprocess(row["raw_output"])
            write_markdown(markdown_dir, row["id"], markdown)
            sink.write(json.dumps({"id": row["id"], "raw_output": row["raw_output"], **fields},
                                  ensure_ascii=False) + "\n")
            count += 1
    print(f"{count} rows -> {args.output}")
    print(f"{count} pages -> {markdown_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
