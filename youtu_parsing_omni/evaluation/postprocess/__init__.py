"""Post-processing of Youtu-Parsing-Omni OmniDocBench outputs (run by ``infer_omnidocbench.py``).

The raw JSON answer is repaired (code fence / special tokens, tail repetition,
malformed JSON) and converted into the per-page Markdown that the official
OmniDocBench evaluator reads from ``predictions/document_markdown/<page>.md``.
"""

from .pipeline import DOCUMENT_MARKDOWN_DIR, markdown_path, postprocess, write_markdown

__all__ = ["DOCUMENT_MARKDOWN_DIR", "markdown_path", "postprocess", "write_markdown"]
