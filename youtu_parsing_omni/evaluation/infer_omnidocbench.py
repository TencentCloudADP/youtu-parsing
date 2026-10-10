#!/usr/bin/env python3
"""Youtu-Parsing-Omni inference on OmniDocBench v1.6.

Every page image in ``<omnidocbench-root>/images`` is parsed. Writes ``<output-dir>/predictions/document.jsonl`` with one
``{"id", "raw_output", "parsed_json", "parse_status"}`` per page, and one Markdown file
per page in ``<output-dir>/predictions/document_markdown/<page>.md``, which is the
``prediction.data_path`` of the official OmniDocBench evaluator
(``scripts/eval_omnidocbench.sh``). ``raw_output`` is the verbatim answer; the rest
comes from ``postprocess`` (JSON repair, tail-repetition trimming, JSON -> Markdown).
Finished pages are skipped when the command is rerun.

    python evaluation/infer_omnidocbench.py --api-base http://127.0.0.1:8000/v1 \
        --omnidocbench-root /path/to/OmniDocBench --output-dir outputs/omnidocbench
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import requests

from postprocess import DOCUMENT_MARKDOWN_DIR, postprocess, write_markdown

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "youtu_parsing_omni.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_NAME = "Youtu-Parsing-Omni"  # --served-model-name of scripts/vllm.sh
CHAT_TEMPLATE_KWARGS = {"enable_parsing": True, "enable_thinking": False}
EMPTY_RETRIES = 5

SESSION = requests.Session()
SESSION.trust_env = False  # local servers: never go through HTTP(S)_PROXY


def load_samples(data_root: Path) -> list[dict]:
    """One sample per page image in ``<data_root>/images`` (id = image file name)."""
    prompt = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))["prompts"]["document"]
    image_dir = data_root / "images"
    if not image_dir.is_dir():
        raise SystemExit(f"{image_dir} not found: --omnidocbench-root must contain images/")
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no page images in {image_dir}")
    return [{"id": p.name, "path": p.resolve(), "prompt": prompt} for p in images]


def post_chat(api_base: str, payload: dict, retries: int = 8) -> dict:
    for attempt in range(retries):
        try:
            response = SESSION.post(f"{api_base}/chat/completions", json=payload, timeout=21600,
                                    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"})
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            body = response.json()
            choice = body["choices"][0]
            choice["usage"] = body.get("usage") or {}
            return choice
        except Exception as error:  # noqa: BLE001
            print(f"  request failed {api_base} ({attempt + 1}/{retries}): {error}", flush=True)
            if attempt + 1 == retries:
                raise
            time.sleep(5)
    raise AssertionError("unreachable")


def build_payload(sample: dict) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": sample["path"].as_uri()}},
            {"type": "text", "text": sample["prompt"]},
        ]}],
        "max_tokens": 32768,
        "min_tokens": 16,
        "temperature": 0.0,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "skip_special_tokens": False,
        "logprobs": True,
    }


def generate(api_base: str, sample: dict) -> str:
    """The answer is rebuilt from the per-token logprobs so that special tokens
    (coordinates) are kept verbatim; empty answers are retried."""
    for attempt in range(EMPTY_RETRIES):
        choice = post_chat(api_base, build_payload(sample))
        tokens = (choice.get("logprobs") or {}).get("content") or []
        text = "".join(t.get("token") or bytes(t.get("bytes") or []).decode("utf-8", "replace")
                       for t in tokens) or choice["message"].get("content") or ""
        if text.strip():
            usage = choice.get("usage") or {}
            finish = choice.get("finish_reason")
            warn = "  WARNING hit max_tokens" if finish == "length" else ""
            print(f"  {sample['id']} prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')} finish={finish}{warn}", flush=True)
            return text
        print(f"  empty response {sample['id']} ({attempt + 1}/{EMPTY_RETRIES})", flush=True)
    raise RuntimeError("empty response")


def wait_ready(api_base: str) -> None:
    started = time.time()
    for attempt in range(720):
        try:
            if SESSION.get(f"{api_base}/models", timeout=5).ok:
                print(f"endpoint ready: {api_base} ({time.time() - started:.0f}s)", flush=True)
                return
        except requests.RequestException:
            pass
        if attempt % 12 == 0:
            print(f"waiting for {api_base} ({time.time() - started:.0f}s)", flush=True)
        time.sleep(5)
    raise TimeoutError(f"{api_base} is not ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1",
                        help="comma-separated OpenAI-compatible endpoints (one worker each)")
    parser.add_argument("--workers-per-endpoint", type=int, default=1)
    parser.add_argument("--omnidocbench-root", type=Path, required=True,
                        help="OmniDocBench dataset directory (contains images/ and OmniDocBench.json)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "document.jsonl"
    markdown_dir = pred_dir / DOCUMENT_MARKDOWN_DIR

    done = {json.loads(line)["id"] for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()} \
        if pred_file.exists() else set()
    samples = [s for s in load_samples(args.omnidocbench_root.resolve()) if s["id"] not in done]
    print(f"document: {len(done)} done, {len(samples)} to run", flush=True)
    if not samples:
        return 0
    print(f"  first: id={samples[0]['id']} media={samples[0]['path']} exists={samples[0]['path'].is_file()} "
          f"prompt={samples[0]['prompt'][:60]!r}", flush=True)
    missing = [s["id"] for s in samples if not s["path"].is_file()]
    if missing:
        print(f"  WARNING {len(missing)} media files missing, e.g. {missing[:3]}", flush=True)

    todo: queue.Queue = queue.Queue()
    for sample in samples:
        todo.put(sample)
    total = todo.qsize()

    endpoints = [u.rstrip("/") for u in args.api_base.split(",")]
    for url in endpoints:
        wait_ready(url)

    lock = threading.Lock()
    failed: list[str] = []
    finished = [0]
    started = time.time()

    def worker(url: str) -> None:
        while True:
            try:
                sample = todo.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            try:
                text = generate(url, sample)
            except Exception as error:  # noqa: BLE001
                with lock:
                    failed.append(sample["id"])
                    finished[0] += 1
                print(f"FAILED {sample['id']} endpoint={url} after {time.time() - t0:.0f}s: {error}", flush=True)
                continue
            row = {"id": sample["id"], "raw_output": text}
            try:
                fields, markdown = postprocess(text)
                # written before the row, so that every finished row has its page
                write_markdown(markdown_dir, sample["id"], markdown)
                row.update(fields)
            except Exception as error:  # noqa: BLE001
                row["postprocess_error"] = f"{type(error).__name__}: {error}"
                print(f"  WARNING postprocess failed {sample['id']}: {row['postprocess_error']}", flush=True)
            with lock:
                with pred_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                finished[0] += 1
                elapsed = time.time() - started
                eta = elapsed / finished[0] * (total - finished[0])
                print(f"[{finished[0]}/{total}] {sample['id']} port={url.rsplit(':', 1)[-1].split('/')[0]} "
                      f"{time.time() - t0:.1f}s chars={len(text)} tail={text[-30:]!r} "
                      f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)

    threads = [threading.Thread(target=worker, args=(url,), daemon=True)
               for url in endpoints for _ in range(args.workers_per_endpoint)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    status = [r.get("parse_status") or {} for r in rows]
    print(f"done: {total - len(failed)}/{total} succeeded in {(time.time() - started) / 60:.1f}m; "
          f"{len(rows)} pages in {pred_file}", flush=True)
    print(f"  json repaired={sum(bool(s.get('is_repaired')) for s in status)} "
          f"tail repetition trimmed={sum(bool(s.get('is_tail_repetition_trimmed')) for s in status)} "
          f"pathological repetition (empty page)={sum(bool(s.get('is_pathological_repetition')) for s in status)} "
          f"fatal (empty page)={sum(bool(s.get('is_fatal')) for s in status)} "
          f"postprocess errors={sum('postprocess_error' in r for r in rows)}", flush=True)
    print(f"  page Markdown: {markdown_dir}", flush=True)
    if failed:
        print(f"{len(failed)} pages failed (e.g. {failed[:5]}); rerun the same command to retry them", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
