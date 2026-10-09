#!/usr/bin/env python3
"""Minimal Hugging Face Transformers inference for Youtu-Parsing-Omni.

Requires ``transformers==5.10.2`` (``bash scripts/setup_env.sh --transformers``),
installed in an environment separate from vLLM (which needs ``transformers==5.2.0``).
The reported results were produced with vLLM (``scripts/vllm.sh``).

    python examples/infer_transformers.py --model tencent/Youtu-Parsing-Omni --task document --media page.png
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "youtu_parsing_omni.json"
MODALITY = {"document": "image", "natural_image": "image", "graphics_chart": "image",
            "graphics_flowchart": "image", "graphics_geometric": "image", "audio": "audio",
            "natural_video": "video", "textrich_video": "video"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="local checkpoint directory or Hub repo id (e.g. tencent/Youtu-Parsing-Omni)")
parser.add_argument("--media", type=Path, required=True)
parser.add_argument("--task", default="document", choices=sorted(MODALITY))
parser.add_argument("--revision", default=None,
                    help="commit hash of the model repository (recommended when --model is a Hub id: "
                         "trust_remote_code=True executes Python code shipped with the checkpoint)")
parser.add_argument("--download-dir", type=Path, default=Path.home() / ".cache" / "youtu-parsing-omni",
                    help="where Hub repo ids are materialised (ignored when --model is a local directory)")
args = parser.parse_args()

modality = MODALITY[args.task]
prompt = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"][args.task]

if Path(args.model).is_dir():
    model_dir = str(args.model)
else:
    model_dir = snapshot_download(args.model, revision=args.revision,
                                  local_dir=args.download_dir / Path(args.model).name)

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto").eval()

messages = [{"role": "user", "content": [
    {"type": modality, modality: str(args.media)},
    {"type": "text", "text": prompt},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False, enable_parsing=True)
media_arg = {"image": "images", "audio": "audio", "video": "videos"}[modality]
inputs = processor(text=text, **{media_arg: [str(args.media)]}, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32768, do_sample=False)
# keep special tokens: the <box><x_..><y_..></box> coordinates are special tokens
print(processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)[0])
