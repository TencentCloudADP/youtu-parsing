#!/usr/bin/env python3
"""Parse one image / audio / video with a running Youtu-Parsing-Omni server.

Start a server first (``GPUS=0 MODEL=... DATA_ROOT=/path/to/media bash scripts/vllm.sh``), then::

    python examples/infer_vllm.py --task document       --media page.png
    python examples/infer_vllm.py --task natural_image  --media photo.jpg
    python examples/infer_vllm.py --task graphics_chart --media chart.png
    python examples/infer_vllm.py --task audio          --media speech.wav
    python examples/infer_vllm.py --task natural_video  --media clip.mp4
    python examples/infer_vllm.py --task textrich_video --media lecture.mp4
"""

import argparse
import json
from pathlib import Path

import requests
from openai import OpenAI

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "youtu_parsing_omni.json"
MODALITY = {"document": "image", "natural_image": "image", "graphics_chart": "image",
            "graphics_flowchart": "image", "graphics_geometric": "image", "audio": "audio",
            "natural_video": "video", "textrich_video": "video"}

parser = argparse.ArgumentParser()
parser.add_argument("--media", type=Path, required=True)
parser.add_argument("--task", default="document", choices=sorted(MODALITY))
parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
args = parser.parse_args()

# decoding stops at <|image_pad|> / <|end_of_text|>, as in the reported results
tokenized = requests.post(f"{args.api_base.removesuffix('/v1')}/tokenize", timeout=60, json={
    "model": "Youtu-Parsing-Omni", "prompt": "<|image_pad|><|end_of_text|>", "add_special_tokens": False})
tokenized.raise_for_status()
stop_token_ids = sorted(tokenized.json()["tokens"])

modality = MODALITY[args.task]
prompt = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"][args.task]
media = args.media.resolve()
extra_body = {
    "chat_template_kwargs": {"enable_parsing": True, "enable_thinking": False},
    "skip_special_tokens": False,
    "min_tokens": 16,
    "stop_token_ids": stop_token_ids,
}
if modality == "video":  # the audio track is decoded together with the frames
    extra_body["mm_processor_kwargs"] = {"nframes": 64, "use_audio_in_video": True,
                                         "use_vision_in_video": True}

client = OpenAI(base_url=args.api_base, api_key="EMPTY")
response = client.chat.completions.create(
    model="Youtu-Parsing-Omni",
    messages=[{"role": "user", "content": [
        {"type": f"{modality}_url", f"{modality}_url": {"url": media.as_uri()}},  # media first
        {"type": "text", "text": prompt},
    ]}],
    temperature=0.0,
    max_tokens=32768,
    extra_body=extra_body,
)
print(response.choices[0].message.content)
