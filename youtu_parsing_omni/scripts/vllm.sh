#!/usr/bin/env bash
# Start a Youtu-Parsing-Omni vLLM server in the foreground on GPU (port PORT).
#
#   MODEL=/path/to/Youtu-Parsing-Omni DATA_ROOT=./data bash scripts/vllm.sh
#
# DATA_ROOT is the only directory the server may read local media from.
# These are the serving settings of the reported results (max-num-seqs 1,
# gpu util 0.70, len 65536, seed 1234, limit-mm {"image": 8, "video": 4,
# "audio": 8}, vLLM defaults for chunked prefill / prefix caching / multimodal
# profiling / multimodal cache / chat-template-content-format).
set -euo pipefail

MODEL=${MODEL:?set MODEL to the checkpoint directory}
DATA_ROOT=$(realpath "${DATA_ROOT:?set DATA_ROOT to the directory containing the media}")
GPU=${GPU:-0}
PORT=${PORT:-8000}

# media token budget read by vllm-plugin-vita-omni
export YOUTU_VITA_IMAGE_MAX_TOKENS=16384
export YOUTU_VITA_VIDEO_TOTAL_MAX_TOKENS=16384
export YOUTU_VITA_VIDEO_MAX_TOKENS=16384
export YOUTU_VITA_VIDEO_IMAGE_MAX_TOKENS=512
export YOUTU_VITA_VIDEO_IMAGE_MIN_TOKENS=4
export YOUTU_VITA_VIDEO_MIN_TOKENS=4
export VLLM_DISABLE_FLASH_ATTN_3=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT

# keys read by the plugin processor; the media token budgets are set by the YOUTU_VITA_* variables above
MM_KWARGS='{"video_audio_chunk_max_second":29.5,"video_audio_chunk_min_second":1.5}'

ENGINE_ARGS=(
    --gpu-memory-utilization 0.70
    --max-model-len 65536 --max-num-batched-tokens 65536 --max-num-seqs 1
    --limit-mm-per-prompt '{"image": 8, "video": 4, "audio": 8}'
    --seed 1234
)

echo "GPU ${GPU} -> http://127.0.0.1:${PORT}/v1"
export CUDA_VISIBLE_DEVICES=${GPU}
export VLLM_PORT=${VLLM_PORT:-30000}
exec vllm serve "${MODEL}" \
    --served-model-name Youtu-Parsing-Omni \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --tensor-parallel-size 1 \
    --allowed-local-media-path "${DATA_ROOT}" \
    --mm-processor-kwargs "${MM_KWARGS}" \
    "${ENGINE_ARGS[@]}"
