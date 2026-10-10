#!/usr/bin/env bash
# OmniDocBench inference + scoring with the official evaluator (Docker image).
#   MODEL=/path/to/checkpoint OMNIDOCBENCH=/path/to/OmniDocBench [OUTPUT_DIR=...] [GPUS=0,1,...] \
#       bash scripts/eval_omnidocbench.sh
# Rerun with the same OUTPUT_DIR to resume. Logs: <OUTPUT_DIR>/logs/{infer,vllm_<port>,eval}.log
set -eo pipefail
set -x

EVAL_IMAGE=ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
docker image inspect "${EVAL_IMAGE}" > /dev/null 2>&1 || docker pull "${EVAL_IMAGE}"

cd "$(dirname "$0")/.."
DATA=$(realpath "${OMNIDOCBENCH}")
OUT=$(realpath -m "${OUTPUT_DIR:-outputs/omnidocbench_$(date +%Y%m%d_%H%M%S)}")
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
mkdir -p "${OUT}/logs"

# inference: one scripts/vllm.sh server per GPU (ports 8000, 8001, ...)
PIDS=() API="" i=0
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
for gpu in ${GPUS//,/ }; do
    MODEL=${MODEL} DATA_ROOT=${DATA} GPU=${gpu} PORT=$((8000 + i)) VLLM_PORT=$((30000 + 16 * i)) \
        bash scripts/vllm.sh > "${OUT}/logs/vllm_$((8000 + i)).log" 2>&1 &
    PIDS+=($!) API+="${API:+,}http://127.0.0.1:$((8000 + i))/v1" i=$((i + 1))
done
python evaluation/infer_omnidocbench.py --api-base "${API}" --omnidocbench-root "${DATA}" \
    --output-dir "${OUT}" 2>&1 | tee -a "${OUT}/logs/infer.log"
kill "${PIDS[@]}" 2>/dev/null || true

# scoring: official configs/end2end.yaml, with GT / predictions mounted onto the paths it reads.
# OMNIDOCBENCH_CDM_WORKERS=1: serial CDM (its process pool can deadlock on small pipes).
rm -rf "${OUT}/omnidocbench_eval" && mkdir -p "${OUT}/omnidocbench_eval"
docker run --rm -e OMNIDOCBENCH_MATCH_WORKERS=13 -e OMNIDOCBENCH_CDM_WORKERS=4 -e OMNIDOCBENCH_TEDS_WORKERS=13 --entrypoint bash \
    -v "${DATA}/OmniDocBench.json":/workspace/demo_data/omnidocbench_demo/OmniDocBench_demo.json:ro \
    -v "${OUT}/predictions/document_markdown":/workspace/demo_data/end2end:ro \
    -v "${OUT}/omnidocbench_eval":/workspace/result \
    "${EVAL_IMAGE}" \
    -lc 'python pdf_validation.py --config configs/end2end.yaml' 2>&1 | tee "${OUT}/logs/eval.log"
sed -n '/^========== FINAL_EVAL_RUN_REPORT/,/^========== END_FINAL_EVAL_RUN_REPORT/p' \
    "${OUT}/logs/eval.log" | tee "${OUT}/results.md"
