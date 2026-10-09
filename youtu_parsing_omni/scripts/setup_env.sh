#!/usr/bin/env bash
# Install an environment into the active Python env.
#
#   bash scripts/setup_env.sh                 # vLLM serving (transformers==5.2.0) + plugin
#   bash scripts/setup_env.sh --transformers  # Transformers inference (transformers==5.10.2)
#
# vLLM and Transformers inference pin different transformers versions;
# install them into separate Python environments.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
PYTHON=${PYTHON:-python}
MODE=${1:-vllm}

case "${MODE}" in
    vllm)
        "${PYTHON}" -m pip install -r requirements/vllm.txt
        "${PYTHON}" -m pip install transformers==5.2.0   # overrides vllm's transformers<5 pin
        "${PYTHON}" -m pip uninstall -y peft >/dev/null 2>&1 || true   # breaks the pinned vLLM runtime
        "${PYTHON}" -m pip install --no-deps ./vllm-plugin-vita-omni
        command -v ffmpeg >/dev/null 2>&1 || echo "WARNING: ffmpeg not found (needed for audio / video)"
        ;;
    --transformers)
        "${PYTHON}" -m pip install -r requirements/transformers.txt
        ;;
    -h|--help)
        sed -n '2,8p' "$0"
        ;;
    *)
        echo "Usage: $0 [vllm | --transformers]" >&2
        exit 1
        ;;
esac
