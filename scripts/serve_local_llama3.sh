#!/usr/bin/env bash
# Serve the local Hugging Face Llama 3 checkpoint with vLLM's OpenAI API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/../../../Meta-Llama-3-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Meta-Llama-3-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vLLM is not installed or vllm is not on PATH." >&2
    echo "Install a vLLM build compatible with your CUDA/PyTorch environment, then rerun this script." >&2
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: model directory not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to the Hugging Face checkpoint directory." >&2
    exit 1
fi

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 8192
