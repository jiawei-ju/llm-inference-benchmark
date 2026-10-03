#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
RESULT_DIR="${1:-results/vllm_online/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULT_DIR}"

# Refuse to overwrite results from an earlier run.
for concurrency in 1 2 4 8; do
    result_file="${RESULT_DIR}/concurrency_${concurrency}.json"
    if [[ -e "${result_file}" ]]; then
        echo "Result file already exists: ${result_file}" >&2
        echo "Choose a new result directory to preserve the existing run." >&2
        exit 1
    fi
done

for concurrency in 1 2 4 8; do
    echo "Running vLLM online benchmark with concurrency=${concurrency}"

    vllm bench serve \
        --backend vllm \
        --base-url "${VLLM_BASE_URL}" \
        --endpoint /v1/completions \
        --model "${MODEL_NAME}" \
        --tokenizer "${MODEL_NAME}" \
        --dataset-name random \
        --random-input-len 128 \
        --random-output-len 64 \
        --random-range-ratio 0 \
        --random-prefix-len 0 \
        --seed 0 \
        --ignore-eos \
        --temperature 0.0 \
        --request-rate inf \
        --max-concurrency "${concurrency}" \
        --num-warmups "${NUM_WARMUPS}" \
        --num-prompts "${NUM_PROMPTS}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --ready-check-timeout-sec 600 \
        --label "vllm-online-c${concurrency}" \
        --save-result \
        --save-detailed \
        --result-dir "${RESULT_DIR}" \
        --result-filename "concurrency_${concurrency}.json"

    test -f "${RESULT_DIR}/concurrency_${concurrency}.json"
done

echo "Results saved in ${RESULT_DIR}:"
find "${RESULT_DIR}" -maxdepth 1 -type f -name 'concurrency_*.json' -print | sort
