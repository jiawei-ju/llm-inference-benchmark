#!/usr/bin/env bash
set -euo pipefail

# Use an already-running server. Never start or restart it between groups.
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
RESULT_DIR="${1:-results/vllm_online_input_concurrency/$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

command -v vllm >/dev/null
command -v python >/dev/null
# Atomically claim a fresh run directory after creating its parents.
mkdir -p -- "$(dirname -- "${RESULT_DIR}")"
if ! mkdir -- "${RESULT_DIR}"; then
    echo "Cannot exclusively create result directory: ${RESULT_DIR}. Choose a new directory." >&2
    exit 1
fi
printf 'Environment scope: local benchmark client; represents server only when colocated in the same Colab/T4 software environment\n' > "${RESULT_DIR}/environment.txt"
python -c 'import torch, transformers, vllm; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__); print("Transformers:", transformers.__version__); print("vLLM:", vllm.__version__)' >> "${RESULT_DIR}/environment.txt"
printf 'Model: %s\nBase URL: %s\nInput lengths: 128,512,1024\nConcurrency: 1,4,8\nOutput: 64\nWarm-ups: 8\nRequests: 100\nSeed: 0\nRequest rate: inf\nServer configuration: see companion document and server.log\n' "${MODEL_NAME}" "${VLLM_BASE_URL}" >> "${RESULT_DIR}/environment.txt"

python "${SCRIPT_DIR}/summarize_vllm_input_concurrency.py" "${RESULT_DIR}" \
    --check-server "${VLLM_BASE_URL}" >> "${RESULT_DIR}/environment.txt"

for input_length in 128 512 1024; do
    group_dir="${RESULT_DIR}/input_${input_length}"
    mkdir -p "${group_dir}"
    for concurrency in 1 4 8; do
        echo "Running input=${input_length}, concurrency=${concurrency}"
        vllm bench serve \
            --backend vllm --base-url "${VLLM_BASE_URL}" \
            --endpoint /v1/completions \
            --model "${MODEL_NAME}" --tokenizer "${MODEL_NAME}" \
            --dataset-name random --random-input-len "${input_length}" \
            --random-output-len 64 --random-range-ratio 0 --random-prefix-len 0 \
            --seed 0 --ignore-eos --temperature 0.0 --request-rate inf \
            --max-concurrency "${concurrency}" --num-warmups 8 --num-prompts 100 \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --ready-check-timeout-sec 600 \
            --label "vllm-online-i${input_length}-c${concurrency}" \
            --save-result --save-detailed --result-dir "${group_dir}" \
            --result-filename "concurrency_${concurrency}.json" \
            2>&1 | tee "${group_dir}/concurrency_${concurrency}.log"
        python "${SCRIPT_DIR}/summarize_vllm_input_concurrency.py" "${RESULT_DIR}" \
            --validate-group "${input_length}" "${concurrency}"
    done
done

python "${SCRIPT_DIR}/summarize_vllm_input_concurrency.py" "${RESULT_DIR}"
echo "Results and summary.csv saved in ${RESULT_DIR}"
