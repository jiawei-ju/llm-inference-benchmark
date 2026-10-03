import time
from statistics import median

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_PROMPT_TOKENS = 128
GENERATED_TOKENS = 64
NUM_WARMUP_RUNS = 3
NUM_RUNS = 10
BASE_PROMPT = (
    "Explain how efficient language model inference helps interactive applications. "
    "Discuss latency, throughput, batching, and hardware utilization in clear English. "
) * 100


def build_inputs(tokenizer):
    # Estimate the fixed token overhead added by the model's chat template.
    empty_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    template_tokens = empty_inputs["input_ids"].shape[-1]

    # Trim English text so the complete formatted prompt is near 128 tokens.
    content_token_ids = tokenizer.encode(BASE_PROMPT, add_special_tokens=False)
    content_length = max(TARGET_PROMPT_TOKENS - template_tokens, 1)
    prompt = tokenizer.decode(
        content_token_ids[:content_length],
        skip_special_tokens=True,
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def main() -> None:
    # Require CUDA explicitly so the benchmark never falls back to CPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")

    # Load the model in FP16 and move it explicitly to the first CUDA device.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )
    model.to(device)
    model.eval()

    model_parameter = next(model.parameters())
    print(f"Model device: {model_parameter.device}")
    print(f"Model dtype: {model_parameter.dtype}")

    # Build and move the fixed-length prompt before measuring generation.
    inputs = build_inputs(tokenizer).to(device)
    actual_prompt_tokens = inputs["input_ids"].shape[-1]
    latencies = []
    throughputs = []

    with torch.inference_mode():
        # Run three warm-ups and wait for CUDA work after every generation.
        for _ in range(NUM_WARMUP_RUNS):
            model.generate(
                **inputs,
                min_new_tokens=GENERATED_TOKENS,
                max_new_tokens=GENERATED_TOKENS,
            )
            torch.cuda.synchronize(device)

        # Synchronize around each generate call for accurate GPU wall-clock timing.
        for run_number in range(1, NUM_RUNS + 1):
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            outputs = model.generate(
                **inputs,
                min_new_tokens=GENERATED_TOKENS,
                max_new_tokens=GENERATED_TOKENS,
            )
            torch.cuda.synchronize(device)
            latency = time.perf_counter() - start_time

            actual_generated_tokens = outputs.shape[-1] - actual_prompt_tokens
            throughput = actual_generated_tokens / latency
            latencies.append(latency)
            throughputs.append(throughput)
            print(
                f"Run {run_number}: actual_prompt_tokens={actual_prompt_tokens}, "
                f"actual_generated_tokens={actual_generated_tokens}, "
                f"latency={latency:.4f}s, "
                f"throughput={throughput:.2f} tokens/s"
            )

    # Report arithmetic means and medians across the measured runs.
    average_latency = sum(latencies) / len(latencies)
    average_throughput = sum(throughputs) / len(throughputs)
    median_latency = median(latencies)
    median_throughput = median(throughputs)
    print(f"Average latency: {average_latency:.4f}s")
    print(f"Average throughput: {average_throughput:.2f} tokens/s")
    print(f"Median latency: {median_latency:.4f}s")
    print(f"Median throughput: {median_throughput:.2f} tokens/s")


if __name__ == "__main__":
    main()
