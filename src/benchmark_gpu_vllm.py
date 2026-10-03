import time
from statistics import median

import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_PROMPT_TOKENS = 128
GENERATED_TOKENS = 64
NUM_WARMUP_RUNS = 3
NUM_RUNS = 10
CONCURRENCY = 1
BASE_PROMPT = (
    "Explain how efficient language model inference helps interactive applications. "
    "Discuss latency, throughput, batching, and hardware utilization in clear English. "
) * 100


def build_prompt_token_ids(tokenizer) -> list[int]:
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
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs["input_ids"][0].tolist()


def main() -> None:
    # Require CUDA explicitly so the benchmark never falls back to CPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"vLLM version: {vllm.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_NAME}")
    print("Model dtype: float16")
    print(f"Concurrency: {CONCURRENCY}")

    # Build the tokenized prompt before starting any timed generation.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_token_ids = build_prompt_token_ids(tokenizer)
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    # Load one FP16 vLLM engine on the single available GPU.
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
    )
    # Use greedy decoding, force 64 tokens, and match token-only Transformers output.
    sampling_params = SamplingParams(
        temperature=0.0,
        min_tokens=GENERATED_TOKENS,
        max_tokens=GENERATED_TOKENS,
        detokenize=False,
    )

    # Warm up the same single-request generation path three times.
    for _ in range(NUM_WARMUP_RUNS):
        llm.generate([prompt], sampling_params, use_tqdm=False)

    latencies = []
    throughputs = []

    # LLM.generate is synchronous, so its return marks request completion.
    for run_number in range(1, NUM_RUNS + 1):
        start_time = time.perf_counter()
        request_output = llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )[0]
        latency = time.perf_counter() - start_time

        actual_prompt_tokens = len(request_output.prompt_token_ids)
        actual_generated_tokens = len(request_output.outputs[0].token_ids)
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
