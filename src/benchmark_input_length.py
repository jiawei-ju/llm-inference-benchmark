import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_INPUT_LENGTHS = [64, 128, 256, 512]
MAX_NEW_TOKENS = 32
NUM_RUNS = 3
BASE_PROMPT = (
    "Explain how efficient language model inference helps interactive applications. "
    "Discuss latency, throughput, batching, and hardware utilization in clear English. "
) * 100


def build_inputs(tokenizer, target_tokens: int):
    # Estimate the fixed token overhead added by the model's chat template.
    empty_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    template_tokens = empty_inputs["input_ids"].shape[-1]

    # Trim English text so the complete formatted prompt is near the target length.
    content_token_ids = tokenizer.encode(BASE_PROMPT, add_special_tokens=False)
    content_length = max(target_tokens - template_tokens, 1)
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
    # Load the tokenizer and model only once before running all benchmarks.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    for target_tokens in TARGET_INPUT_LENGTHS:
        # Build and tokenize each prompt before measuring inference latency.
        inputs = build_inputs(tokenizer, target_tokens).to(model.device)
        actual_prompt_tokens = inputs["input_ids"].shape[-1]
        latencies = []
        throughputs = []

        with torch.inference_mode():
            # Warm up once for this input length.
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MAX_NEW_TOKENS,
            )

            # Measure three generations with the same prompt.
            for _ in range(NUM_RUNS):
                start_time = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MAX_NEW_TOKENS,
                )
                latency = time.perf_counter() - start_time

                generated_tokens = outputs.shape[-1] - actual_prompt_tokens
                latencies.append(latency)
                throughputs.append(generated_tokens / latency)

        # Print one summary for the current target input length.
        average_latency = sum(latencies) / len(latencies)
        average_throughput = sum(throughputs) / len(throughputs)
        print(
            f"target_input_length={target_tokens}, "
            f"actual_prompt_tokens={actual_prompt_tokens}, "
            f"average_latency={average_latency:.4f}s, "
            f"average_throughput={average_throughput:.2f} tokens/s"
        )


if __name__ == "__main__":
    main()
