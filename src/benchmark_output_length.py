import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_PROMPT_TOKENS = 128
TARGET_OUTPUT_LENGTHS = [32, 64, 128, 256]
NUM_RUNS = 3
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
    # Load the tokenizer and model only once before running all benchmarks.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Build and tokenize the fixed-length prompt before measuring latency.
    inputs = build_inputs(tokenizer).to(model.device)
    actual_prompt_tokens = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        for target_output_tokens in TARGET_OUTPUT_LENGTHS:
            # Warm up once for this output length.
            model.generate(
                **inputs,
                min_new_tokens=target_output_tokens,
                max_new_tokens=target_output_tokens,
            )

            latencies = []
            throughputs = []
            generated_token_counts = []

            # Measure three generations with the same prompt and output length.
            for _ in range(NUM_RUNS):
                start_time = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    min_new_tokens=target_output_tokens,
                    max_new_tokens=target_output_tokens,
                )
                latency = time.perf_counter() - start_time

                generated_tokens = outputs.shape[-1] - actual_prompt_tokens
                latencies.append(latency)
                throughputs.append(generated_tokens / latency)
                generated_token_counts.append(generated_tokens)

            # Print one summary for the current target output length.
            average_latency = sum(latencies) / len(latencies)
            average_throughput = sum(throughputs) / len(throughputs)
            actual_generated_tokens = generated_token_counts[0]
            print(
                f"actual_prompt_tokens={actual_prompt_tokens}, "
                f"target_output_tokens={target_output_tokens}, "
                f"actual_generated_tokens={actual_generated_tokens}, "
                f"average_latency={average_latency:.4f}s, "
                f"average_throughput={average_throughput:.2f} tokens/s"
            )


if __name__ == "__main__":
    main()
