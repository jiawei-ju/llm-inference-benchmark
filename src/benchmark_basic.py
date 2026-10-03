import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
NUM_RUNS = 3


def main() -> None:
    # Load the tokenizer and model before measuring inference latency.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Prepare and tokenize a fixed English prompt before timing begins.
    messages = [
        {
            "role": "user",
            "content": "Explain in one sentence what large language model inference is.",
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    latencies = []
    throughputs = []

    with torch.inference_mode():
        # Run one warm-up generation before collecting measurements.
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        # Measure three generations with the same prompt and settings.
        for run_number in range(1, NUM_RUNS + 1):
            start_time = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            latency = time.perf_counter() - start_time

            # Count only newly generated tokens and calculate token throughput.
            generated_tokens = outputs.shape[-1] - prompt_tokens
            throughput = generated_tokens / latency
            latencies.append(latency)
            throughputs.append(throughput)

            print(
                f"Run {run_number}: prompt_tokens={prompt_tokens}, "
                f"generated_tokens={generated_tokens}, "
                f"latency={latency:.4f}s, "
                f"throughput={throughput:.2f} tokens/s"
            )

    # Report arithmetic means across the measured runs.
    average_latency = sum(latencies) / len(latencies)
    average_throughput = sum(throughputs) / len(throughputs)
    print(f"Average latency: {average_latency:.4f}s")
    print(f"Average throughput: {average_throughput:.2f} tokens/s")


if __name__ == "__main__":
    main()
