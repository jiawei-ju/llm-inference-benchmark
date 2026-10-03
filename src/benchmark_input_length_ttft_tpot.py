import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_PROMPT_LENGTHS = [64, 128, 256, 512]
GENERATED_TOKENS = 64
NUM_RUNS = 3
BASE_PROMPT = (
    "Explain how efficient language model inference helps interactive applications. "
    "Discuss latency, throughput, batching, and hardware utilization in clear English. "
) * 100


class TokenTimestampStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.next_value_is_prompt = True
        self.token_timestamps: list[float] = []

    def put(self, value: torch.Tensor) -> None:
        # generate() first sends the complete prompt input ids; do not time them.
        if self.next_value_is_prompt:
            self.next_value_is_prompt = False
            return

        # The standard batch-size-1 decode path sends exactly one generated token.
        timestamp = time.perf_counter()
        if value.numel() != 1:
            raise ValueError("Expected one generated token per streamer call")
        self.token_timestamps.append(timestamp)

    def end(self) -> None:
        # No buffered text needs to be flushed because this streamer stores timestamps only.
        pass


def build_inputs(tokenizer, target_prompt_tokens: int):
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
    content_length = max(target_prompt_tokens - template_tokens, 1)
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

    for target_prompt_tokens in TARGET_PROMPT_LENGTHS:
        # Build and tokenize each prompt before measuring any metrics.
        inputs = build_inputs(tokenizer, target_prompt_tokens).to(model.device)
        actual_prompt_tokens = inputs["input_ids"].shape[-1]
        ttfts = []
        tpots = []
        latencies = []

        with torch.inference_mode():
            # Warm up once with the same prompt and streamer path.
            model.generate(
                **inputs,
                min_new_tokens=GENERATED_TOKENS,
                max_new_tokens=GENERATED_TOKENS,
                streamer=TokenTimestampStreamer(),
            )

            # Measure three generations with a fresh streamer for each run.
            for run_number in range(1, NUM_RUNS + 1):
                streamer = TokenTimestampStreamer()
                start_time = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    min_new_tokens=GENERATED_TOKENS,
                    max_new_tokens=GENERATED_TOKENS,
                    streamer=streamer,
                )
                latency = time.perf_counter() - start_time

                # Validate that the streamer recorded every generated token exactly once.
                generated_tokens = len(streamer.token_timestamps)
                output_generated_tokens = outputs.shape[-1] - actual_prompt_tokens
                if generated_tokens != output_generated_tokens:
                    raise RuntimeError("Streamer token count does not match generated output")

                # Calculate TTFT and the mean interval between generated tokens.
                first_token_time = streamer.token_timestamps[0]
                last_token_time = streamer.token_timestamps[-1]
                ttft = first_token_time - start_time
                tpot = (last_token_time - first_token_time) / (generated_tokens - 1)

                ttfts.append(ttft)
                tpots.append(tpot)
                latencies.append(latency)
                print(
                    f"target_prompt_tokens={target_prompt_tokens}, "
                    f"actual_prompt_tokens={actual_prompt_tokens}, "
                    f"run={run_number}, generated_tokens={generated_tokens}, "
                    f"TTFT={ttft:.4f}s, TPOT={tpot:.4f}s/token, "
                    f"latency={latency:.4f}s"
                )

        # Report arithmetic means for the current input length.
        average_ttft = sum(ttfts) / len(ttfts)
        average_tpot = sum(tpots) / len(tpots)
        average_latency = sum(latencies) / len(latencies)
        print(
            f"Average for target_prompt_tokens={target_prompt_tokens}, "
            f"actual_prompt_tokens={actual_prompt_tokens}: "
            f"TTFT={average_ttft:.4f}s, "
            f"TPOT={average_tpot:.4f}s/token, "
            f"latency={average_latency:.4f}s"
        )


if __name__ == "__main__":
    main()
