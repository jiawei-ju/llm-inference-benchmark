"""Validate detailed serving results and summarize either comparison axis."""

import argparse
import csv
import json
import math
from pathlib import Path
from urllib.request import urlopen

INPUT_LENGTHS = (128, 512, 1024)
CONCURRENCIES = (1, 4, 8)
METRICS = (
    "request_throughput", "output_throughput", "total_token_throughput",
    "duration",
) + tuple(
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "p90", "p99")
)


def read_result(path, input_length, concurrency):
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "completed": 100, "failed": 0, "max_concurrency": concurrency,
        "total_input_tokens": 100 * input_length, "total_output_tokens": 6400,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"{path}: {key} must be {value}, got {data.get(key)}")
    for key, length in (("input_lens", input_length), ("output_lens", 64)):
        values = data.get(key, [])
        if len(values) != 100 or set(values) != {length}:
            raise ValueError(f"{path}: {key} must contain 100 lengths of {length}")
    for key in METRICS:
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{path}: invalid or missing metric {key}")
    return {
        "input_length": input_length, "concurrency": concurrency,
        **{key: data[key] for key in METRICS},
        **{key: data[key] for key in expected},
    }


def check_server(result_dir, base_url):
    with urlopen(base_url.rstrip("/") + "/v1/models", timeout=30) as response:
        data = json.load(response)
    with (result_dir / "server_models.json").open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    models = [model for model in data.get("data", [])
              if model.get("id") == "Qwen/Qwen2.5-0.5B-Instruct"]
    if len(models) != 1:
        raise ValueError("Server must expose the expected model exactly once")
    if models[0].get("max_model_len") != 2048:
        raise ValueError("/v1/models must report max_model_len=2048; missing or mismatched value")
    print("Server model and max-model-len=2048: API verified via /v1/models")
    print("Server dtype=float16: 由启动命令保证 (not API verified)")
    print("Server prefix caching disabled: 由启动命令保证 (not API verified)")
    print("Scheduler/KV cache settings and uninterrupted server identity: "
          "由启动命令及运行流程保证 (not API verified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--validate-group", nargs=2, type=int,
                        metavar=("INPUT_LENGTH", "CONCURRENCY"))
    parser.add_argument("--view", choices=("input", "concurrency"), default="input")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--check-server", metavar="BASE_URL",
                        help="Check model/max_model_len, save API response, and exit")
    args = parser.parse_args()
    if args.check_server:
        if args.validate_group or args.print_only:
            parser.error("--check-server cannot be combined with result analysis options")
        check_server(args.result_dir, args.check_server)
        return
    combinations = ([tuple(args.validate_group)] if args.validate_group else
                    [(length, c) for length in INPUT_LENGTHS for c in CONCURRENCIES])
    rows = []
    for length, concurrency in combinations:
        if length not in INPUT_LENGTHS or concurrency not in CONCURRENCIES:
            parser.error("unsupported input length or concurrency")
        path = args.result_dir / f"input_{length}" / f"concurrency_{concurrency}.json"
        rows.append(read_result(path, length, concurrency))
    if args.validate_group:
        print(f"Validated input={length}, concurrency={concurrency}")
        return
    if not args.print_only:
        # Exclusive creation preserves existing summaries; validation happens first.
        with (args.result_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    axis = "input_length" if args.view == "input" else "concurrency"
    other = "concurrency" if axis == "input_length" else "input_length"
    for row in sorted(rows, key=lambda r: (r[axis], r[other])):
        print(f"input={row['input_length']:4} concurrency={row['concurrency']} "
              f"req/s={row['request_throughput']:.3f} "
              f"output tok/s={row['output_throughput']:.2f} "
              f"TTFT={row['mean_ttft_ms']:.2f} ms "
              f"TPOT={row['mean_tpot_ms']:.3f} ms/token "
              f"E2EL={row['mean_e2el_ms']:.2f} ms")


if __name__ == "__main__":
    main()
