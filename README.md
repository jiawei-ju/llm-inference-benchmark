# LLM Inference Benchmark

通过实验学习和比较 LLM inference 的 latency、TTFT、TPOT、throughput，并逐步研究 Transformers 与 vLLM。

## Experiment 1: Input Length

实验目的：研究 input length 对 LLM inference latency 和 throughput 的影响。

实验设置：使用 `Qwen/Qwen2.5-0.5B-Instruct`，固定 `max_new_tokens=32`（同时设置 `min_new_tokens=32` 以固定输出长度），input length 分别为 64、128、256 和 512；每组 warm-up 1 次，正式运行 3 次。

| Input length (tokens) | Average latency (s) | Average throughput (tokens/s) |
| ---: | ---: | ---: |
| 64 | 1.4634 | 21.87 |
| 128 | 1.4958 | 21.39 |
| 256 | 1.5740 | 20.33 |
| 512 | 1.7464 | 18.32 |

随着 input length 增大，end-to-end latency 上升，throughput 下降。当前测量包含 Prefill 和 Decode，因此不能把 latency 增长直接解释为纯 Prefill latency；更长上下文也会增加 Decode 阶段的 context/KV Cache 开销。

## Experiment 2: Output Length

实验目的：研究 output length 对 LLM inference latency 和 throughput 的影响。

实验设置：使用 `Qwen/Qwen2.5-0.5B-Instruct`，固定 prompt length 为 128 tokens，output length 分别为 32、64、128 和 256；每组 warm-up 1 次，正式运行 3 次。

| Prompt length (tokens) | Output length (tokens) | Average latency (s) | Average throughput (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 32 | 1.6596 | 19.28 |
| 128 | 64 | 3.2534 | 19.69 |
| 128 | 128 | 6.4286 | 19.91 |
| 128 | 256 | 12.3965 | 20.67 |

随着 output length 增大，end-to-end latency 近似线性增加，因为 autoregressive decode 需要执行更多 sequential decode steps。Throughput 整体保持在约 19–21 tokens/s，没有随 output length 明显下降。较长输出中固定的 Prefill 开销被更多生成 token 摊薄，但当前只有 3 次 CPU 测试，不应对 throughput 的小幅变化做过度解释。

## Experiment 3: TTFT and TPOT

实验目的：分别观察首次生成延迟和后续 token 生成速度。

实验设置：使用 `Qwen/Qwen2.5-0.5B-Instruct`，prompt length 为 128 tokens，generated tokens 为 64；warm-up 1 次，正式运行 3 次。

| Run | TTFT (s) | TPOT (s/token) | Latency (s) |
| ---: | ---: | ---: | ---: |
| 1 | 0.1107 | 0.0445 | 2.9132 |
| 2 | 0.1169 | 0.0447 | 2.9304 |
| 3 | 0.1171 | 0.0444 | 2.9171 |
| **Average** | **0.1149** | **0.0445** | **2.9202** |

TTFT 表示从请求开始到第一个 generated token 出现的时间，但不能直接视为纯 Prefill latency。TPOT 表示第一个 token 之后，后续 token 的平均生成间隔。`TTFT + 63 × TPOT` 与总 latency 基本一致，说明当前 token-level 计时逻辑是自洽的。

## Experiment 4: Input Length vs TTFT/TPOT

实验目的：研究 input length 对 TTFT、TPOT 和 end-to-end latency 的影响。

实验设置：使用 `Qwen/Qwen2.5-0.5B-Instruct`，input length 分别为 64、128、256 和 512 tokens，固定生成 64 tokens；每组 warm-up 1 次，正式运行 3 次。

| Input length (tokens) | Average TTFT (s) | Average TPOT (s/token) | Average latency (s) |
| ---: | ---: | ---: | ---: |
| 64 | 0.0894 | 0.0487 | 3.1592 |
| 128 | 0.1314 | 0.0485 | 3.1895 |
| 256 | 0.1922 | 0.0498 | 3.3268 |
| 512 | 0.3502 | 0.0501 | 3.5058 |

随 input length 增大，TTFT 明显增加，说明长 prompt 会显著增加请求首次响应时间，但 TTFT 不能直接等同于纯 Prefill latency。TPOT 只轻微增加，当前范围内 Decode 每 token 的速度相对稳定；更长 context 会带来更大的 KV Cache 和更长的 attention context，因此 TPOT 仍可能受到影响，但不应过度解读 CPU 上的小幅差异。固定输出 64 tokens 后，总 latency 仍主要由 sequential Decode 时间构成。

## Experiment 5: Transformers GPU Baseline

实验环境：Google Colab，NVIDIA Tesla T4，PyTorch `2.13.0+cu130`，Transformers `5.17.0`。模型为 `Qwen/Qwen2.5-0.5B-Instruct`，运行在 `cuda:0`，使用 `torch.float16`；prompt length 为 128 tokens，固定生成 64 tokens，warm-up 3 次，正式运行 10 次。

| Metric | Average | Median |
| --- | ---: | ---: |
| Latency (s) | 2.2891 | 2.2063 |
| Throughput (tokens/s) | 28.09 | 29.01 |

Median 对偶发慢 run 不敏感，可以减小异常波动对典型性能判断的影响，因此与 average 一起报告。CPU 与 GPU 实验的硬件、运行环境和 dtype 不同，不应对两者的性能差异做过度解读或直接归因。

## Transformers vs vLLM: Single-request GPU Baseline

实验环境：Google Colab，NVIDIA Tesla T4，PyTorch `2.13.0+cu130`，Transformers `5.17.0`，vLLM `0.30.0`。两组测试均使用 `Qwen/Qwen2.5-0.5B-Instruct` 和 FP16，input length 为 128 tokens，output length 为 64 tokens，concurrency 为 1；warm-up 3 次，正式运行 10 次。vLLM 在 T4 上使用 `TRITON_ATTN` backend，而不是 FlashAttention 2。

| Backend | Average latency (s) | Median latency (s) | Average throughput (tokens/s) | Median throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 2.2891 | 2.2063 | 28.09 | 29.01 |
| vLLM | 0.3677 | 0.3676 | 174.08 | 174.11 |

在这组相同 GPU、模型、input/output length 和 concurrency=1 的 workload 下，vLLM 显示出明显更低的 latency 和更高的 output-token throughput；该结果只描述当前实验条件，不代表 vLLM 在所有 workload 下都保持固定倍数的优势。表中数据是 warm-up 后的 steady-state 请求性能，vLLM 首次启动时的 engine initialization、compilation 和 CUDA graph capture 等冷启动开销未计入请求 latency。当前结果也仅代表 single-request performance，不代表高并发 serving 表现。

## TODO

- [x] 开展 input length 实验
- [x] 开展 output length 实验
- [x] 开展 TTFT 和 TPOT 实验
- [x] 开展 input length 对 TTFT/TPOT 影响实验
- [x] 完成 Transformers GPU baseline
- [ ] 开展 concurrency 实验
