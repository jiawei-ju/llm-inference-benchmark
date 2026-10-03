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

## TODO

- [x] 开展 input length 实验
- [x] 开展 output length 实验
- [ ] 开展 concurrency 实验
