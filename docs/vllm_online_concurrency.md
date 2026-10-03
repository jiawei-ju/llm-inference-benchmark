# vLLM Online Serving Concurrency Benchmark

本实验通过 vLLM OpenAI-compatible server 测量真正的 online serving concurrency。四组测试使用相同的模型和 workload，只改变最大并发请求数。

## 实验设置

- GPU：NVIDIA Tesla T4
- vLLM：`0.30.0`
- Model：`Qwen/Qwen2.5-0.5B-Instruct`
- dtype：FP16
- Input length：128 tokens
- Output length：64 tokens
- Concurrency：1、2、4、8
- Warm-up：每组 8 个请求
- Measured requests：每组 100 个请求
- Request rate：`inf`，持续填满当前 concurrency 上限
- Prefix caching：关闭，避免不同测试之间复用相同 prompt 的 KV Cache

不要在 server 运行期间同时执行 offline benchmark；server 会占用这块 GPU。

## 1. 检查环境

在 Colab 中进入项目目录，然后运行：

```python
!nvidia-smi --query-gpu=name --format=csv,noheader
!python -c "import torch, transformers, vllm; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('vLLM:', vllm.__version__)"
```

## 2. 启动 vLLM server

在一个新的 Colab cell 中运行：

```bash
%%bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --no-enable-prefix-caching \
  > /tmp/vllm_server.log 2>&1 &

echo $! > /tmp/vllm_server.pid
echo "Server PID: $(cat /tmp/vllm_server.pid)"
```

模型加载、compilation 和 CUDA graph capture 都发生在 benchmark 之前，不计入请求 latency。

## 3. 等待 server 就绪

```bash
%%bash
timeout 900 bash -c '
  until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do
    sleep 5
  done
'

curl -s http://127.0.0.1:8000/v1/models
tail -n 50 /tmp/vllm_server.log
```

如果等待失败，先检查完整日志：

```python
!cat /tmp/vllm_server.log
```

## 4. 运行四组 concurrency benchmark

脚本会依次运行 concurrency 1、2、4、8，并为每组保留独立 JSON 文件：

```python
!bash scripts/run_vllm_online_benchmark.sh results/vllm_online/t4_run_01
```

输出目录应包含：

```text
results/vllm_online/t4_run_01/concurrency_1.json
results/vllm_online/t4_run_01/concurrency_2.json
results/vllm_online/t4_run_01/concurrency_4.json
results/vllm_online/t4_run_01/concurrency_8.json
```

如需保留下一轮结果，改用新的目录名，例如 `t4_run_02`。如果目标目录中已存在任一同名结果文件，脚本会直接报错，以免覆盖前一轮。

## 5. 验证 workload 和查看关键指标

运行以下 Colab Python cell：

```python
import json
from pathlib import Path

result_dir = Path("results/vllm_online/t4_run_01")

for concurrency in (1, 2, 4, 8):
    path = result_dir / f"concurrency_{concurrency}.json"
    result = json.loads(path.read_text())

    assert result["completed"] == 100
    assert result["failed"] == 0
    assert set(result["input_lens"]) == {128}
    assert set(result["output_lens"]) == {64}
    assert result["total_input_tokens"] == 12800
    assert result["total_output_tokens"] == 6400

    print(f"\nConcurrency {concurrency}")
    print(f"  Request throughput: {result['request_throughput']:.2f} req/s")
    print(f"  Output throughput:  {result['output_throughput']:.2f} tok/s")
    print(f"  Total throughput:   {result['total_token_throughput']:.2f} tok/s")
    for metric in ("ttft", "tpot", "itl", "e2el"):
        print(
            f"  {metric.upper():4}: "
            f"mean={result[f'mean_{metric}_ms']:.2f} ms, "
            f"median={result[f'median_{metric}_ms']:.2f} ms, "
            f"P90={result[f'p90_{metric}_ms']:.2f} ms, "
            f"P99={result[f'p99_{metric}_ms']:.2f} ms"
        )
```

如果任何 assertion 失败，不要把该组结果写入最终对比；先检查对应 JSON 和 server 日志。

重点记录以下指标：

- Request throughput
- Output token throughput
- Total token throughput
- TTFT 的 mean、median、P90、P99
- TPOT 的 mean、median、P90、P99
- ITL 的 mean、median、P90、P99
- E2EL 的 mean、median、P90、P99

这里的 E2EL 是从客户端实际发出请求到接收完整响应的时间，不包含客户端等待 `max-concurrency` 信号量的排队时间；request throughput 则反映整组饱和负载的完成速率。

## 6. 停止 server

所有测试完成后，通过启动时保存的 PID 停止该 server：

```bash
%%bash
if [[ -f /tmp/vllm_server.pid ]]; then
  server_pid=$(cat /tmp/vllm_server.pid)
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -f /tmp/vllm_server.pid
fi
```

结果 JSON 不会被删除。
