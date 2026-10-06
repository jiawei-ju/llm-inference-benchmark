# vLLM Online Serving: Input Length × Concurrency

本实验使用一个持续运行的 vLLM OpenAI-compatible server，依次测量 input length 128、512、1024 × concurrency 1、4、8，共九组。输出固定为 64 tokens，每组 warm-up 8 个请求、正式请求 100 个。使用随机数据、seed 0、random-range-ratio=0、random-prefix-len=0、ignore-eos、temperature=0、request-rate=inf，关闭 prefix caching。

## 固定环境和服务配置

使用 NVIDIA Tesla T4、PyTorch 2.13.0+cu130、Transformers 5.17.0、vLLM 0.31.0、Qwen/Qwen2.5-0.5B-Instruct、FP16。先确认安装环境，不要在九组之间升级依赖或改变服务参数。

```bash
python -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__); print(torch.cuda.get_device_name(0))'
nvidia-smi
```

当前设计假设 benchmark client 和 vLLM server 在同一台 Colab/T4 runtime，使用相同软件环境。`environment.txt` 记录本机/客户端环境；仅在上述同机条件下才能视为 server 环境。远程服务不在本设计假设内，不能用客户端记录代替远程环境。

在项目根目录严格按以下固定命令启动服务一次（Colab 中使用 `%%bash` cell）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 --tensor-parallel-size 1 \
  --max-model-len 2048 --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --host 127.0.0.1 --port 8000 --no-enable-prefix-caching \
  > /tmp/vllm_input_concurrency_server.log 2>&1 &
echo $! > /tmp/vllm_input_concurrency_server.pid
```

固定 `max-model-len=2048`，足以容纳最大 1024 输入加 64 输出。九组均使用同一实例、同一调度配置和同一 KV cache 容量；不要按组重启或调整参数。记录启动日志中的 attention backend、chunked prefill、KV cache 和调度配置实际值。以上配置明确限制最大序列数为 8；与此前使用默认调度配置的历史实验比较时，应说明这个差异，以本轮重新测量的 input=128 组作为基线。

等待就绪并检查日志：

```bash
timeout 900 bash -c 'until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done'
tail -n 50 /tmp/vllm_input_concurrency_server.log
```

确认模型、FP16、prefix caching disabled 及服务配置正确，GPU 无其他任务。服务初始化和编译发生在正式 benchmark 之前。

第一组测试前，脚本查询 `/v1/models`，自动验证模型 ID 和 `max_model_len=2048`，并将原始响应保存为 `server_models.json`。接口失败、字段缺失或配置不符均停止。vLLM 0.31.0 的字段来自实际 model config，见 [官方实现](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/models/serving.py)。

本方案未采用能可靠查询 dtype 和 prefix caching 的接口，因此二者在环境记录中明确写为“由启动命令保证”，不宣称自动验证；请按固定命令启动并人工核对日志。其他调度/KV cache 配置及全程实例未重启也由启动命令和运行流程保证，API 查询不能证明进程身份一直不变。

运行目录使用不带 `-p` 的 `mkdir` 原子创建；已有目录或同名并行启动的失败进程会立即退出，不写入该目录。父目录可以共享，运行目录不可共享。

## 执行九组测试

```bash
bash scripts/run_vllm_online_input_concurrency_benchmark.sh \
  results/vllm_online_input_concurrency/t4_run_01
cp /tmp/vllm_input_concurrency_server.log \
  results/vllm_online_input_concurrency/t4_run_01/server.log
```

脚本不启动或停止服务，按 input length 升序、每个长度内 concurrency 升序串行执行。默认使用 localhost:8000，可通过 `VLLM_BASE_URL` 指定地址。要求全新的结果目录，防止覆盖结果。命令失败或 JSON 验证失败立即停止，不生成最终 summary；排查后用新的运行目录重新执行。即使运行失败，也应保存 server 日志。

```text
results/vllm_online_input_concurrency/t4_run_01/
├── environment.txt
├── server_models.json
├── server.log
├── input_128/
│   ├── concurrency_1.json
│   ├── concurrency_1.log
│   ├── concurrency_4.json
│   ├── concurrency_4.log
│   ├── concurrency_8.json
│   └── concurrency_8.log
├── input_512/    # 同样六个文件
├── input_1024/   # 同样六个文件
└── summary.csv
```

每组必须 completed=100、failed=0、max_concurrency 与组合一致；input_lens/output_lens 各含 100 个正确长度，总输入为 `100 × input_length`、总输出为 6400。汇总工具验证九组后从原始 JSON 写入 CSV，保留原始精度，不修改 JSON。CSV 包含：

- input_length、concurrency、completed、failed、max_concurrency、总输入/输出 tokens。
- request_throughput（req/s）、output_throughput（tok/s）、total_token_throughput（tok/s）、duration（s）。
- TTFT、TPOT、ITL、E2EL 的 mean、median、P90、P99；列名沿用 JSON 的 `*_ms`，TPOT 单位为 ms/token，其余为 ms。

## 两种分析视角

从 JSON 重新验证并查看结果，不覆盖已存在的 summary：

```bash
# 固定 input length，比较 concurrency
python scripts/summarize_vllm_input_concurrency.py \
  results/vllm_online_input_concurrency/t4_run_01 --print-only --view input

# 固定 concurrency，比较 input length
python scripts/summarize_vllm_input_concurrency.py \
  results/vllm_online_input_concurrency/t4_run_01 --print-only --view concurrency
```

两种 `--view` 命令均重新读取并验证九组 JSON，再按所选维度排序输出，不读取 summary.csv；`--print-only` 不创建或覆盖 CSV。summary.csv 由默认汇总命令从同一批 JSON 生成，行顺序固定为输入长度优先，`--view` 只影响终端显示。绘图可使用 CSV，分别以 concurrency 或 input_length 为横轴，另一维度为曲线分组；优先展示 output throughput、mean TTFT、mean TPOT、mean E2EL，并检查 P90/P99。图可保存至本轮目录下的 `figures/`，本脚本不自动生成图。

公平比较只改变输入长度和并发上限。同一输入长度下固定 seed 和生成参数；不同输入长度使用相同生成方式，不假设随机文本完全相同。每组独立预热，固定输出长度，九组之间无重叠请求。模型、tokenizer、服务配置、客户端机器和网络路径均保持一致。

优先比较 request throughput 和 output throughput；total token throughput 包含输入 tokens，不能单凭它判断生成能力提升。TTFT 不等同于纯 prefill latency；E2EL 包含客户端发出请求后的服务排队、处理和响应时间，不包含客户端等待并发信号量的时间。这是饱和 online serving 测试，与直接调用 LLM.generate 的 offline single-request benchmark 口径不同。

每组 100 个请求的 P99 稳定性有限。建议用 t4_run_02、t4_run_03 重复完整实验，记录执行顺序及 GPU 温度/频率；后续轮次可轮换组合顺序，分析时间漂移。本轮脚本固定升序，所有结果先按单轮观察解读。

## 测试完成后停止服务

仅在全部九组完成并保存日志后，停止本次启动的实例：

```bash
if [[ -f /tmp/vllm_input_concurrency_server.pid ]]; then
  server_pid=$(cat /tmp/vllm_input_concurrency_server.pid)
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
  fi
fi
```
