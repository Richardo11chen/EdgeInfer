# EdgeInfer Experiment Report

## Setup

- Local repository: `EdgeInfer`
- Remote machine: `nju@10.48.1.151`
- GPU: `NVIDIA GeForce RTX 3070 8GB`
- Runtime environment: `uv`-managed Python environment in `~/EdgeInfer`
- Models:
  - `models/qwen3-1.7b`
  - `models/qwen3-8b`
- Shared prompt set: [eval/prompts/required_prompts.jsonl](/home/aromatic/Applications/Homework/DeepLearning/EdgeInfer/eval/prompts/required_prompts.jsonl)

## Measurement Definition

- `ttft_ms = prefill_latency_ms`
- `decode_tokens_per_sec = (generated_tokens - 1) / (total_generation_time - ttft)`
- Peak memory uses `torch.cuda.max_memory_allocated()` for EdgeInfer
- vLLM `ttft_ms` is approximated using a separate greedy `max_tokens=1` run, and this is recorded in `measurement_notes`

## Correctness

- `Qwen3-1.7B` correctness runner now treats greedy token alignment as the pass criterion.
- Current remote result:
  - `tokens passed = True`
  - `mismatch_count = 0`
  - `logits passed = False` under `float16`, but token generation matches Hugging Face greedy decoding.

## Completed Results

### Qwen3-1.7B Baseline

| Framework | Mode | Avg TTFT (ms) | Avg Decode Tokens/s | Peak Memory (MiB) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| EdgeInfer | resident | 48.8706 | 43.3166 | 3338.0039 | Native resident baseline |
| vLLM | default | 14.9028 | 102.5326 | 6303.3125 | TTFT estimated by separate `max_tokens=1` run |

### 1.7B Observation

- On the same `3070 8GB` machine, vLLM shows substantially lower TTFT and higher decode throughput than the current EdgeInfer resident runtime.
- EdgeInfer resident keeps peak memory lower than vLLM in this setup, but the throughput gap is still large enough that the current implementation is primarily useful as a teaching/runtime-systems baseline rather than a performance leader.
- The prompt set mixes Chinese and English tasks, so the reported averages are not dominated by a single prompt style.

## In Progress

- `Qwen3-8B naive offload` is currently running on the remote `3070` machine.
- Remaining required experiments after that:
  - `Qwen3-8B prefetch offload`
  - `Qwen3-8B vLLM cpu offload`
  - `Qwen3-8B budget scan`
  - result aggregation and figure generation

## Next Fill-ins

- Add `8B naive/prefetch/vLLM` summary table
- Add budget-scan figure references
- Add final comparison analysis across resident, naive offload, prefetch offload, and vLLM

## Expected Final Discussion Points

- Whether `naive` and `prefetch` both satisfy the assignment's offloading semantics on an `8GB` GPU
- How much `prefetch` improves TTFT and decode throughput relative to `naive`
- How GPU layer budget changes throughput and peak memory
- Where EdgeInfer still trails vLLM and which parts of the runtime likely explain the gap
