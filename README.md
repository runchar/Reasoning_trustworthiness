# Shorter, but Still Trustworthy?

## An Empirical Study of Chain-of-Thought Compression

**COLM 2026 — Accepted**

**Authors:** Lingjie Zeng, Xiaofan Chen, Yanbo Wang, and Xiuying Chen

## Overview

Long chain-of-thought (Long-CoT) reasoning can improve difficult-task performance, but its inference cost motivates compression. This work asks whether shortening a reasoning trace also preserves the model's trustworthiness. We evaluate compressed models against matched uncompressed baselines under a common protocol, covering three dimensions: safety, hallucination resistance, and multilingual robustness.

The camera-ready study finds that CoT compression frequently introduces trustworthiness regressions, and that the degradation profile depends on the compression method and evaluation dimension. It introduces a per-dimension normalized efficiency score so that length savings are not allowed to hide a trustworthiness loss. As a feasibility result, an alignment-aware DPO variant reduces mean CoT token count by **19.3%** on reasoning benchmarks with substantially smaller trustworthiness loss.

## Main Results

The table below is a transcription of Table 1 in the camera-ready source (`main.tex`, commit `038543a`). Values are percentages. A dash means that the benchmark was not reported for that model under the paper's benchmark scope.

| Base model | Paradigm | Model | HarmBench ↑ | AgentHarm ↓ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DeepScaleR-1.5B-Preview | — | Baseline | 21.5 | — | 26.2 | 51.0 |
| DeepScaleR-1.5B-Preview | L1 | L1-Qwen-1.5B-Exact | 20.3 | — | 25.7 | 34.6 |
| DeepScaleR-1.5B-Preview | L1 | L1-Qwen-1.5B-Max | 20.8 | — | 28.2 | 36.6 |
| DeepScaleR-1.5B-Preview | BeConcise | — | 21.5 | — | 23.7 | 48.8 |
| DeepSeek-R1-Distill-Qwen-1.5B | — | Baseline | 18.3 | — | 20.6 | 53.2 |
| DeepSeek-R1-Distill-Qwen-1.5B | Thinkless | Thinkless-1.5B | 15.3 | — | 19.8 | 36.4 |
| DeepSeek-R1-Distill-Qwen-7B | — | Baseline | 30.5 | — | 35.5 | 60.2 |
| DeepSeek-R1-Distill-Qwen-7B | L1 | L1-Qwen-7B-Exact | 29.0 | — | 39.7 | 56.4 |
| DeepSeek-R1-Distill-Qwen-7B | L1 | L1-Qwen-7B-Max | 28.8 | — | 39.0 | 58.4 |
| DeepSeek-R1-Distill-Qwen-7B | BeConcise | — | 31.0 | — | 30.6 | 60.4 |
| DeepSeek-R1-Distill-Qwen-7B | VeriThinker | VeriThinker-7B | 29.5 | — | 25.3 | 62.8 |
| DeepSeek-R1-Distill-Qwen-7B | DPO | — | 34.8 | — | 37.3 | 63.4 |
| Qwen3-8B | — | Baseline | 68.8 | 55.5 | 65.9 | 75.8 |
| Qwen3-8B | L1 | L1-Qwen3-8B-Exact | 58.8 | 61.7 | 68.2 | 79.0 |
| Qwen3-8B | L1 | L1-Qwen3-8B-Max | 58.0 | 61.3 | 67.2 | 76.2 |
| Qwen3-8B | BeConcise | — | 63.0 | 52.0 | 69.0 | 73.4 |
| Qwen3-8B | TokenSkip | TokenSkip | 65.3 | 54.3 | 60.3 | 75.0 |
| Qwen3-8B | DPO | — | 65.0 | 55.4 | 65.6 | 75.0 |
| DeepSeek-R1-0528-Qwen3-8B | — | Baseline | 71.5 | — | 40.7 | 68.4 |
| DeepSeek-R1-0528-Qwen3-8B | Merged | average-Merged | 75.2 | — | 40.6 | 71.4 |
| DeepSeek-R1-0528-Qwen3-8B | Merged | DARE-TIES-Merged | 64.5 | — | 66.2 | 74.0 |
| DeepSeek-R1-0528-Qwen3-8B | Merged | TA-Merged | 75.2 | — | 35.1 | 67.2 |
| DeepSeek-R1-0528-Qwen3-8B | Merged | TIES-Merged | 64.0 | — | 66.1 | 73.8 |
| DeepSeek-R1-0528-Qwen3-8B | Merged | TIES-origin-Merged | 62.0 | — | 66.3 | 75.0 |
| DeepSeek-R1-0528-Qwen3-8B | DPO | — | 82.2 | — | 45.0 | 75.2 |
| Qwen3-32B | — | Baseline | 61.4 | — | 73.0 | 71.5 |
| Qwen3-32B | BeConcise | — | 55.0 | — | 70.9 | 69.9 |
| Qwen3-32B | TokenSkip | TokenSkip | 61.0 | — | 76.8 | 71.5 |
| QwQ-32B-Preview | — | Baseline | 64.2 | — | 61.3 | 67.4 |
| QwQ-32B-Preview | O1-Pruned | QwQ-32B-Preview-Pruned | 54.2 | — | 69.0 | 73.0 |

### Metric explanations

- **HarmBench ↑:** refusal accuracy; higher is better.
- **AgentHarm ↓:** harmful-task completion rate; lower is safer. The paper reports this only where the model has sufficient agentic capability for an informative comparison.
- **MMLU-ProX ↑:** multilingual accuracy across 29 languages; higher is better.
- **FaithEval ↑:** truthfulness rate on unanswerable questions; higher is better.

## Results Workbook

The [original experiment workbook](docs/assets/Reasoning_Trustworthiness.xlsx?raw=1) contains the study's results. It is provided as the original experiment workbook, rather than as a replacement for the evaluation code or benchmark requirements.

## Repository Structure and Entrypoints

The repository layout below records the known legacy repository components. This README intentionally does not invent installation commands; consult `BENCHMARK_TEST_REQUIREMENTS.md` for benchmark-specific requirements.

| Path | Role |
| --- | --- |
| `dataset/` | Dataset assets and benchmark inputs. |
| `eval/` | Evaluation code and shared evaluation helpers. |
| `hallucination/` | Hallucination-resistance evaluation components. |
| `multilingual/` | Multilingual evaluation components. |
| `safety/` | Safety evaluation components. |
| `tools/` | Supporting tools used by the evaluation workflow. |
| `run_batch_hallucination.sh` | Batch hallucination-evaluation entrypoint. |
| `run_batch_multilingual.sh` | Batch multilingual-evaluation entrypoint. |
| `run_batch_safety.sh` | Batch safety-evaluation entrypoint. |
| `run_eval.sh` | Evaluation entrypoint. |
| `BENCHMARK_TEST_REQUIREMENTS.md` | Benchmark test requirements and evaluation notes. |

## Citation

```bibtex
@inproceedings{zeng2026shorter,
  title     = {Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression},
  author    = {Zeng, Lingjie and Chen, Xiaofan and Wang, Yanbo and Chen, Xiuying},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  note      = {Accepted at COLM 2026}
}
```
