<h1 align="center">Shorter, but Still Trustworthy?</h1>
<p align="center"><strong>An Empirical Study of Chain-of-Thought Compression</strong></p>
<p align="center">
  Lingjie Zeng, Xiaofan Chen, Yanbo Wang, and Xiuying Chen<br>
  <strong>COLM 2026</strong>
</p>
<p align="center">
  <a href="#results">Results</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="docs/assets/Reasoning_Trustworthiness.xlsx?raw=1">Results workbook</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

Evaluation code for our paper accepted at COLM 2026.

Compressing chain-of-thought (CoT) reduces inference cost. We study how it affects safety, hallucination resistance, and multilingual robustness, comparing compressed models with their uncompressed baselines under the same evaluation protocol.

Compression often reduces trustworthiness, with the effects varying across methods and benchmarks. We use a normalized efficiency score for each dimension to measure token savings alongside changes in trustworthiness. Our alignment-aware DPO variant reduces mean CoT length by **19.3%** on reasoning benchmarks, with smaller losses in trustworthiness.

## Results

Results from Table 1 of the camera-ready paper, grouped by base model. All values are percentages; arrows indicate the preferred direction.

| Benchmark | Metric |
| --- | --- |
| HarmBench ↑ | Refusal accuracy |
| AgentHarm ↓ | Harmful-task completion rate |
| MMLU-ProX ↑ | Multilingual accuracy across 29 languages |
| FaithEval ↑ | Truthfulness rate on unanswerable questions |

AgentHarm results are reported for the Qwen3-8B group, where the model's agentic capability supports a meaningful comparison. Each section below contains an uncompressed baseline and its compressed variants.

<details>
<summary><strong>DeepScaleR-1.5B-Preview</strong></summary>

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 21.5 | 26.2 | 51.0 |
| L1-Qwen-1.5B-Exact | 20.3 | 25.7 | 34.6 |
| L1-Qwen-1.5B-Max | 20.8 | 28.2 | 36.6 |
| BeConcise | 21.5 | 23.7 | 48.8 |

</details>

<details>
<summary><strong>DeepSeek-R1-Distill-Qwen-1.5B</strong></summary>

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 18.3 | 20.6 | 53.2 |
| Thinkless-1.5B | 15.3 | 19.8 | 36.4 |

</details>

<details>
<summary><strong>DeepSeek-R1-Distill-Qwen-7B</strong></summary>

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 30.5 | 35.5 | 60.2 |
| L1-Qwen-7B-Exact | 29.0 | 39.7 | 56.4 |
| L1-Qwen-7B-Max | 28.8 | 39.0 | 58.4 |
| BeConcise | 31.0 | 30.6 | 60.4 |
| VeriThinker-7B | 29.5 | 25.3 | 62.8 |
| DPO | 34.8 | 37.3 | 63.4 |

</details>

<details open>
<summary><strong>Qwen3-8B</strong></summary>

| Variant | HarmBench ↑ | AgentHarm ↓ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: | ---: |
| **Baseline** | 68.8 | 55.5 | 65.9 | 75.8 |
| L1-Qwen3-8B-Exact | 58.8 | 61.7 | 68.2 | 79.0 |
| L1-Qwen3-8B-Max | 58.0 | 61.3 | 67.2 | 76.2 |
| BeConcise | 63.0 | 52.0 | 69.0 | 73.4 |
| TokenSkip | 65.3 | 54.3 | 60.3 | 75.0 |
| DPO | 65.0 | 55.4 | 65.6 | 75.0 |

</details>

<details>
<summary><strong>DeepSeek-R1-0528-Qwen3-8B</strong></summary>

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 71.5 | 40.7 | 68.4 |
| average-Merged | 75.2 | 40.6 | 71.4 |
| DARE-TIES-Merged | 64.5 | 66.2 | 74.0 |
| TA-Merged | 75.2 | 35.1 | 67.2 |
| TIES-Merged | 64.0 | 66.1 | 73.8 |
| TIES-origin-Merged | 62.0 | 66.3 | 75.0 |
| DPO | 82.2 | 45.0 | 75.2 |

</details>

<details>
<summary><strong>Qwen3-32B</strong></summary>

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 61.4 | 73.0 | 71.5 |
| BeConcise | 55.0 | 70.9 | 69.9 |
| TokenSkip | 61.0 | 76.8 | 71.5 |

</details>

<details>
<summary><strong>QwQ-32B-Preview</strong></summary>

O1-Pruned uses the `QwQ-32B-Preview-Pruned` checkpoint.

| Variant | HarmBench ↑ | MMLU-ProX ↑ | FaithEval ↑ |
| --- | ---: | ---: | ---: |
| **Baseline** | 64.2 | 61.3 | 67.4 |
| O1-Pruned | 54.2 | 69.0 | 73.0 |

</details>

The [original results workbook](docs/assets/Reasoning_Trustworthiness.xlsx?raw=1) contains the experiment results. The tables above follow `main.tex` at paper-source commit `038543a`.

## Evaluation

The [benchmark guide](BENCHMARK_TEST_REQUIREMENTS.md) documents input fields, scoring rules, and evaluation settings in Chinese. The evaluators use LMDeploy. Before running a batch script, set its model paths and check the GPU configuration for your machine.

| Evaluation | Batch script | Evaluator |
| --- | --- | --- |
| Safety | [run_batch_safety.sh](run_batch_safety.sh) | [SafetyBench_lmdeploy.py](safety/SafetyBench_lmdeploy.py) |
| Multilingual | [run_batch_multilingual.sh](run_batch_multilingual.sh) | [MultilingualBench_lmdeploy.py](multilingual/MultilingualBench_lmdeploy.py) |
| Hallucination | [run_batch_hallucination.sh](run_batch_hallucination.sh) | [HallucinationBench_lmdeploy.py](hallucination/HallucinationBench_lmdeploy.py) |

For safety evaluation, set `MODELS_CSV` or edit `DEFAULT_MODELS` in the batch script. For multilingual and hallucination evaluation, edit the script's `MODELS` list. `DATA_PATH` and `OUTPUT_DIR` select the input file and output directory.

The batch scripts validate input fields before evaluation. To check a dataset separately, run:

```bash
python tools/validate_benchmark_dataset.py --task safety --data-path dataset/HarmBench400.json
python tools/validate_benchmark_dataset.py --task multilingual --data-path dataset/MMLU-PROX.json
python tools/validate_benchmark_dataset.py --task hallucination --data-path dataset/FaithEval_unanswerable.json
```

### Repository layout

| Path | Contents |
| --- | --- |
| [dataset/](dataset/) | Benchmark inputs |
| [eval/](eval/) | Reasoning benchmark evaluation scripts and configuration |
| [safety/](safety/) | Safety evaluators and judging scripts |
| [multilingual/](multilingual/) | Multilingual evaluators |
| [hallucination/](hallucination/) | Hallucination evaluators |
| [tools/](tools/) | Dataset validation |
| [run_eval.sh](run_eval.sh) | Single-model evaluation script; currently runs hallucination evaluation |

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
