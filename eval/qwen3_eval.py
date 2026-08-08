import argparse
import json
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    # math-verify 用于符号等价验证，避免格式差异导致误判
    from math_verify import parse as mv_parse, verify as mv_verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from latex2sympy2_extended.math_normalization import NormalizationConfig

    _mv_available = True
    MV_EXTRACTION_CONFIG: Sequence[Any] = (
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=True,
                nits=True,
                boxed="all",
                equations=False,
            )
        ),
        ExprExtractionConfig(),
    )
except Exception:  # pragma: no cover - 可选依赖
    mv_parse = mv_verify = None
    _mv_available = False
# lmdeploy imports are deferred to avoid hard failure on environments without it


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # local import to avoid mandatory dependency before use

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    split: str
    question_field: str
    answer_field: str
    prompt_template: str
    answer_pattern: str = r"\[\[(.*?)\]\]"
    postprocess: Optional[str] = None
    config_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)


# ---------------------- Answer post-processors ----------------------

def gsm8k_number(text: str) -> str:
    # gsm8k 推理中可能出现多个数字；优先取 \boxed{...} 或末尾数字作为最终答案
    boxed = math_box(text)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", boxed):
        return boxed

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    return text.strip()


def math_box(text: str) -> str:
    # extract LaTeX boxed answer if exists
    # 使用非贪婪匹配并允许嵌套括号内出现大括号，尽量捕获 \boxed{...} 的完整内容
    m = re.search(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if m:
        content = m.group(1)
        # 尝试截断到成对的大括号闭合处，避免在 \frac 等内部提前结束
        depth = 0
        collected = []
        for ch in content:
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    break
                depth -= 1
            collected.append(ch)
        parsed = "".join(collected).strip()
        # 将 \\frac 等双反斜杠序列收敛成单反斜杠，便于比较和展示
        parsed = parsed.replace("\\\\", "\\")
        return parsed
    return text.strip()


POSTPROCESSORS: Dict[str, Callable[[str], str]] = {
    "gsm8k_number": gsm8k_number,
    "math_box": math_box,
}


# ---------------------- Dataset registry ----------------------

def build_default_configs() -> Dict[str, DatasetConfig]:
    math_cfg = DatasetConfig(
        name="math500",
        hf_repo="HuggingFaceH4/MATH-500",
        split="test",
        config_name=None,
        question_field="problem",
        answer_field="solution",
        prompt_template=(
            "You are a helpful math tutor. Solve the following problem step by step.\n"
            "Problem: {question}\n"
            "Show your reasoning. Put the final answer inside [[...]]."
        ),
        answer_pattern=r"\[\[(.*?)\]\]",
        postprocess="math_box",
    )
    return {
        "gsm8k": DatasetConfig(
            name="gsm8k",
            hf_repo="openai/gsm8k",
            split="test",
            config_name="main",
            question_field="question",
            answer_field="answer",
            prompt_template=(
                "You are a helpful math tutor. Solve the following problem step by step.\n"
                "Question: {question}\n"
                "Show reasoning clearly. Put the final numeric answer inside [[...]]."
            ),
            answer_pattern=r"\[\[(.*?)\]\]",
            postprocess="gsm8k_number",
        ),
        "math500": math_cfg,
        "math": math_cfg,  # alias to avoid breaking shorthand
    }


def merge_yaml_configs(defaults: Dict[str, DatasetConfig], yaml_path: Optional[Path]) -> Dict[str, DatasetConfig]:
    if yaml_path is None:
        return defaults
    cfg = load_yaml(yaml_path)
    datasets = cfg.get("datasets", {}) if isinstance(cfg, dict) else {}
    merged = dict(defaults)
    for name, item in datasets.items():
        merged[name] = DatasetConfig(
            name=name,
            hf_repo=item["hf_repo"],
            split=item.get("split", "test"),
            config_name=item.get("config_name"),
            question_field=item.get("question_field", "question"),
            answer_field=item.get("answer_field", "answer"),
            prompt_template=item["prompt_template"],
            answer_pattern=item.get("answer_pattern", r"\[\[(.*?)\]\]"),
            postprocess=item.get("postprocess"),
            extra={k: v for k, v in item.items() if k not in {"hf_repo", "split", "question_field", "answer_field", "prompt_template", "answer_pattern", "postprocess"}},
        )
    return merged


# ---------------------- Token length utils ----------------------

def tokens_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def bucket_counts(lengths: List[int], buckets: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024)) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for l in lengths:
        placed = False
        for b in buckets:
            if l <= b:
                key = f"<= {b}"
                counts[key] = counts.get(key, 0) + 1
                placed = True
                break
        if not placed:
            key = f"> {buckets[-1]}"
            counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------- Answer parsing ----------------------

def parse_answer(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_final_answer(text: str, cfg: DatasetConfig) -> str:
    extracted = parse_answer(text, cfg.answer_pattern) or text.strip()
    if cfg.postprocess and cfg.postprocess in POSTPROCESSORS:
        extracted = POSTPROCESSORS[cfg.postprocess](extracted)
    return extracted.strip()


# ---------------------- Answer equivalence ----------------------

def _normalize_equiv_string(text: str) -> str:
    """轻量级字符串归一化，用于 math-verify 不可用或解析失败时的兜底比较。"""
    t = math_box(text)  # 先处理 \boxed 包裹
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"^\$+|\$+$", "", t)  # 去掉成对 $ / $$ 包裹
    t = t.replace("\\\\", "\\")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def verify_equiv_answer(gold: str, pred: str) -> bool:
    """使用 math-verify 做符号等价验证，失败时回退到字符串归一化比较。"""
    gold_norm = gold.strip()
    pred_norm = pred.strip()

    if _mv_available:
        try:
            gold_expr = mv_parse(gold_norm, extraction_config=MV_EXTRACTION_CONFIG, raise_on_error=False)
            pred_expr = mv_parse(pred_norm, extraction_config=MV_EXTRACTION_CONFIG, raise_on_error=False)
            if gold_expr and pred_expr:
                if mv_verify(gold_expr, pred_expr, raise_on_error=False):
                    return True
        except Exception:
            # 解析或验证异常时回退到字符串比较
            pass

    return _normalize_equiv_string(gold_norm) == _normalize_equiv_string(pred_norm)


# ---------------------- Main evaluation ----------------------

def run_eval(args: argparse.Namespace) -> None:
    defaults = build_default_configs()
    configs = merge_yaml_configs(defaults, Path(args.dataset_config) if args.dataset_config else None)

    for ds_name in args.dataset:
        if ds_name not in configs:
            raise ValueError(f"Dataset {ds_name} not configured. Available: {list(configs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # detect thinking support
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    supports_thinking = "enable_thinking" in chat_template or "<think>" in chat_template

    # lazy import lmdeploy
    import lmdeploy
    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    # 在部分 lmdeploy 版本中，pipeline 内部会向 AsyncEngine 传递 backend 参数，
    # 若调用时也传 backend 可能触发 “multiple values for backend” 报错。
    # 这里对 turbomind 使用 backend_config 避免重复传参。
    if args.backend == "turbomind":
        tm_cfg = TurbomindEngineConfig(tp=args.tp, session_len=args.session_len)
        pipe = pipeline(args.model_path, backend_config=tm_cfg)
    else:
        pipe = pipeline(
            args.model_path,
            backend=args.backend,
            tp=args.tp,
            session_len=args.session_len,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for ds_name in args.dataset:
        cfg = configs[ds_name]
        if cfg.config_name:
            ds = load_dataset(cfg.hf_repo, cfg.config_name, split=cfg.split, cache_dir=args.hf_cache)
        else:
            ds = load_dataset(cfg.hf_repo, split=cfg.split, cache_dir=args.hf_cache)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        correct = 0
        total = 0
        answer_lengths: List[int] = []
        think_lengths: List[int] = []
        thinking_on_ds = supports_thinking and args.enable_thinking

        ds_out_path = output_dir / f"{ds_name}-results.jsonl"
        with ds_out_path.open("w", encoding="utf-8") as fout:
            for item in tqdm(ds, desc=f"{ds_name} eval", unit="sample"):
                question = item[cfg.question_field]
                gold = extract_final_answer(item[cfg.answer_field], cfg)
                prompt = cfg.render_prompt(question)

                gen_cfg = GenerationConfig(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                if thinking_on_ds:
                    gen_cfg.enable_thinking = True

                resp = pipe([prompt], gen_config=gen_cfg)[0]
                # lmdeploy pipeline returns response object or string depending on version
                if isinstance(resp, dict):
                    content = resp.get("response", "") or resp.get("text", "")
                elif hasattr(resp, "text"):
                    content = getattr(resp, "text")
                else:
                    content = str(resp)

                parsed = extract_final_answer(content, cfg)

                is_correct = verify_equiv_answer(gold, parsed)
                correct += int(is_correct)
                total += 1

                # lengths
                answer_len = tokens_len(tokenizer, content)
                answer_lengths.append(answer_len)

                think_len = None
                if thinking_on_ds:
                    m = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                    if m:
                        think_text = m.group(1)
                        think_len = tokens_len(tokenizer, think_text)
                        think_lengths.append(think_len)

                rec = {
                    "question": question,
                    "prompt": prompt,
                    "gold": gold,
                    "response": content,
                    "parsed_answer": parsed,
                    "is_correct": is_correct,
                    "answer_len_tokens": answer_len,
                    "think_len_tokens": think_len,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        acc = correct / total if total else 0.0
        answer_stats = {
            "mean": float(np.mean(answer_lengths)) if answer_lengths else 0.0,
            "median": float(np.median(answer_lengths)) if answer_lengths else 0.0,
            "distribution": bucket_counts(answer_lengths),
        }
        think_stats = None
        if thinking_on_ds:
            think_stats = {
                "mean": float(np.mean(think_lengths)) if think_lengths else 0.0,
                "median": float(np.median(think_lengths)) if think_lengths else 0.0,
                "distribution": bucket_counts(think_lengths) if think_lengths else {},
            }

        all_metrics[ds_name] = {
            "accuracy": acc,
            "count": total,
            "answer_length_tokens": answer_stats,
            "think_length_tokens": think_stats,
            "supports_thinking": supports_thinking,
            "thinking_enabled": thinking_on_ds,
        }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LMDeploy Qwen3 evaluation (single-node TP, with thinking support when available). GPU & CUDA required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/lmdeploy_qwen3_eval.py \\\n"
            "    --model-path /path/to/qwen3 \\\n"
            "    --dataset gsm8k math500 \\\n"
            "    --tp 2 --backend turbomind \\\n"
            "    --output-dir outputs/qwen3-tp2 \\\n"
            "    --enable-thinking\n\n"
            "If offline, pre-download datasets and model into local cache and pass --hf-cache.\n"
        ),
    )
    parser.add_argument("--model-path", required=True, help="Path or repo to Qwen3 model")
    parser.add_argument("--dataset", nargs="+", default=["gsm8k", "math500"], help="Datasets to run (e.g., gsm8k math500)")
    parser.add_argument("--dataset-config", help="YAML config to extend/override datasets")
    parser.add_argument("--hf-cache", help="HF datasets cache dir")
    parser.add_argument("--limit", type=int, help="Limit number of samples per dataset")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism for turbomind")
    parser.add_argument("--backend", default="turbomind", help="LMDeploy backend")
    parser.add_argument("--session-len", type=int, default=81640, help="Context window")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking when template supports (default: on if supported).",
    )
    parser.add_argument(
        "--disable-thinking-override",
        dest="enable_thinking",
        action="store_false",
        help="Force disable thinking even if template supports.",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--output-dir", default="outputs/lmdeploy_eval", help="Where to save results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_eval(args)


if __name__ == "__main__":
    main()
