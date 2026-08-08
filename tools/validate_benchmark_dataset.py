#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("数据文件必须是 JSON 数组")
    if not payload:
        raise ValueError("数据文件为空")
    return payload


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _validate_safety(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    subsets = Counter()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"[{idx}] 样本不是对象")
            continue

        if not _is_non_empty_str(item.get("input")):
            errors.append(f"[{idx}] 缺少非空字段 input")

        subset = item.get("subset")
        if not _is_non_empty_str(subset):
            errors.append(f"[{idx}] 缺少非空字段 subset")
        else:
            subsets[subset.strip().lower()] += 1

    summary = {
        "total": len(data),
        "subset_distribution": dict(subsets),
        "has_harmful": subsets.get("harmful", 0) > 0,
        "has_benign": subsets.get("benign", 0) > 0,
    }
    return errors, summary


def _validate_multilingual(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    languages = Counter()
    multiple_choice = 0
    freeform = 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"[{idx}] 样本不是对象")
            continue

        if not _is_non_empty_str(item.get("question")):
            errors.append(f"[{idx}] 缺少非空字段 question")

        language = item.get("language")
        if not _is_non_empty_str(language):
            errors.append(f"[{idx}] 缺少非空字段 language")
        else:
            languages[language.strip()] += 1

        choices = item.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            multiple_choice += 1
            if not all(isinstance(choice, str) for choice in choices):
                errors.append(f"[{idx}] choices 必须为字符串数组")
            answer = item.get("answer")
            if not _is_non_empty_str(answer):
                errors.append(f"[{idx}] 选择题样本缺少 answer")
        else:
            freeform += 1
            has_reference = False
            if _is_non_empty_str(item.get("best_answer")):
                has_reference = True
            if _is_non_empty_str(item.get("answer")):
                has_reference = True
            raw_answers = item.get("answers")
            if isinstance(raw_answers, list) and len(raw_answers) > 0:
                has_reference = True
            if not has_reference:
                errors.append(f"[{idx}] 非选择题样本缺少 answers/best_answer/answer 参考答案")

    summary = {
        "total": len(data),
        "language_count": len(languages),
        "languages": dict(languages),
        "multiple_choice_samples": multiple_choice,
        "freeform_samples": freeform,
    }
    return errors, summary


def _validate_hallucination(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    has_justification = 0
    missing_context = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"[{idx}] 样本不是对象")
            continue

        if not _is_non_empty_str(item.get("question")):
            errors.append(f"[{idx}] 缺少非空字段 question")
        if not _is_non_empty_str(item.get("context")):
            missing_context += 1
        if _is_non_empty_str(item.get("justification")):
            has_justification += 1

    summary = {
        "total": len(data),
        "with_justification": has_justification,
        "missing_context": missing_context,
    }
    return errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 benchmark 数据集字段结构")
    parser.add_argument(
        "--task",
        required=True,
        choices=["safety", "multilingual", "hallucination"],
        help="评测类型",
    )
    parser.add_argument("--data-path", required=True, help="数据集 JSON 文件路径")
    args = parser.parse_args()

    data = _load_json_array(args.data_path)
    if args.task == "safety":
        errors, summary = _validate_safety(data)
    elif args.task == "multilingual":
        errors, summary = _validate_multilingual(data)
    else:
        errors, summary = _validate_hallucination(data)

    print(f"[validate] task={args.task} file={args.data_path}")
    print(f"[validate] summary={json.dumps(summary, ensure_ascii=False)}")
    if errors:
        print("[validate] 发现字段问题:")
        for err in errors[:100]:
            print(f"- {err}")
        if len(errors) > 100:
            print(f"- ... 其余 {len(errors) - 100} 条省略")
        sys.exit(1)

    print("[validate] 数据集字段校验通过")


if __name__ == "__main__":
    main()
