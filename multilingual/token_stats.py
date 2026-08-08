import statistics
from typing import Iterable, List, Optional, Tuple


def _clean_numbers(values: Iterable[Optional[int]]) -> List[int]:
    return [int(v) for v in values if v is not None]


def aggregate_token_stats(generated_tokens: Iterable[Optional[int]], thinking_tokens: Iterable[Optional[int]]) -> Tuple[dict, dict]:
    """Return mean/median for generated and thinking tokens.

    Missing (None) values are ignored. Empty lists yield 0 for both stats.
    """
    gen_list = _clean_numbers(generated_tokens)
    think_list = _clean_numbers(thinking_tokens)

    def stats(nums: List[int]) -> dict:
        if not nums:
            return {"mean": 0.0, "median": 0.0}
        return {
            "mean": statistics.mean(nums),
            "median": statistics.median(nums),
        }

    return stats(gen_list), stats(think_list)
