from typing import Callable, Any


def exact_match_binary(pred: Any, gold: Any, comparator: Callable[[Any, Any], bool]) -> int:
    return int(bool(comparator(pred, gold)))


def exact_match_rate(binary_values: list[int]) -> float:
    if not binary_values:
        return 0.0
    return (sum(binary_values) / len(binary_values)) * 100.0