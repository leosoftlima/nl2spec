from pathlib import Path
from typing import List, Tuple
import json


def mmr_select_from_similarity_matrix(
    relevance_scores: List[float],
    pairwise_similarity: List[List[float]],
    k: int = 3,
    mmr_weight: float = 0.7,
) -> List[Tuple[int, float]]:
    n = len(relevance_scores)
    if n == 0:
        return []

    selected: List[Tuple[int, float]] = []
    selected_idx: List[int] = []
    candidates = list(range(n))

    while len(selected_idx) < min(k, n):
        best_idx = None
        best_score = float("-inf")

        for i in candidates:
            if i in selected_idx:
                continue

            relevance = relevance_scores[i]

            redundancy = 0.0
            if selected_idx:
                redundancy = max(pairwise_similarity[i][j] for j in selected_idx)

            score = mmr_weight * relevance - (1.0 - mmr_weight) * redundancy

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        selected.append((best_idx, float(best_score)))

    return selected


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _distance_to_similarity(dist: float) -> float:
    dist = float(dist)
    if dist < 0:
        raise ValueError(f"IRSP distance must be non-negative, got {dist}.")
    return 1.0 / (1.0 + dist)


def _selector_similarity(selector, ir_a: dict, ir_b: dict) -> float:
    vec_a = selector.extract_vector(ir_a)
    vec_b = selector.extract_vector(ir_b)
    dist = selector.distance(vec_a, vec_b)
    return 1.0 / (1.0 + dist)


def select_mmr_irsp(
    selector,
    files: List[Path],
    ir_base: dict,
    k: int,
    return_scores: bool = False,
    mmr_weight: float = 0.7,
):
    valid_files: List[Path] = []
    candidate_irs: List[dict] = []

    for f in files:
        candidate_ir = _load_json_file(f)
        valid_files.append(f)
        candidate_irs.append(candidate_ir)

    relevance_scores: List[float] = [
        _selector_similarity(selector, ir_base, cand_ir)
        for cand_ir in candidate_irs
    ]

    n = len(candidate_irs)
    pairwise_similarity: List[List[float]] = [
        [0.0 for _ in range(n)] for _ in range(n)
    ]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                sim = _selector_similarity(selector, candidate_irs[i], candidate_irs[j])

            pairwise_similarity[i][j] = sim
            pairwise_similarity[j][i] = sim

    chosen_with_scores = mmr_select_from_similarity_matrix(
        relevance_scores=relevance_scores,
        pairwise_similarity=pairwise_similarity,
        k=k,
        mmr_weight=mmr_weight,
    )

    if return_scores:
        return [(valid_files[i], score) for i, score in chosen_with_scores]

    return [valid_files[i] for i, _ in chosen_with_scores]