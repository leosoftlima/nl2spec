from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import json
import re


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_select(
    query_emb: np.ndarray,
    candidate_embs: List[np.ndarray],
    k: int = 3,
    mmr_weight: float = 0.7,
) -> List[Tuple[int, float]]:
    selected: List[Tuple[int, float]] = []
    selected_idx: List[int] = []
    candidates = list(range(len(candidate_embs)))

    while len(selected_idx) < min(k, len(candidates)):
        best_idx = None
        best_score = -np.inf

        for i in candidates:
            if i in selected_idx:
                continue

            relevance = cosine_sim(query_emb, candidate_embs[i])

            diversity = 0.0
            if selected_idx:
                diversity = max(
                    cosine_sim(candidate_embs[i], candidate_embs[j])
                    for j in selected_idx
                )

            score = mmr_weight * relevance - (1.0 - mmr_weight) * diversity

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        selected.append((best_idx, float(best_score)))

    return selected


def _serialize_ir_for_mmr(ir_base: Any) -> str:
    if ir_base is None:
        return ""

    try:
        return json.dumps(
            ir_base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError:
        return str(ir_base)


def _embed_text(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)

    if not text:
        return vec

    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    for tok in tokens:
        idx = hash(tok) % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def select_mmr(
    files: List[Path],
    ir_base: dict,
    k: int,
    return_scores: bool = False,
    mmr_weight: float = 0.7,
):
    query_text = _serialize_ir_for_mmr(ir_base)
    query_emb = _embed_text(query_text)

    valid_files: List[Path] = []
    candidate_embs: List[np.ndarray] = []

    for f in files:
        candidate_text = f.read_text(encoding="utf-8")
        valid_files.append(f)
        candidate_embs.append(_embed_text(candidate_text))

    chosen_with_scores = mmr_select(
        query_emb=query_emb,
        candidate_embs=candidate_embs,
        k=k,
        mmr_weight=mmr_weight,
    )

    if return_scores:
        return [(valid_files[i], score) for i, score in chosen_with_scores]

    return [valid_files[i] for i, _ in chosen_with_scores]