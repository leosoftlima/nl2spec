# run:
# python -m nl2spec.scripts.run_compare_specs_semantic_only_dashboard_ast

import base64
import itertools
import json
import os
import pickle
import re
import stat
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None

try:
    from code_bert_score import score as code_bert_score_fn
except Exception:
    code_bert_score_fn = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

ORIGINAL_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\datasets\baseline_ir")

STRATEGY_PATHS = {
    "random": Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\random\few_k3"),
    "mmr": Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\mmr\few_k3"),
    "irsp": Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\irsp\few_k3"),
    "mmr_irsp": Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\mmr_irsp\few_k3"),
}

RESULT_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\results_codeBertScore_dashboard")
INDIVIDUAL_METRICS_PATH = RESULT_PATH / "individual" / "metrics"
INDIVIDUAL_MATRICES_PATH = RESULT_PATH / "individual" / "matrices"
PLOTS_PATH = RESULT_PATH / "plots"
DASHBOARD_PATH = RESULT_PATH / "dashboard"
CACHE_DIR = RESULT_PATH / "_cache"

PRIMARY_SCORE = "semantic_mean_score"
MIN_SCENARIOS_FOR_STATS = 5
# These lists are configured at runtime by configure_metric_selection().
COMPOSITE_METRICS: List[str] = []
OPTIONAL_PAIR_METRICS: List[str] = []

AVAILABLE_SCORE_METRICS = {
    "bertscore_f1": "BERTScore",
    "embedding_cosine_similarity": "EmbeddingCosine",
    "codebertscore_f1": "CodeBERTScore",
    "ast_similarity_score": "JSONTreeSimilarity",
}

DISPLAY_METRICS = {
    "bertscore_f1": "BERTScore",
    "embedding_cosine_similarity": "EmbeddingCosine",
    "codebertscore_f1": "CodeBERTScore",
    "ast_similarity_score": "JSONTreeSimilarity",
}

METRIC_SELECTION_OPTIONS = {
    "1": ["bertscore_f1"],
    "2": ["embedding_cosine_similarity"],
    "3": ["codebertscore_f1"],
    "4": ["ast_similarity_score"],
    "5": ["bertscore_f1", "embedding_cosine_similarity"],
    "6": ["bertscore_f1", "codebertscore_f1"],
    "7": ["embedding_cosine_similarity", "codebertscore_f1"],
    "8": ["bertscore_f1", "embedding_cosine_similarity", "codebertscore_f1"],
    "9": ["bertscore_f1", "embedding_cosine_similarity", "codebertscore_f1", "ast_similarity_score"],
    "all": ["bertscore_f1", "embedding_cosine_similarity", "codebertscore_f1", "ast_similarity_score"],
}

ACTIVE_SCORE_METRICS: List[str] = ["bertscore_f1", "embedding_cosine_similarity"]

# -----------------------------
# Model options
# -----------------------------
BERTSCORE_MODEL_OPTIONS = {
    "1": "roberta-large",
    "2": "microsoft/deberta-xlarge-mnli",
    "3": "microsoft/deberta-large-mnli",
    "4": "distilbert-base-uncased",
}

EMBEDDING_MODEL_OPTIONS = {
    "1": "sentence-transformers/all-MiniLM-L6-v2",
    "2": "sentence-transformers/all-mpnet-base-v2",
    "3": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "4": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

CODEBERTSCORE_LANG_OPTIONS = {
    "1": "java",
    "2": "python",
    "3": "javascript",
    "4": "php",
    "5": "ruby",
    "6": "go",
}

BERTSCORE_MODEL_TYPE = None
EMBEDDING_MODEL_NAME = None
CODEBERTSCORE_LANG = None

SEMANTIC_WEIGHT_BERTSCORE = 0.6
SEMANTIC_WEIGHT_EMBEDDING = 0.4

# -----------------------------
# Standardized color palette
# -----------------------------
LABEL_COLORS = {
    "errado": "#2EC4B6",
    "parcialmente_semelhante": "#0B3C6F",
    "parecido": "#2E6FA3",
    "muito_parecido": "#7FB3D5",
    "quase_igual": "#D6E6F2",
    "identico": "#BDBDBD",
}

LABELS_ORDER = [
    "errado",
    "parcialmente_semelhante",
    "parecido",
    "muito_parecido",
    "quase_igual",
    "identico",
]

SEMANTIC_LABEL_FOOTNOTE = (
    "Semantic label thresholds: Errado < 70; Parcialmente Semelhante = 70–84.99; "
    "Parecido = 85–92.99; Muito Parecido = 93–96.99; "
    "Quase Igual = 97–99.99; Idêntico = 100."
)

# Strategy colors harmonized with the same palette
STRATEGY_COLORS = {
    "random": "#0B3C6F",
    "mmr": "#2E6FA3",
    "irsp": "#7FB3D5",
    "mmr_irsp": "#D6E6F2",
}

PAIRWISE_COLORS = {
    "a_wins": "#0B3C6F",
    "b_wins": "#2E6FA3",
    "ties": "#BDBDBD",
}

REQUIRED_TOP_LEVEL_KEYS = ["id", "formalism", "domain", "signature", "ir"]
REQUIRED_IR_KEYS_ERE = ["events", "ere", "violation"]

STRUCTURAL_MODE_OPTIONS = {
    "1": "syntax_schema",
    "2": "ast",
    "3": "both",
}

STRUCTURAL_EVALUATION_MODE = "both"


def path_depth(path: Path) -> int:
    return len(path.parts)


def canonical_sort_key(item: Any) -> str:
    return canonical_json_string(item)


def candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, str]:
    return candidate["score_raw"], candidate["strategy"]


def safe_model_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def get_embedding_cache_file() -> Path:
    model_name = safe_model_name(EMBEDDING_MODEL_NAME or "unset_embedding_model")
    return CACHE_DIR / f"embedding_cache_{model_name}.pkl"


def get_bertscore_cache_file() -> Path:
    model_name = safe_model_name(BERTSCORE_MODEL_TYPE or "unset_bertscore_model")
    return CACHE_DIR / f"bertscore_cache_{model_name}.pkl"


def get_codebertscore_cache_file() -> Path:
    lang_name = safe_model_name(CODEBERTSCORE_LANG or "unset_codebertscore_lang")
    return CACHE_DIR / f"codebertscore_cache_{lang_name}.pkl"


def safe_delete_tree(path: Path, retries: int = 5, wait: float = 0.8):
    if not path.exists():
        return []
    for _ in range(retries):
        failed = []
        for item in sorted(path.rglob("*"), key=path_depth, reverse=True):
            try:
                if item.is_file() or item.is_symlink():
                    try:
                        os.chmod(item, stat.S_IWRITE)
                    except Exception:
                        pass
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            except Exception:
                failed.append(str(item))
        try:
            path.rmdir()
        except Exception:
            pass
        if not path.exists():
            return []
        time.sleep(wait)
    return [str(item) for item in sorted(path.rglob("*"), key=path_depth, reverse=True) if item.exists()]


def prepare_results():
    if RESULT_PATH.exists():
        while True:
            resp = input("Results folder exists. Delete and recreate? (y/n): ").strip().lower()
            if resp == "y":
                failed = safe_delete_tree(RESULT_PATH)
                if failed:
                    print("\nCould not delete some files because they are open or locked.")
                    print("Close Excel / editor / preview using these files and run again.")
                    print("\nLocked files:")
                    for f in failed[:20]:
                        print(" -", f)
                    if len(failed) > 20:
                        print(f" - ... and {len(failed) - 20} more")
                    return False
                break
            if resp == "n":
                print("Returning to start...")
                return "restart"
            print("Please answer with y or n.")
    for p in [INDIVIDUAL_METRICS_PATH, INDIVIDUAL_MATRICES_PATH, PLOTS_PATH, DASHBOARD_PATH, CACHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    return True


def choose_option(prompt_text: str, options: Dict[str, str]) -> str:
    print(f"\n{prompt_text}")
    for key, value in options.items():
        print(f" {key} - {value}")

    while True:
        choice = input("Choose option: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid option. Please choose one of:", ", ".join(options.keys()))



def metric_label(metric_name: str) -> str:
    return DISPLAY_METRICS.get(metric_name, metric_name)


def selected_metric_names() -> str:
    if not ACTIVE_SCORE_METRICS:
        return "none"
    return ", ".join(metric_label(metric) for metric in ACTIVE_SCORE_METRICS)


def configure_metric_selection():
    global ACTIVE_SCORE_METRICS, COMPOSITE_METRICS, OPTIONAL_PAIR_METRICS

    print("\nSelect semantic score metrics:")
    print(" 1 - BERTScore only")
    print(" 2 - Embedding cosine only")
    print(" 3 - CodeBERTScore only")
    print(" 4 - AST similarity only")
    print(" 5 - BERTScore + Embedding cosine")
    print(" 6 - BERTScore + CodeBERTScore")
    print(" 7 - Embedding cosine + CodeBERTScore")
    print(" 8 - BERTScore + Embedding cosine + CodeBERTScore")
    print(" 9 - all metrics: BERTScore + Embedding cosine + CodeBERTScore + AST similarity")
    print(" custom - type metric ids separated by comma, for example: 1,3,4")

    metric_id_map = {
        "1": "bertscore_f1",
        "2": "embedding_cosine_similarity",
        "3": "codebertscore_f1",
        "4": "ast_similarity_score",
    }

    while True:
        choice = input("Choose metric option: ").strip().lower()

        if choice in METRIC_SELECTION_OPTIONS:
            ACTIVE_SCORE_METRICS = list(METRIC_SELECTION_OPTIONS[choice])
        elif choice == "custom":
            raw = input("Type metric ids separated by comma: ").strip()
            selected = []
            invalid = []
            for part in raw.split(","):
                key = part.strip()
                metric = metric_id_map.get(key)
                if metric is None:
                    invalid.append(key)
                elif metric not in selected:
                    selected.append(metric)
            if invalid:
                print("Invalid metric ids:", ", ".join(invalid))
                continue
            if not selected:
                print("Select at least one metric.")
                continue
            ACTIVE_SCORE_METRICS = selected
        else:
            print("Invalid option. Please choose 1-9, all, or custom.")
            continue

        COMPOSITE_METRICS = [
            metric for metric in ACTIVE_SCORE_METRICS
            if metric in ["bertscore_f1", "embedding_cosine_similarity", "codebertscore_f1"]
        ]
        OPTIONAL_PAIR_METRICS = [
            metric for metric in ACTIVE_SCORE_METRICS
            if metric in ["ast_similarity_score"]
        ]

        print(f" - Selected semantic metrics: {selected_metric_names()}")
        return


def configure_models():
    global BERTSCORE_MODEL_TYPE, EMBEDDING_MODEL_NAME, CODEBERTSCORE_LANG

    if "bertscore_f1" in ACTIVE_SCORE_METRICS:
        BERTSCORE_MODEL_TYPE = choose_option(
            "Select BERTScore model:",
            BERTSCORE_MODEL_OPTIONS
        )
    else:
        BERTSCORE_MODEL_TYPE = None

    if "embedding_cosine_similarity" in ACTIVE_SCORE_METRICS:
        EMBEDDING_MODEL_NAME = choose_option(
            "Select embedding model:",
            EMBEDDING_MODEL_OPTIONS
        )
    else:
        EMBEDDING_MODEL_NAME = None

    if "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        CODEBERTSCORE_LANG = choose_option(
            "Select CodeBERTScore language:",
            CODEBERTSCORE_LANG_OPTIONS
        )
    else:
        CODEBERTSCORE_LANG = None

    print("\nSelected semantic metrics:", selected_metric_names())
    print(f" - BERTScore model: {BERTSCORE_MODEL_TYPE or 'disabled'}")
    print(f" - Embedding model: {EMBEDDING_MODEL_NAME or 'disabled'}")
    print(f" - CodeBERTScore language: {CODEBERTSCORE_LANG or 'disabled'}")


def configure_structural_mode():
    global STRUCTURAL_EVALUATION_MODE
    print("\nSelect structural robustness mode:")
    print(" 1 - syntax_schema  (Total, Syntax OK, Schema OK, Different)")
    print(" 2 - ast            (Total, Structural Exact, Different, AST Similarity)")
    print(" 3 - both           (syntax/schema + AST counts and AST similarity)")
    while True:
        choice = input("Choose option: ").strip()
        if choice in STRUCTURAL_MODE_OPTIONS:
            STRUCTURAL_EVALUATION_MODE = STRUCTURAL_MODE_OPTIONS[choice]
            print(f" - Structural mode: {STRUCTURAL_EVALUATION_MODE}")
            return
        print("Invalid option. Please choose one of: 1, 2, 3")


def norm_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    return re.sub(r"\s+", " ", s)


def compact_logic(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([\(\)\[\]\{\},;:\->\+\*\?\|&=!<>])\s*", r"\1", s)
    return s.replace("\n", "").replace("\t", "").strip().lower()


def normalize_json(obj):
    if isinstance(obj, dict):
        return {k: normalize_json(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [normalize_json(x) for x in obj]
    return obj


def canonical_json_string(obj):
    return json.dumps(normalize_json(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sort_canonical_list(items):
    return sorted(items, key=canonical_sort_key)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def safe_pct(x: Optional[float]) -> float:
    if x is None or pd.isna(x):
        return np.nan
    return round(float(x) * 100, 2)


def semantic_label(score) -> str:
    if score is None or pd.isna(score):
        return "sem_score"
    score = float(score)
    if score < 70:
        return "errado"
    if score < 85:
        return "parcialmente_semelhante"
    if score < 93:
        return "parecido"
    if score < 97:
        return "muito_parecido"
    if score < 100:
        return "quase_igual"
    return "identico"


def classify_api(name):
    name = name.lower()
    if any(x in name for x in ["socket", "net", "http", "url", "datagram", "inet"]):
        return "NET"
    if any(x in name for x in ["file", "stream", "reader", "writer", "console"]):
        return "IO"
    if any(x in name for x in ["collection", "collections", "list", "map", "set", "queue", "deque", "iterator", "vector", "arraydeque", "treemap", "treeset"]):
        return "UTIL"
    return "LANG"


def compute_coverage(baseline_files, processed_count):
    total = len(baseline_files)
    coverage = (processed_count / total) * 100 if total > 0 else 0
    return total, processed_count, round(coverage, 2)


@dataclass
class SemanticResources:
    embedding_model: Optional[object] = None
    embedding_cache: Optional[Dict[str, np.ndarray]] = None
    bertscore_cache: Optional[Dict[Tuple[str, str], Optional[float]]] = None
    codebertscore_cache: Optional[Dict[Tuple[str, str, str], Optional[float]]] = None


def load_pickle_cache(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default


def save_pickle_cache(path: Path, obj):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass


SEMANTIC_RESOURCES = SemanticResources(
    embedding_model=None,
    embedding_cache={},
    bertscore_cache={},
    codebertscore_cache={},
)


def reset_semantic_resources_for_selected_models():
    SEMANTIC_RESOURCES.embedding_model = None
    SEMANTIC_RESOURCES.embedding_cache = load_pickle_cache(get_embedding_cache_file(), {})
    SEMANTIC_RESOURCES.bertscore_cache = load_pickle_cache(get_bertscore_cache_file(), {})
    SEMANTIC_RESOURCES.codebertscore_cache = load_pickle_cache(get_codebertscore_cache_file(), {})


def flush_semantic_caches():
    save_pickle_cache(get_embedding_cache_file(), SEMANTIC_RESOURCES.embedding_cache)
    save_pickle_cache(get_bertscore_cache_file(), SEMANTIC_RESOURCES.bertscore_cache)
    save_pickle_cache(get_codebertscore_cache_file(), SEMANTIC_RESOURCES.codebertscore_cache)


def get_embedding_model():
    if SentenceTransformer is None:
        return None
    if SEMANTIC_RESOURCES.embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        SEMANTIC_RESOURCES.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return SEMANTIC_RESOURCES.embedding_model


def get_text_embedding(text: str) -> Optional[np.ndarray]:
    if text in SEMANTIC_RESOURCES.embedding_cache:
        return SEMANTIC_RESOURCES.embedding_cache[text]
    model = get_embedding_model()
    if model is None:
        return None
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0]
    SEMANTIC_RESOURCES.embedding_cache[text] = vec
    return vec


def bertscore_f1_single(text_a: str, text_b: str) -> Optional[float]:
    if bert_score_fn is None:
        return None
    key = (text_a, text_b)
    if key in SEMANTIC_RESOURCES.bertscore_cache:
        return SEMANTIC_RESOURCES.bertscore_cache[key]
    try:
        _, _, f1 = bert_score_fn(
            [text_a],
            [text_b],
            lang="en",
            model_type=BERTSCORE_MODEL_TYPE,
            verbose=False,
            rescale_with_baseline=True,
        )
        value = float(f1[0].item())
    except Exception:
        value = None
    SEMANTIC_RESOURCES.bertscore_cache[key] = value
    return value


def embedding_cosine_similarity(text_a: str, text_b: str) -> Optional[float]:
    vec_a = get_text_embedding(text_a)
    vec_b = get_text_embedding(text_b)
    if vec_a is None or vec_b is None:
        return None
    return cosine_similarity(vec_a, vec_b)


def codebertscore_f1_single(text_a: str, text_b: str) -> Optional[float]:
    if code_bert_score_fn is None:
        return None
    if CODEBERTSCORE_LANG is None:
        return None

    key = (text_a, text_b, CODEBERTSCORE_LANG)
    if key in SEMANTIC_RESOURCES.codebertscore_cache:
        return SEMANTIC_RESOURCES.codebertscore_cache[key]

    try:
        # code_bert_score.score returns P, R, F1, F3 tensors.
        _, _, f1, _ = code_bert_score_fn(
            [text_a],
            [text_b],
            lang=CODEBERTSCORE_LANG,
            verbose=False,
        )
        value = float(f1[0].item())
    except Exception:
        value = None

    SEMANTIC_RESOURCES.codebertscore_cache[key] = value
    return value


def selected_semantic_score_pct(metric_values: Dict[str, Optional[float]]) -> float:
    selected_values = []
    for metric in ACTIVE_SCORE_METRICS:
        value = metric_values.get(metric)
        if value is not None and not pd.isna(value):
            selected_values.append(float(value))
    if not selected_values:
        return np.nan
    return round(float(np.mean(selected_values)), 2)


def normalize_parameter(param):
    if not isinstance(param, dict):
        return {"raw": norm_text(param)}
    return {"type": norm_text(param.get("type")), "name": norm_text(param.get("name"))}


def normalize_argument(arg):
    if not isinstance(arg, dict):
        return {"raw": norm_text(arg)}
    return {"value": norm_text(arg.get("value"))}


def normalize_function_item(item):
    if not isinstance(item, dict):
        return {"raw": compact_logic(canonical_json_string(item))}
    return {
        "name": norm_text(item.get("name")),
        "arguments": sort_canonical_list([normalize_argument(a) for a in item.get("arguments", [])]),
    }


def normalize_operation_list(ops):
    if not isinstance(ops, list):
        return [norm_text(ops)] if ops is not None else []
    return [norm_text(x) for x in ops]


def normalize_method(method):
    if not isinstance(method, dict):
        return {"raw": compact_logic(canonical_json_string(method))}
    return {
        "action": norm_text(method.get("action")),
        "name": norm_text(method.get("name")),
        "timing": norm_text(method.get("timing")),
        "parameters": sort_canonical_list([normalize_parameter(p) for p in method.get("parameters", [])]),
        "procediments": norm_text(method.get("procediments")),
        "function": sort_canonical_list([normalize_function_item(f) for f in method.get("function", [])]),
        "operation": normalize_operation_list(method.get("operation", [])),
    }


def normalize_signature(signature):
    if not isinstance(signature, dict):
        return {"raw": compact_logic(canonical_json_string(signature))}
    return {
        "name": norm_text(signature.get("name")),
        "parameters": sort_canonical_list([normalize_parameter(p) for p in signature.get("parameters", [])]),
    }


def normalize_violation(violation):
    if not isinstance(violation, dict):
        return {"raw": compact_logic(canonical_json_string(violation))}
    body = violation.get("body", {})
    statements = body.get("statements", [])
    return {
        "tag": norm_text(violation.get("tag")),
        "body": {
            "statements": normalize_json(statements),
            "has_reset": bool(body.get("has_reset", False)),
        },
    }


def detect_formalism(ir_root):
    formalism = norm_text(ir_root.get("formalism"))
    if formalism:
        return formalism
    ir_type = norm_text(ir_root.get("ir", {}).get("type"))
    if ir_type:
        return ir_type
    return "unknown"


def extract_event_signature(ev):
    if not isinstance(ev, dict):
        return [{"raw_event": compact_logic(canonical_json_string(ev))}]
    methods = ev.get("body", {}).get("methods", [])
    if methods:
        return [normalize_method(m) for m in methods]
    return [{"raw_event": compact_logic(canonical_json_string(ev))}]


def extract_events(ir_root):
    ir = ir_root.get("ir", {})
    events = ir.get("events", [])
    extracted = []
    for ev in events:
        extracted.extend(extract_event_signature(ev))
    return sort_canonical_list(extracted)


def extract_logic_block(ir, keys):
    for key in keys:
        block = ir.get(key)
        if block is None:
            continue
        if isinstance(block, str):
            return compact_logic(block)
        if isinstance(block, dict):
            if "expression" in block:
                return compact_logic(block["expression"])
            return compact_logic(canonical_json_string(block))
    return ""


def extract_ere_semantics(ir_root):
    ir = ir_root.get("ir", {})
    return {
        "formalism": "ere",
        "domain": norm_text(ir_root.get("domain")),
        "signature": normalize_signature(ir_root.get("signature", {})),
        "events": extract_events(ir_root),
        "logic": extract_logic_block(ir, ["ere"]),
        "violation": normalize_violation(ir.get("violation", {})),
    }


def extract_semantic_components(ir_root):
    formalism = detect_formalism(ir_root)
    if formalism == "ere":
        return extract_ere_semantics(ir_root)
    return {
        "formalism": formalism,
        "domain": norm_text(ir_root.get("domain")),
        "signature": normalize_signature(ir_root.get("signature", {})),
        "events": extract_events(ir_root),
        "violation": normalize_violation(ir_root.get("ir", {}).get("violation", {})),
        "raw_ir": normalize_json(ir_root.get("ir", {})),
    }


def semantic_component_string(ir_root):
    return canonical_json_string(extract_semantic_components(ir_root))


# -----------------------------
# Code-oriented representation for CodeBERTScore
# -----------------------------
def code_fragment_has_signal(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    code_markers = [
        "(", ")", "..", "*", "+", ".", "&&", "||", "!", "==", "!=", "<", ">",
        "instanceof", "call", "args", "target", "returning", "new", "before", "after", "|",
    ]
    if any(marker in text for marker in code_markers):
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return True
    return False


def collect_code_fragments(obj: Any, parent_key: str = "", fragments: Optional[List[str]] = None) -> List[str]:
    if fragments is None:
        fragments = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            collect_code_fragments(value, str(key), fragments)
        return fragments

    if isinstance(obj, list):
        for item in obj:
            collect_code_fragments(item, parent_key, fragments)
        return fragments

    if isinstance(obj, str):
        raw = obj.strip()
        if not raw:
            return fragments

        key = parent_key.lower()
        relevant_keys = {
            "name", "type", "value", "expression", "procediments", "pointcut_raw",
            "operation", "returning", "arguments", "parameters", "statements",
        }
        ignored_keys = {"id", "domain", "formalism", "tag"}

        if key in ignored_keys:
            return fragments

        if key in relevant_keys or code_fragment_has_signal(raw):
            normalized = compact_logic(raw)
            if normalized and normalized not in fragments:
                fragments.append(normalized)

    return fragments


def code_component_string(ir_root: Dict[str, Any]) -> str:
    """
    Extracts only code-like/specification fragments for CodeBERTScore.
    This avoids feeding the full canonical JSON to a code-oriented metric.
    Examples of retained fragments include Java pointcuts, expressions,
    operations, argument values, method names, type names, and ERE expressions.
    """
    fragments = collect_code_fragments(ir_root)
    if not fragments:
        return ""
    return "\n".join(fragments)


def find_file(base, filename):
    for f in base.rglob(filename):
        return f
    return None


def load_json_quick(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def filter_ere_baselines(files: List[Path]) -> Tuple[List[Path], pd.DataFrame]:
    eligible = []
    ignored_rows = []
    for file in files:
        data = load_json_quick(file)
        detected = detect_formalism(data or {}) if data else "unknown"
        if detected == "ere":
            eligible.append(file)
        else:
            ignored_rows.append({
                "spec": file.name,
                "baseline_formalism_detected": detected,
                "reason": "Ignored_Non_ERE_Baseline",
            })
    ignored_df = pd.DataFrame(ignored_rows, columns=["spec", "baseline_formalism_detected", "reason"])
    return eligible, ignored_df


def validate_json_file(path: Path, expected_formalism: str = "ere") -> Dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path is not None and path.exists(),
        "syntax_success": 0,
        "schema_success": 0,
        "processable": 0,
        "error_type": "",
        "error_detail": "",
        "loaded_json": None,
        "formalism_detected": "unknown",
    }

    if path is None or not path.exists():
        result["error_type"] = "Missing_File"
        result["error_detail"] = "Output file not found."
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result["syntax_success"] = 1
        result["loaded_json"] = data
    except Exception as exc:
        result["error_type"] = "Syntax_Error"
        result["error_detail"] = str(exc)
        return result

    if not isinstance(data, dict):
        result["error_type"] = "Schema_Error"
        result["error_detail"] = "Top-level JSON must be an object."
        return result

    formalism = detect_formalism(data)
    result["formalism_detected"] = formalism
    if formalism != expected_formalism:
        result["error_type"] = "Wrong_Formalism"
        result["error_detail"] = f"Expected {expected_formalism}, found {formalism}."
        return result

    missing = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            missing.append(key)

    signature = data.get("signature")
    if not isinstance(signature, dict):
        missing.append("signature")
    else:
        if not non_empty_str(signature.get("name")):
            missing.append("signature.name")
        if not isinstance(signature.get("parameters", []), list):
            missing.append("signature.parameters")

    ir = data.get("ir")
    if not isinstance(ir, dict):
        missing.append("ir")
    else:
        for key in REQUIRED_IR_KEYS_ERE:
            if key not in ir:
                missing.append(f"ir.{key}")
        if not isinstance(ir.get("events", []), list):
            missing.append("ir.events")
        ere_block = ir.get("ere")
        if not isinstance(ere_block, dict) or not non_empty_str(ere_block.get("expression")):
            missing.append("ir.ere.expression")
        violation = ir.get("violation")
        if not isinstance(violation, dict):
            missing.append("ir.violation")

    if missing:
        result["error_type"] = "Schema_Error"
        result["error_detail"] = "Missing or invalid required fields: " + ", ".join(sorted(set(missing)))
        return result

    result["schema_success"] = 1
    result["processable"] = 1
    return result


def structural_exact_match(a: Any, b: Any, path: str = "root") -> Tuple[bool, str]:
    if type(a) != type(b):
        return False, f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}"

    if isinstance(a, dict):
        keys_a = list(a.keys())
        keys_b = list(b.keys())
        if keys_a != keys_b:
            return False, f"{path}: key order mismatch {keys_a} vs {keys_b}"
        for key in keys_a:
            ok, msg = structural_exact_match(a[key], b[key], f"{path}.{key}")
            if not ok:
                return False, msg
        return True, "ok"

    if isinstance(a, list):
        if len(a) != len(b):
            return False, f"{path}: list size mismatch {len(a)} vs {len(b)}"
        for index, pair in enumerate(zip(a, b)):
            left, right = pair
            ok, msg = structural_exact_match(left, right, f"{path}[{index}]")
            if not ok:
                return False, msg
        return True, "ok"

    if a != b:
        return False, f"{path}: value mismatch {repr(a)} vs {repr(b)}"
    return True, "ok"


def ast_flatten_nodes(obj: Any, path: str = "root") -> Set[str]:
    nodes = set()
    if isinstance(obj, dict):
        nodes.add(f"{path}::<dict>")
        for key in obj.keys():
            child_path = f"{path}.{key}"
            nodes.add(f"{child_path}::<key>")
            nodes.update(ast_flatten_nodes(obj[key], child_path))
        return nodes
    if isinstance(obj, list):
        nodes.add(f"{path}::<list>")
        for index, value in enumerate(obj):
            child_path = f"{path}[{index}]"
            nodes.add(f"{child_path}::<index>")
            nodes.update(ast_flatten_nodes(value, child_path))
        return nodes
    nodes.add(f"{path}::<value>={repr(obj)}")
    return nodes


def ast_similarity_score(baseline_json: Any, predicted_json: Any) -> float:
    baseline_nodes = ast_flatten_nodes(baseline_json)
    predicted_nodes = ast_flatten_nodes(predicted_json)
    if not baseline_nodes and not predicted_nodes:
        return 100.0
    union = baseline_nodes.union(predicted_nodes)
    if not union:
        return 100.0
    intersection = baseline_nodes.intersection(predicted_nodes)
    return round((len(intersection) / len(union)) * 100.0, 2)


def list_ere_baseline_names(baseline_dir: Path) -> List[str]:
    names = []
    for path in sorted(baseline_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("formalism", "")).strip().lower() == "ere":
            names.append(path.name)
    return names


def list_strategy_existing_names(strategy_dir: Path, allowed_names: Set[str]) -> Set[str]:
    existing = set()
    for path in strategy_dir.rglob("*.json"):
        if path.name in allowed_names:
            existing.add(path.name)
    return existing


def count_total_specs_for_strategy(strategy_dir: Path, ere_baseline_names: List[str]) -> int:
    allowed = set(ere_baseline_names)
    existing = list_strategy_existing_names(strategy_dir, allowed)
    return len(existing)


def count_syntax_ok_for_strategy(strategy_dir: Path, ere_baseline_names: List[str]) -> int:
    allowed = set(ere_baseline_names)
    count = 0
    for path in strategy_dir.rglob("*.json"):
        if path.name not in allowed:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
            count += 1
        except Exception:
            pass
    return count


def count_schema_ok_for_strategy(strategy_dir: Path, ere_baseline_names: List[str]) -> int:
    allowed = set(ere_baseline_names)
    count = 0
    for path in strategy_dir.rglob("*.json"):
        if path.name not in allowed:
            continue
        result = validate_json_file(path, expected_formalism="ere")
        if int(result.get("schema_success", 0)) == 1:
            count += 1
    return count


def get_baseline_path_map(baseline_dir: Path, allowed_names: Set[str]) -> Dict[str, Path]:
    baseline_map = {}
    for path in baseline_dir.rglob("*.json"):
        if path.name in allowed_names:
            baseline_map[path.name] = path
    return baseline_map


def compute_ast_stats_for_strategy(strategy_dir: Path, baseline_dir: Path, ere_baseline_names: List[str]) -> Dict[str, Any]:
    allowed = set(ere_baseline_names)
    baseline_map = get_baseline_path_map(baseline_dir, allowed)
    structural_exact = 0
    ast_compared = 0
    similarities = []

    for path in strategy_dir.rglob("*.json"):
        if path.name not in allowed:
            continue
        baseline_path = baseline_map.get(path.name)
        if baseline_path is None:
            continue
        pred_result = validate_json_file(path, expected_formalism="ere")
        base_result = validate_json_file(baseline_path, expected_formalism="ere")
        if int(pred_result.get("processable", 0)) != 1:
            continue
        if int(base_result.get("processable", 0)) != 1:
            continue
        ast_compared += 1
        ok, _detail = structural_exact_match(base_result["loaded_json"], pred_result["loaded_json"])
        if ok:
            structural_exact += 1
        similarities.append(ast_similarity_score(base_result["loaded_json"], pred_result["loaded_json"]))

    mean_similarity = round(float(np.mean(similarities)), 2) if similarities else np.nan
    median_similarity = round(float(np.median(similarities)), 2) if similarities else np.nan
    return {
        "ast_compared": ast_compared,
        "structural_exact": structural_exact,
        "ast_similarity_mean_%": mean_similarity,
        "ast_similarity_median_%": median_similarity,
    }


def build_structural_counts_table(baseline_dir: Path, strategy_paths: Dict[str, Path], mode: str) -> pd.DataFrame:
    ere_baseline_names = list_ere_baseline_names(baseline_dir)
    rows = []
    for strategy, strategy_dir in strategy_paths.items():
        total = count_total_specs_for_strategy(strategy_dir, ere_baseline_names)
        row = {"Strategy": strategy, "Total": total}

        if mode in ["syntax_schema", "both"]:
            syntax_ok = count_syntax_ok_for_strategy(strategy_dir, ere_baseline_names)
            schema_ok = count_schema_ok_for_strategy(strategy_dir, ere_baseline_names)
            row["Syntax OK"] = syntax_ok
            row["Schema OK"] = schema_ok
            if mode == "syntax_schema":
                row["Different"] = total - schema_ok

        if mode in ["ast", "both"]:
            ast_stats = compute_ast_stats_for_strategy(strategy_dir, baseline_dir, ere_baseline_names)
            row["Structural Exact"] = ast_stats["structural_exact"]
            row["Different"] = total - ast_stats["structural_exact"]
            row["AST Similarity Mean (%)"] = ast_stats["ast_similarity_mean_%"]
            row["AST Similarity Median (%)"] = ast_stats["ast_similarity_median_%"]

        rows.append(row)
    return pd.DataFrame(rows)


def compute_dashboard_counts(
    baseline_total_all: int,
    baseline_ere_total_all: int,
    compared_scenarios: int,
    strategies_count: int,
    total_files_analyzed: int,
) -> pd.DataFrame:
    coverage = 0.0
    if baseline_ere_total_all > 0:
        coverage = compared_scenarios / baseline_ere_total_all * 100.0
    return pd.DataFrame([{
        "baseline_total_specs_all": int(baseline_total_all),
        "baseline_ere_total_specs": int(baseline_ere_total_all),
        "scenarios_compared": int(compared_scenarios),
        "comparison_strategies": int(strategies_count),
        "total_files_analyzed": int(total_files_analyzed),
        "coverage_rate_ere_%": round(float(coverage), 2),
    }])


def compute_robustness_summary(_global_df_unused: pd.DataFrame, _strategies_unused: List[str]) -> pd.DataFrame:
    df = build_structural_counts_table(ORIGINAL_PATH, STRATEGY_PATHS, STRUCTURAL_EVALUATION_MODE)
    df.to_csv(RESULT_PATH / "strategy_robustness_summary.csv", index=False)
    return df


def pair_metrics(label_a, label_b, json_a, json_b, spec_name, formalism):
    sem_text_a = semantic_component_string(json_a)
    sem_text_b = semantic_component_string(json_b)

    # CodeBERTScore is code-oriented, so it receives only code/spec fragments,
    # not the full canonical JSON representation.
    code_text_a = code_component_string(json_a)
    code_text_b = code_component_string(json_b)

    metric_values: Dict[str, Optional[float]] = {}

    if "bertscore_f1" in ACTIVE_SCORE_METRICS:
        metric_values["bertscore_f1"] = safe_pct(bertscore_f1_single(sem_text_a, sem_text_b))

    if "embedding_cosine_similarity" in ACTIVE_SCORE_METRICS:
        metric_values["embedding_cosine_similarity"] = safe_pct(embedding_cosine_similarity(sem_text_a, sem_text_b))

    if "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        if code_text_a and code_text_b:
            metric_values["codebertscore_f1"] = safe_pct(codebertscore_f1_single(code_text_a, code_text_b))
        elif not code_text_a and not code_text_b:
            # CodeBERTScore is not applicable when neither side contains code/spec fragments.
            # Keep NaN so this non-applicable metric does not affect the selected mean.
            metric_values["codebertscore_f1"] = np.nan
        else:
            # If only one side contains code/spec fragments, the generated artifact omitted
            # required code-like content or hallucinated extra code-like content. Penalize it.
            metric_values["codebertscore_f1"] = 0.0

    if "ast_similarity_score" in ACTIVE_SCORE_METRICS:
        metric_values["ast_similarity_score"] = ast_similarity_score(json_a, json_b)

    semantic_score_pct = selected_semantic_score_pct(metric_values)

    row = {
        "spec": spec_name,
        "formalism": formalism,
        "api": classify_api(spec_name),
        "file_a": label_a,
        "file_b": label_b,
        "comparison": f"{label_a}_vs_{label_b}",
        "semantic_mean_score": semantic_score_pct,
        "semantic_label": semantic_label(semantic_score_pct),
    }

    for metric in ACTIVE_SCORE_METRICS:
        row[metric] = metric_values.get(metric, np.nan)

    return row

def compare_files_multi(baseline_file, strategy_files):
    jsons = {}
    all_files = {"baseline": baseline_file, **strategy_files}
    for label, path in all_files.items():
        txt = Path(path).read_text(encoding="utf-8")
        jsons[label] = json.loads(txt)
    spec_name = Path(baseline_file).name
    formalism = detect_formalism(jsons["baseline"])
    rows = []
    names = list(jsons.keys())
    for a, b in itertools.combinations(names, 2):
        rows.append(pair_metrics(a, b, jsons[a], jsons[b], spec_name, formalism))
    df = pd.DataFrame(rows)
    matrix = build_similarity_matrix(df, names, PRIMARY_SCORE)
    return df, matrix


def build_similarity_matrix(df: pd.DataFrame, names: List[str], score_col: str) -> pd.DataFrame:
    matrix = pd.DataFrame(index=names, columns=names)
    for a in names:
        for b in names:
            if a == b:
                matrix.loc[a, b] = 100.0
            else:
                r = df[((df.file_a == a) & (df.file_b == b)) | ((df.file_a == b) & (df.file_b == a))]
                value = r[score_col].iloc[0]
                matrix.loc[a, b] = round(float(value), 2) if not pd.isna(value) else np.nan
    return matrix


def compute_strategy_scores(global_df, strategies):
    baseline_rows = global_df[(global_df["file_a"] == "baseline") | (global_df["file_b"] == "baseline")].copy()
    baseline_rows["strategy"] = np.where(baseline_rows["file_a"] == "baseline", baseline_rows["file_b"], baseline_rows["file_a"])
    baseline_rows = baseline_rows[baseline_rows["strategy"].isin(strategies)].copy()
    baseline_rows["mean_score"] = pd.to_numeric(baseline_rows[PRIMARY_SCORE], errors="coerce")
    baseline_rows["semantic_label"] = baseline_rows["mean_score"].apply(semantic_label)
    baseline_rows["original_ir"] = 100.0
    keep_cols = ["spec", "formalism", "api", "strategy", "original_ir", "mean_score", "semantic_label"] + COMPOSITE_METRICS
    for optional_metric in OPTIONAL_PAIR_METRICS:
        if optional_metric in baseline_rows.columns:
            keep_cols.append(optional_metric)
    strategy_scores = baseline_rows[keep_cols].copy().sort_values(["spec", "strategy"]).reset_index(drop=True)
    strategy_scores.to_csv(RESULT_PATH / "strategy_scores_long.csv", index=False)

    score_pivot = strategy_scores.pivot_table(
        index=["spec", "formalism", "api"],
        columns="strategy",
        values="mean_score",
        aggfunc="first"
    ).reset_index()
    score_pivot = score_pivot.rename(
        columns={c: f"{c}_mean_score" for c in score_pivot.columns if c not in ["spec", "formalism", "api"]}
    )

    wide_df = score_pivot.copy()
    for metric in COMPOSITE_METRICS + OPTIONAL_PAIR_METRICS:
        metric_pivot = strategy_scores.pivot_table(
            index=["spec", "formalism", "api"],
            columns="strategy",
            values=metric,
            aggfunc="first"
        ).reset_index()
        metric_pivot = metric_pivot.rename(
            columns={c: f"{c}_{metric}" for c in metric_pivot.columns if c not in ["spec", "formalism", "api"]}
        )
        wide_df = wide_df.merge(metric_pivot, on=["spec", "formalism", "api"], how="left")

    wide_df.insert(3, "original_ir", 100.0)
    score_pivot.to_csv(RESULT_PATH / "strategy_scores_wide.csv", index=False)
    wide_df.to_csv(RESULT_PATH / "scenario_similarity_by_strategy.csv", index=False)
    return strategy_scores, wide_df


def compute_scenario_organization(strategy_wide_df, strategies):
    rows = []
    for _, row in strategy_wide_df.iterrows():
        strategy_scores = [float(row.get(f"{strategy}_mean_score", np.nan)) for strategy in strategies]
        best_score = max(strategy_scores)
        worst_score = min(strategy_scores)
        best_score_strategies = [
            strategy
            for strategy in strategies
            if abs(float(row.get(f"{strategy}_mean_score", np.nan)) - float(best_score)) < 1e-9
        ]
        out = {
            "spec": row["spec"],
            "formalism": row["formalism"],
            "api": row["api"],
            "original_ir": 100.0,
            "best_mean_score": round(best_score, 2),
            "best_score_label": semantic_label(best_score),
            "worst_mean_score": round(worst_score, 2),
            "best_score_strategies": ", ".join(best_score_strategies),
        }
        metric_cols = [c for c in strategy_wide_df.columns if any(c.startswith(f"{s}_") for s in strategies)]
        for col in metric_cols:
            value = row.get(col, np.nan)
            out[col] = round(float(value), 2) if not pd.isna(value) and isinstance(value, (int, float, np.floating)) else value
        rows.append(out)

    scenario_df = pd.DataFrame(rows).sort_values(["best_mean_score", "spec"], ascending=[False, True]).reset_index(drop=True)
    scenario_df.to_csv(RESULT_PATH / "scenario_summary.csv", index=False)
    return scenario_df


def compute_winners(strategy_wide_df, strategies):
    winner_rows = []
    for _, row in strategy_wide_df.iterrows():
        candidates = [{"strategy": strategy, "score_raw": float(row.get(f"{strategy}_mean_score", np.nan))} for strategy in strategies]
        candidates_sorted = sorted(candidates, key=candidate_sort_key, reverse=True)
        best_score_raw = candidates_sorted[0]["score_raw"]
        best_winners = [c["strategy"] for c in candidates_sorted if abs(c["score_raw"] - best_score_raw) < 1e-9]

        if len(best_winners) > 1:
            winner_name = "tie"
            tied_strategies = ", ".join(sorted(best_winners))
            tie_type = "tie"
        else:
            winner_name = best_winners[0]
            tied_strategies = ""
            tie_type = "single"

        out = {
            "spec": row["spec"],
            "formalism": row["formalism"],
            "api": row["api"],
            "winner_by_mean_score": winner_name,
            "winner_mean_score": round(best_score_raw, 2),
            "winner_label": semantic_label(best_score_raw),
            "tie_type": tie_type,
            "tied_strategies": tied_strategies,
            "original_ir": 100.0,
        }

        for strategy in strategies:
            score_val = row.get(f"{strategy}_mean_score", np.nan)
            out[f"{strategy}_mean_score"] = score_val
            out[f"{strategy}_SemanticLabel"] = semantic_label(score_val)
            for metric in COMPOSITE_METRICS + OPTIONAL_PAIR_METRICS:
                label = DISPLAY_METRICS.get(metric, metric)
                out[f"{strategy}_{label}"] = row.get(f"{strategy}_{metric}", np.nan)

        winner_rows.append(out)

    winners_df = pd.DataFrame(winner_rows).sort_values(["winner_mean_score", "spec"], ascending=[False, True])
    winners_df.to_csv(RESULT_PATH / "winner_per_spec.csv", index=False)
    return winners_df


def compute_overall_scoreboard(winners_df, strategy_scores, strategies):
    winner_counts = (
        winners_df["winner_by_mean_score"].value_counts(dropna=False).rename_axis("strategy").reset_index(name="wins_by_mean_score")
        if not winners_df.empty
        else pd.DataFrame(columns=["strategy", "wins_by_mean_score"])
    )

    overall_rows = []
    for strategy in strategies:
        strategy_df = strategy_scores[strategy_scores["strategy"] == strategy].copy()
        mean_scores = pd.to_numeric(strategy_df["mean_score"], errors="coerce")
        wins_row = winner_counts[winner_counts["strategy"] == strategy]
        wins = int(wins_row["wins_by_mean_score"].iloc[0]) if not wins_row.empty else 0

        row = {
            "strategy": strategy,
            "comparisons_total": int(strategy_df.shape[0]),
            "wins_by_mean_score": wins,
            "avg_score_total_%": round(float(mean_scores.mean()), 2),
            "median_score_total_%": round(float(mean_scores.median()), 2),
            "std_score_total_%": round(float(mean_scores.std(ddof=0)), 2),
            "dominant_label": strategy_df["semantic_label"].mode().iloc[0] if not strategy_df.empty else "sem_score",
        }

        for metric in COMPOSITE_METRICS + OPTIONAL_PAIR_METRICS:
            if metric in strategy_df.columns:
                vals_total = pd.to_numeric(strategy_df[metric], errors="coerce")
                row[f"avg_{metric}_total_%"] = round(float(vals_total.mean()), 2)

        overall_rows.append(row)

    scoreboard = pd.DataFrame(overall_rows).sort_values(
        by=["avg_score_total_%", "wins_by_mean_score", "strategy"],
        ascending=[False, False, True],
    )
    scoreboard.to_csv(RESULT_PATH / "overall_scoreboard.csv", index=False)
    return scoreboard


def compute_pairwise_advantage(strategy_wide_df, strategies, result_root: Path):
    rows = []
    for s1, s2 in itertools.combinations(strategies, 2):
        col1 = f"{s1}_mean_score"
        col2 = f"{s2}_mean_score"
        sub = strategy_wide_df[[col1, col2]].dropna()
        n = len(sub)
        if n == 0:
            continue
        diff = sub[col1] - sub[col2]
        rows.append({
            "strategy_a": s1,
            "strategy_b": s2,
            "a_wins": int((diff > 0).sum()),
            "b_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "n_scenarios": n,
        })
    df = pd.DataFrame(rows)
    df.to_csv(result_root / "pairwise_semantic_advantage.csv", index=False)
    return df


def build_metric_tables(strategy_wide_df, strategies):
    scenario_rows = []
    for _, row in strategy_wide_df.iterrows():
        out = {"spec": row["spec"], "formalism": row["formalism"], "api": row["api"], "original_ir": 100.0}
        for strategy in strategies:
            score_val = row.get(f"{strategy}_mean_score", np.nan)
            out[f"{strategy}_ScoreTotal"] = score_val
            out[f"{strategy}_SemanticLabel"] = semantic_label(score_val)
            for metric in COMPOSITE_METRICS + OPTIONAL_PAIR_METRICS:
                metric_label = DISPLAY_METRICS.get(metric, metric)
                out[f"{strategy}_{metric_label}"] = row.get(f"{strategy}_{metric}", np.nan)
        scenario_rows.append(out)

    metrics_by_scenario = pd.DataFrame(scenario_rows)

    strategy_rows = []
    for strategy in strategies:
        total_vals = pd.to_numeric(strategy_wide_df.get(f"{strategy}_mean_score"), errors="coerce")
        row = {
            "strategy": strategy,
            "ScoreTotal": round(float(total_vals.mean()), 2),
            "DominantSemanticLabel": total_vals.apply(semantic_label).mode().iloc[0],
        }
        for metric in COMPOSITE_METRICS + OPTIONAL_PAIR_METRICS:
            metric_label = DISPLAY_METRICS.get(metric, metric)
            vals_total = pd.to_numeric(strategy_wide_df.get(f"{strategy}_{metric}"), errors="coerce")
            row[metric_label] = round(float(vals_total.mean()), 2)
        strategy_rows.append(row)

    metrics_by_strategy = pd.DataFrame(strategy_rows)
    metrics_by_scenario.to_csv(RESULT_PATH / "metric_means_by_scenario.csv", index=False)
    metrics_by_strategy.to_csv(RESULT_PATH / "metric_means_by_strategy.csv", index=False)
    return metrics_by_scenario, metrics_by_strategy


def _color_boxplot(bp, labels):
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(STRATEGY_COLORS.get(label, "#7FB3D5"))
        patch.set_edgecolor("#3A3A3A")
        patch.set_alpha(0.95)

    for median in bp["medians"]:
        median.set_color("#3A3A3A")
        median.set_linewidth(1.5)

    for whisker in bp["whiskers"]:
        whisker.set_color("#555555")
    for cap in bp["caps"]:
        cap.set_color("#555555")


def statistical_analysis_multi(strategy_wide_df, strategies):
    available = [f"{s}_mean_score" for s in strategies if f"{s}_mean_score" in strategy_wide_df.columns]
    rows = []

    for s1, s2 in itertools.combinations(available, 2):
        sub = strategy_wide_df[[s1, s2]].dropna()
        n = len(sub)
        if n < MIN_SCENARIOS_FOR_STATS:
            rows.append({
                "strategy_a": s1.replace("_mean_score", ""),
                "strategy_b": s2.replace("_mean_score", ""),
                "n_scenarios": n,
                "test_used": "insufficient_data",
                "statistic": np.nan,
                "p_value": np.nan,
                "shapiro_stat": np.nan,
                "shapiro_p": np.nan,
                "warning": f"Need at least {MIN_SCENARIOS_FOR_STATS} scenarios",
            })
            continue

        diff = sub[s1].values - sub[s2].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shapiro_stat, shapiro_p = shapiro(diff)

        if shapiro_p > 0.05:
            test = "paired_t_test"
            stat, p = ttest_rel(sub[s1].values, sub[s2].values)
        else:
            test = "wilcoxon"
            stat, p = wilcoxon(sub[s1].values, sub[s2].values)

        rows.append({
            "strategy_a": s1.replace("_mean_score", ""),
            "strategy_b": s2.replace("_mean_score", ""),
            "n_scenarios": n,
            "test_used": test,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "shapiro_stat": round(float(shapiro_stat), 6),
            "shapiro_p": round(float(shapiro_p), 6),
            "warning": "",
        })

    pd.DataFrame(rows).to_csv(RESULT_PATH / "statistical_tests_pairwise.csv", index=False)

    if not available:
        return

    labels = [s.replace("_mean_score", "") for s in available]
    data = [strategy_wide_df[s].dropna().values for s in available]

    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    _color_boxplot(bp, labels)
    plt.ylabel("Mean Semantic Score (%)")
    plt.title("Semantic Score Distribution per Strategy")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "boxplot_semantic_score_per_strategy.png", bbox_inches="tight")
    plt.close()


def plot_to_base64(path: Path):
    if path is None or (not path.exists()):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def style_table(df, max_rows=None):
    if df is None or df.empty:
        return "<p>No data available.</p>"
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df.to_html(index=False, border=0)


def make_score_distribution_boxplot(strategy_scores, strategies):
    if strategy_scores.empty:
        return None

    data = []
    labels = []
    for strategy in strategies:
        vals = pd.to_numeric(
            strategy_scores.loc[strategy_scores["strategy"] == strategy, "mean_score"],
            errors="coerce"
        )
        data.append(vals.values)
        labels.append(strategy)

    plt.figure(figsize=(8, 4.5))
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    _color_boxplot(bp, labels)
    plt.ylabel("Mean Semantic Score (%)")
    plt.title("Semantic Score Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_PATH / "semantic_score_distribution_boxplot.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def make_pairwise_advantage_plot(pairwise_df, title, filename):
    if pairwise_df.empty:
        return None

    labels = [f"{a} vs {b}" for a, b in zip(pairwise_df["strategy_a"], pairwise_df["strategy_b"])]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, pairwise_df["a_wins"], width=width, label="a_wins", color=PAIRWISE_COLORS["a_wins"])
    plt.bar(x, pairwise_df["b_wins"], width=width, label="b_wins", color=PAIRWISE_COLORS["b_wins"])
    plt.bar(x + width, pairwise_df["ties"], width=width, label="ties", color=PAIRWISE_COLORS["ties"])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def make_label_distribution_plot(strategy_wide_df, strategies):
    rows = []

    for strategy in strategies:
        labels = strategy_wide_df[f"{strategy}_mean_score"].apply(semantic_label)
        counts = labels.value_counts()
        for label in LABELS_ORDER:
            rows.append({
                "strategy": strategy,
                "label": label,
                "count": int(counts.get(label, 0))
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    pivot = (
        df.pivot(index="strategy", columns="label", values="count")
        .fillna(0)[LABELS_ORDER]
    )

    plt.figure(figsize=(11, 5))
    left = np.zeros(len(pivot.index))

    for label in LABELS_ORDER:
        values = pivot[label].values
        plt.barh(
            pivot.index,
            values,
            left=left,
            label=label.replace("_", " ").title(),
            color=LABEL_COLORS[label]
        )
        left += values

    plt.xlabel("Quantidade")
    plt.ylabel("Estratégia")
    plt.title("Distribuição Semântica por Estratégia")
    plt.legend(title="Rótulo", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out = PLOTS_PATH / "semantic_label_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def make_success_rate_plot(robustness_df: pd.DataFrame):
    if robustness_df is None or robustness_df.empty:
        return None
    labels = robustness_df["Strategy"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 4.5))
    if "Structural Exact" in robustness_df.columns:
        exact = pd.to_numeric(robustness_df["Structural Exact"], errors="coerce").fillna(0).values
        different = pd.to_numeric(robustness_df["Different"], errors="coerce").fillna(0).values
        plt.bar(x - width / 2, exact, width=width, label="Structural Exact", color="#0B3C6F")
        plt.bar(x + width / 2, different, width=width, label="Different", color="#BDBDBD")
        plt.title("AST Structural Counts by Strategy")
    else:
        schema_ok = pd.to_numeric(robustness_df["Schema OK"], errors="coerce").fillna(0).values
        different = pd.to_numeric(robustness_df["Different"], errors="coerce").fillna(0).values
        plt.bar(x - width / 2, schema_ok, width=width, label="Schema OK", color="#0B3C6F")
        plt.bar(x + width / 2, different, width=width, label="Different", color="#BDBDBD")
        plt.title("Syntax/Schema Counts by Strategy")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = PLOTS_PATH / "structural_counts_by_strategy.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def export_excel_book(coverage_df, robustness_df, scoreboard, scenario_df, winners_df, pairwise_df, metrics_by_scenario, metrics_by_strategy, missing_df):
    out = RESULT_PATH / "semantic_dashboard_tables.xlsx"
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(out, engine=engine) as writer:
        coverage_df.to_excel(writer, sheet_name="00_coverage", index=False)
        robustness_df.to_excel(writer, sheet_name="01_structural_counts", index=False)
        scoreboard.to_excel(writer, sheet_name="02_strategy_summary", index=False)
        scenario_df.to_excel(writer, sheet_name="03_scenario_summary", index=False)
        winners_df.to_excel(writer, sheet_name="04_winner_by_mean_score", index=False)
        pairwise_df.to_excel(writer, sheet_name="05_pairwise_semantic", index=False)
        metrics_by_scenario.to_excel(writer, sheet_name="06_metrics_by_scenario", index=False)
        metrics_by_strategy.to_excel(writer, sheet_name="07_metrics_by_strategy", index=False)
        if missing_df is not None and not missing_df.empty:
            missing_df.to_excel(writer, sheet_name="08_missing_scenarios", index=False)
    return out


def generate_dashboard(
    coverage_df,
    robustness_df,
    scoreboard,
    winners_df,
    pairwise_df,
    strategy_scores,
    metrics_by_scenario,
    metrics_by_strategy,
    strategy_wide_df,
    strategies,
    excel_book
):
    score_boxplot = make_score_distribution_boxplot(strategy_scores, strategies)
    pairwise_plot = make_pairwise_advantage_plot(
        pairwise_df,
        "Pairwise Semantic Advantage",
        "pairwise_semantic_advantage.png"
    )
    label_plot = make_label_distribution_plot(strategy_wide_df, strategies)
    structural_plot = make_success_rate_plot(robustness_df)

    baseline_total_all = int(coverage_df["baseline_total_specs_all"].iloc[0]) if not coverage_df.empty else 0
    baseline_ere_total = int(coverage_df["baseline_ere_total_specs"].iloc[0]) if not coverage_df.empty else 0
    compared = int(coverage_df["scenarios_compared"].iloc[0]) if not coverage_df.empty else 0
    strategies_count = int(coverage_df["comparison_strategies"].iloc[0]) if not coverage_df.empty else 0
    total_files_analyzed = int(coverage_df["total_files_analyzed"].iloc[0]) if not coverage_df.empty else 0
    coverage = float(coverage_df["coverage_rate_ere_%"].iloc[0]) if not coverage_df.empty else 0.0

    warning_html = ""
    if coverage < 100:
        warning_html = """
        <div class="warning">⚠️ This analysis is based on partial coverage of the ERE baseline scenarios.
        Results reflect the subset where all strategies produced outputs for semantic comparison.</div>
        """

    label_note = SEMANTIC_LABEL_FOOTNOTE

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Semantic Similarity Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-bottom: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(6, minmax(170px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; background: #fafafa; }}
.metric {{ font-size: 28px; font-weight: bold; }}
.small {{ color: #666; font-size: 13px; }}
.section {{ margin-top: 28px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
.plot-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fff; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #f1f1f1; }}
.note {{ color: #444; font-size: 14px; margin-top: 8px; }}
.footnote {{ color: #555; font-size: 12px; margin-top: 8px; font-style: italic; }}
.warning {{ background: #fff4d6; border: 1px solid #e7c96b; color: #6a5312; padding: 12px 14px; border-radius: 10px; margin: 16px 0 24px 0; }}
</style>
</head>
<body>
<h1>Semantic Similarity Dashboard</h1>
<p>This dashboard evaluates semantic similarity using only the metrics selected at execution time: <strong>{selected_metric_names()}</strong>.
The semantic mean score is the arithmetic mean of the selected metrics, using the same 0 to 100 scale.</p>
{warning_html}
<div class="grid">
  <div class="card"><div class="small">Baseline scenarios</div><div class="metric">{baseline_total_all}</div></div>
  <div class="card"><div class="small">ERE scenarios</div><div class="metric">{baseline_ere_total}</div></div>
  <div class="card"><div class="small">Scenarios compared</div><div class="metric">{compared}</div></div>
  <div class="card"><div class="small">Comparison strategies</div><div class="metric">{strategies_count}</div></div>
  <div class="card"><div class="small">Total files analyzed</div><div class="metric">{total_files_analyzed}</div></div>
  <div class="card"><div class="small">Coverage (compared / ERE)</div><div class="metric">{coverage}%</div></div>
</div>
<div class="section"><h2>Structural robustness</h2><div class="note">Mode: {STRUCTURAL_EVALUATION_MODE}. Counts are computed by strategy directory. In AST mode, Structural Exact is binary, while AST Similarity is continuous.</div>{style_table(robustness_df)}<div class="footnote">{label_note}</div><img src="data:image/png;base64,{plot_to_base64(structural_plot)}" alt="Structural counts"/><div class="footnote">{label_note}</div></div>
<div class="section"><h2>Semantic Similarity Overview</h2><div class="note">Strategies are ranked by mean semantic score across all compared scenarios.</div>{style_table(scoreboard)}<div class="footnote">{label_note}</div></div>
<div class="section"><h2>Semantic score and label distribution</h2><div class="two-col"><div class="plot-card"><h3>Semantic score distribution</h3><img src="data:image/png;base64,{plot_to_base64(score_boxplot)}" alt="Semantic score distribution"/><div class="footnote">{label_note}</div></div><div class="plot-card"><h3>Semantic label distribution</h3><img src="data:image/png;base64,{plot_to_base64(label_plot)}" alt="Semantic label distribution"/><div class="footnote">{label_note}</div></div></div></div>
<div class="section"><h2>Pairwise Semantic Advantage</h2><div class="note">The column n_scenarios shows the comparison base for each pair.</div>{style_table(pairwise_df)}<div class="footnote">{label_note}</div><img src="data:image/png;base64,{plot_to_base64(pairwise_plot)}" alt="Pairwise semantic advantage"/><div class="footnote">{label_note}</div></div>
<div class="section"><h2>Winner by mean semantic score</h2>{style_table(winners_df, max_rows=100)}<div class="footnote">{label_note}</div></div>
<div class="section"><h2>All scenarios with consolidated metrics</h2>{style_table(metrics_by_scenario, max_rows=1000)}<div class="footnote">{label_note}</div></div>
<div class="section"><h2>Final mean metrics by strategy</h2>{style_table(metrics_by_strategy)}<div class="footnote">{label_note}</div></div>
<div class="section"><h2>Workbook</h2><p><a href="{excel_book.name}">{excel_book.name}</a></p></div>
</body>
</html>
"""
    out = DASHBOARD_PATH / "semantic_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out

def main():
    while True:
        prep = prepare_results()
        if prep == "restart":
            continue
        if prep is False:
            return
        break

    configure_metric_selection()
    configure_models()
    configure_structural_mode()
    reset_semantic_resources_for_selected_models()

    print("\nSemantic metrics configuration:")
    print(f" - Selected semantic metrics: {selected_metric_names()}")
    print(f" - BERTScore available: {bert_score_fn is not None}")
    print(f" - Embedding model available: {SentenceTransformer is not None}")
    print(f" - CodeBERTScore available: {code_bert_score_fn is not None}")
    print(f" - BERTScore model: {BERTSCORE_MODEL_TYPE or 'disabled'}")
    print(f" - Embedding model: {EMBEDDING_MODEL_NAME or 'disabled'}")
    print(f" - CodeBERTScore language: {CODEBERTSCORE_LANG or 'disabled'}")
    print(f" - Structural robustness mode: {STRUCTURAL_EVALUATION_MODE}")

    if bert_score_fn is None and "bertscore_f1" in ACTIVE_SCORE_METRICS:
        print("   WARNING: bert_score is not installed. BERTScore will be unavailable.")
    if SentenceTransformer is None and "embedding_cosine_similarity" in ACTIVE_SCORE_METRICS:
        print("   WARNING: sentence-transformers is not installed. Embedding cosine will be unavailable.")
    if code_bert_score_fn is None and "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        print("   WARNING: code-bert-score is not installed. Install it with: pip install code-bert-score")

    while True:
        raw_n = input("How many specs do you want to compare? ").strip()
        try:
            n = int(raw_n)
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number, for example: 10 or 182.")

    all_baseline_files = sorted(ORIGINAL_PATH.rglob("*.json"))
    baseline_total_all = len(all_baseline_files)
    all_ere_baseline_files, _ignored_all_non_ere = filter_ere_baselines(all_baseline_files)
    baseline_ere_total_all = len(all_ere_baseline_files)
    baseline_files = all_ere_baseline_files[:n]
    global_metrics = []
    missing_scenarios = []
    processed = 0
    generated_files_total = 0
    strategies = list(STRATEGY_PATHS.keys())

    for file in baseline_files:
        name = file.name
        print("Processing", name)
        strategy_files = {}
        missing = []

        for strategy, strategy_path in STRATEGY_PATHS.items():
            found = find_file(strategy_path, name)
            if found is None:
                missing.append(strategy)
            else:
                generated_files_total += 1
                strategy_files[strategy] = found

        if missing:
            print("Missing:", name, "->", ", ".join(missing))
            row = {"spec": name, "baseline_exists": True}
            for strategy in strategies:
                row[f"{strategy}_exists"] = strategy not in missing
            missing_scenarios.append(row)
            continue

        processed += 1
        df, matrix = compare_files_multi(file, strategy_files)
        global_metrics.append(df)
        df.to_csv(INDIVIDUAL_METRICS_PATH / f"{name}_metrics.csv", index=False)
        matrix.to_csv(INDIVIDUAL_MATRICES_PATH / f"{name}_matrix.csv")

    flush_semantic_caches()

    missing_df = pd.DataFrame(missing_scenarios)
    if not missing_df.empty:
        missing_df.to_csv(RESULT_PATH / "missing_scenarios.csv", index=False)

    if not global_metrics:
        print("No complete scenarios available.")
        return

    coverage_df = compute_dashboard_counts(
        baseline_total_all=baseline_total_all,
        baseline_ere_total_all=baseline_ere_total_all,
        compared_scenarios=processed,
        strategies_count=len(strategies),
        total_files_analyzed=generated_files_total,
    )
    coverage_df.to_csv(RESULT_PATH / "coverage_summary.csv", index=False)

    global_df = pd.concat(global_metrics, ignore_index=True)
    global_df.to_csv(RESULT_PATH / "global_metrics.csv", index=False)

    global_similarity_matrix = global_df.pivot_table(
        index="file_a",
        columns="file_b",
        values=PRIMARY_SCORE,
        aggfunc="mean"
    ).round(2)
    global_similarity_matrix.to_csv(RESULT_PATH / "global_similarity_matrix.csv")

    strategy_scores, strategy_wide_df = compute_strategy_scores(global_df, strategies)
    robustness_df = compute_robustness_summary(global_df, strategies)
    scenario_df = compute_scenario_organization(strategy_wide_df, strategies)
    winners_df = compute_winners(strategy_wide_df, strategies)
    scoreboard = compute_overall_scoreboard(winners_df, strategy_scores, strategies)
    pairwise_df = compute_pairwise_advantage(strategy_wide_df, strategies, RESULT_PATH)
    metrics_by_scenario, metrics_by_strategy = build_metric_tables(strategy_wide_df, strategies)
    statistical_analysis_multi(strategy_wide_df, strategies)

    excel_book = export_excel_book(
        coverage_df,
        robustness_df,
        scoreboard,
        scenario_df,
        winners_df,
        pairwise_df,
        metrics_by_scenario,
        metrics_by_strategy,
        missing_df,
    )

    dashboard_file = generate_dashboard(
        coverage_df,
        robustness_df,
        scoreboard,
        winners_df,
        pairwise_df,
        strategy_scores,
        metrics_by_scenario,
        metrics_by_strategy,
        strategy_wide_df,
        strategies,
        excel_book,
    )

    print("\nBaseline scenarios:", baseline_total_all)
    print("ERE scenarios:", baseline_ere_total_all)
    print("Scenarios compared:", processed)
    print("Coverage:", round((processed / baseline_ere_total_all * 100) if baseline_ere_total_all else 0.0, 2), "%")
    print("Requested scenarios:", n)
    print("Files processed:", processed)
    print("Missing scenarios:", len(missing_scenarios))
    print("Total files analyzed:", generated_files_total)
    print("Structural robustness mode:", STRUCTURAL_EVALUATION_MODE)
    print("\nAnalysis complete.")
    print("Files generated:")
    print(" - coverage_summary.csv")
    print(" - strategy_robustness_summary.csv")
    print(" - global_metrics.csv")
    print(" - strategy_scores_long.csv")
    print(" - strategy_scores_wide.csv")
    print(" - scenario_similarity_by_strategy.csv")
    print(" - scenario_summary.csv")
    print(" - winner_per_spec.csv")
    print(" - overall_scoreboard.csv")
    print(" - pairwise_semantic_advantage.csv")
    print(" - global_similarity_matrix.csv")
    print(" - statistical_tests_pairwise.csv")
    print(" - metric_means_by_scenario.csv")
    print(" - metric_means_by_strategy.csv")
    print(" - semantic_dashboard_tables.xlsx")
    print(" - dashboard/semantic_dashboard.html")
    print(f" - {get_embedding_cache_file().name}")
    print(f" - {get_bertscore_cache_file().name}")
    print(f" - {get_codebertscore_cache_file().name}")
    print("Dashboard:", dashboard_file)
    print("Workbook:", excel_book)


if __name__ == "__main__":
    main()