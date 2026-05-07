#!/usr/bin/env python3
# run:
# python -m nl2spec.scripts.run_analisy_specs_all_dashboard

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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

try:
    from code_bert_score import score as code_bert_score_fn
except Exception:
    code_bert_score_fn = None

try:
    from codebleu import calc_codebleu
except Exception:
    calc_codebleu = None


# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(r"C:\UFPE\Siesta\project_LLM_spec")

ORIGINAL_PATH = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"
LLM_ROOT = PROJECT_ROOT / "nl2spec" / "output" / "llm"

RESULT_PATH = PROJECT_ROOT / "nl2spec" / "output" / "llm" / "results_llm_runs_dashboard_leo"
INDIVIDUAL_METRICS_PATH = RESULT_PATH / "individual" / "metrics"
PLOTS_PATH = RESULT_PATH / "plots"
DASHBOARD_PATH = RESULT_PATH / "dashboard"
CACHE_DIR = RESULT_PATH / "_cache"


# ======================================================
# EXPERIMENT CONFIG
# ======================================================

TARGET_FORMALISM = "ere"
PRIMARY_SCORE = "semantic_mean_score"
MIN_SCENARIOS_FOR_STATS = 5

MODEL_RUNS = {
    "gpt4o": {
        "provider": "openAI",
        "model": "gpt-4o",
        "enabled": True,
    },
    "claude": {
        "provider": "Claude",
        "model": "claude-sonnet-4-6",
        "enabled": True,
    },
}

RUN_CONFIGS = [
    {"selection": "mmr", "shot_dir": "few_k3", "shot_label": "mmr"},
    {"selection": "mmr_irsp", "shot_dir": "few_k3", "shot_label": "mmr_irsp"},
    {"selection": "irsp", "shot_dir": "few_k3", "shot_label": "irsp"},
    {"selection": "random", "shot_dir": "few_k3", "shot_label": "random"},
]

ACTIVE_SCORE_METRICS: List[str] = ["codebertscore_f1"]
COMPOSITE_METRICS: List[str] = []
OPTIONAL_PAIR_METRICS: List[str] = []

METRIC_SELECTION_OPTIONS = {
    "1": ["codebertscore_f1"],
    "2": ["codebleu_score"],
    "3": ["codebertscore_f1", "codebleu_score"],
    "all": ["codebertscore_f1", "codebleu_score"],
}

DISPLAY_METRICS = {
    "codebertscore_f1": "CodeBERTScore",
    "codebleu_score": "CodeBLEU",
}

CODE_LANGUAGE_OPTIONS = {
    "1": "java",
    "2": "python",
    "3": "javascript",
    "4": "php",
    "5": "ruby",
    "6": "go",
}

CODEBERTSCORE_LANG = None
CODEBLEU_LANG = None

REQUIRED_TOP_LEVEL_KEYS = ["id", "formalism", "domain", "signature", "ir"]
REQUIRED_IR_KEYS_ERE = ["events", "ere", "violation"]

LABEL_COLORS = {
    "errado": "#2EC4B6",
    "parcialmente_semelhante": "#0B3C6F",
    "parecido": "#2E6FA3",
    "muito_parecido": "#7FB3D5",
    "quase_igual": "#D6E6F2",
    "identico": "#BDBDBD",
}

PAIRWISE_COLORS = {
    "a_wins": "#0B3C6F",
    "b_wins": "#2E6FA3",
    "ties": "#BDBDBD",
}


# ======================================================
# RUN DISCOVERY
# ======================================================

def build_run_paths() -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}

    for model_label, model_cfg in MODEL_RUNS.items():
        if not model_cfg.get("enabled", True):
            continue

        provider = model_cfg["provider"]
        model = model_cfg["model"]

        for run_cfg in RUN_CONFIGS:
            selection = run_cfg["selection"]
            shot_dir = run_cfg["shot_dir"]
            shot_label = run_cfg["shot_label"]

            run_id = f"{provider}_{model_label}_{shot_label}"
            path = LLM_ROOT / provider / model / selection / shot_dir

            runs[run_id] = {
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "model_label": model_label,
                "selection": selection,
                "shot_dir": shot_dir,
                "shot_label": shot_label,
                "path": path,
                "exists": path.exists(),
            }

    return runs


RUN_PATHS = build_run_paths()


def run_color(run_id: str) -> str:
    palette = ["#0B3C6F", "#2E6FA3", "#7FB3D5", "#D6E6F2", "#2EC4B6", "#BDBDBD"]
    keys = list(RUN_PATHS.keys())
    if run_id in keys:
        return palette[keys.index(run_id) % len(palette)]
    return "#7FB3D5"


# ======================================================
# BASIC UTILS
# ======================================================

def path_depth(path: Path) -> int:
    return len(path.parts)


def safe_model_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def get_codebertscore_cache_file() -> Path:
    lang_name = safe_model_name(CODEBERTSCORE_LANG or "unset_codebertscore_lang")
    return CACHE_DIR / f"codebertscore_cache_{lang_name}.pkl"


def get_codebleu_cache_file() -> Path:
    lang_name = safe_model_name(CODEBLEU_LANG or "unset_codebleu_lang")
    return CACHE_DIR / f"codebleu_cache_{lang_name}.pkl"


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
                    for f in failed[:20]:
                        print(" -", f)
                    return False
                break

            if resp == "n":
                print("Returning to start...")
                return "restart"

            print("Please answer with y or n.")

    for p in [INDIVIDUAL_METRICS_PATH, PLOTS_PATH, DASHBOARD_PATH, CACHE_DIR]:
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

    print("\nSelect code-oriented semantic score metrics:")
    print(" 1 - CodeBERTScore only")
    print(" 2 - CodeBLEU only")
    print(" 3/all - CodeBERTScore + CodeBLEU")
    print(" custom - type metric ids separated by comma, for example: 1,2")

    metric_id_map = {
        "1": "codebertscore_f1",
        "2": "codebleu_score",
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
            print("Invalid option. Please choose 1-3, all, or custom.")
            continue

        COMPOSITE_METRICS = list(ACTIVE_SCORE_METRICS)
        OPTIONAL_PAIR_METRICS = []

        print(f" - Selected semantic metrics: {selected_metric_names()}")
        return


def configure_models():
    global CODEBERTSCORE_LANG, CODEBLEU_LANG

    if "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        CODEBERTSCORE_LANG = choose_option("Select CodeBERTScore language:", CODE_LANGUAGE_OPTIONS)
    else:
        CODEBERTSCORE_LANG = None

    if "codebleu_score" in ACTIVE_SCORE_METRICS:
        CODEBLEU_LANG = choose_option("Select CodeBLEU language:", CODE_LANGUAGE_OPTIONS)
    else:
        CODEBLEU_LANG = None

    print("\nSelected semantic metrics:", selected_metric_names())
    print(f" - CodeBERTScore language: {CODEBERTSCORE_LANG or 'disabled'}")
    print(f" - CodeBLEU language: {CODEBLEU_LANG or 'disabled'}")


# ======================================================
# JSON AND NORMALIZATION
# ======================================================

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


def canonical_sort_key(item: Any) -> str:
    return canonical_json_string(item)


def sort_canonical_list(items):
    return sorted(items, key=canonical_sort_key)


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

    if any(x in name for x in [
        "collection", "collections", "list", "map", "set", "queue",
        "deque", "iterator", "vector", "arraydeque", "treemap", "treeset"
    ]):
        return "UTIL"

    return "LANG"


def detect_formalism(ir_root):
    formalism = norm_text(ir_root.get("formalism")) if isinstance(ir_root, dict) else ""
    if formalism:
        return formalism

    ir_type = norm_text(ir_root.get("ir", {}).get("type")) if isinstance(ir_root, dict) else ""
    if ir_type:
        return ir_type

    return "unknown"


def load_json_quick(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def list_ere_baseline_files(baseline_dir: Path) -> List[Path]:
    files = []

    for path in sorted(baseline_dir.rglob("*.json")):
        data = load_json_quick(path)
        if detect_formalism(data or {}) == TARGET_FORMALISM:
            files.append(path)

    return files


def non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


# ======================================================
# OUTPUT FILE DISCOVERY: JSON + TXT
# ======================================================

def find_file(base, filename):
    if base is None or not Path(base).exists():
        return None

    for f in Path(base).rglob(filename):
        return f

    return None


def find_output_file(base: Path, baseline_filename: str) -> Optional[Path]:
    stem = Path(baseline_filename).stem

    for ext in [".json", ".txt"]:
        found = find_file(base, stem + ext)
        if found is not None:
            return found

    return None


def read_json_or_txt(path: Path) -> Tuple[Optional[Dict[str, Any]], str, str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return None, "syntax_error", f"Could not read file: {exc}", ""

    try:
        data = json.loads(text)
        return data, "json_ok", "", text
    except Exception as exc:
        return None, "syntax_error", str(exc), text


def validate_output_file(path: Optional[Path], expected_formalism: str = "ere") -> Dict[str, Any]:
    result = {
        "path": str(path) if path else "",
        "exists": path is not None and path.exists(),
        "extension": path.suffix.lower() if path else "",
        "syntax_success": 0,
        "schema_success": 0,
        "processable": 0,
        "error_type": "",
        "error_detail": "",
        "loaded_json": None,
        "raw_text": "",
        "formalism_detected": "unknown",
    }

    if path is None or not path.exists():
        result["error_type"] = "Missing_File"
        result["error_detail"] = "Output file not found."
        return result

    data, syntax_status, syntax_error, raw_text = read_json_or_txt(path)
    result["raw_text"] = raw_text

    if syntax_status != "json_ok":
        result["error_type"] = "Syntax_Error"
        result["error_detail"] = syntax_error
        return result

    result["syntax_success"] = 1
    result["loaded_json"] = data

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


# ======================================================
# SEMANTIC REPRESENTATION
# ======================================================

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


def code_fragment_has_signal(value: str) -> bool:
    text = str(value).strip()

    if not text:
        return False

    code_markers = [
        "(", ")", "..", "*", "+", ".", "&&", "||", "!", "==", "!=", "<", ">",
        "instanceof", "call", "args", "target", "returning", "new",
        "before", "after", "|", "ere", "violation", "event",
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
    fragments = collect_code_fragments(ir_root)

    if not fragments:
        return ""

    return "\n".join(fragments)


def raw_output_fragment_string(raw_text: str) -> str:
    if not raw_text:
        return ""

    fragments = []

    for line in raw_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if code_fragment_has_signal(stripped):
            normalized = compact_logic(stripped)

            if normalized and normalized not in fragments:
                fragments.append(normalized)

    if fragments:
        return "\n".join(fragments)

    return compact_logic(raw_text)


# ======================================================
# METRICS
# ======================================================

@dataclass
class SemanticResources:
    codebertscore_cache: Optional[Dict[Tuple[str, str, str], Optional[float]]] = None
    codebleu_cache: Optional[Dict[Tuple[str, str, str], Optional[float]]] = None


SEMANTIC_RESOURCES = SemanticResources(
    codebertscore_cache={},
    codebleu_cache={},
)


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


def reset_semantic_resources_for_selected_models():
    SEMANTIC_RESOURCES.codebertscore_cache = load_pickle_cache(get_codebertscore_cache_file(), {})
    SEMANTIC_RESOURCES.codebleu_cache = load_pickle_cache(get_codebleu_cache_file(), {})


def flush_semantic_caches():
    save_pickle_cache(get_codebertscore_cache_file(), SEMANTIC_RESOURCES.codebertscore_cache)
    save_pickle_cache(get_codebleu_cache_file(), SEMANTIC_RESOURCES.codebleu_cache)


def codebertscore_f1_single(text_a: str, text_b: str) -> Optional[float]:
    if code_bert_score_fn is None or CODEBERTSCORE_LANG is None:
        return None

    key = (text_a, text_b, CODEBERTSCORE_LANG)

    if key in SEMANTIC_RESOURCES.codebertscore_cache:
        return SEMANTIC_RESOURCES.codebertscore_cache[key]

    try:
        _, _, f1, _ = code_bert_score_fn([text_a], [text_b], lang=CODEBERTSCORE_LANG, verbose=False)
        value = float(f1[0].item())
    except Exception:
        value = None

    SEMANTIC_RESOURCES.codebertscore_cache[key] = value
    return value


def codebleu_single(reference_code: str, predicted_code: str) -> Optional[float]:
    if calc_codebleu is None or CODEBLEU_LANG is None:
        return None

    key = (reference_code, predicted_code, CODEBLEU_LANG)

    if key in SEMANTIC_RESOURCES.codebleu_cache:
        return SEMANTIC_RESOURCES.codebleu_cache[key]

    try:
        result = calc_codebleu([reference_code], [predicted_code], CODEBLEU_LANG)
        value = float(result.get("codebleu", np.nan))
    except Exception:
        value = None

    SEMANTIC_RESOURCES.codebleu_cache[key] = value
    return value


def code_metric_value_from_fragments(metric_name: str, reference_code: str, predicted_code: str) -> float:
    if not reference_code and not predicted_code:
        return np.nan

    if (reference_code and not predicted_code) or (not reference_code and predicted_code):
        return 0.0

    if metric_name == "codebertscore_f1":
        return safe_pct(codebertscore_f1_single(reference_code, predicted_code))

    if metric_name == "codebleu_score":
        return safe_pct(codebleu_single(reference_code, predicted_code))

    return np.nan


def selected_semantic_score_pct(metric_values: Dict[str, Optional[float]]) -> float:
    selected_values = []

    for metric in ACTIVE_SCORE_METRICS:
        value = metric_values.get(metric)

        if value is not None and not pd.isna(value):
            selected_values.append(float(value))

    if not selected_values:
        return np.nan

    return round(float(np.mean(selected_values)), 2)


def pair_metrics(
    label_a,
    label_b,
    json_a,
    json_b,
    spec_name,
    formalism,
    semantic_scope="processable_json",
    structural_status="Processable",
):
    code_text_a = code_component_string(json_a)
    code_text_b = code_component_string(json_b)

    metric_values: Dict[str, Optional[float]] = {}

    if "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        metric_values["codebertscore_f1"] = code_metric_value_from_fragments(
            "codebertscore_f1",
            code_text_a,
            code_text_b,
        )

    if "codebleu_score" in ACTIVE_SCORE_METRICS:
        metric_values["codebleu_score"] = code_metric_value_from_fragments(
            "codebleu_score",
            code_text_a,
            code_text_b,
        )

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
        "semantic_scope": semantic_scope,
        "structural_status": structural_status,
    }

    for metric in ACTIVE_SCORE_METRICS:
        row[metric] = metric_values.get(metric, np.nan)

    return row


def compare_baseline_to_run(baseline_file: Path, run_id: str, run_json: Dict[str, Any]) -> pd.DataFrame:
    baseline_json = json.loads(baseline_file.read_text(encoding="utf-8"))
    formalism = detect_formalism(baseline_json)

    row = pair_metrics(
        "baseline",
        run_id,
        baseline_json,
        run_json,
        baseline_file.name,
        formalism,
        semantic_scope="processable_json",
        structural_status="Processable",
    )

    return pd.DataFrame([row])


def fragment_level_pair_metrics(
    baseline_file: Path,
    run_id: str,
    raw_output_text: str,
    spec_name: str,
    error_type: str,
) -> pd.DataFrame:
    baseline_json = json.loads(baseline_file.read_text(encoding="utf-8"))
    formalism = detect_formalism(baseline_json)

    reference_code = code_component_string(baseline_json)
    predicted_code = raw_output_fragment_string(raw_output_text)

    metric_values: Dict[str, Optional[float]] = {}

    if "codebertscore_f1" in ACTIVE_SCORE_METRICS:
        metric_values["codebertscore_f1"] = code_metric_value_from_fragments(
            "codebertscore_f1",
            reference_code,
            predicted_code,
        )

    if "codebleu_score" in ACTIVE_SCORE_METRICS:
        metric_values["codebleu_score"] = code_metric_value_from_fragments(
            "codebleu_score",
            reference_code,
            predicted_code,
        )

    semantic_score_pct = selected_semantic_score_pct(metric_values)

    row = {
        "spec": spec_name,
        "formalism": formalism,
        "api": classify_api(spec_name),
        "file_a": "baseline",
        "file_b": run_id,
        "comparison": f"baseline_vs_{run_id}",
        "semantic_mean_score": semantic_score_pct,
        "semantic_label": semantic_label(semantic_score_pct),
        "semantic_scope": "fragment_level_invalid_output",
        "structural_status": error_type,
    }

    for metric in ACTIVE_SCORE_METRICS:
        row[metric] = metric_values.get(metric, np.nan)

    return pd.DataFrame([row])


# ======================================================
# TABLES
# ======================================================

def build_coverage_table(run_status_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for run_id, group in run_status_df.groupby("run_id"):
        first = group.iloc[0]

        rows.append({
            "run_id": run_id,
            "provider": first["provider"],
            "model": first["model"],
            "selection": first["selection"],
            "shot": first["shot_label"],
            "run_dir_exists": bool(first["run_dir_exists"]),
            "total_baseline": int(group.shape[0]),
            "files_found": int(group["exists"].sum()),
            "syntax_ok": int(group["syntax_success"].sum()),
            "schema_ok": int(group["schema_success"].sum()),
            "processable": int(group["processable"].sum()),
            "missing": int((group["error_type"] == "Missing_File").sum()),
            "missing_directory": int((group["error_type"] == "Missing_Directory").sum()),
            "syntax_fail": int((group["error_type"] == "Syntax_Error").sum()),
            "schema_fail": int((group["error_type"] == "Schema_Error").sum()),
            "wrong_formalism": int((group["error_type"] == "Wrong_Formalism").sum()),
        })

    return pd.DataFrame(rows)


def compute_run_scores(global_df: pd.DataFrame, run_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for _, row in global_df.iterrows():
        if row["file_a"] == "baseline":
            run_id = row["file_b"]
        elif row["file_b"] == "baseline":
            run_id = row["file_a"]
        else:
            continue

        if run_id not in run_ids:
            continue

        out = row.to_dict()
        out["run_id"] = run_id
        out["mean_score"] = row[PRIMARY_SCORE]
        rows.append(out)

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        return long_df, pd.DataFrame()

    long_df.to_csv(RESULT_PATH / "run_scores_long.csv", index=False)

    wide = long_df.pivot_table(
        index=["spec", "formalism", "api"],
        columns="run_id",
        values="mean_score",
        aggfunc="first"
    ).reset_index()

    wide = wide.rename(
        columns={c: f"{c}_mean_score" for c in wide.columns if c not in ["spec", "formalism", "api"]}
    )

    wide.to_csv(RESULT_PATH / "scenario_similarity_by_run.csv", index=False)
    return long_df, wide


def compute_overall_scoreboard(run_scores: pd.DataFrame, run_status_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    status_summary = build_coverage_table(run_status_df)

    for run_id in RUN_PATHS.keys():
        sub = run_scores[run_scores["run_id"] == run_id] if not run_scores.empty else pd.DataFrame()
        vals = pd.to_numeric(sub["mean_score"], errors="coerce") if not sub.empty else pd.Series(dtype=float)
        status_row = status_summary[status_summary["run_id"] == run_id]

        processable_semantic_total = 0
        fragment_level_total = 0

        if not sub.empty and "semantic_scope" in sub.columns:
            processable_semantic_total = int((sub["semantic_scope"] == "processable_json").sum())
            fragment_level_total = int((sub["semantic_scope"] == "fragment_level_invalid_output").sum())

        base = {
            "run_id": run_id,
            "provider": RUN_PATHS[run_id]["provider"],
            "model": RUN_PATHS[run_id]["model"],
            "selection": RUN_PATHS[run_id]["selection"],
            "shot": RUN_PATHS[run_id]["shot_label"],
            "comparisons_total": int(sub.shape[0]) if not sub.empty else 0,
            "processable_semantic_total": processable_semantic_total,
            "fragment_level_semantic_total": fragment_level_total,
            "avg_score_total_%": round(float(vals.mean()), 2) if len(vals.dropna()) else np.nan,
            "median_score_total_%": round(float(vals.median()), 2) if len(vals.dropna()) else np.nan,
            "std_score_total_%": round(float(vals.std(ddof=0)), 2) if len(vals.dropna()) else np.nan,
        }

        if not status_row.empty:
            for col in [
                "files_found", "syntax_ok", "schema_ok", "processable",
                "missing", "missing_directory", "syntax_fail", "schema_fail"
            ]:
                base[col] = int(status_row[col].iloc[0])

        rows.append(base)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH / "overall_run_scoreboard.csv", index=False)
    return df


def compute_final_mean_metrics_by_model(run_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the final mean metrics table segmented by model and strategy.
    Each row represents one run: provider + model + selection + shot.
    """
    rows = []

    for run_id, info in RUN_PATHS.items():
        sub = run_scores[run_scores["run_id"] == run_id] if run_scores is not None and not run_scores.empty else pd.DataFrame()

        score_vals = pd.to_numeric(sub.get("mean_score", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)
        codebert_vals = pd.to_numeric(sub.get("codebertscore_f1", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)
        codebleu_vals = pd.to_numeric(sub.get("codebleu_score", pd.Series(dtype=float)), errors="coerce") if not sub.empty else pd.Series(dtype=float)

        valid_json_rows = 0
        fragment_rows = 0
        if not sub.empty and "semantic_scope" in sub.columns:
            valid_json_rows = int((sub["semantic_scope"] == "processable_json").sum())
            fragment_rows = int((sub["semantic_scope"] == "fragment_level_invalid_output").sum())

        rows.append({
            "provider": info["provider"],
            "model": info["model"],
            "model_label": info["model_label"],
            "selection": info["selection"],
            "shot": info["shot_label"],
            "run_id": run_id,
            "comparisons_total": int(sub.shape[0]) if not sub.empty else 0,
            "valid_json_semantic_rows": valid_json_rows,
            "fragment_level_semantic_rows": fragment_rows,
            "ScoreTotal_mean_%": round(float(score_vals.mean()), 2) if len(score_vals.dropna()) else np.nan,
            "ScoreTotal_median_%": round(float(score_vals.median()), 2) if len(score_vals.dropna()) else np.nan,
            "CodeBERTScore_mean_%": round(float(codebert_vals.mean()), 2) if len(codebert_vals.dropna()) else np.nan,
            "CodeBLEU_mean_%": round(float(codebleu_vals.mean()), 2) if len(codebleu_vals.dropna()) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH / "final_mean_metrics_by_model.csv", index=False)
    return df


def build_consolidated_metrics_by_model(run_scores: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Builds one large scenario-level table per model.
    Columns follow the requested style:
    spec, formalism, api, original_ir,
    random_ScoreTotal, random_CodeBERTScore, mmr_ScoreTotal, ...
    """
    tables: Dict[str, pd.DataFrame] = {}

    if run_scores is None or run_scores.empty:
        return tables

    for model_label in sorted({info["model_label"] for info in RUN_PATHS.values()}):
        model_run_ids = [
            run_id for run_id, info in RUN_PATHS.items()
            if info["model_label"] == model_label
        ]
        model_df = run_scores[run_scores["run_id"].isin(model_run_ids)].copy()

        if model_df.empty:
            tables[model_label] = pd.DataFrame()
            continue

        base = model_df[["spec", "formalism", "api"]].drop_duplicates().sort_values(["spec", "api"]).reset_index(drop=True)
        base.insert(3, "original_ir", 100.0)

        for run_id in model_run_ids:
            info = RUN_PATHS[run_id]
            selection = info["selection"]
            sub = model_df[model_df["run_id"] == run_id].copy()

            if sub.empty:
                continue

            score_cols = ["spec", "formalism", "api", "mean_score"]
            score_part = sub[score_cols].rename(columns={"mean_score": f"{selection}_ScoreTotal"})
            base = base.merge(score_part, on=["spec", "formalism", "api"], how="left")

            if "codebertscore_f1" in sub.columns:
                cb_part = sub[["spec", "formalism", "api", "codebertscore_f1"]].rename(
                    columns={"codebertscore_f1": f"{selection}_CodeBERTScore"}
                )
                base = base.merge(cb_part, on=["spec", "formalism", "api"], how="left")

            if "codebleu_score" in sub.columns:
                bleu_part = sub[["spec", "formalism", "api", "codebleu_score"]].rename(
                    columns={"codebleu_score": f"{selection}_CodeBLEU"}
                )
                base = base.merge(bleu_part, on=["spec", "formalism", "api"], how="left")

        out_file = RESULT_PATH / f"all_scenarios_consolidated_metrics_{safe_model_name(model_label)}.csv"
        base.to_csv(out_file, index=False)
        tables[model_label] = base

    return tables


def compute_pairwise_advantage(run_wide_df: pd.DataFrame, run_ids: List[str]) -> pd.DataFrame:
    rows = []

    for r1, r2 in itertools.combinations(run_ids, 2):
        col1 = f"{r1}_mean_score"
        col2 = f"{r2}_mean_score"

        if col1 not in run_wide_df.columns or col2 not in run_wide_df.columns:
            continue

        sub = run_wide_df[[col1, col2]].dropna()
        n = len(sub)

        if n == 0:
            continue

        diff = sub[col1] - sub[col2]

        rows.append({
            "run_a": r1,
            "run_b": r2,
            "a_wins": int((diff > 0).sum()),
            "b_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "n_scenarios": n,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH / "pairwise_semantic_advantage.csv", index=False)
    return df


def statistical_analysis_multi(run_wide_df: pd.DataFrame, run_ids: List[str]):
    available = [f"{r}_mean_score" for r in run_ids if f"{r}_mean_score" in run_wide_df.columns]
    rows = []

    for s1, s2 in itertools.combinations(available, 2):
        sub = run_wide_df[[s1, s2]].dropna()
        n = len(sub)

        if n < MIN_SCENARIOS_FOR_STATS:
            rows.append({
                "run_a": s1.replace("_mean_score", ""),
                "run_b": s2.replace("_mean_score", ""),
                "n_scenarios": n,
                "test_used": "insufficient_data",
                "statistic": np.nan,
                "p_value": np.nan,
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
            "run_a": s1.replace("_mean_score", ""),
            "run_b": s2.replace("_mean_score", ""),
            "n_scenarios": n,
            "test_used": test,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "shapiro_stat": round(float(shapiro_stat), 6),
            "shapiro_p": round(float(shapiro_p), 6),
            "warning": "",
        })

    pd.DataFrame(rows).to_csv(RESULT_PATH / "statistical_tests_pairwise.csv", index=False)


# ======================================================
# PLOTS AND DASHBOARD
# ======================================================

def plot_to_base64(path: Optional[Path]):
    if path is None or not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def style_table(df, max_rows=None):
    if df is None or df.empty:
        return "<p>No data available.</p>"

    if max_rows is not None:
        df = df.head(max_rows).copy()

    return df.to_html(index=False, border=0)


def make_run_score_boxplot(run_scores: pd.DataFrame, run_ids: List[str]):
    if run_scores.empty:
        return None

    data = []
    labels = []

    for run_id in run_ids:
        vals = pd.to_numeric(run_scores.loc[run_scores["run_id"] == run_id, "mean_score"], errors="coerce").dropna()

        if len(vals) == 0:
            continue

        data.append(vals.values)
        labels.append(run_id)

    if not data:
        return None

    plt.figure(figsize=(12, 5))
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(run_color(label))
        patch.set_edgecolor("#3A3A3A")
        patch.set_alpha(0.95)

    for median in bp["medians"]:
        median.set_color("#3A3A3A")
        median.set_linewidth(1.5)

    plt.ylabel("Mean Semantic Score (%)")
    plt.title("Semantic Score Distribution by Run")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "semantic_score_distribution_by_run.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()

    return out


def make_structural_plot(coverage_df: pd.DataFrame):
    if coverage_df.empty:
        return None

    labels = coverage_df["run_id"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, coverage_df["syntax_ok"], width=width, label="Syntax OK", color="#0B3C6F")
    plt.bar(x, coverage_df["schema_ok"], width=width, label="Schema OK", color="#2E6FA3")
    plt.bar(x + width, coverage_df["syntax_fail"], width=width, label="Syntax Fail", color="#BDBDBD")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Syntax and Schema Robustness by Run")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "syntax_schema_by_run.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()

    return out


def make_pairwise_advantage_plot(pairwise_df: pd.DataFrame):
    if pairwise_df.empty:
        return None

    labels = [f"{a} vs {b}" for a, b in zip(pairwise_df["run_a"], pairwise_df["run_b"])]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, pairwise_df["a_wins"], width=width, label="a_wins", color=PAIRWISE_COLORS["a_wins"])
    plt.bar(x, pairwise_df["b_wins"], width=width, label="b_wins", color=PAIRWISE_COLORS["b_wins"])
    plt.bar(x + width, pairwise_df["ties"], width=width, label="ties", color=PAIRWISE_COLORS["ties"])

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Pairwise Semantic Advantage")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "pairwise_semantic_advantage.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()

    return out



def render_tabbed_tables(tab_id: str, tables: Dict[str, pd.DataFrame], max_rows: Optional[int] = None) -> str:
    """Render large tables as HTML tabs."""
    if not tables:
        return "<p>No data available.</p>"

    buttons = []
    contents = []

    for idx, (label, df) in enumerate(tables.items()):
        safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", f"{tab_id}_{label}")
        active_class = " active" if idx == 0 else ""
        display_style = "block" if idx == 0 else "none"

        buttons.append(
            f'<button class="tab-button{active_class}" onclick="openTab(event, \'{safe_id}\')">{label}</button>'
        )

        contents.append(
            f'<div id="{safe_id}" class="tab-content" style="display:{display_style};">'
            f'{style_table(df, max_rows=max_rows)}'
            f'</div>'
        )

    return '<div class="tabs">' + "".join(buttons) + '</div>' + "".join(contents)

def export_excel_book(
    coverage_df,
    scoreboard,
    pairwise_df,
    run_scores,
    run_wide_df,
    run_status_df,
    final_mean_by_model_df,
    consolidated_by_model,
):
    out = RESULT_PATH / "llm_runs_dashboard_tables.xlsx"

    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(out, engine=engine) as writer:
        coverage_df.to_excel(writer, sheet_name="00_coverage_by_run", index=False)
        scoreboard.to_excel(writer, sheet_name="01_run_summary", index=False)
        final_mean_by_model_df.to_excel(writer, sheet_name="02_mean_by_model", index=False)
        pairwise_df.to_excel(writer, sheet_name="03_pairwise", index=False)
        run_scores.to_excel(writer, sheet_name="04_scores_long", index=False)
        run_wide_df.to_excel(writer, sheet_name="05_scores_wide", index=False)
        run_status_df.to_excel(writer, sheet_name="06_status_by_spec", index=False)

        for model_label, df in consolidated_by_model.items():
            sheet = f"scenarios_{safe_model_name(model_label)}"[:31]
            if df is None or df.empty:
                pd.DataFrame().to_excel(writer, sheet_name=sheet, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet, index=False)

    return out




def add_rate_columns(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage rates used by the RQ pages."""
    if coverage_df is None or coverage_df.empty:
        return pd.DataFrame()

    df = coverage_df.copy()
    denom = pd.to_numeric(df.get("total_baseline", 0), errors="coerce").replace(0, np.nan)

    for col, out_col in [
        ("files_found", "coverage_rate_%"),
        ("syntax_ok", "syntax_ok_rate_%"),
        ("schema_ok", "schema_ok_rate_%"),
        ("processable", "processable_rate_%"),
        ("syntax_fail", "syntax_fail_rate_%"),
        ("schema_fail", "schema_fail_rate_%"),
    ]:
        if col in df.columns:
            df[out_col] = (pd.to_numeric(df[col], errors="coerce") / denom * 100).round(2)
        else:
            df[out_col] = np.nan

    return df


def make_structural_rate_plot(coverage_df: pd.DataFrame):
    """RQ1 plot: syntax/schema/processable rates by run."""
    if coverage_df is None or coverage_df.empty:
        return None

    df = add_rate_columns(coverage_df)
    labels = df["run_id"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, df["syntax_ok_rate_%"], width=width, label="Syntax OK rate")
    plt.bar(x, df["schema_ok_rate_%"], width=width, label="Schema OK rate")
    plt.bar(x + width, df["processable_rate_%"], width=width, label="Processable JSON rate")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Rate (%)")
    plt.ylim(0, 105)
    plt.title("Structural Validity Rates by Run")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "structural_validity_rates_by_run.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out


def make_tradeoff_plot(coverage_df: pd.DataFrame, scoreboard: pd.DataFrame):
    """RQ3 plot: structural validity versus semantic quality."""
    if coverage_df is None or coverage_df.empty or scoreboard is None or scoreboard.empty:
        return None

    cov = add_rate_columns(coverage_df)
    cols = ["run_id", "schema_ok_rate_%", "processable_rate_%", "syntax_ok_rate_%"]
    cols = [c for c in cols if c in cov.columns]
    df = scoreboard.merge(cov[cols], on="run_id", how="left")

    if "avg_score_total_%" not in df.columns:
        return None

    x = pd.to_numeric(df.get("processable_rate_%"), errors="coerce")
    y = pd.to_numeric(df["avg_score_total_%"], errors="coerce")

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=90)

    for _, row in df.iterrows():
        xv = row.get("processable_rate_%")
        yv = row.get("avg_score_total_%")
        if pd.notna(xv) and pd.notna(yv):
            label = str(row.get("run_id", ""))
            plt.annotate(label, (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel("Processable JSON rate (%)")
    plt.ylabel("Mean semantic score (%)")
    plt.title("Trade-off Between Structural Validity and Semantic Quality")
    plt.xlim(-2, 105)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "tradeoff_structural_vs_semantic.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out


def make_strategy_sensitivity_plot(pairwise_df: pd.DataFrame):
    """RQ4 plot: pairwise advantage restricted to strategies within the same model/provider."""
    if pairwise_df is None or pairwise_df.empty:
        return None

    df = pairwise_df.copy()

    def _same_model(a, b):
        if a not in RUN_PATHS or b not in RUN_PATHS:
            return False
        return (
            RUN_PATHS[a]["provider"] == RUN_PATHS[b]["provider"]
            and RUN_PATHS[a]["model"] == RUN_PATHS[b]["model"]
        )

    df = df[df.apply(lambda r: _same_model(r["run_a"], r["run_b"]), axis=1)]
    if df.empty:
        return None

    labels = [f'{a.replace("_", " ")}\nvs\n{b.replace("_", " ")}' for a, b in zip(df["run_a"], df["run_b"])]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, df["a_wins"], width=width, label="Run A wins")
    plt.bar(x, df["b_wins"], width=width, label="Run B wins")
    plt.bar(x + width, df["ties"], width=width, label="Ties")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Number of scenarios")
    plt.title("Strategy Sensitivity Within Each Model")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = PLOTS_PATH / "strategy_sensitivity_within_model.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    return out


def make_semantic_scope_table(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """RQ2 support table: separates valid-JSON semantic rows from fragment-level rows."""
    if scoreboard is None or scoreboard.empty:
        return pd.DataFrame()

    keep = [
        "run_id", "provider", "model", "selection", "comparisons_total",
        "processable_semantic_total", "fragment_level_semantic_total",
        "avg_score_total_%", "median_score_total_%", "std_score_total_%"
    ]
    keep = [c for c in keep if c in scoreboard.columns]
    df = scoreboard[keep].copy()

    total = pd.to_numeric(df.get("comparisons_total", 0), errors="coerce").replace(0, np.nan)
    if "processable_semantic_total" in df.columns:
        df["valid_json_semantic_rate_%"] = (
            pd.to_numeric(df["processable_semantic_total"], errors="coerce") / total * 100
        ).round(2)
    if "fragment_level_semantic_total" in df.columns:
        df["fragment_level_rate_%"] = (
            pd.to_numeric(df["fragment_level_semantic_total"], errors="coerce") / total * 100
        ).round(2)

    return df



def save_html_page(filename: str, title: str, active: str, body: str) -> Path:
    """Save one dashboard page with a shared navigation bar."""
    DASHBOARD_PATH.mkdir(parents=True, exist_ok=True)

    nav_items = [
        ("index", "Principal", "index.html"),
        ("methodology", "Metodologia", "methodology.html"),
        ("rq1", "RQ1 Robustez", "rq1_structural_validity.html"),
        ("rq2", "RQ2 Semântica", "rq2_semantic_quality.html"),
        ("rq3", "RQ3 Trade-off", "rq3_tradeoff.html"),
        ("rq4", "RQ4 Estratégias", "rq4_strategy_sensitivity.html"),
        ("tables", "Tabelas", "data_tables.html"),
    ]

    nav_html = "".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, label, href in nav_items
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
:root {{
  --border: #d9dee7;
  --muted: #5f6b7a;
  --bg: #f7f9fc;
  --card: #ffffff;
  --accent: #143b73;
}}
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: var(--bg);
  color: #1f2937;
}}
.header {{
  background: #ffffff;
  border-bottom: 1px solid var(--border);
  padding: 18px 28px;
}}
.header h1 {{
  margin: 0;
  font-size: 22px;
}}
.header p {{
  margin: 6px 0 0 0;
  color: var(--muted);
}}
.nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 28px;
  background: #ffffff;
  border-bottom: 1px solid var(--border);
}}
.nav-link {{
  text-decoration: none;
  color: #243447;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  background: #ffffff;
}}
.nav-link.active {{
  background: var(--accent);
  color: #ffffff;
  border-color: var(--accent);
}}
.container {{
  max-width: 1240px;
  margin: 0 auto;
  padding: 24px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
  gap: 12px;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
}}
.card .label {{
  color: var(--muted);
  font-size: 12px;
}}
.card .value {{
  font-size: 24px;
  font-weight: 700;
  margin-top: 6px;
}}
.rqbox, .note, .warning {{
  border-radius: 10px;
  padding: 14px 16px;
  margin: 14px 0;
  border: 1px solid var(--border);
  background: #ffffff;
}}
.rqbox {{
  border-left: 5px solid var(--accent);
}}
.warning {{
  border-left: 5px solid #b7791f;
  background: #fffaf0;
}}
.note {{
  border-left: 5px solid #64748b;
}}
img {{
  max-width: 100%;
  height: auto;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  margin: 12px 0 24px 0;
}}
.table-wrap {{
  overflow-x: auto;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 12px;
}}
th, td {{
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}}
th {{
  background: #f1f5f9;
  position: sticky;
  top: 0;
}}
.tabs {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
}}
.tab-button {{
  border: 1px solid var(--border);
  background: #ffffff;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}}
.tab-button.active {{
  background: var(--accent);
  color: #ffffff;
}}
.tab-content {{
  margin-top: 8px;
}}
</style>
<script>
function openTab(evt, tabId) {{
  var root = evt.target.closest('.container') || document;
  var contents = root.getElementsByClassName('tab-content');
  for (var i = 0; i < contents.length; i++) {{
    if (contents[i].id.startsWith(tabId.split('_')[0])) {{
      contents[i].style.display = 'none';
    }}
  }}
  var buttons = root.getElementsByClassName('tab-button');
  for (var j = 0; j < buttons.length; j++) {{
    buttons[j].className = buttons[j].className.replace(' active', '');
  }}
  document.getElementById(tabId).style.display = 'block';
  evt.currentTarget.className += ' active';
}}
</script>
</head>
<body>
<header class="header">
  <h1>Evaluating Few-Shot Selection Strategies for LLM-Based Runtime Verification Specification Generation</h1>
  <p>{title}</p>
</header>
<nav class="nav">{nav_html}</nav>
<main class="container">
{body}
</main>
</body>
</html>
"""
    out = DASHBOARD_PATH / filename
    out.write_text(html, encoding="utf-8")
    return out


def generate_dashboard(
    coverage_df,
    scoreboard,
    pairwise_df,
    run_scores,
    run_wide_df,
    run_status_df,
    final_mean_by_model_df,
    consolidated_by_model,
    excel_book
):
    # Core plots
    score_plot = make_run_score_boxplot(run_scores, list(RUN_PATHS.keys()))
    structural_count_plot = make_structural_plot(coverage_df)
    structural_rate_plot = make_structural_rate_plot(coverage_df)
    pairwise_plot = make_pairwise_advantage_plot(pairwise_df)
    tradeoff_plot = make_tradeoff_plot(coverage_df, scoreboard)
    sensitivity_plot = make_strategy_sensitivity_plot(pairwise_df)

    coverage_rates_df = add_rate_columns(coverage_df)
    semantic_scope_df = make_semantic_scope_table(scoreboard)

    total_runs = len(RUN_PATHS)
    total_existing_dirs = int(coverage_df["run_dir_exists"].sum()) if not coverage_df.empty and "run_dir_exists" in coverage_df.columns else 0
    total_files_found = int(coverage_df["files_found"].sum()) if not coverage_df.empty and "files_found" in coverage_df.columns else 0
    total_syntax_ok = int(coverage_df["syntax_ok"].sum()) if not coverage_df.empty and "syntax_ok" in coverage_df.columns else 0
    total_schema_ok = int(coverage_df["schema_ok"].sum()) if not coverage_df.empty and "schema_ok" in coverage_df.columns else 0
    total_processable = int(coverage_df["processable"].sum()) if not coverage_df.empty and "processable" in coverage_df.columns else 0
    total_syntax_fail = int(coverage_df["syntax_fail"].sum()) if not coverage_df.empty and "syntax_fail" in coverage_df.columns else 0
    total_schema_fail = int(coverage_df["schema_fail"].sum()) if not coverage_df.empty and "schema_fail" in coverage_df.columns else 0

    total_fragment_semantic = (
        int(scoreboard["fragment_level_semantic_total"].sum())
        if scoreboard is not None and not scoreboard.empty and "fragment_level_semantic_total" in scoreboard.columns
        else 0
    )

    total_strategies = len(set(info["selection"] for info in RUN_PATHS.values()))
    total_providers = len(set(info["provider"] for info in RUN_PATHS.values()))
    total_models = len(set(info["model"] for info in RUN_PATHS.values()))
    total_ere_scenarios = int(coverage_df["total_baseline"].max()) if not coverage_df.empty and "total_baseline" in coverage_df.columns else 0
    total_run_scenarios = int(coverage_df["total_baseline"].sum()) if not coverage_df.empty and "total_baseline" in coverage_df.columns else total_ere_scenarios * total_runs

    codebert_rows = int(run_scores["codebertscore_f1"].notna().sum()) if run_scores is not None and not run_scores.empty and "codebertscore_f1" in run_scores.columns else 0
    codebleu_rows = int(run_scores["codebleu_score"].notna().sum()) if run_scores is not None and not run_scores.empty and "codebleu_score" in run_scores.columns else 0

    processable_semantic_rows = int((run_scores["semantic_scope"] == "processable_json").sum()) if run_scores is not None and not run_scores.empty and "semantic_scope" in run_scores.columns else 0
    fragment_semantic_rows = int((run_scores["semantic_scope"] == "fragment_level_invalid_output").sum()) if run_scores is not None and not run_scores.empty and "semantic_scope" in run_scores.columns else 0

    # Pending / missing runs block
    missing_runs_df = coverage_df[coverage_df["run_dir_exists"] == False].copy() if not coverage_df.empty and "run_dir_exists" in coverage_df.columns else pd.DataFrame()

    index_body = f"""
<div class="rqbox">
Main research goal: evaluate how few-shot selection strategies affect structural validity and semantic quality of LLM-generated ERE runtime verification specifications.
</div>

<div class="warning">
Invalid outputs are not counted as valid IR. When textual content exists, fragment-level semantic similarity is computed only to verify whether specification-relevant fragments were preserved despite syntax/schema failure.
</div>

<h2>Experimental scope</h2>
<div class="grid">
  <div class="card"><div class="label">Formalism</div><div class="value">ERE</div></div>
  <div class="card"><div class="label">ERE scenarios</div><div class="value">{total_ere_scenarios}</div></div>
  <div class="card"><div class="label">Few-shot strategies</div><div class="value">{total_strategies}</div></div>
  <div class="card"><div class="label">Providers</div><div class="value">{total_providers}</div></div>
  <div class="card"><div class="label">Models</div><div class="value">{total_models}</div></div>
  <div class="card"><div class="label">Total run-scenarios</div><div class="value">{total_run_scenarios}</div></div>
</div>

<h2>Structural summary</h2>
<div class="grid">
  <div class="card"><div class="label">Configured runs</div><div class="value">{total_runs}</div></div>
  <div class="card"><div class="label">Existing run dirs</div><div class="value">{total_existing_dirs}</div></div>
  <div class="card"><div class="label">Files found</div><div class="value">{total_files_found}</div></div>
  <div class="card"><div class="label">Syntax OK</div><div class="value">{total_syntax_ok}</div></div>
  <div class="card"><div class="label">Schema OK</div><div class="value">{total_schema_ok}</div></div>
  <div class="card"><div class="label">Processable JSON</div><div class="value">{total_processable}</div></div>
  <div class="card"><div class="label">Syntax Fail</div><div class="value">{total_syntax_fail}</div></div>
  <div class="card"><div class="label">Schema Fail</div><div class="value">{total_schema_fail}</div></div>
  <div class="card"><div class="label">Fragment-level rows</div><div class="value">{total_fragment_semantic}</div></div>
</div>

<h2>Semantic metrics summary</h2>
<div class="grid">
  <div class="card"><div class="label">CodeBERTScore rows</div><div class="value">{codebert_rows}</div></div>
  <div class="card"><div class="label">CodeBLEU rows</div><div class="value">{codebleu_rows}</div></div>
  <div class="card"><div class="label">Valid JSON semantic rows</div><div class="value">{processable_semantic_rows}</div></div>
  <div class="card"><div class="label">Fragment-level semantic rows</div><div class="value">{fragment_semantic_rows}</div></div>
  <div class="card"><div class="label">Semantic metric mode</div><div class="value">{selected_metric_names()}</div></div>
</div>

<h2>Pending or missing runs</h2>
<div class="note">
Missing runs are kept visible for reproducibility. They should be marked as pending execution, not interpreted as model failure.
</div>
<div class="table-wrap">{style_table(missing_runs_df) if not missing_runs_df.empty else "<p>No missing run directories.</p>"}</div>

<h2>Quick visual overview</h2>
<img src="data:image/png;base64,{plot_to_base64(structural_rate_plot)}" alt="Structural rates"/>
<img src="data:image/png;base64,{plot_to_base64(tradeoff_plot)}" alt="Tradeoff plot"/>

<h2>Workbook</h2>
<p><a href="{excel_book.name}">{excel_book.name}</a></p>
"""
    index_out = save_html_page("index.html", "Dashboard principal", "index", index_body)

    methodology_body = f"""
<h2>Methodology</h2>
<div class="rqbox">Evaluation target: JSON-based ERE Runtime Verification specifications.</div>

<h3>Structural validation</h3>
<p>The structural layer checks whether each generated output exists, can be parsed as JSON, satisfies the expected schema, and is processable as a valid ERE IR.</p>

<h3>Semantic evaluation</h3>
<p>The primary semantic metric is CodeBERTScore over normalized specification content. CodeBLEU is kept as an exploratory metric only when it produces valid rows.</p>

<div class="warning">
CodeBLEU may return zero or no usable rows for JSON-based ERE specifications because it was designed around programming-language assumptions such as AST and data-flow features. For this reason, CodeBERTScore and fragment-level similarity are the primary semantic evidence.
</div>

<h3>Fragment-level semantic analysis</h3>
<p>Fragment-level analysis is applied only to invalid outputs. It extracts specification-relevant textual/code fragments from the raw output and compares them with the baseline fragments. This does not make the output valid; it only measures whether the model preserved useful semantic content despite syntax or schema failure.</p>

<h3>Configured runs</h3>
<div class="table-wrap">{style_table(coverage_rates_df)}</div>
"""
    save_html_page("methodology.html", "Metodologia", "methodology", methodology_body)

    rq1_body = f"""
<div class="rqbox">
RQ1: How do few-shot selection strategies impact the structural validity — syntax and schema compliance — of LLM-generated runtime verification specifications?
</div>

<h2>Metrics</h2>
<p>syntax_ok rate, schema_ok rate, and processable JSON rate.</p>

<img src="data:image/png;base64,{plot_to_base64(structural_rate_plot)}" alt="RQ1 structural rates"/>
<img src="data:image/png;base64,{plot_to_base64(structural_count_plot)}" alt="RQ1 structural counts"/>

<h2>Structural validity by run</h2>
<div class="table-wrap">{style_table(coverage_rates_df)}</div>
"""
    save_html_page("rq1_structural_validity.html", "RQ1 — Robustez estrutural", "rq1", rq1_body)

    rq2_body = f"""
<div class="rqbox">
RQ2: How does the semantic quality of generated specifications vary across models and few-shot selection strategies?
</div>

<h2>Metrics</h2>
<p>CodeBERTScore for valid/processable JSON and fragment-level similarity for invalid outputs with recoverable textual content.</p>

<div class="note">
Fragment-level similarity is diagnostic evidence: it shows whether the model kept meaningful ERE fragments even when the final file was not valid JSON or did not satisfy the schema.
</div>

<img src="data:image/png;base64,{plot_to_base64(score_plot)}" alt="RQ2 semantic score distribution"/>

<h2>Semantic summary by run</h2>
<div class="table-wrap">{style_table(semantic_scope_df)}</div>

<h2>Final mean metrics by model</h2>
<div class="table-wrap">{style_table(final_mean_by_model_df)}</div>
"""
    save_html_page("rq2_semantic_quality.html", "RQ2 — Qualidade semântica", "rq2", rq2_body)

    rq3_body = f"""
<div class="rqbox">
RQ3: Is there a trade-off between structural validity and semantic quality across different LLMs and selection strategies?
</div>

<h2>Trade-off view</h2>
<p>The x-axis shows processable JSON rate; the y-axis shows mean semantic score. Runs in the upper-right region combine robustness and semantic fidelity.</p>

<img src="data:image/png;base64,{plot_to_base64(tradeoff_plot)}" alt="RQ3 tradeoff"/>

<h2>Data used in the trade-off analysis</h2>
<div class="table-wrap">{style_table(coverage_rates_df.merge(semantic_scope_df, on='run_id', how='left'))}</div>
"""
    save_html_page("rq3_tradeoff.html", "RQ3 — Trade-off crítico", "rq3", rq3_body)

    rq4_body = f"""
<div class="rqbox">
RQ4: Are advanced few-shot selection strategies — MMR and IRSP — more beneficial than random selection across different LLMs?
</div>

<h2>Pairwise semantic advantage</h2>
<p>This analysis counts, scenario by scenario, whether one run outperforms another, ties, or loses.</p>

<img src="data:image/png;base64,{plot_to_base64(sensitivity_plot)}" alt="RQ4 strategy sensitivity"/>
<img src="data:image/png;base64,{plot_to_base64(pairwise_plot)}" alt="Pairwise semantic advantage"/>

<h2>Pairwise table</h2>
<div class="table-wrap">{style_table(pairwise_df)}</div>
"""
    save_html_page("rq4_strategy_sensitivity.html", "RQ4 — Sensibilidade à estratégia", "rq4", rq4_body)

    data_body = f"""
<h2>Consolidated scenario-level metrics</h2>
<div class="note">Large scenario-level tables are organized by model.</div>
{render_tabbed_tables("consolidated", consolidated_by_model, max_rows=1000)}

<h2>Status by scenario and run</h2>
<div class="table-wrap">{style_table(run_status_df, max_rows=1000)}</div>

<h2>Run-wide metrics</h2>
<div class="table-wrap">{style_table(run_wide_df, max_rows=1000)}</div>
"""
    save_html_page("data_tables.html", "Tabelas consolidadas", "data", data_body)

    # Keep the previous filename as a redirect/alias to the new main page.
    redirect = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<meta http-equiv="refresh" content="0; url=index.html"/>
<title>Redirecting</title></head><body>
<p>Dashboard moved to <a href="index.html">index.html</a>.</p>
</body></html>"""
    legacy_out = DASHBOARD_PATH / "llm_runs_dashboard.html"
    legacy_out.write_text(redirect, encoding="utf-8")

    return index_out


# ======================================================
# MAIN
# ======================================================

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
    reset_semantic_resources_for_selected_models()

    print("\nConfigured runs:")
    for run_id, info in RUN_PATHS.items():
        status = "exists" if info["exists"] else "MISSING_DIR"
        print(f" - {run_id}: {info['path']} [{status}]")

    while True:
        raw_n = input("How many ERE specs do you want to compare? ").strip()

        try:
            n = int(raw_n)

            if n <= 0:
                print("Please enter a positive integer.")
                continue

            break

        except ValueError:
            print("Invalid input. Please enter a number, for example: 10 or 72.")

    all_baseline_files = sorted(ORIGINAL_PATH.rglob("*.json"))
    baseline_total_all = len(all_baseline_files)

    ere_baseline_files = list_ere_baseline_files(ORIGINAL_PATH)
    baseline_ere_total_all = len(ere_baseline_files)
    baseline_files = ere_baseline_files[:n]

    global_metrics = []
    run_status_rows = []

    for baseline_file in baseline_files:
        name = baseline_file.name
        print(f"\nProcessing {name}")
        print("  [1/2] Checking syntax and schema...")

        for run_id, run_info in RUN_PATHS.items():
            run_path = run_info["path"]

            if not run_info["exists"]:
                print(f"  → {run_id}: missing run directory")

                run_status_rows.append({
                    "spec": name,
                    **{k: run_info[k] for k in ["run_id", "provider", "model", "model_label", "selection", "shot_dir", "shot_label"]},
                    "run_dir_exists": False,
                    "file_path": "",
                    "extension": "",
                    "exists": 0,
                    "syntax_success": 0,
                    "schema_success": 0,
                    "processable": 0,
                    "error_type": "Missing_Directory",
                    "error_detail": str(run_path),
                })

                continue

            output_file = find_output_file(run_path, name)
            validation = validate_output_file(output_file, expected_formalism=TARGET_FORMALISM)

            run_status_rows.append({
                "spec": name,
                **{k: run_info[k] for k in ["run_id", "provider", "model", "model_label", "selection", "shot_dir", "shot_label"]},
                "run_dir_exists": True,
                "file_path": validation["path"],
                "extension": validation["extension"],
                "exists": int(validation["exists"]),
                "syntax_success": int(validation["syntax_success"]),
                "schema_success": int(validation["schema_success"]),
                "processable": int(validation["processable"]),
                "error_type": validation["error_type"],
                "error_detail": validation["error_detail"],
            })

            if int(validation["processable"]) != 1:
                print(f"  → {run_id}: {validation['error_type']}")

                if int(validation["exists"]) == 1 and validation.get("raw_text", "").strip():
                    print("  [2/2] Computing semantic metrics from available fragments...")

                    df = fragment_level_pair_metrics(
                        baseline_file=baseline_file,
                        run_id=run_id,
                        raw_output_text=validation["raw_text"],
                        spec_name=name,
                        error_type=validation["error_type"],
                    )

                    global_metrics.append(df)

                    df.to_csv(
                        INDIVIDUAL_METRICS_PATH / f"{Path(name).stem}_{run_id}_fragment_metrics.csv",
                        index=False,
                    )

                continue

            print(f"  → {run_id}: processable")
            print("  [2/2] Computing semantic metrics from valid JSON...")

            df = compare_baseline_to_run(
                baseline_file=baseline_file,
                run_id=run_id,
                run_json=validation["loaded_json"],
            )

            global_metrics.append(df)

            df.to_csv(
                INDIVIDUAL_METRICS_PATH / f"{Path(name).stem}_{run_id}_metrics.csv",
                index=False,
            )

    flush_semantic_caches()

    run_status_df = pd.DataFrame(run_status_rows)
    run_status_df.to_csv(RESULT_PATH / "run_status_by_spec.csv", index=False)

    coverage_df = build_coverage_table(run_status_df)
    coverage_df.to_csv(RESULT_PATH / "coverage_by_run.csv", index=False)

    if not global_metrics:
        print("\nNo output text available for semantic comparison.")
        print("Still generated status and coverage files.")
        return

    global_df = pd.concat(global_metrics, ignore_index=True)
    global_df.to_csv(RESULT_PATH / "global_metrics.csv", index=False)

    run_ids = list(RUN_PATHS.keys())

    run_scores, run_wide_df = compute_run_scores(global_df, run_ids)
    scoreboard = compute_overall_scoreboard(run_scores, run_status_df)
    final_mean_by_model_df = compute_final_mean_metrics_by_model(run_scores)
    consolidated_by_model = build_consolidated_metrics_by_model(run_scores)
    pairwise_df = compute_pairwise_advantage(run_wide_df, run_ids)
    statistical_analysis_multi(run_wide_df, run_ids)

    excel_book = export_excel_book(
        coverage_df=coverage_df,
        scoreboard=scoreboard,
        pairwise_df=pairwise_df,
        run_scores=run_scores,
        run_wide_df=run_wide_df,
        run_status_df=run_status_df,
        final_mean_by_model_df=final_mean_by_model_df,
        consolidated_by_model=consolidated_by_model,
    )

    dashboard_file = generate_dashboard(
        coverage_df=coverage_df,
        scoreboard=scoreboard,
        pairwise_df=pairwise_df,
        run_scores=run_scores,
        run_wide_df=run_wide_df,
        run_status_df=run_status_df,
        final_mean_by_model_df=final_mean_by_model_df,
        consolidated_by_model=consolidated_by_model,
        excel_book=excel_book,
    )

    print("\nBaseline total specs:", baseline_total_all)
    print("ERE baseline specs:", baseline_ere_total_all)
    print("Requested ERE specs:", n)
    print("Configured runs:", len(RUN_PATHS))
    print("\nAnalysis complete.")
    print("Files generated:")
    print(" - coverage_by_run.csv")
    print(" - run_status_by_spec.csv")
    print(" - global_metrics.csv")
    print(" - run_scores_long.csv")
    print(" - scenario_similarity_by_run.csv")
    print(" - overall_run_scoreboard.csv")
    print(" - final_mean_metrics_by_model.csv")
    print(" - all_scenarios_consolidated_metrics_<model>.csv")
    print(" - pairwise_semantic_advantage.csv")
    print(" - statistical_tests_pairwise.csv")
    print(" - llm_runs_dashboard_tables.xlsx")
    print(" - dashboard/llm_runs_dashboard.html")
    print("Dashboard:", dashboard_file)
    print("Workbook:", excel_book)


if __name__ == "__main__":
    main()
