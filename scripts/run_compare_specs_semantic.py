# run:
# python -m nl2spec.scripts.run_compare_specs_semantic

import json
import difflib
import itertools
import os
import re
import shutil
import stat
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


# ---------------- CONFIG PATHS ----------------

ORIGINAL_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\datasets\baseline_ir"
)

RANDOM_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\random\one_k1"
)

STRUCTURAL_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\structural\one_k1"
)

RESULT_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\results_semantic"
)

INDIVIDUAL_METRICS_PATH = RESULT_PATH / "individual" / "metrics"
INDIVIDUAL_MATRICES_PATH = RESULT_PATH / "individual" / "matrices"

EQUAL_CASES_PATH = RESULT_PATH / "equal_cases"
DIFFERENT_CASES_PATH = RESULT_PATH / "different_cases"
PLOTS_PATH = RESULT_PATH / "plots"


# ---------------- PREPARE RESULT FOLDER ----------------

def safe_delete_tree(path: Path, retries: int = 5, wait: float = 0.8):
    if not path.exists():
        return []

    for _ in range(retries):
        failed = []

        for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
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

    failed = []
    if path.exists():
        for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if item.exists():
                failed.append(str(item))

    return failed


def prepare_results():
    if RESULT_PATH.exists():
        resp = input("Results folder exists. Delete and recreate? (y/n): ")
        if resp.lower() == "y":
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

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_MATRICES_PATH.mkdir(parents=True, exist_ok=True)
    EQUAL_CASES_PATH.mkdir(parents=True, exist_ok=True)
    DIFFERENT_CASES_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    return True


# ---------------- BASIC HELPERS ----------------

def norm_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compact_logic(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([\(\)\[\]\{\},;:\->\+\*\?\|&=!<>])\s*", r"\1", s)
    s = s.replace("\n", "").replace("\t", "")
    return s.strip().lower()


def normalize_json(obj):
    if isinstance(obj, dict):
        return {k: normalize_json(obj[k]) for k in sorted(obj)}

    if isinstance(obj, list):
        return [normalize_json(x) for x in obj]

    return obj


def canonical_json_string(obj):
    return json.dumps(
        normalize_json(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_json_equal(a, b):
    return canonical_json_string(a) == canonical_json_string(b)


def sort_canonical_list(items):
    return sorted(items, key=lambda x: canonical_json_string(x))


# ---------------- ALGORITHMS ----------------

def levenshtein(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))

    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n

        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1]

            if a[j - 1] != b[i - 1]:
                change += 1

            current[j] = min(add, delete, change)

    return current[n]


def levenshtein_similarity(a, b):
    denom = max(len(a), len(b), 1)
    return 1 - levenshtein(a, b) / denom


def sequence_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def flatten_json(obj, prefix="root"):
    items = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            items += flatten_json(v, f"{prefix}.{k}")

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items += flatten_json(v, f"{prefix}[{i}]")

    else:
        items.append(prefix)

    return items


def structural_similarity(a, b):
    s1 = set(flatten_json(a))
    s2 = set(flatten_json(b))
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def tokenize(text):
    return set(re.findall(r"[A-Za-z_]+", text))


def jaccard_similarity(a, b):
    t1 = tokenize(a)
    t2 = tokenize(b)
    union = len(t1 | t2)
    return len(t1 & t2) / union if union else 1.0


def tree_similarity(a, b):
    p1 = set(flatten_json(a))
    p2 = set(flatten_json(b))
    max_nodes = max(len(p1), len(p2), 1)
    return len(p1 & p2) / max_nodes


# ---------------- SEMANTIC EXTRACTION HELPERS ----------------

def normalize_parameter(param):
    if not isinstance(param, dict):
        return {"raw": norm_text(param)}

    return {
        "type": norm_text(param.get("type")),
        "name": norm_text(param.get("name")),
        "value": norm_text(param.get("value")),
        "raw": compact_logic(canonical_json_string(param)),
    }


def normalize_argument(arg):
    if not isinstance(arg, dict):
        return {"raw": norm_text(arg)}

    return {
        "name": norm_text(arg.get("name")),
        "type": norm_text(arg.get("type")),
        "value": norm_text(arg.get("value")),
        "raw": compact_logic(canonical_json_string(arg)),
    }


def normalize_function_item(item):
    if not isinstance(item, dict):
        return {"raw": compact_logic(canonical_json_string(item))}

    return {
        "name": norm_text(item.get("name")),
        "negated": bool(item.get("negated", False)),
        "arguments": [normalize_argument(a) for a in item.get("arguments", [])],
        "raw": compact_logic(canonical_json_string(item)),
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
        "kind": norm_text(method.get("kind")),
        "name": norm_text(method.get("name")),
        "timing": norm_text(method.get("timing")),
        "procediments": norm_text(method.get("procediments")),
        "pointcut_raw": norm_text(method.get("pointcut_raw")),
        "parameters": sort_canonical_list(
            [normalize_parameter(p) for p in method.get("parameters", [])]
        ),
        "function": sort_canonical_list(
            [normalize_function_item(f) for f in method.get("function", [])]
        ),
        "operation": normalize_operation_list(method.get("operation", [])),
        "returning": normalize_json(method.get("returning"))
        if isinstance(method.get("returning"), dict) else None,
        "throws": normalize_json(method.get("throws", [])),
        "raw": compact_logic(canonical_json_string(method)),
    }


# ---------------- SEMANTIC IR EXTRACTION ----------------

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

    return [{
        "kind": norm_text(ev.get("kind")),
        "name": norm_text(ev.get("name")),
        "timing": norm_text(ev.get("timing")),
        "parameters": sort_canonical_list(
            [normalize_parameter(p) for p in ev.get("parameters", [])]
        ),
        "pointcut_raw": norm_text(ev.get("pointcut_raw")),
        "returning": normalize_json(ev.get("returning"))
        if isinstance(ev.get("returning"), dict) else None,
        "raw": compact_logic(canonical_json_string(ev)),
    }]


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

            if "raw_block" in block:
                if isinstance(block["raw_block"], list):
                    return compact_logic(" ".join(map(str, block["raw_block"])))
                return compact_logic(block["raw_block"])

            if "raw_lines" in block:
                return compact_logic(" ".join(map(str, block["raw_lines"])))

            return compact_logic(canonical_json_string(block))

    return ""


def extract_ere_semantics(ir_root):
    ir = ir_root.get("ir", {})

    return {
        "formalism": "ere",
        "domain": norm_text(ir_root.get("domain")),
        "signature": normalize_json(ir_root.get("signature", {})),
        "events": extract_events(ir_root),
        "logic": extract_logic_block(ir, ["ere", "regex", "logic"]),
        "violation": normalize_json(ir.get("violation", {})),
    }


def extract_semantic_components(ir_root):
    formalism = detect_formalism(ir_root)

    if formalism == "ere":
        return extract_ere_semantics(ir_root)

    return {
        "formalism": formalism,
        "domain": norm_text(ir_root.get("domain")),
        "signature": normalize_json(ir_root.get("signature", {})),
        "events": extract_events(ir_root),
        "violation": normalize_json(ir_root.get("ir", {}).get("violation", {})),
        "raw_ir": normalize_json(ir_root.get("ir", {})),
    }


def semantic_ir_match(a, b):
    return (
        canonical_json_string(extract_semantic_components(a))
        == canonical_json_string(extract_semantic_components(b))
    )


def semantic_ir_string(ir_root):
    return canonical_json_string(extract_semantic_components(ir_root))


def semantic_ir_similarity(ir_a, ir_b):
    s1 = semantic_ir_string(ir_a)
    s2 = semantic_ir_string(ir_b)
    return sequence_similarity(s1, s2)


def semantic_similarity(a, b):
    comp_a = extract_semantic_components(a)
    comp_b = extract_semantic_components(b)
    return sequence_similarity(
        canonical_json_string(comp_a),
        canonical_json_string(comp_b),
    )


# ---------------- FIND FILE ----------------

def find_file(base, filename):
    for f in base.rglob(filename):
        return f
    return None


# ---------------- API CLASSIFICATION ----------------

def classify_api(name):
    name = name.lower()

    if (
        "socket" in name or "net" in name or "http" in name or "url" in name
        or "datagram" in name or "inet" in name
    ):
        return "NET"

    if (
        "file" in name or "stream" in name or "reader" in name
        or "writer" in name or "console" in name
    ):
        return "IO"

    if (
        "collection" in name or "collections" in name or "list" in name
        or "map" in name or "set" in name or "queue" in name
        or "deque" in name or "iterator" in name or "vector" in name
        or "arraydeque" in name or "treemap" in name or "treeset" in name
    ):
        return "UTIL"

    return "LANG"


# ---------------- COMPARE PAIRS ----------------

def pair_metrics(label_a, label_b, json_a, json_b, text_a, text_b, spec_name, formalism):
    json_equal = canonical_json_equal(json_a, json_b)
    sem_equal = semantic_ir_match(json_a, json_b)

    lev = levenshtein_similarity(text_a, text_b)
    seq = sequence_similarity(text_a, text_b)
    struct = structural_similarity(json_a, json_b)
    jacc = jaccard_similarity(text_a, text_b)
    tree = tree_similarity(json_a, json_b)
    sem = semantic_similarity(json_a, json_b)
    sem_ir = semantic_ir_similarity(json_a, json_b)

    row = {
        "spec": spec_name,
        "formalism": formalism,
        "api": classify_api(spec_name),
        "file_a": label_a,
        "file_b": label_b,
        "comparison": f"{label_a}_vs_{label_b}",
        "json_exact_match": int(json_equal),
        "semantic_exact_match": int(sem_equal),
        "levenshtein_similarity": round(lev * 100, 2),
        "sequence_similarity": round(seq * 100, 2),
        "structural_similarity": round(struct * 100, 2),
        "jaccard_similarity": round(jacc * 100, 2),
        "tree_similarity": round(tree * 100, 2),
        "semantic_similarity": round(sem * 100, 2),
        "semantic_ir_similarity": round(sem_ir * 100, 2),
        "case_type": "equal" if sem_equal else "different",
    }

    return row


def compare_files(f1, f2, f3):
    texts = {}
    jsons = {}

    for label, path in zip(
        ["baseline", "random", "structural"],
        [f1, f2, f3],
    ):
        txt = Path(path).read_text(encoding="utf-8")
        texts[label] = txt
        jsons[label] = json.loads(txt)

    spec_name = Path(f1).name
    formalism = detect_formalism(jsons["baseline"])

    rows = []
    names = list(texts.keys())

    for a, b in itertools.combinations(names, 2):
        rows.append(
            pair_metrics(
                label_a=a,
                label_b=b,
                json_a=jsons[a],
                json_b=jsons[b],
                text_a=texts[a],
                text_b=texts[b],
                spec_name=spec_name,
                formalism=formalism,
            )
        )

    df = pd.DataFrame(rows)
    matrix = pd.DataFrame(index=names, columns=names)

    for a in names:
        for b in names:
            if a == b:
                matrix.loc[a, b] = 100.0
            else:
                r = df[
                    ((df.file_a == a) & (df.file_b == b)) |
                    ((df.file_a == b) & (df.file_b == a))
                ]
                if r.empty:
                    matrix.loc[a, b] = np.nan
                else:
                    value = r["semantic_ir_similarity"].iloc[0]
                    matrix.loc[a, b] = round(float(value), 2) if not pd.isna(value) else np.nan

    return df, matrix


# ---------------- CROSS STRATEGY ANALYSIS ----------------

def compute_cross_strategy_correction(global_df):
    scenarios = []
    specs = global_df["spec"].unique()

    for spec in specs:
        subset = global_df[global_df["spec"] == spec]
        formalism = subset["formalism"].iloc[0]
        api = subset["api"].iloc[0]

        br = subset[subset["comparison"] == "baseline_vs_random"]
        bs = subset[subset["comparison"] == "baseline_vs_structural"]
        rs = subset[subset["comparison"] == "random_vs_structural"]

        if br.empty or bs.empty or rs.empty:
            continue

        br_equal = int(br["semantic_exact_match"].iloc[0])
        bs_equal = int(bs["semantic_exact_match"].iloc[0])
        rs_equal = int(rs["semantic_exact_match"].iloc[0])

        br_score = float(br["semantic_ir_similarity"].iloc[0])
        bs_score = float(bs["semantic_ir_similarity"].iloc[0])
        rs_score = float(rs["semantic_ir_similarity"].iloc[0])

        if br_equal == 1 and bs_equal == 1:
            category = "both_correct"
        elif br_equal == 0 and bs_equal == 0:
            category = "both_wrong"
        elif br_equal == 0 and bs_equal == 1:
            category = "corrected_by_structural"
        elif br_equal == 1 and bs_equal == 0:
            category = "degraded_by_structural"
        else:
            category = "unknown"

        if bs_score > br_score:
            closer_to_baseline = "structural"
        elif br_score > bs_score:
            closer_to_baseline = "random"
        else:
            closer_to_baseline = "tie"

        difference = round(abs(bs_score - br_score), 2)

        scenarios.append({
            "scenario": spec,
            "formalism": formalism,
            "api": api,
            "category": category,
            "closer_to_baseline": closer_to_baseline,
            "difference_%": difference,
            "baseline_vs_random_semantic_ir_similarity": br_score,
            "baseline_vs_structural_semantic_ir_similarity": bs_score,
            "random_vs_structural_semantic_ir_similarity": rs_score,
            "baseline_vs_random_equal": br_equal,
            "baseline_vs_structural_equal": bs_equal,
            "random_vs_structural_equal": rs_equal,
        })

    return pd.DataFrame(scenarios)


# ---------------- OVERALL ANALYSIS ----------------

def compute_overall_stats(cross_df):
    structural_wins = (cross_df["closer_to_baseline"] == "structural").sum()
    random_wins = (cross_df["closer_to_baseline"] == "random").sum()
    ties = (cross_df["closer_to_baseline"] == "tie").sum()

    non_ties = cross_df[cross_df["closer_to_baseline"] != "tie"]
    avg_advantage = round(non_ties["difference_%"].mean(), 2) if not non_ties.empty else 0.0

    scoreboard = pd.DataFrame([{
        "structural_closer_to_baseline": structural_wins,
        "random_closer_to_baseline": random_wins,
        "ties": ties,
        "average_advantage_%": avg_advantage,
    }])

    scoreboard.to_csv(RESULT_PATH / "overall_scoreboard.csv", index=False)
    return scoreboard


# ---------------- METRIC ANALYSIS ----------------

def metric_analysis(all_df):
    metric_cols = [
        "levenshtein_similarity",
        "sequence_similarity",
        "structural_similarity",
        "jaccard_similarity",
        "tree_similarity",
        "semantic_similarity",
        "semantic_ir_similarity",
    ]

    valid_df = all_df.dropna(subset=metric_cols).copy()

    if valid_df.empty:
        print("No valid cases available for metric analysis.")
        return

    metric_corr = valid_df[metric_cols].corr()
    metric_corr.to_csv(RESULT_PATH / "metric_correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(metric_corr)
    plt.colorbar()
    plt.xticks(range(len(metric_cols)), metric_cols, rotation=45)
    plt.yticks(range(len(metric_cols)), metric_cols)
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "metric_correlation_heatmap.png", bbox_inches="tight")
    plt.close()


# ---------------- FORMALISM ANALYSIS ----------------

def formalism_analysis(cross_df):
    summary = cross_df.groupby(["formalism", "closer_to_baseline"]).size().unstack(fill_value=0)
    summary.to_csv(RESULT_PATH / "formalism_wins.csv")

    winrate = summary.div(summary.sum(axis=1), axis=0) * 100
    winrate = winrate.round(2)
    winrate.to_csv(RESULT_PATH / "formalism_winrate.csv")

    advantage = cross_df.groupby("formalism")["difference_%"].mean().round(2)
    advantage.to_csv(RESULT_PATH / "formalism_average_advantage.csv")

    ax = summary.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Closer to Baseline per Formalism")
    ax.set_ylabel("Number of Scenarios")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "formalism_wins_plot.png")
    plt.close()


# ---------------- FORMALISM DIFFICULTY ----------------

def formalism_difficulty(cross_df):
    cross_df = cross_df.copy()
    cross_df["difficulty"] = 100 - cross_df[
        [
            "baseline_vs_random_semantic_ir_similarity",
            "baseline_vs_structural_semantic_ir_similarity",
        ]
    ].max(axis=1)

    formalisms = list(cross_df["formalism"].dropna().unique())
    data = [cross_df[cross_df["formalism"] == f]["difficulty"] for f in formalisms]

    if not data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=formalisms)
    plt.ylabel("Difficulty")
    plt.title("Difficulty per Formalism")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "formalism_difficulty_boxplot.png", bbox_inches="tight")
    plt.close()


# ---------------- API DIFFICULTY ----------------

def api_difficulty(cross_df):
    cross_df = cross_df.copy()
    cross_df["difficulty"] = 100 - cross_df[
        [
            "baseline_vs_random_semantic_ir_similarity",
            "baseline_vs_structural_semantic_ir_similarity",
        ]
    ].max(axis=1)

    difficulty = cross_df.groupby("api")[
        [
            "baseline_vs_random_semantic_ir_similarity",
            "baseline_vs_structural_semantic_ir_similarity",
            "difficulty",
        ]
    ].mean().round(2)

    difficulty.to_csv(RESULT_PATH / "api_difficulty.csv")

    plt.figure(figsize=(8, 5))
    ordered_apis = list(difficulty.index)
    data = [cross_df[cross_df["api"] == a]["difficulty"] for a in ordered_apis]
    plt.boxplot(data, tick_labels=ordered_apis)
    plt.ylabel("Difficulty")
    plt.title("API Difficulty for LLM")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "api_difficulty_boxplot.png", bbox_inches="tight")
    plt.close()

    return cross_df


# ---------------- API × FORMALISM HEATMAP ----------------

def api_formalism_heatmap(cross_df):
    pivot = cross_df.pivot_table(
        index="api",
        columns="formalism",
        values="difference_%",
        aggfunc="mean",
    ).round(2)

    pivot.to_csv(RESULT_PATH / "api_formalism_matrix.csv")

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot)
    plt.colorbar()
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("API × Formalism Difficulty")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "api_formalism_heatmap.png")
    plt.close()


# ---------------- STATISTICAL ANALYSIS ----------------

def statistical_analysis(cross_df):
    random_scores = cross_df["baseline_vs_random_semantic_ir_similarity"].values
    structural_scores = cross_df["baseline_vs_structural_semantic_ir_similarity"].values
    diff = structural_scores - random_scores

    if len(diff) < 3:
        pd.DataFrame([{
            "test_used": "insufficient_data",
            "statistic": np.nan,
            "p_value": np.nan,
            "shapiro_stat": np.nan,
            "shapiro_p": np.nan,
        }]).to_csv(RESULT_PATH / "statistical_test.csv", index=False)
        return

    shapiro_stat, shapiro_p = shapiro(diff)

    if shapiro_p > 0.05:
        test = "paired_t_test"
        stat, p = ttest_rel(structural_scores, random_scores)
    else:
        test = "wilcoxon"
        stat, p = wilcoxon(structural_scores, random_scores)

    pd.DataFrame([{
        "test_used": test,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "shapiro_stat": round(float(shapiro_stat), 6),
        "shapiro_p": round(float(shapiro_p), 6),
    }]).to_csv(RESULT_PATH / "statistical_test.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.boxplot([random_scores, structural_scores], tick_labels=["Random", "Structural"])
    plt.ylabel("Semantic IR Similarity (%)")
    plt.title("Similarity Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "boxplot_similarity.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(diff, bins=15)
    plt.xlabel("Structural - Random (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Differences")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "difference_histogram.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    better_struct = diff > 0
    better_random = diff < 0
    ties = diff == 0

    plt.scatter(
        random_scores[better_struct],
        structural_scores[better_struct],
        marker="o",
        label="Structural better",
    )

    plt.scatter(
        random_scores[better_random],
        structural_scores[better_random],
        marker="x",
        label="Random better",
    )

    if np.any(ties):
        plt.scatter(
            random_scores[ties],
            structural_scores[ties],
            marker="s",
            label="Tie",
        )

    max_val = max(max(random_scores), max(structural_scores), 100)
    plt.plot([0, max_val], [0, max_val], label="Equal performance")

    plt.xlabel("Baseline vs Random")
    plt.ylabel("Baseline vs Structural")
    plt.title("Random vs Structural")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "random_vs_structural.png", bbox_inches="tight")
    plt.close()


# ---------------- SUMMARY TABLES ----------------

def save_equal_and_different_cases(global_df):
    equal_df = global_df[global_df["case_type"] == "equal"].copy()
    diff_df = global_df[global_df["case_type"] == "different"].copy()

    equal_df.to_csv(EQUAL_CASES_PATH / "equal_cases.csv", index=False)
    diff_df.to_csv(DIFFERENT_CASES_PATH / "different_cases.csv", index=False)

    summary_equal = (
        equal_df.groupby(["formalism", "comparison"])
        .size()
        .reset_index(name="count")
    )
    summary_diff = (
        diff_df.groupby(["formalism", "comparison"])
        .size()
        .reset_index(name="count")
    )

    summary_equal.to_csv(EQUAL_CASES_PATH / "equal_cases_summary.csv", index=False)
    summary_diff.to_csv(DIFFERENT_CASES_PATH / "different_cases_summary.csv", index=False)

    return equal_df, diff_df


# ---------------- AUDIT / VERIFICATION ----------------

def save_audit_views(global_df, cross_df):
    audit_rows = []

    for spec in global_df["spec"].unique():
        subset = global_df[global_df["spec"] == spec].copy()

        row = {"spec": spec}
        for comp in ["baseline_vs_random", "baseline_vs_structural", "random_vs_structural"]:
            sub = subset[subset["comparison"] == comp]
            if sub.empty:
                continue

            row[f"{comp}_semantic_exact_match"] = int(sub["semantic_exact_match"].iloc[0])
            row[f"{comp}_semantic_ir_similarity"] = float(sub["semantic_ir_similarity"].iloc[0])
            row[f"{comp}_json_exact_match"] = int(sub["json_exact_match"].iloc[0])

        audit_rows.append(row)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(RESULT_PATH / "audit_pairwise_view.csv", index=False)

    verification = cross_df.copy()
    verification["structural_advantage_over_random"] = (
        verification["baseline_vs_structural_semantic_ir_similarity"]
        - verification["baseline_vs_random_semantic_ir_similarity"]
    ).round(2)

    verification["verified_label"] = np.where(
        verification["structural_advantage_over_random"] > 0,
        "structural",
        np.where(
            verification["structural_advantage_over_random"] < 0,
            "random",
            "tie",
        ),
    )

    verification["label_matches_rule"] = (
        verification["verified_label"] == verification["closer_to_baseline"]
    )

    verification.to_csv(RESULT_PATH / "winner_verification.csv", index=False)


# ---------------- MAIN ----------------

def main():
    ok = prepare_results()
    if not ok:
        return

    n = int(input("How many specs do you want to compare? "))

    files = sorted(ORIGINAL_PATH.rglob("*.json"))[:n]

    global_metrics = []
    missing_scenarios = []
    processed = 0

    for file in files:
        name = file.name
        print("Processing", name)

        rand = find_file(RANDOM_PATH, name)
        struct = find_file(STRUCTURAL_PATH, name)

        if not rand or not struct:
            print("Missing:", name)

            missing_scenarios.append({
                "spec": name,
                "baseline_exists": True,
                "random_exists": rand is not None,
                "structural_exists": struct is not None,
            })
            continue

        processed += 1

        df, matrix = compare_files(file, rand, struct)

        global_metrics.append(df)

        df.to_csv(INDIVIDUAL_METRICS_PATH / f"{name}_metrics.csv", index=False)
        matrix.to_csv(INDIVIDUAL_MATRICES_PATH / f"{name}_matrix.csv")

    if missing_scenarios:
        pd.DataFrame(missing_scenarios).to_csv(
            RESULT_PATH / "missing_scenarios.csv",
            index=False,
        )

    if not global_metrics:
        return

    global_df = pd.concat(global_metrics, ignore_index=True)
    global_df.to_csv(RESULT_PATH / "global_metrics.csv", index=False)

    semantic_matrix = global_df.pivot_table(
        index="file_a",
        columns="file_b",
        values="semantic_ir_similarity",
        aggfunc="mean",
    ).round(2)
    semantic_matrix.to_csv(RESULT_PATH / "global_similarity_matrix.csv")

    equal_df, diff_df = save_equal_and_different_cases(global_df)

    cross_df = compute_cross_strategy_correction(global_df)
    cross_df.to_csv(RESULT_PATH / "cross_strategy_correction.csv", index=False)

    compute_overall_stats(cross_df)
    metric_analysis(global_df)
    formalism_analysis(cross_df)
    formalism_difficulty(cross_df)

    cross_with_api = api_difficulty(cross_df)
    api_formalism_heatmap(cross_with_api)

    statistical_analysis(cross_df)
    save_audit_views(global_df, cross_df)

    print("\nTotal scenarios analysed:", len(cross_df))
    print("Requested scenarios:", n)
    print("Files processed:", processed)
    print("\nAnalysis complete.")
    print("Files generated:")
    print(" - global_metrics.csv")
    print(" - global_similarity_matrix.csv")
    print(" - cross_strategy_correction.csv")
    print(" - overall_scoreboard.csv")
    print(" - statistical_test.csv")
    print(" - metric_correlation_matrix.csv")
    print(" - api_difficulty.csv")
    print(" - formalism_wins.csv")
    print(" - formalism_winrate.csv")
    print(" - formalism_average_advantage.csv")
    print(" - audit_pairwise_view.csv")
    print(" - winner_verification.csv")
    print(" - equal_cases/equal_cases.csv")
    print(" - equal_cases/equal_cases_summary.csv")
    print(" - different_cases/different_cases.csv")
    print(" - different_cases/different_cases_summary.csv")
    print(" - individual/metrics/")
    print(" - individual/matrices/")
    print(" - plots/")
    print("Missing scenarios:", len(missing_scenarios))


if __name__ == "__main__":
    main()