import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from nl2spec.logging_utils import get_logger
from nl2spec.analysis.metrics.exact_match import exact_match_binary
from nl2spec.analysis.comparators.canonical_json import canonical_json_equal

log = get_logger(__name__)


# =====================================================
# BUILD RAW DATA
# =====================================================

def _build_raw(generated_root: Path, baseline_root: Path, rq_dir: Path):

    rows = []
    match_logs = []
    detailed_logs = []

    start_time = time.time()

    for baseline_file in baseline_root.rglob("*.json"):

        domain = baseline_file.parent.name
        spec_id = baseline_file.stem

        log.info("Analyzing baseline file: %s", baseline_file.name)

        base_ir = json.loads(baseline_file.read_text())

        matching_llm_files = list(
            generated_root.rglob(f"{spec_id}.json")
        )

        if not matching_llm_files:
            log.warning("No LLM output found for baseline: %s", spec_id)
            continue

        for gen_file in matching_llm_files:

            parts = gen_file.parts
            llm_idx = parts.index("llm")

            provider = parts[llm_idx + 1]
            model = parts[llm_idx + 2]
            strategy = parts[llm_idx + 3]
            shot_mode = parts[llm_idx + 4]
            formalism = parts[llm_idx + 6]

            log.info(
                "Comparing baseline=%s with llm=%s | provider=%s | model=%s",
                baseline_file.name,
                gen_file.name,
                provider,
                model
            )

            gen_ir = json.loads(gen_file.read_text())

            exact = exact_match_binary(
                gen_ir,
                base_ir,
                canonical_json_equal
            )

            log.info(
                "Result: spec_id=%s | exact_match=%d",
                spec_id,
                exact
            )

            rows.append({
                "spec_id": spec_id,
                "provider": provider,
                "model": model,
                "strategy": strategy,
                "shot_mode": shot_mode,
                "domain": domain,
                "formalism": formalism,
                "exact_match": exact
            })

            match_logs.append({
                "spec_id": spec_id,
                "baseline_path": str(baseline_file),
                "llm_path": str(gen_file),
                "provider": provider,
                "model": model,
                "strategy": strategy
            })

            detailed_logs.append({
                "spec_id": spec_id,
                "baseline_file": baseline_file.name,
                "llm_file": gen_file.name,
                "provider": provider,
                "model": model,
                "strategy": strategy,
                "exact_match": exact
            })

    # -------------------------
    # Saving logs
    # -------------------------

    pd.DataFrame(match_logs).to_csv(
        rq_dir / "log_match_file.csv",
        index=False
    )

    pd.DataFrame(detailed_logs).to_csv(
        rq_dir / "log_detailed_comparison.csv",
        index=False
    )

    df = pd.DataFrame(rows)

    log.info("Raw dataframe built with %d rows", len(df))
    log.info("RQ1 execution time: %.2f seconds", time.time() - start_time)

    return df


# =====================================================
# GROUPING UTIL
# =====================================================

def _group(df, cols):

    grouped = (
        df.groupby(cols)["exact_match"]
        .mean()
        .reset_index()
    )

    grouped["exact_match_percent"] = (
        grouped["exact_match"] * 100
    ).round(2)

    return grouped


# =====================================================
# SAVE TABLE
# =====================================================

def _save_table(table, name, output_dir, tex_dir):

    csv_path = output_dir / f"{name}.csv"
    tex_path = tex_dir / f"{name}.tex"

    table.to_csv(csv_path, index=False)

    tex = table.to_latex(
        index=False,
        float_format="%.2f",
        caption=f"RQ1 Exact Match grouped by {name}",
        label=f"tab:rq1_{name}"
    )

    tex_path.write_text(tex)

    log.info("Saved table: %s", name)


# =====================================================
# GENERATE ALL GROUPINGS
# =====================================================

def _generate_all_groupings(df, output_dir, tex_dir):

    group_configs = {
        "per_formalism": ["formalism"],
        "per_strategy": ["strategy"],
        "per_model": ["model"],
        "per_provider": ["provider"],

        "model_strategy": ["model", "strategy"],
        "model_formalism": ["model", "formalism"],
        "strategy_formalism": ["strategy", "formalism"],

        "provider_model": ["provider", "model"],
        "provider_strategy": ["provider", "strategy"],

        "provider_model_strategy": ["provider", "model", "strategy"],
        "full_comparison": ["provider", "model", "strategy", "formalism"],
    }

    for name, cols in group_configs.items():
        table = _group(df, cols)
        _save_table(table, name, output_dir, tex_dir)


# =====================================================
# OVERALL SUMMARY
# =====================================================

def _generate_overall_summary(df, output_dir, tex_dir):

    overall = (df["exact_match"].mean() * 100).round(2)

    summary_df = pd.DataFrame([{
        "overall_exact_match_percent": overall,
        "total_samples": len(df)
    }])

    _save_table(summary_df, "overall_summary", output_dir, tex_dir)


# =====================================================
# HEATMAP (Model x Strategy)
# =====================================================

def _heatmap_model_strategy(df, img_dir):

    pivot = (
        df.groupby(["model", "strategy"])["exact_match"]
        .mean()
        .unstack()
        * 100
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.fillna(0).values)

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center")

    plt.colorbar()
    plt.title("RQ1 - Exact Match (%) Model x Strategy")
    plt.tight_layout()

    path = img_dir / "heatmap_model_strategy.png"
    plt.savefig(path)
    plt.close()

    log.info("Saved heatmap: %s", path.resolve())


# =====================================================
# MAIN
# =====================================================

def run(ctx, results_root: Path):

    log.info("========== RQ1 START ==========")

    rq_dir = results_root / "rq1"
    tex_dir = rq_dir / "tex"
    img_dir = rq_dir / "images"

    rq_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    generated_root = Path("nl2spec/output/llm")
    baseline_root = Path("nl2spec/datasets/baseline_ir")

    raw_csv = rq_dir / "exact_match_raw.csv"

    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        log.info("Loaded existing raw CSV")
    else:
        df = _build_raw(generated_root, baseline_root, rq_dir)
        df.to_csv(raw_csv, index=False)
        log.info("Built and saved raw CSV")

    if df.empty:
        log.warning("No data for RQ1")
        return

    _generate_all_groupings(df, rq_dir, tex_dir)
    _generate_overall_summary(df, rq_dir, tex_dir)
    _heatmap_model_strategy(df, img_dir)

    log.info("RQ1 completed successfully.")
    log.info("========== RQ1 END ==========")