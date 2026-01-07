import json
import csv
from pathlib import Path
from typing import Dict, List


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
RESULTS_DIR = "datasets/results/openai/run_01"
OUT_DIR = "datasets/results/openai/run_01/metrics"

OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def infer_domain(spec_id: str) -> str:
    # id vem como "io/FileInputStream_Close"
    return spec_id.split("/")[0]


def load_diffs(results_dir: Path) -> List[Dict]:
    diffs = []
    for f in results_dir.glob("*.diff.json"):
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
            d["_file"] = f.name
            diffs.append(d)
    return diffs


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    results_dir = Path(RESULTS_DIR)
    diffs = load_diffs(results_dir)

    if not diffs:
        print("[WARN] No diff files found.")
        return

    # ---------------------------------------------------
    # CSV 1 — Per specification (ENRICHED)
    # ---------------------------------------------------
    per_spec_csv = OUT_DIR / "per_spec_metrics.csv"

    with open(per_spec_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "spec_id",
            "domain",
            "category",
            "equal",
            "num_errors",
            "num_warnings"
        ])

        for d in diffs:
            spec_id = d.get("id")
            domain = infer_domain(spec_id)
            category = d.get("category", "UNKNOWN")

            writer.writerow([
                spec_id,
                domain,
                category,
                int(d.get("equal", False)),
                len(d.get("errors", [])),
                len(d.get("warnings", []))
            ])

    # ---------------------------------------------------
    # Aggregated metrics (run-level)
    # ---------------------------------------------------
    total = len(diffs)
    equal = sum(1 for d in diffs if d.get("equal"))
    avg_errors = sum(len(d.get("errors", [])) for d in diffs) / total
    avg_warnings = sum(len(d.get("warnings", [])) for d in diffs) / total

    summary = {
        "total_specs": total,
        "exact_match": equal,
        "exact_match_ratio": round(equal / total, 4),
        "avg_errors_per_spec": round(avg_errors, 3),
        "avg_warnings_per_spec": round(avg_warnings, 3)
    }

    summary_csv = OUT_DIR / "summary_metrics.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print("[OK] Metrics generated with domain and category")
    print(" -", per_spec_csv)
    print(" -", summary_csv)


if __name__ == "__main__":
    main()
