import csv
from pathlib import Path


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
LLM_RUNS = {
    "openai": "datasets/results/openai/run_01/metrics/per_spec_metrics.csv",
    "gemini": "datasets/results/gemini/run_01/metrics/per_spec_metrics.csv",
    "deepseek": "datasets/results/deepseek/run_01/metrics/per_spec_metrics.csv",
}

OUT_FILE = "datasets/results/merged_llm_metrics.csv"


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    rows = []

    for llm, csv_path in LLM_RUNS.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {llm}: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["llm"] = llm
                rows.append(r)

    if not rows:
        print("[ERROR] No data loaded.")
        return

    fieldnames = ["llm", "spec_id", "domain", "category", "equal", "num_errors", "num_warnings"]

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] Merged CSV generated:")
    print(" -", OUT_FILE)


if __name__ == "__main__":
    main()
