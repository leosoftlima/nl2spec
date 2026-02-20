#!/usr/bin/env python3
# run: python -m nl2spec.scripts.run_fewshot_selection_experiment

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from nl2spec.logging_utils import get_logger, setup_logging
from nl2spec.core.handlers.fewshot_loader import FewShotLoader

log = get_logger(__name__)

# ======================================================
# ROOT DISCOVERY
# ======================================================

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        has_nl2spec = (candidate / "nl2spec").exists()
        has_marker = any(
            (candidate / marker).exists()
            for marker in [
                ".git",
                "pyproject.toml",
                "setup.cfg",
                "requirements.txt",
                "README.md",
            ]
        )
        if has_nl2spec and has_marker:
            return candidate

    for candidate in [cur] + list(cur.parents):
        if (candidate / "nl2spec").exists():
            return candidate

    return start.resolve()


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(SCRIPT_PATH.parent)

BASELINE_DIR = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"
FEWSHOT_DIR = PROJECT_ROOT / "nl2spec" / "datasets" / "fewshot"
RESULTS_DIR = PROJECT_ROOT / "results"
FEWSHOT_RESULTS_DIR = RESULTS_DIR / "fewshot"

# ======================================================
# CONFIGURAÇÃO EXPERIMENTAL
# ======================================================

TARGET_FORMALISM = "fsm"     # "ltl " | "fsm" | "event" | "ere" | "all"
SHOT_MODE = "one"            # "zero" | "one" | "few"
K = 1                        #    0   |   1   | 3
SELECTION = "structural"     # "structural" | "random"

# ======================================================
# UTILIDADES
# ======================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ir_type(spec_json: dict) -> str:
    return spec_json.get("ir", {}).get("type", "").lower()


def _ensure_results_dir() -> None:
    FEWSHOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(rows: List[dict], formalism: str) -> Path:
    _ensure_results_dir()

    filename = f"{SELECTION}_{SHOT_MODE}_{formalism}.csv"
    output_file = FEWSHOT_RESULTS_DIR / filename

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset",
                "formalism",
                "rank",
                "selected_fewshot",
                "distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_file


# ======================================================
# MAIN
# ======================================================

def main():

    setup_logging("INFO")

    print("\n=== Testing Few-Shot Selection ===\n")
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Baseline dir:  {BASELINE_DIR}")
    print(f"Fewshot dir:   {FEWSHOT_DIR}")
    print(f"Results dir:   {RESULTS_DIR}\n")

    allowed = {"ltl", "fsm", "event", "ere", "all"}
    if TARGET_FORMALISM not in allowed:
        raise ValueError(f"Invalid TARGET_FORMALISM='{TARGET_FORMALISM}'")

    loader = FewShotLoader(str(FEWSHOT_DIR))

    print(f"Target formalism: {TARGET_FORMALISM}")
    print(f"Shot mode: {SHOT_MODE} | k={K} | selection={SELECTION}\n")

    # 👇 agora inclui ERE
    rows_by_formalism: Dict[str, List[dict]] = {
        "ltl": [],
        "fsm": [],
        "event": [],
        "ere": [],
    }

    total = 0

    for domain_dir in BASELINE_DIR.iterdir():

        if not domain_dir.is_dir():
            continue

        for file_path in domain_dir.glob("*.json"):

            spec_json = load_json(file_path)
            ir_type = get_ir_type(spec_json)

            if ir_type not in rows_by_formalism:
                continue

            if TARGET_FORMALISM != "all" and ir_type != TARGET_FORMALISM:
                continue

            total += 1

            print(f"Dataset: {file_path.name}")
            print(f"  Formalism: {ir_type}")

            try:

                selected: List[Tuple[Path, float]] = loader.get(
                    ir_type=ir_type,
                    shot_mode=SHOT_MODE,
                    k=K,
                    selection=SELECTION,
                    ir_base=spec_json,
                    return_scores=True,
                )

                if not selected:

                    print("  → No few-shot selected")

                    rows_by_formalism[ir_type].append({
                        "dataset": file_path.stem,
                        "formalism": ir_type,
                        "rank": 0,
                        "selected_fewshot": "",
                        "distance": "",
                    })

                else:
                    for rank, (path, dist) in enumerate(selected, start=1):

                        print(
                            f"  → Selected (rank {rank}): {path.name} "
                            f"| distance={dist:.4f}"
                        )

                        rows_by_formalism[ir_type].append({
                            "dataset": file_path.stem,
                            "formalism": ir_type,
                            "rank": rank,
                            "selected_fewshot": path.name,
                            "distance": f"{dist:.6f}",
                        })

            except Exception as e:

                print(f"  → ERROR: {str(e)}")

                rows_by_formalism[ir_type].append({
                    "dataset": file_path.stem,
                    "formalism": ir_type,
                    "rank": -1,
                    "selected_fewshot": f"ERROR: {str(e)}",
                    "distance": "",
                })

            print()

    print(f"\nTotal specifications analyzed: {total}\n")

    # ======================================================
    # SALVAR CSV
    # ======================================================

    if TARGET_FORMALISM == "all":
        for f in ["ltl", "fsm", "event", "ere"]:
            out = save_results(rows_by_formalism[f], f)
            print(f"✅ CSV saved at: {out} (rows={len(rows_by_formalism[f])})")
    else:
        out = save_results(rows_by_formalism[TARGET_FORMALISM], TARGET_FORMALISM)
        print(f"✅ CSV saved at: {out} (rows={len(rows_by_formalism[TARGET_FORMALISM])})")


if __name__ == "__main__":
    main()