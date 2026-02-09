"""
Build canonical baseline IR from JavaMOP specifications.

This script performs an OFFLINE and DETERMINISTIC conversion of JavaMOP
(.mop) properties into a unified Intermediate Representation (IR).

The generated IR files constitute the BASELINE DATASET used in NL2Spec
experiments and are:

- independent from any LLM
- validated against the IR JSON schema
- reused across all experimental runs
- NOT regenerated automatically by the main pipeline

This script should be executed manually whenever the JavaMOP corpus
or the IR mapping rules are updated.
"""

# executed 
# C:/Python39/python.exe -m nl2spec.scripts.convert_mop_to_ir
# python.exe -m nl2spec.scripts.convert_mop_to_ir
#

from pathlib import Path
import shutil
import sys

from nl2spec.core.comparator.mop_to_ir import convert_mop_dir_to_ir
from nl2spec.core.inspection.validate_ir import IRValidator


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input: original JavaMOP specifications
MOP_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "raw_mop"

# Output: canonical baseline IR dataset
BASELINE_OUT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"

# IR schema used for validation
SCHEMA_PATH = PROJECT_ROOT / "nl2spec" / "core" / "schemas" / "ir.schema.json"


# ==========================================================
# UTILS
# ==========================================================

def ask_overwrite(path: Path) -> bool:
    """Ask user whether an existing dataset should be overwritten."""
    answer = input(
        f"[WARN] Baseline dataset already exists:\n"
        f"       {path}\n"
        f"Do you want to DELETE and REGENERATE it? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ==========================================================
# MAIN PROCEDURE
# ==========================================================

def main() -> None:
    print("=" * 70)
    print("[INFO] Building baseline IR dataset from JavaMOP specifications")
    print("[INFO] Source directory :", MOP_ROOT)
    print("[INFO] Output directory :", BASELINE_OUT)
    print("=" * 70)

    # ------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------
    if not MOP_ROOT.exists():
        print(f"[ERROR] JavaMOP root directory not found:\n        {MOP_ROOT}")
        sys.exit(1)

    if not SCHEMA_PATH.exists():
        print(f"[ERROR] IR schema not found:\n        {SCHEMA_PATH}")
        sys.exit(1)

    # ------------------------------------------------------
    # Handle existing baseline dataset
    # ------------------------------------------------------
    if BASELINE_OUT.exists():
        if not ask_overwrite(BASELINE_OUT):
            print("[INFO] Baseline generation aborted by user.")
            return

        print("[INFO] Removing existing baseline dataset...")
        shutil.rmtree(BASELINE_OUT)

    BASELINE_OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # Step 1: Convert JavaMOP → IR
    # ------------------------------------------------------
    print("[INFO] Converting .mop files to canonical IR...")
    count = convert_mop_dir_to_ir(
        mop_root=str(MOP_ROOT),
        out_dir=str(BASELINE_OUT),
        keep_structure=True
    )

    print(f"[OK] Conversion completed: {count} properties converted.")

    # ------------------------------------------------------
    # Step 2: Validate generated IR files
    # ------------------------------------------------------
    print("[INFO] Validating generated IR files against schema...")
    validator = IRValidator(str(SCHEMA_PATH))

    total = 0
    errors = 0

    for json_file in BASELINE_OUT.rglob("*.json"):
        total += 1
        result = validator.validate_file(str(json_file))
        if not result.valid:
            errors += 1
            print(f"[ERROR] Invalid IR file: {json_file}")
            for e in result.errors:
                print(f"        - {e}")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    print("=" * 70)
    print(f"[SUMMARY] Total IR files generated : {total}")
    print(f"[SUMMARY] Invalid IR files        : {errors}")

    if errors == 0:
        print("[OK] Baseline IR dataset successfully generated and validated.")
    else:
        print("[WARN] Baseline generated with validation errors.")

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
