"""
Convert JavaMOP (.mop) specifications into canonical IR specifications.

This script performs an offline conversion of JavaMOP properties into
a unified Intermediate Representation (IR) used as baseline for NL2Spec
experiments.

The generated IR files are reused across experimental runs and are NOT
regenerated automatically by the main pipeline.
"""

from pathlib import Path
import shutil
import sys

from nl2spec.core.comparator.mop_to_ir import convert_mop_dir_to_ir
from nl2spec.core.inspection.validate_ir import IRValidator


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOP_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "raw_mop"
BASELINE_OUT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"
SCHEMA_PATH = PROJECT_ROOT / "nl2spec" / "core" / "schemas" / "ir.schema.json"



# ==========================================================
# UTILS
# ==========================================================

def ask_overwrite(path: Path) -> bool:
    """Ask user whether to overwrite an existing directory."""
    answer = input(
        f"[WARN] Baseline directory already exists:\n"
        f"       {path}\n"
        f"Do you want to delete and regenerate it? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    print("=" * 60)
    print("[INFO] Converting JavaMOP specifications to IR")
    print("[INFO] Source  :", MOP_ROOT)
    print("[INFO] Output  :", BASELINE_OUT)
    print("=" * 60)

    # ------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------
    if not MOP_ROOT.exists():
        print(f"[ERROR] MOP root directory not found: {MOP_ROOT}")
        sys.exit(1)

    if not SCHEMA_PATH.exists():
        print(f"[ERROR] IR schema not found: {SCHEMA_PATH}")
        sys.exit(1)

    # ------------------------------------------------------
    # Handle existing baseline
    # ------------------------------------------------------
    if BASELINE_OUT.exists():
        if not ask_overwrite(BASELINE_OUT):
            print("[INFO] Conversion aborted by user.")
            return

        print("[INFO] Removing existing baseline directory...")
        shutil.rmtree(BASELINE_OUT)

    BASELINE_OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # Step 1: Convert .mop → IR
    # ------------------------------------------------------
    print("[INFO] Converting .mop files to IR...")
    count = convert_mop_dir_to_ir(
        mop_root=str(MOP_ROOT),
        out_dir=str(BASELINE_OUT),
        keep_structure=True
    )

    print(f"[OK] Converted {count} .mop files into IR JSON")

    # ------------------------------------------------------
    # Step 2: Validate generated IR files
    # ------------------------------------------------------
    print("[INFO] Validating generated IR files...")
    validator = IRValidator(str(SCHEMA_PATH))

    errors = 0
    total = 0

    for json_file in BASELINE_OUT.rglob("*.json"):
        total += 1
        result = validator.validate_file(str(json_file))
        if not result.valid:
            errors += 1
            print(f"[ERROR] Invalid IR: {json_file}")
            for e in result.errors:
                print(f"        - {e}")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    print("=" * 60)
    print(f"[SUMMARY] IR files validated : {total}")
    print(f"[SUMMARY] Invalid IR files  : {errors}")

    if errors == 0:
        print("[OK] All baseline IR files are valid.")
    else:
        print("[WARN] Baseline generated with validation errors.")

    print("=" * 60)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
