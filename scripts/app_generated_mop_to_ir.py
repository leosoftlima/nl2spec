from pathlib import Path
import shutil
import sys
import json

from core.convert.mop_to_ir import convert_mop_file_to_ir, detect_formalism


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MOP_ROOT = PROJECT_ROOT / "datasets" / "raw_mop"
OUT_ROOT = PROJECT_ROOT / "datasets" / "baseline_ir_temp"


# ==========================================================
# UTILS
# ==========================================================

def ask_overwrite(path: Path) -> bool:
    answer = input(
        f"[WARN] Output directory exists:\n"
        f"       {path}\n"
        f"Delete and regenerate? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("[INFO] Baseline IR Test Runner (LTL + FSM + ERE + EVENT)")
    print("[INFO] Source :", MOP_ROOT)
    print("[INFO] Output :", OUT_ROOT)
    print("=" * 70)

    if not MOP_ROOT.exists():
        print("[ERROR] raw_mop directory not found.")
        sys.exit(1)

    if OUT_ROOT.exists():
        if not ask_overwrite(OUT_ROOT):
            print("[INFO] Aborted.")
            return
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    converted_ltl = 0
    converted_fsm = 0
    converted_ere = 0
    converted_event = 0

    SUPPORTED = {"ltl", "fsm", "ere", "event"}

    all_files = list(MOP_ROOT.rglob("*.mop"))
    total_files = len(all_files)

    for mop_file in all_files:
        text = mop_file.read_text(encoding="utf-8", errors="replace")
        formalism = detect_formalism(text)

        if formalism not in SUPPORTED:
            continue

        print(f"Processing {formalism.upper()}:", mop_file)

        try:
            ir = convert_mop_file_to_ir(mop_file)

            relative = mop_file.relative_to(MOP_ROOT)
            target = OUT_ROOT / relative.with_suffix(".json")
            target.parent.mkdir(parents=True, exist_ok=True)

            with open(target, "w", encoding="utf-8") as f:
                json.dump(ir, f, indent=2, ensure_ascii=False)

            if formalism == "ltl":
                converted_ltl += 1
            elif formalism == "fsm":
                converted_fsm += 1
            elif formalism == "ere":
                converted_ere += 1
            elif formalism == "event":
                converted_event += 1

        except Exception as e:
            print(f"[ERROR] {mop_file}")
            print(f"        {e}")

    total_converted = (
        converted_ltl
        + converted_fsm
        + converted_ere
        + converted_event
    )

    print("=" * 70)
    print("[SUMMARY]")
    print(f"  Converted LTL   : {converted_ltl}")
    print(f"  Converted FSM   : {converted_fsm}")
    print(f"  Converted ERE   : {converted_ere}")
    print(f"  Converted EVENT : {converted_event}")
    print("-" * 70)
    print(f"  Total Converted : {total_converted}")
    print(f"  Total Files     : {total_files}")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()