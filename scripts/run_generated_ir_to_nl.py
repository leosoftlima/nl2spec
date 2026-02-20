from pathlib import Path
import sys

from nl2spec.core.convert.ir_to_nl import IRToNL

#run: python -m nl2spec.scripts.run_generated_ir_to_nl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IR_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"
NL_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_nl"
TEMPLATE_DIR = PROJECT_ROOT / "nl2spec"  / "prompts" / "templates"


def ask_confirmation(path: Path) -> bool:
    answer = input(
        f"[WARN] The output directory will be deleted:\n"
        f"       {path}\n"
        f"Continue? [y/N]: "
    ).strip().lower()
    return answer in {"y", "yes"}


def main():
    print("=" * 70)
    print("[INFO] IR → NL generation (interactive)")
    print("=" * 70)

    builder = IRToNL(TEMPLATE_DIR)

    if NL_ROOT.exists():
        if not ask_confirmation(NL_ROOT):
            print("[INFO] Operation cancelled.")
            sys.exit(0)

    total = builder.generate_from_directory(IR_ROOT, NL_ROOT)

    print(f"[OK] Generated {total} NL files.")


if __name__ == "__main__":
    main()