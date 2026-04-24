from pathlib import Path

from nl2spec.core.convert.nl.mop_to_nl import MOPToNL

# run: 
# python -m nl2spec.scripts.run_generated_mop_to_nl

PROJECT_ROOT = Path(r"C:\UFPE\Siesta\project_LLM_spec")
MOP_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "dataset_mop"
NL_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_nl_temp"
TEMPLATE_DIR = PROJECT_ROOT / "nl2spec" / "prompts" / "templates"


def main():
    print("=" * 70)
    print("[INFO] MOP -> NL generation (interactive)")
    print("=" * 70)

    builder = MOPToNL(TEMPLATE_DIR)
    total = builder.generate_from_directory(MOP_ROOT, NL_ROOT)

    print(f"[OK] Generated {total} NL files in: {NL_ROOT}")


if __name__ == "__main__":
    main()