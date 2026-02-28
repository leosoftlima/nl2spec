# python -m nl2spec.scripts.run_generated_prompt_to_llm

import yaml
from types import SimpleNamespace
from pathlib import Path
from nl2spec.config import load_config
from nl2spec.core.llms.stage_llm import stage_llm


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=== Running LLM Stage (manual execution) ===")

    cfg = load_config()
    ctx = SimpleNamespace(config=cfg)

   
    # Simula contexto mínimo esperado pelo stage
    ctx = SimpleNamespace(config=cfg)

    flags = {}  # se quiser futuramente usar flags

    stage_llm(ctx, flags)

    print("=== Execution finished ===")
    print("Check output/llm/ directory.")


if __name__ == "__main__":
    main()