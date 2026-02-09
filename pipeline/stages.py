from __future__ import annotations

import subprocess
import json
import random
from pathlib import Path
from typing import List

from nl2spec.pipeline_types import PipelineFlags
from nl2spec.logging_utils import get_logger
from nl2spec.pipeline.generate import generate_one
from nl2spec.core.llms.factory.llm_factory import create_llm
from nl2spec.pipeline.nl_loader import load_nl_scenarios_by_domain
from nl2spec.prompts.build_prompt import build_prompt

log = get_logger(__name__)

# ==========================================================
# FEW-SHOT LOADER (VOCÊ JÁ TEM ESSE CÓDIGO)
# ==========================================================
log = get_logger(__name__)

class FewShotLoader:
    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.root = Path(fewshot_dir)
        self.seed = seed
        self._rng = random.Random(seed)

        log.info("Few-shot root directory: %s", self.root)
        log.info("Few-shot random seed: %d", self.seed)

    def get(self, ir_type: str, k: int) -> List[Path]:
        ir_type = ir_type.lower()
        ir_dir = self.root / ir_type

        log.info(
            "Resolving few-shot examples for IR type '%s' (k=%d)",
            ir_type, k,
        )

        if not ir_dir.exists():
            log.warning(
                "Few-shot directory not found for IR type '%s' (%s). Using zero-shot.",
                ir_type, ir_dir,
            )
            return []

        files = sorted(ir_dir.glob("*.json"))

        if not files:
            log.warning(
                "Few-shot directory for IR type '%s' exists but has no JSON files. Using zero-shot.",
                ir_type,
            )
            return []

        total = len(files)

        # Caso 1: menos exemplos do que k → usa todos
        if total <= k:
            selected = files
            log.info(
                "Only %d few-shot example(s) available for IR type '%s'. Using all.",
                total, ir_type,
            )
        else:
            # Caso 2: mais exemplos que k → amostragem aleatória
            selected = self._rng.sample(files, k)
            log.info(
                "Randomly selected %d out of %d few-shot example(s) for IR type '%s'.",
                k, total, ir_type,
            )

        # Log detalhado (útil para debug/reprodutibilidade)
        for p in selected:
            log.debug("  few-shot: %s", p.name)

        return selected


# ==========================================================
# GENERATE STAGE
# ==========================================================
def stage_generate(ctx, flags: PipelineFlags) -> None:
    log.info("Stage: generate")

    if not flags.generate:
        log.info("Generate stage skipped")
        return

    cfg = ctx.config
    llm = create_llm(cfg)

    log.info("Using LLM: %s", llm.__class__.__name__)

    # instanciar loader de few-shot (UMA VEZ)
    fewshot_loader = FewShotLoader(
        fewshot_dir=cfg["prompting"]["fewshot"]["dataset_dir"],
        seed=int(cfg.get("seed", 42)),
    )

    scenarios_by_domain = load_nl_scenarios_by_domain(
        cfg["paths"]["baseline_nl"]
    )

    baseline_ir_root = Path(cfg["paths"]["baseline_ir"])
    generated = []

    for domain, scenarios in scenarios_by_domain.items():
        log.info(
            "Processing domain '%s' (%d scenarios)",
            domain, len(scenarios)
        )

        for scenario in scenarios:
            sid = scenario["id"]

            try:
                log.info("Generating IR for scenario '%s'", sid)

                # inferir ir_type via baseline IR
                gt_path = baseline_ir_root / domain / f"{sid}.json"
                if not gt_path.exists():
                    raise FileNotFoundError(
                        f"Baseline IR not found: {gt_path}"
                    )

                gt = json.loads(gt_path.read_text(encoding="utf-8"))
                ir_type = gt["ir"]["type"].lower()

                log.info("IR type resolved: %s", ir_type)

                # resolver few-shot PELO LOADER QUE VOCÊ TEM
                k = int(cfg["prompting"].get("k", 0))
                fewshot_files = fewshot_loader.get(
                    ir_type=ir_type,
                    k=k,
                )

                # gerar IR (generate_one já usa build_prompt internamente)
                result = generate_one(
                    scenario=scenario,
                    ir_type=ir_type,
                    fewshot_files=fewshot_files,
                    llm=llm,
                    schema_path=cfg["paths"]["schema_ir"],
                )

                # metadados
                result["id"] = sid
                result["domain"] = domain
                result["ir_type"] = ir_type

                generated.append(result)

            except Exception as e:
                log.error(
                    "Generation failed for %s/%s: %s",
                    domain, sid, e,
                )

    ctx.artifacts["generated_ir"] = generated
    log.info("Generate stage finished (%d IRs)", len(generated))


# ==========================================================
# COMPARE STAGE
# ==========================================================
def stage_compare(ctx, flags: PipelineFlags) -> None:
    log.info("Stage: compare")

    if not flags.compare:
        log.info("Compare stage skipped")
        return

    # Placeholder correto
    ctx.artifacts["diff"] = []

    log.info("Compare stage completed")


# ==========================================================
# CSV EXPORT STAGE
# ==========================================================
def stage_export_csv(ctx, flags: PipelineFlags) -> None:
    if not flags.csv:
        return

    log.info("Stage: csv export")
    log.info("CSV export completed")


# ==========================================================
# STATS STAGE
# ==========================================================
def stage_stats(ctx, flags: PipelineFlags) -> None:
    if not flags.stats:
        return

    log.info("Stage: stats")
    log.info("Stats stage completed")


# ==========================================================
# TEST STAGE (pytest)
# ==========================================================
def stage_tests(ctx, flags: PipelineFlags) -> None:
    log.info("Stage: tests")

    project_root = Path(__file__).resolve().parents[2]
    tests_dir = project_root / "nl2spec" / "tests"

    args = ["pytest"]

    if flags.generate:
        args.append(str(tests_dir / "test_manual_generate.py"))

    if flags.compare:
        args.append(str(tests_dir / "test_manual_compare.py"))

    if not flags.generate and not flags.compare:
        args.append(str(tests_dir))

    log.info("Running tests: %s", " ".join(args))
    subprocess.run(args, check=True)

    log.info("Tests completed successfully")