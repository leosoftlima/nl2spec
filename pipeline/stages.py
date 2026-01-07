from __future__ import annotations

import subprocess
from pathlib import Path

from nl2spec.pipeline_types import PipelineFlags
from nl2spec.logging_utils import get_logger
from nl2spec.pipeline.generate import generate_one
from nl2spec.core.llms.factory.llm_factory import create_llm

log = get_logger(__name__)


# ==========================================================
# GENERATE STAGE
# ==========================================================
def stage_generate(ctx, flags: PipelineFlags) -> None:
    log.info("Stage: generate")

    if not flags.generate:
        log.info("Generate stage skipped")
        return
    log.info("CONFIG LOADED = %s", ctx.config)

    cfg = ctx.config

    #  LLM definido EXCLUSIVAMENTE pelo config.yaml
    llm = create_llm(cfg)

    #  aqui ainda é 1 cenário (ok para agora)
    ir = generate_one(
        scenario_text="A file must be closed after being opened.",
        llm=llm,
        fewshot_examples=[],
        schema_path=cfg["paths"]["schema_ir"],
    )

    ctx.artifacts["generated_ir"] = ir

    log.info("Generate stage completed")


# ==========================================================
# COMPARE STAGE
# ==========================================================
def stage_compare(ctx, flags: PipelineFlags) -> None:
    log.info("Stage: compare")

    if not flags.compare:
        log.info("Compare stage skipped")
        return

    # Placeholder correto (depende do generate completo)
    # Aqui você vai:
    # - carregar baseline_ir
    # - comparar IR vs IR
    # - salvar diff estruturado

    ctx.artifacts["diff"] = []

    log.info("Compare stage completed")


# ==========================================================
# CSV EXPORT STAGE
# ==========================================================
def stage_export_csv(ctx, flags: PipelineFlags) -> None:
    if not flags.csv:
        return

    log.info("Stage: csv export")

    # TODO:
    # - converter ctx.artifacts["diff"] em CSV
    # - salvar em cfg["paths"]["output_dir"]

    log.info("CSV export completed")


# ==========================================================
# STATS STAGE
# ==========================================================
def stage_stats(ctx, flags: PipelineFlags) -> None:
    if not flags.stats:
        return

    log.info("Stage: stats")

    # TODO:
    # - estatísticas sobre diffs
    # - agregações
    # - métricas finais

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
