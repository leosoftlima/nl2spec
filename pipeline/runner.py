from __future__ import annotations

from dataclasses import asdict

from nl2spec.config import load_config
from nl2spec.pipeline.stages import (
    stage_generate,
    stage_compare,
    stage_export_csv,
    stage_tests
)
from nl2spec.pipeline_types import PipelineContext, PipelineFlags
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)


def run_pipeline(config_path: str, flags: PipelineFlags) -> None:
    cfg = load_config(config_path)
    ctx = PipelineContext(config=cfg, artifacts={})

    log.info("Pipeline start | flags=%s", asdict(flags))

    if flags.test:
        stage_tests(ctx, flags)
        log.info("Pipeline finished (tests).")
        return

    if flags.generate or flags.llm:
        stage_generate(ctx, flags)

    if flags.compare:
        stage_compare(ctx, flags)

    if flags.csv:
        stage_export_csv(ctx, flags)

    if flags.stats:
        stage_stats(ctx, flags)

    log.info("Pipeline finished.")
