from pathlib import Path

from nl2spec.logging_utils import get_logger
from nl2spec.analysis import rq1_canonical_equivalence

log = get_logger(__name__)


def _list_available_providers(llm_root: Path):

    if not llm_root.exists():
        return []

    return [
        p.name for p in llm_root.iterdir()
        if p.is_dir()
    ]


def _has_generated_data(llm_root: Path):

    if not llm_root.exists():
        return False

    return any(llm_root.rglob("*.json"))


def stage_compare(ctx, flags):

    log.info("========== STAGE COMPARE START ==========")

    results_root = Path("nl2spec/output/results")
    results_root.mkdir(parents=True, exist_ok=True)

    llm_root = Path("nl2spec/output/llm")

    log.info("LLM root: %s", llm_root.resolve())

    # -------------------------------------------------
    # Check providers
    # -------------------------------------------------

    providers = _list_available_providers(llm_root)

    if not providers:
        log.warning("No providers found in output/llm. Nothing to analyze.")
        log.info("========== STAGE COMPARE END ==========")
        return

    log.info("Providers detected: %s", ", ".join(providers))

    # -------------------------------------------------
    # Check JSON data
    # -------------------------------------------------

    if not _has_generated_data(llm_root):
        log.warning("No generated JSON files found. Nothing to analyze.")
        log.info("========== STAGE COMPARE END ==========")
        return

    # -------------------------------------------------
    # Run RQ1
    # -------------------------------------------------

    try:
        log.info("Running RQ1 - Canonical Equivalence")
        rq1_canonical_equivalence.run(ctx, results_root)
        log.info("RQ1 completed successfully.")

    except Exception as e:
        log.exception("Stage compare failed: %s", e)
        raise

    log.info("========== STAGE COMPARE END ==========")