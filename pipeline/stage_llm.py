import json
import csv
import time
import shutil
from pathlib import Path

from nl2spec.logging_utils import get_logger
from nl2spec.core.llms.factory.llm_factory import create_llm

log = get_logger(__name__)

BASE_OUTPUT = Path("nl2spec/output")


def _call_llm(llm, prompt, spec_id, provider, model):
    start_ts = time.time()
    raw = llm.generate(prompt)
    end_ts = time.time()

    elapsed_ms = round((end_ts - start_ts) * 1000, 3)

    log.info(
        "[LLM] id=%s | provider=%s | model=%s | start=%.6f | end=%.6f | elapsed_ms=%.3f",
        spec_id, provider, model, start_ts, end_ts, elapsed_ms
    )

    if not raw:
        raise RuntimeError(f"Empty response for {spec_id}")

    try:
        ir = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON for {spec_id}: {e}")

    return ir, start_ts, end_ts, elapsed_ms


def stage_llm(ctx, flags):
    log.info("========== STAGE LLM START ==========")

    cfg = ctx.config

    provider = cfg["llm"]["provider"]
    shot_mode = cfg["prompting"]["shot_mode"]
    k = cfg["prompting"]["k"]
    selection = cfg["prompting"]["fewshot"]["selection"]

    if shot_mode == "zero":
        selection = "none"

    prompts_root = BASE_OUTPUT / "prompt" / selection / shot_mode
    if not prompts_root.exists():
        raise RuntimeError(f"Prompt directory not found: {prompts_root}")

    llm = create_llm(cfg)
    model_name = getattr(llm, "model", llm.__class__.__name__)

    log.info(
        "LLM CONFIG | provider=%s | model=%s | shot_mode=%s | selection=%s | k=%s",
        provider, model_name, shot_mode, selection, k
    )

    # ----------------------------
    # OUTPUT ROOT (CLEAN FIRST)
    # ----------------------------
    output_root = (
        BASE_OUTPUT
        / "llm"
        / provider
        / model_name
        / selection
        / f"{shot_mode}_k{k}"
    )

    if output_root.exists():
        log.info("Deleting previous LLM output folder: %s", output_root.resolve())
        shutil.rmtree(output_root)

    # ----------------------------
    # STATS CSV (CLEAN FIRST)
    # ----------------------------
    stats_root = BASE_OUTPUT / "statistics"
    stats_root.mkdir(parents=True, exist_ok=True)

    csv_path = stats_root / "generation_times.csv"

    if csv_path.exists():
        log.info("Deleting previous generation_times.csv: %s", csv_path.resolve())
        csv_path.unlink()

    # ✅ compute AFTER deleting
    write_header = True

    total = 0
    total_time = 0.0

    log.info("Saving statistics to: %s", csv_path.resolve())

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "id",
                "provider",
                "model",
                "selection",
                "shot_mode",
                "k",
                "domain",
                "ir_type",
                "start_timestamp",
                "end_timestamp",
                "generation_time_ms"
            ])
            f.flush()

        # ----------------------------
        # PROCESS ALL PROMPTS
        # ----------------------------
        for prompt_file in prompts_root.rglob("*.txt"):
            spec_id = prompt_file.stem
            ir_type = prompt_file.parent.name
            prompt = prompt_file.read_text(encoding="utf-8")

            log.info(
                "[PROMPT] file=%s | id=%s | type=%s | provider=%s | model=%s",
                str(prompt_file), spec_id, ir_type, provider, model_name
            )

            try:
                ir, start_ts, end_ts, elapsed_ms = _call_llm(
                    llm, prompt, spec_id, provider, model_name
                )

                domain = ir.get("domain", "unknown")

                target_dir = output_root / domain / ir_type
                target_dir.mkdir(parents=True, exist_ok=True)

                out_file = target_dir / f"{spec_id}.json"
                out_file.write_text(json.dumps(ir, indent=2), encoding="utf-8")

                writer.writerow([
                    spec_id,
                    provider,
                    model_name,
                    selection,
                    shot_mode,
                    k,
                    domain,
                    ir_type,
                    start_ts,
                    end_ts,
                    elapsed_ms
                ])
                f.flush()

                total += 1
                total_time += elapsed_ms

                log.info(
                    "[SAVED] id=%s | domain=%s | type=%s | elapsed_ms=%.3f | accumulated_ms=%.3f",
                    spec_id, domain, ir_type, elapsed_ms, total_time
                )

            except Exception as e:
                log.error("LLM failed for %s: %s", spec_id, e)

    if hasattr(llm, "close"):
        llm.close()

    log.info("========== STAGE LLM END ==========")
    log.info("TOTAL FILES PROCESSED: %d", total)
    log.info("TOTAL ACCUMULATED TIME (ms): %.3f", total_time)