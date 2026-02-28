import json
import csv
import time
from pathlib import Path

from nl2spec.core.inspection.validate_ir import IRValidator
from nl2spec.logging_utils import get_logger
from nl2spec.core.llms.factory.llm_factory import create_llm

log = get_logger(__name__)


# =====================================================
# CONFIG EXTRACTION
# =====================================================

def _extract_config(cfg):
    provider = cfg["llm"]["provider"]
    shot_mode = cfg["prompting"]["shot_mode"]
    k = cfg["prompting"]["k"]
    selection = cfg["prompting"]["fewshot"]["selection"]

    if shot_mode == "zero":
        selection = "none"

    schema_path = cfg["paths"]["schema_ir"]

    log.info("LLM Configuration:")
    log.info("  Provider  : %s", provider)
    log.info("  Shot mode : %s", shot_mode)
    log.info("  k         : %s", k)
    log.info("  Selection : %s", selection)

    return provider, shot_mode, k, selection, schema_path


# =====================================================
# PROMPT RESOLUTION
# =====================================================

def _resolve_prompts_root(selection, shot_mode):
    root = Path("nl2spec/output/prompt") / selection / shot_mode
    log.info("Looking for prompts in: %s", root.resolve())
    return root


def _validate_prompts(prompts_root):
    if not prompts_root.exists():
        raise RuntimeError(f"Prompt directory not found: {prompts_root}")

    txt_files = list(prompts_root.rglob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No prompt files found in {prompts_root}")

    log.info("Total prompt files found: %d", len(txt_files))


# =====================================================
# OUTPUT STRUCTURE
# =====================================================

def _build_output_root(provider, model, selection, shot_mode, k):
    output_root = (
        Path("output")
        / "llm"
        / provider
        / model
        / selection
        / f"{shot_mode}_k{k}"
    )
    log.info("Generated IRs will be saved to: %s", output_root.resolve())
    return output_root


# =====================================================
# LLM CALL
# =====================================================

def _call_llm(llm, prompt, spec_id):
    start = time.time()
    raw = llm.generate(prompt)
    end = time.time()

    elapsed_ms = int((end - start) * 1000)

    if not raw:
        raise RuntimeError(f"Empty response for {spec_id}")

    try:
        ir = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON for {spec_id}: {e}")

    return ir, elapsed_ms


# =====================================================
# PROCESS PROMPTS BY TYPE
# =====================================================

def _process_prompt_type(
    ir_type,
    prompts_root,
    llm,
    validator,
    output_root,
    writer,
    provider,
    model,
    selection,
    shot_mode,
    k
):
    type_path = prompts_root / ir_type.upper()

    if not type_path.exists():
        log.warning("Prompt type folder missing: %s", type_path)
        return 0

    log.info("Processing prompt type: %s", ir_type)

    total = 0

    for prompt_file in type_path.glob("*.txt"):
        spec_id = prompt_file.stem
        prompt = prompt_file.read_text(encoding="utf-8")

        try:
            log.info("Calling LLM for %s (%s)", spec_id, ir_type)

            ir, elapsed_ms = _call_llm(llm, prompt, spec_id)

            result = True #validator.validate_dict(ir)
            if not result.valid:
                log.error("Invalid IR for %s", spec_id)
                for err in result.errors:
                    log.error("  - %s", err)
                continue

            domain = ir.get("domain", "unknown")

            target_dir = (
                output_root
                / f"domain_{domain}"
                / ir_type
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            out_file = target_dir / f"{spec_id}.json"
            out_file.write_text(json.dumps(ir, indent=2), encoding="utf-8")

            writer.writerow([
                spec_id,
                provider,
                model,
                selection,
                shot_mode,
                k,
                domain,
                ir_type,
                elapsed_ms
            ])

            total += 1

        except Exception as e:
            log.error("LLM failed for %s: %s", spec_id, e)

    return total


# =====================================================
# MAIN STAGE
# =====================================================

def stage_llm(ctx, flags):
    log.info("Stage: llm")

    cfg = ctx.config

    provider, shot_mode, k, selection, schema_path = _extract_config(cfg)

    prompts_root = _resolve_prompts_root(selection, shot_mode)
    _validate_prompts(prompts_root)

    llm = create_llm(cfg)
    model_name = getattr(llm, "model", llm.__class__.__name__)

    #validator = IRValidator(schema_path)

    output_root = _build_output_root(
        provider,
        model_name,
        selection,
        shot_mode,
        k
    )

    stats_root = Path("output/statistics")
    stats_root.mkdir(parents=True, exist_ok=True)

    csv_path = stats_root / "generation_time.csv"
    write_header = not csv_path.exists()

    total = 0

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
                "generation_time_ms"
            ])

        for ir_type in ["event", "fsm", "ere", "ltl"]:
            total += _process_prompt_type(
                ir_type,
                prompts_root,
                llm,
                "",
                output_root,
                writer,
                provider,
                model_name,
                selection,
                shot_mode,
                k
            )

    if hasattr(llm, "close"):
        llm.close()

    log.info(
        "LLM stage finished (%d IRs generated) | Provider: %s | Model: %s | Strategy: %s | Shot: %s | k=%s",
        total,
        provider,
        model_name,
        selection,
        shot_mode,
        k
    )