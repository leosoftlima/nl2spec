import json
import csv
import time
from pathlib import Path

from nl2spec.core.inspection.validate_ir import IRValidator
from nl2spec.logging_utils import get_logger
from nl2spec.core.llms.factory.llm_factory import create_llm

log = get_logger(__name__)


def stage_llm(ctx, flags):
    log.info("Stage: llm")

    cfg = ctx.config
    prompts_root = Path("nl2spec/prompts/generated")
    schema_path = cfg["paths"]["schema_ir"]

    if not prompts_root.exists():
        raise RuntimeError(
            f"Prompt directory not found: {prompts_root}\n"
            "Run with -g first."
        )

    llm = create_llm(cfg)
    model_name = llm.model if hasattr(llm, "model") else llm.__class__.__name__
    provider_name = cfg["llm"]["provider"]

    output_root = Path("artifacts") / provider_name / model_name / "generated_ir"
    stats_dir = Path("artifacts/statistics")
    stats_dir.mkdir(parents=True, exist_ok=True)

    csv_path = stats_dir / "generation_time.csv"
    write_header = not csv_path.exists()

    validator = IRValidator(schema_path)

    total = 0

    # Abrimos CSV uma única vez
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "id",
                "provider",
                "model",
                "domain",
                "ir_type",
                "start_timestamp",
                "end_timestamp",
                "generation_time_ms"
            ])

        for ir_type_dir in ["event", "fsm", "ere", "ltl"]:

            type_path = prompts_root / ir_type_dir.upper()
            if not type_path.exists():
                log.warning("Prompt type folder missing: %s", type_path)
                continue

            log.info("Processing prompt type: %s", ir_type_dir)

            for prompt_file in type_path.glob("*.txt"):

                try:
                    spec_id = prompt_file.stem
                    prompt = prompt_file.read_text(encoding="utf-8")

                    log.info("Calling LLM for %s (%s)", spec_id, ir_type_dir)

                    start_ts = time.time()
                    raw = llm.generate(prompt)
                    end_ts = time.time()

                    elapsed_ms = int((end_ts - start_ts) * 1000)

                    ir = json.loads(raw)

                    result = validator.validate_dict(ir)

                    if not result.valid:
                        log.error("Invalid IR for %s", spec_id)
                        for e in result.errors:
                            log.error("  - %s", e)
                        continue

                    # Extrai domínio do próprio IR
                    domain = ir.get("domain", "unknown")

                    # Salva por tipo
                    target_dir = output_root / ir_type_dir
                    target_dir.mkdir(parents=True, exist_ok=True)

                    out_file = target_dir / f"{spec_id}.json"
                    out_file.write_text(
                        json.dumps(ir, indent=2),
                        encoding="utf-8"
                    )

                    # Escreve estatística
                    writer.writerow([
                        spec_id,
                        provider_name,
                        model_name,
                        domain,
                        ir_type_dir,
                        start_ts,
                        end_ts,
                        elapsed_ms
                    ])

                    total += 1

                except Exception as e:
                    log.error("LLM failed for %s: %s", prompt_file, e)

    if hasattr(llm, "close"):
        llm.close()

    log.info(
        "LLM stage finished (%d IRs generated) | Provider: %s | Model: %s",
        total,
        provider_name,
        model_name
    )