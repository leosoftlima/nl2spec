import json
import time
from typing import Dict, Any

from nl2spec.core.inspection.validate_ir import IRValidator


class GenerationError(Exception):
    pass


def generate_one(
    prompt: str,
    llm,
    schema_path: str
) -> Dict[str, Any]:

    # 1. Call LLM
    start_ts = time.time()

    try:
        raw_output = llm.generate(prompt)
    except Exception as e:
        raise GenerationError(f"LLM call failed: {e}")

    end_ts = time.time()
    elapsed_ms = int((end_ts - start_ts) * 1000)

    # 2. Clean possible markdown fences
    raw_output = raw_output.strip()

    if raw_output.startswith("```"):
        parts = raw_output.split("```")
        if len(parts) >= 2:
            raw_output = parts[1].strip()

    # 3. Parse JSON
    try:
        ir = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise GenerationError(
            f"Generated output is not valid JSON:\n{raw_output}\nError: {e}"
        )

    # 4. Validate IR
    validator = IRValidator(schema_path)
    result = validator.validate_dict(ir)

    if not result.valid:
        raise GenerationError(
            "Generated IR does not conform to schema:\n"
            + "\n".join(result.errors)
        )

    return {
        "ir": ir,
        "generation_time_ms": elapsed_ms,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }