import json
import time
from typing import Dict, Any

from nl2spec.core.inspection.validate_ir import IRValidator
from nl2spec.prompts.build_prompt import build_prompt


class GenerationError(Exception):
    pass


def generate_one(
    scenario: Dict[str, Any],
    ir_type: str,
    fewshot_files: list,
    llm,
    schema_path: str
) -> Dict[str, Any]:

    # 1. Build prompt
   # prompt = build_prompt(
   #   ir_type=ir_type,
   #   nl_text=scenario["natural_language"],
   #   fewshot_files=fewshot_files
   # )

    prompt = build_prompt(
      ir_type=ir_type,
      nl_text=scenario["natural_language"],
      fewshot_files=fewshot_files,
      scenario_id=scenario["id"],
      save=True,
    )

    # 2. Call LLM
    start = time.time()
    try:
        raw_output = llm.generate(prompt)
    except Exception as e:
        raise GenerationError(f"LLM call failed: {e}")
    elapsed_ms = int((time.time() - start) * 1000)

    # 3. Parse JSON
    try:
        ir = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise GenerationError(f"Generated output is not valid JSON: {e}")

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
        "prompt": prompt,
        "generation_time_ms": elapsed_ms
    }
