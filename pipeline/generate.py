import json
from typing import Dict, Any

from core.inspection.validate_ir import IRValidator
from core.prompts.build_prompt import build_prompt


class GenerationError(Exception):
    """Raised when IR generation fails."""


def generate_one(
    scenario_text: str,
    fewshot_examples: list,
    llm,
    schema_path: str,
    instruction: str = None
) -> Dict[str, Any]:
    """
    Generate a single IR from a natural language scenario.

    Parameters:
    - scenario_text: Natural language description
    - fewshot_examples: List of validated IR examples
    - llm: An object implementing generate(prompt: str) -> str
    - schema_path: Path to the IR schema
    - instruction: Optional custom instruction text

    Returns:
    - IR dictionary (validated)

    Raises:
    - GenerationError if generation or validation fails
    """

    # ------------------------------------------------------------------
    # 1. Build prompt
    # ------------------------------------------------------------------
    prompt = build_prompt(
        scenario_text=scenario_text,
        fewshot_examples=fewshot_examples,
        schema_path=schema_path,
        instruction=instruction
    )

    # ------------------------------------------------------------------
    # 2. Call LLM (mock or real)
    # ------------------------------------------------------------------
    try:
        raw_output = llm.generate(prompt)
    except Exception as e:
        raise GenerationError(f"LLM call failed: {e}")

    # ------------------------------------------------------------------
    # 3. Parse JSON
    # ------------------------------------------------------------------
    try:
        ir = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise GenerationError(f"Generated output is not valid JSON: {e}")

    # ------------------------------------------------------------------
    # 4. Validate IR
    # ------------------------------------------------------------------
    validator = IRValidator(schema_path)
    result = validator.validate_dict(ir)

    if not result.valid:
        raise GenerationError(
            "Generated IR does not conform to schema:\n"
            + "\n".join(result.errors)
        )

    return ir
