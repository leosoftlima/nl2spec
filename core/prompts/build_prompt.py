import json
from typing import List, Dict, Optional


def load_schema_text(schema_path: str) -> str:
     with open(schema_path, "r", encoding="utf-8") as f:
         schema = json.load(f)
     return json.dumps(schema, indent=2)




def format_fewshot_examples(examples: List[Dict]) -> str:
    """
    Format few-shot IR examples as prompt text.
    """
    blocks = []

    for i, ex in enumerate(examples, start=1):
        ir_text = json.dumps(ex, indent=2)
        block = (
            f"Example {i}:\n"
            f"{ir_text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(
    scenario_text: str,
    fewshot_examples: List[Dict],
    schema_path: str,
    instruction: Optional[str] = None
) -> str:
    """
    Build the final prompt sent to the LLM.

    Parameters:
    - scenario_text: natural language description of the scenario
    - fewshot_examples: list of IR dicts (validated)
    - schema_path: path to ir.schema.json
    - instruction: optional custom instruction header
    """
    if instruction is None:
        instruction = (
            "You are an expert in Java runtime verification.\n"
            "Your task is to convert the given natural language description "
            "into a JSON specification that strictly follows the provided IR schema.\n"
            "Return ONLY valid JSON. Do not include explanations or comments."
        )

    schema_text = load_schema_text(schema_path)

    fewshot_text = ""
    if fewshot_examples:
        fewshot_text = (
            "Below are examples of valid specifications expressed in the IR format:\n\n"
            f"{format_fewshot_examples(fewshot_examples)}"
        )

    prompt = (
        f"{instruction}\n\n"
        f"IR Schema:\n"
        f"{schema_text}\n\n"
        f"{fewshot_text}\n\n"
        f"Now generate the IR JSON for the following scenario:\n"
        f"{scenario_text}"
    )

    return prompt
