from nl2spec.prompts.build_prompt import build_prompt
from nl2spec.handlers.fewshot_loader import FewShotLoader

def test_prompt_contains_required_sections():
    loader = FewShotLoader(
        fewshot_dir="datasets/fewshot",
        schema_path="nl2spec/schemas/ir.schema.json"
    )

    examples = loader.sample("FSM", k=1)

    prompt = build_prompt(
        scenario_text="A socket must be configured before sending data.",
        fewshot_examples=examples,
        schema_path="nl2spec/schemas/ir.schema.json"
    )

    assert "IR Schema" in prompt
    assert "Example 1" in prompt
    assert "socket must be configured" in prompt.lower()
