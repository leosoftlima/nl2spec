from core.prompts.build_prompt import build_prompt

def test_prompt_contains_scenario():
    scenario = "A file must be closed after being opened."
    examples = [{"input": "x", "output": {"y": 1}}]

    prompt = build_prompt(
        scenario=scenario,
        fewshot_examples=examples,
        schema={"type": "object"}
    )

    assert scenario in prompt
    assert "JSON" in prompt
