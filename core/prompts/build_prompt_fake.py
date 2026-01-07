from core.prompts.build_prompt import build_prompt

scenario = "A file must be closed after being opened."

fewshot_examples = [
    {
        "scenario": "A socket must be configured before sending data.",
        "ir": {
            "id": "net/Socket_SetTimeout",
            "category": "EVENT",
            "events": []
        }
    }
]

prompt = build_prompt(
    scenario_text=scenario,
    fewshot_examples=fewshot_examples,
    schema_path="core/schemas/ir.schema.json"
)

print(prompt)
