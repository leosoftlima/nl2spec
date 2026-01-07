from nl2spec.pipeline.generate import generate_one
from nl2spec.handlers.fewshot_loader import FewShotLoader


class MockLLM:
    def generate(self, prompt: str) -> str:
        return """
        {
          "category": "EVENT",
          "ir": {
            "type": "single_event",
            "events": [
              { "name": "send", "timing": "before" }
            ],
            "guard": "socket not configured",
            "violation_message": "Socket used before configuration."
          }
        }
        """


def test_generate_one_with_mock_llm():
    loader = FewShotLoader(
        fewshot_dir="datasets/fewshot",
        schema_path="nl2spec/schemas/ir.schema.json"
    )

    examples = loader.sample(category="EVENT", k=1)

    llm = MockLLM()

    ir = generate_one(
        scenario_text="A socket must be configured before sending data.",
        fewshot_examples=examples,
        llm=llm,
        schema_path="nl2spec/schemas/ir.schema.json"
    )

    assert ir["category"] == "EVENT"
    assert ir["ir"]["type"] == "single_event"

