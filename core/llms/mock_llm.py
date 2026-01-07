import json


class MockLLM:
    """
    Fake LLM used for testing and dry-run execution.
    Returns a minimal valid IR according to ir.schema.json.
    """

    def generate(self, prompt: str) -> str:
        return json.dumps({
            "category": "EVENT",
            "ir": {
                "type": "single_event",
                "events": [
                    {
                        "name": "File.close",
                        "timing": "after"
                    }
                ],
                "guard": "true",
                "violation_message": "File must be closed after being opened."
            }
        })
