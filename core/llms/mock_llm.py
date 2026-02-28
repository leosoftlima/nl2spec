import json
import random


class MockLLM:
    """
    Fake LLM used for testing and dry-run execution.
    Returns a realistic IR structure compatible with ir.schema.json.
    """

    model = "mock-model"

    def generate(self, prompt: str) -> str:

        # Simples heurística para variar tipo
        if "fsm" in prompt.lower():
            ir_type = "fsm"
        elif "ere" in prompt.lower():
            ir_type = "ere"
        elif "ltl" in prompt.lower():
            ir_type = "ltl"
        else:
            ir_type = "event"

        domain = random.choice(["io", "util", "lang", "net"])

        base_ir = {
            "id": "MockSpec",
            "domain": domain,
            "category": ir_type.upper(),
            "ir": {}
        }

        if ir_type == "event":
            base_ir["ir"] = {
                "type": "event",
                "events": [
                    {"name": "File.open", "timing": "before"},
                    {"name": "File.close", "timing": "after"}
                ],
                "constraints": [
                    "File.close must occur after File.open"
                ],
                "violation_message": "File must be closed after being opened."
            }

        elif ir_type == "fsm":
            base_ir["ir"] = {
                "type": "fsm",
                "states": ["START", "OPENED", "CLOSED"],
                "initial_state": "START",
                "transitions": [
                    {"from": "START", "event": "File.open", "to": "OPENED"},
                    {"from": "OPENED", "event": "File.close", "to": "CLOSED"}
                ],
                "error_state": "ERROR",
                "violation_message": "Invalid file lifecycle transition."
            }

        elif ir_type == "ere":
            base_ir["ir"] = {
                "type": "ere",
                "pattern": "open close",
                "violation_message": "File operations must follow open then close."
            }

        elif ir_type == "ltl":
            base_ir["ir"] = {
                "type": "ltl",
                "formula": "G(open -> F close)",
                "violation_message": "Every open must eventually be followed by close."
            }

        return json.dumps(base_ir, indent=2)