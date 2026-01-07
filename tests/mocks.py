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
