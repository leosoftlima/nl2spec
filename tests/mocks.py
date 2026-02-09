class MockLLM:
    def generate(self, prompt: str) -> str:
        return """
        {
          "category": "EVENT",
          "ir": {
            "type": "event",
            "events": [
              { "name": "send", "timing": "before" }
            ],
            "guard": "socket not configured",
            "violation_message": "Socket used before configuration."
          }
        }
        """
