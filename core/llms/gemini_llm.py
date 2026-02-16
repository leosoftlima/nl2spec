from google import genai
from nl2spec.core.llms.base import BaseLLM


class GeminiLLM(BaseLLM):

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text.strip()

    def close(self):
        # libera conexões HTTP internas
        self.client.close()
