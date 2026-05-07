from google import genai
from nl2spec.core.llms.base import BaseLLM


class GeminiLLM(BaseLLM):

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> tuple[str, object]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={'temperature' : 0}
        )

        return response.text.strip(), response.usage_metadata

    def close(self):
        # libera conexões HTTP internas
        self.client.close()
