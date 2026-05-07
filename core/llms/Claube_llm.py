import anthropic
from nl2spec.core.llms.base import BaseLLM


class ClaudeLLM(BaseLLM):

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> tuple[str, object]:

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.content[0].text.strip(), response.usage

    def close(self):
        pass