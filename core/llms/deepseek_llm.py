from openai import OpenAI
from nl2spec.core.llms.base import BaseLLM


class DeepSeekLLM(BaseLLM):

    def __init__(self, api_key: str, model: str):
        self.model = model
        # URL base da API da DeepSeek
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate(self, prompt: str) -> tuple[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip(), response.usage

    def close(self):
        # Mantido por compatibilidade com a classe base
        pass