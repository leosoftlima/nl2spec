import openai
from nl2spec.llms.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message["content"]
