from nl2spec.core.llms.llm_registry import load_llm_info


def create_llm(cfg: dict):

    provider = cfg["llm"]["provider"]
    csv_path = cfg["llm"]["registry_csv"]

    info = load_llm_info(provider, csv_path)


    if provider == "gemini":
        from nl2spec.core.llms.gemini_llm import GeminiLLM
        return GeminiLLM(
            api_key=info["api_key"],
            model=info["model"],
        )

    if provider == "openai":
        from nl2spec.core.llms.openai_llm import OpenAILLM
        return OpenAILLM(
            api_key=info["api_key"],
            model=info["model"],
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")