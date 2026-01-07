# nl2spec/core/factory/llm_factory.py
import importlib

def load_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_llm(cfg: dict):
    provider = cfg["llm"]["provider"]
    llm_cfg = cfg["llm"][provider]

    cls_path = llm_cfg["class"]
    LLMClass = load_class(cls_path)

    params = {k: v for k, v in llm_cfg.items() if k != "class"}
    return LLMClass(**params)
