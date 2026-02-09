from pathlib import Path
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

def resolve_fewshot_files(config, loader, ir_type: str):
    prompting = config.get("prompting", {}) or {}
    mode = (prompting.get("shot_mode") or "zero").lower()
    
    log.info("few shot mode: " + mode)

    if mode == "zero":
        return []

    k = 1 if mode == "one" else int(prompting.get("k", 2))

    ir_type = ir_type.lower()
    fewshot_dir = Path(loader.fewshot_dir) / ir_type

    if not fewshot_dir.exists():
        log.warning(
            "Few-shot directory for IR type '%s' not found (%s). "
            "Falling back to zero-shot.",
            ir_type, fewshot_dir
        )
        return []

    files = sorted(fewshot_dir.glob("*.json"))

    if not files:
        log.warning(
            "Few-shot directory for IR type '%s' exists but contains no JSON files. "
            "Using zero-shot.",
            ir_type
        )
        return []

    return files[:k]