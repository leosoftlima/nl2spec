from pathlib import Path
import yaml
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

def _strip_nl2spec_prefix(p: str) -> str:
    p = p.replace("\\", "/")
    if p.startswith("nl2spec/"):
        return p[len("nl2spec/"):]
    return p

def _resolve(base: Path, p: str) -> str:
    if not p:
        return p
    p2 = _strip_nl2spec_prefix(p)
    pp = Path(p2)
    if pp.is_absolute():
        return str(pp)
    return str((base / pp).resolve())

def load_config(config_path: str):
    # 1) encontra o config de forma robusta
    raw = Path(config_path)
    if raw.is_absolute() and raw.exists():
        cfg_path = raw
    else:
        # tenta direto (relativo ao CWD)
        if raw.exists():
            cfg_path = raw
        else:
            # fallback compatível com o seu comportamento antigo
            cfg_path = (Path("nl2spec") / raw)

    log.info("Loading config")
    log.info("Raw path argument = %s", config_path)
    log.info("Resolved path = %s", cfg_path.resolve())
    log.info("Exists? %s", cfg_path.exists())

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path.resolve()}\n"
            f"Hint: run with --config nl2spec/config.yaml"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) base para resolver paths: a pasta onde está o config
    # No seu caso, config fica em nl2spec/config.yaml -> base = .../nl2spec
    base = cfg_path.parent.resolve()
    cfg["_config_dir"] = str(base)

    # 3) normaliza os caminhos que seu pipeline usa
    paths = cfg.get("paths", {}) or {}
    cfg["paths"] = paths

    paths["baseline_nl"] = _resolve(base, paths.get("baseline_nl", "datasets/baseline_nl"))
    paths["baseline_ir"] = _resolve(base, paths.get("baseline_ir", "datasets/baseline_ir"))
    paths["output_dir"]  = _resolve(base, paths.get("output_dir", "outputs/"))
    paths["schema_ir"]   = _resolve(base, paths.get("schema_ir", "core/schemas/ir.schema.json"))

    # prompting fewshot dataset
    prompting = cfg.get("prompting", {}) or {}
    cfg["prompting"] = prompting
    fewshot = prompting.get("fewshot", {}) or {}
    prompting["fewshot"] = fewshot
    fewshot["dataset_dir"] = _resolve(base, fewshot.get("dataset_dir", "datasets/fewshot"))

    return cfg