from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    p = Path(path)

    if not p.is_absolute():
        # agora resolve relativo ao pacote nl2spec
        project_root = Path(__file__).resolve().parent
        p = project_root / p

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


