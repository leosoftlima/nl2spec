import csv
from pathlib import Path


def load_llm_info(provider: str, csv_path: str) -> dict:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"LLM info file not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["provider"].strip().lower() == provider.lower():
                return {
                    "provider": row["provider"],
                    "model": row["model"],
                    "api_key": row["api_key"],
                    "base_url": row.get("base_url", "")
                }

    raise ValueError(f"Provider '{provider}' not found in {csv_path}")