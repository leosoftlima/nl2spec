import json
import random
from pathlib import Path
from typing import List, Optional, Dict

from nl2spec.inspection.validate_ir import IRValidator


class FewShotLoader:
    """
    Loader for few-shot IR examples.

    Responsibilities:
    - Load JSON IR files from a directory tree
    - Validate each IR against the canonical schema
    - Filter by category
    - Provide deterministic sampling
    """

    def __init__(
        self,
        fewshot_dir: str,
        schema_path: str,
        seed: int = 42
    ):
        self.fewshot_dir = Path(fewshot_dir)
        self.schema_path = schema_path
        self.seed = seed

        if not self.fewshot_dir.exists():
            raise FileNotFoundError(f"Few-shot directory not found: {self.fewshot_dir}")

        self.validator = IRValidator(schema_path)
        self._cache: List[Dict] = []

        self._load_all()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """
        Load and validate all JSON files under the few-shot directory.
        Invalid examples are skipped.
        """
        for json_file in self.fewshot_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            result = self.validator.validate_dict(data)
            if result.valid:
                data["_source"] = str(json_file)
                self._cache.append(data)

        if not self._cache:
            raise RuntimeError("No valid few-shot examples found.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def all(self) -> List[Dict]:
        """
        Return all loaded few-shot examples.
        """
        return list(self._cache)

    def by_category(self, category: str) -> List[Dict]:
        """
        Return all few-shot examples matching a category.
        """
        return [
            ex for ex in self._cache
            if ex.get("category") == category
        ]

    def sample(
        self,
        category: Optional[str] = None,
        k: int = 1
    ) -> List[Dict]:
        """
        Deterministically sample k examples.

        If category is None, samples from all examples.
        """
        rng = random.Random(self.seed)

        pool = self._cache
        if category:
            pool = self.by_category(category)

        if not pool:
            raise ValueError(f"No few-shot examples found for category: {category}")

        if k >= len(pool):
            return list(pool)

        return rng.sample(pool, k)
