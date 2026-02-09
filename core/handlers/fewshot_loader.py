from pathlib import Path
from typing import List
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

class FewShotLoader:
    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.root = Path(fewshot_dir)
        self.seed = seed

        log.info("Few-shot root directory: %s", self.root)

    def get(self, ir_type: str, k: int) -> List[Path]:
        ir_type = ir_type.lower()
        ir_dir = self.root / ir_type

        log.info(
            "Resolving few-shot examples for IR type '%s' (k=%d)",
            ir_type, k,
        )

        if not ir_dir.exists():
            log.warning(
                "Few-shot directory not found for IR type '%s' (%s). Using zero-shot.",
                ir_type, ir_dir,
            )
            return []

        files = sorted(ir_dir.glob("*.json"))

        if not files:
            log.warning(
                "Few-shot directory for IR type '%s' exists but has no JSON files. Using zero-shot.",
                ir_type,
            )
            return []

        selected = files[:k]

        log.info(
            "Selected %d few-shot example(s) for IR type '%s'",
            len(selected),
            ir_type,
        )

        return selected