from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random

from nl2spec.core.handlers.fsm_selector import FSMFewShotSelector
from nl2spec.core.handlers.ere_selector import EREFewShotSelector
from nl2spec.core.handlers.ltl_selector import LTLFewShotSelector
from nl2spec.core.handlers.event_selector import EventFewShotSelector
from nl2spec.logging_utils import get_logger
from nl2spec.exceptions import FewShotNotAvailableError, NL2SpecException

log = get_logger(__name__)

ALLOWED_K = {0, 1, 3}
ALLOWED_SELECTION = {"random", "structural"}


class FewShotLoader:

    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.root = Path(fewshot_dir)
        self.seed = seed
        random.seed(seed)

        self.fsm_selector = FSMFewShotSelector()
        self.ere_selector = EREFewShotSelector()
        self.ltl_selector = LTLFewShotSelector()
        self.event_selector = EventFewShotSelector()

        log.info("Few-shot root directory: %s", self.root)

    # =========================================================
    # Configuration Validation
    # =========================================================

    def _validate_configuration(self, shot_mode: str, k: int, selection: str):
        if k not in ALLOWED_K:
            raise NL2SpecException(f"Invalid k={k}. Allowed values are {ALLOWED_K}.")

        if shot_mode == "zero" and k != 0:
            raise NL2SpecException("shot_mode='zero' requires k=0.")

        if shot_mode == "one" and k != 1:
            raise NL2SpecException("shot_mode='one' requires k=1.")

        if shot_mode == "few" and k < 2:
            raise NL2SpecException("shot_mode='few' requires k >= 2.")

        if selection not in ALLOWED_SELECTION:
            raise NL2SpecException(
                f"Invalid selection='{selection}'. Allowed values: {ALLOWED_SELECTION}"
            )

    # =========================================================
    # Public API
    # =========================================================

    def get(
        self,
        ir_type: str,
        shot_mode: str,
        k: int,
        selection: str,
        ir_base: Optional[dict] = None,
        return_scores: bool = False,
    ):
        self._validate_configuration(shot_mode, k, selection)

        if k == 0:
            return []

        ir_type = ir_type.lower()
        ir_dir = self.root / ir_type

        if not ir_dir.exists():
            raise FewShotNotAvailableError(str(ir_dir))

        files = sorted(ir_dir.glob("*.json"))
        if not files:
            raise FewShotNotAvailableError(str(ir_dir))

        # ===============================
        # RANDOM SELECTION (WITH DISTANCE)
        # ===============================
        if selection == "random":
            chosen = self._select_random(files, k)

            if not return_scores:
                return chosen

            if ir_base is None:
                raise NL2SpecException(
                    "Random selection with scoring requires ir_base."
                )

            if ir_type == "fsm":
                return self.fsm_selector.score_random(chosen, ir_base)

            if ir_type == "ere":
                return self.ere_selector.score_random(chosen, ir_base)

            if ir_type == "ltl":
                return self.ltl_selector.score_random(chosen, ir_base)

            if ir_type == "event":
                return self.event_selector.score_random(chosen, ir_base)

            raise NL2SpecException(
                f"Random scoring not implemented for ir_type '{ir_type}'."
            )

        # ===============================
        # STRUCTURAL SELECTION
        # ===============================
        if selection == "structural":
            if ir_base is None:
                raise NL2SpecException(
                    "Structural selection requires ir_base."
                )

            if ir_type == "fsm":
                return self.fsm_selector.select(files, k, ir_base, return_scores)

            if ir_type == "ere":
                return self.ere_selector.select(files, k, ir_base, return_scores)

            if ir_type == "ltl":
                return self.ltl_selector.select(files, k, ir_base, return_scores)

            if ir_type == "event":
                return self.event_selector.select(files, k, ir_base, return_scores)

            raise NL2SpecException(
                f"Structural selector not implemented for ir_type '{ir_type}'."
            )

        raise NL2SpecException("Unexpected selection mode.")

    # =========================================================
    # Random Selection
    # =========================================================

    def _select_random(self, files: List[Path], k: int) -> List[Path]:
        total = len(files)
        if k >= total:
            return files
        return random.sample(files, k)