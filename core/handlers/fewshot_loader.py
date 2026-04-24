from pathlib import Path
from typing import List, Optional
import random

from nl2spec.core.handlers.random import select_random
from nl2spec.core.handlers.mmr import select_mmr
from nl2spec.core.handlers.mmr_irsp import select_mmr_irsp

from nl2spec.core.handlers.irsp.fsm_irsp import FSMFewShotSelector
from nl2spec.core.handlers.irsp.ere_irsp import EREFewShotSelector
from nl2spec.core.handlers.irsp.ltl_irsp import LTLFewShotSelector
from nl2spec.core.handlers.irsp.event_irsp import EventFewShotSelector
from nl2spec.logging_utils import get_logger
from nl2spec.exceptions import FewShotNotAvailableError, NL2SpecException

log = get_logger(__name__)

ALLOWED_K = {0, 1, 3}
ALLOWED_SELECTION = {"random", "irsp", "mmr", "mmr_irsp"}


class FewShotLoader:
    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.root = Path(fewshot_dir)
        self.seed = seed
        self.rng = random.Random(seed)

        self.fsm_selector = FSMFewShotSelector()
        self.ere_selector = EREFewShotSelector()
        self.ltl_selector = LTLFewShotSelector()
        self.event_selector = EventFewShotSelector()

        log.info("Few-shot root directory: %s", self.root)

    def _validate_configuration(self, shot_mode: str, k: int, selection: str):
        log.info("Few-shot MODE Selection: %s", selection)
        
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

        if selection == "random":
            chosen = select_random(files=files, k=k, rng=self.rng)

            if not return_scores:
                return chosen

            if ir_base is None:
                raise NL2SpecException("Random selection with scoring requires ir_base.")

            selector = self._get_selector(ir_type)
            return selector.score_candidates(chosen, ir_base)
   
        if selection == "irsp":
            if ir_base is None:
                raise NL2SpecException("IRSP selection requires ir_base.")

            if ir_type == "fsm":
                return self.fsm_selector.select(files, k, ir_base, return_scores)
            if ir_type == "ere":
                return self.ere_selector.select(files, k, ir_base, return_scores)
            if ir_type == "ltl":
                return self.ltl_selector.select(files, k, ir_base, return_scores)
            if ir_type == "event":
                return self.event_selector.select(files, k, ir_base, return_scores)

            raise NL2SpecException(
                f"IRSP selector not implemented for ir_type '{ir_type}'."
            )

        if selection == "mmr":
            if ir_base is None:
                raise NL2SpecException("MMR selection requires ir_base.")

            chosen = select_mmr(
                files=files,
                ir_base=ir_base,
                k=k,
                return_scores=False,
            )
            
            if not return_scores:
                return chosen
            
            selector = self._get_selector(ir_type)
            return selector.score_candidates(chosen, ir_base)

        if selection == "mmr_irsp":
            if ir_base is None:
                raise NL2SpecException("MMR_IRSP selection requires ir_base.")

            selector = self._get_selector(ir_type)

            chosen = select_mmr_irsp(
                selector=selector,
                files=files,
                ir_base=ir_base,
                k=k,
                return_scores=False,
            )           
            if not return_scores:
                return chosen

            return selector.score_candidates(chosen, ir_base)
        raise NL2SpecException("Unexpected selection mode.")


    def _get_selector(self, ir_type: str):
        if ir_type == "fsm":
            return self.fsm_selector
        if ir_type == "ere":
            return self.ere_selector
        if ir_type == "ltl":
            return self.ltl_selector
        if ir_type == "event":
            return self.event_selector

        raise NL2SpecException(f"Unsupported ir_type '{ir_type}'.")