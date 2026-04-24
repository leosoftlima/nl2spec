from typing import List
import random
from pathlib import Path


def select_random(files: List[Path], k: int,  rng: random.Random) -> List[Path]:

    total = len(files)

    if k >= total:
        return files

    return rng.sample(files, k)