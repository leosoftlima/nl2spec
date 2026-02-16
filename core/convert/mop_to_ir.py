from pathlib import Path
import json
import re
from typing import Dict

from .ltl import extract_ltl_ir
from .fsm import extract_fsm_ir
from .ere import extract_ere_ir
from .event import extract_event_ir


# ==========================================================
# PUBLIC API
# ==========================================================

def convert_mop_dir_to_ir(mop_root: str, out_dir: str, keep_structure: bool = True) -> int:
    mop_root = Path(mop_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for mop_file in mop_root.rglob("*.mop"):
        text = mop_file.read_text(encoding="utf-8", errors="replace")

        formalism = detect_formalism(text)

        # 🔥 Agora suporta as 4 linguagens
        SUPPORTED = {"ltl", "fsm", "ere", "event"}

        if formalism not in SUPPORTED:
            continue

        ir = convert_mop_file_to_ir(mop_file)

        if keep_structure:
            relative = mop_file.relative_to(mop_root)
            target = out_dir / relative.with_suffix(".json")
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = out_dir / f"{mop_file.stem}.json"

        with open(target, "w", encoding="utf-8") as f:
            json.dump(ir, f, indent=2, ensure_ascii=False)

        count += 1

    return count


# ==========================================================
# SINGLE FILE CONVERSION
# ==========================================================

def convert_mop_file_to_ir(mop_path: Path) -> Dict:
    text = mop_path.read_text(encoding="utf-8", errors="replace")

    formalism = detect_formalism(text)
    domain = detect_domain(mop_path)

    if formalism == "ltl":
        return extract_ltl_ir(text, mop_path.stem, domain)

    if formalism == "fsm":
        return extract_fsm_ir(text, mop_path.stem, domain)

    if formalism == "ere":
        return extract_ere_ir(text, mop_path.stem, domain)

    if formalism == "event":
        return extract_event_ir(text, mop_path.stem, domain)

    raise NotImplementedError(f"Formalism not supported yet: {formalism}")


# ==========================================================
# FORMALISM DETECTION (STRICT ORDER)
# ==========================================================

_LTL_HEADER_RE = re.compile(r"(?im)^\s*(ptltl|ltl)\s*:")
_FSM_HEADER_RE = re.compile(r"(?im)^\s*fsm\s*:")
_ERE_HEADER_RE = re.compile(r"(?im)^\s*ere\s*:")

def detect_formalism(text: str) -> str:
    """
    Detection priority:
        1. LTL / PTLTL
        2. FSM
        3. ERE
        4. EVENT (fallback)

    EVENT = absence of temporal header but presence of events + violation.
    """

    if _LTL_HEADER_RE.search(text):
        return "ltl"

    if _FSM_HEADER_RE.search(text):
        return "fsm"

    if _ERE_HEADER_RE.search(text):
        return "ere"

    # Fallback: se tem "event" mas não tem ltl/fsm/ere → é event-based
    if re.search(r"(?im)^\s*event\s+", text):
        return "event"

    return "unknown"


# ==========================================================
# DOMAIN DETECTION
# ==========================================================

def detect_domain(path: Path) -> str:
    for p in path.parts:
        if p in {"io", "lang", "util", "net"}:
            return p
    return "unknown"