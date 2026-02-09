import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------
def _strip_comments(text: str) -> str:
    # remove /* ... */ and // ...
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def _detect_category(text: str) -> str:
    t = text.lower()
    if "fsm :" in t:
        return "FSM"
    if "ere :" in t:
        return "ERE"
    if "ltl :" in t or "ptltl :" in t:
        return "LTL"
    return "EVENT"


def _extract_violation_message(text: str) -> str:
    """
    Best-effort: grabs the first string literal printed in @fail / @violation / @match blocks.
    Falls back to a safe default.
    """
    # Look inside @fail / @violation / @match blocks first
    m = re.search(r"@(fail|violation|match)\s*\{(.*?)\}", text, flags=re.DOTALL | re.IGNORECASE)
    scope = m.group(2) if m else text

    # Extract first quoted string
    s = re.search(r"\"([^\"]+)\"", scope)
    if s:
        return s.group(1).strip()

    return "Violation detected."


def _extract_events(text: str) -> List[Dict[str, str]]:
    """
    Extract event declarations like:
      event set before() :
      creation event getoutput after(Socket sock) returning(OutputStream output) :
    Returns: [{"name": "...", "timing": "before|after"}, ...]
    """
    events = []

    # Normalize whitespace to make regex easier
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Patterns:
    #   event NAME before(...)
    #   creation event NAME after(...)
    ev_re = re.compile(
        r"^(?:creation\s+)?event\s+([A-Za-z_]\w*)\s+(before|after)\b",
        flags=re.IGNORECASE
    )

    for ln in lines:
        m = ev_re.match(ln)
        if m:
            name = m.group(1)
            timing = m.group(2).lower()
            events.append({"name": name, "timing": timing})

    # Deduplicate preserving order
    seen = set()
    out = []
    for e in events:
        key = (e["name"], e["timing"])
        if key not in seen:
            out.append(e)
            seen.add(key)
    return out


def _extract_guard(text: str) -> str:
    """
    Best-effort guard extraction:
    - captures condition(...) inside event pointcut lines: && condition(...)
    If not found, returns "true" (schema requires guard for event).
    """
    m = re.search(r"\bcondition\s*\(\s*(.*?)\s*\)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        g = re.sub(r"\s+", " ", m.group(1)).strip()
        return g if g else "true"
    return "true"


def _extract_ere_pattern(text: str) -> Optional[str]:
    m = re.search(r"\bere\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    # pattern is usually line; cut at line end
    pat = m.group(1).strip()
    pat = pat.splitlines()[0].strip()
    return pat if pat else None


def _extract_ltl_formula(text: str) -> Optional[str]:
    m = re.search(r"\b(ltl|ptltl)\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    formula = m.group(2).strip()
    formula = formula.splitlines()[0].strip()
    return formula if formula else None


def _parse_fsm_block(text: str) -> Tuple[List[str], Optional[str], List[Dict[str, Any]]]:
    """
    Parse the fsm section:
      fsm :
        start [
          getoutput -> unblocked
        ]
        unblocked [
          enter -> blocked
        ]
    Returns: (states, start_state, transitions)
    transitions: [{"from": ..., "event": ..., "to": ..., "timing": None}]
    """
    # Extract from "fsm :" to end or before next annotation block (@...)
    m = re.search(r"\bfsm\s*:\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return [], None, []

    after = m.group(1)

    # Stop at @fail/@violation/@match if present
    cut = re.split(r"\n\s*@\w+\s*\{", after, maxsplit=1)
    fsm_text = cut[0]

    # Find state blocks:  STATE_NAME [ ... ]
    state_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*$", flags=re.IGNORECASE)
    trans_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*$")

    states: List[str] = []
    transitions: List[Dict[str, Any]] = []
    start_state: Optional[str] = None

    current_state: Optional[str] = None
    for raw in fsm_text.splitlines():
        ln = raw.strip()
        if not ln:
            continue

        sm = state_re.match(ln)
        if sm:
            current_state = sm.group(1)
            if current_state not in states:
                states.append(current_state)
            if current_state.lower() == "start":
                start_state = "start"
            continue

        if ln == "]":
            current_state = None
            continue

        tm = trans_re.match(ln)
        if tm and current_state:
            ev = tm.group(1)
            to = tm.group(2)
            transitions.append({
                "from": current_state,
                "event": ev,
                "to": to
                # "timing" omitted (optional in schema)
            })
            # also include target as a state
            if to not in states:
                states.append(to)

    # If start_state not explicitly "start", set to first state if exists
    if start_state is None and states:
        start_state = states[0]

    return states, start_state, transitions

def _infer_domain_from_path(mop_path: Path) -> str:
    parts = [p.lower() for p in mop_path.parts]

    for domain in ("io", "lang", "util", "net", "concurrent"):
        if domain in parts:
            return domain

    return "other"
# -----------------------------
# Public API
# -----------------------------
def mop_text_to_ir(mop_text: str, spec_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a JavaMOP (.mop) specification text into canonical IR (dict).
    """
    raw = mop_text
    text = _strip_comments(raw)

    
    violation_message = _extract_violation_message(text)
    spec_type = _detect_category(text)

    if spec_type == "EVENT":
        events = _extract_events(text)
        guard = _extract_guard(text)

        # EVENT must be "event" in our IR schema
        return {
            "id": spec_id,
            "category": "EVENT",
            "ir": {
                "type": "event",
                "events": events if events else [{"name": "UNKNOWN_EVENT", "timing": "before"}],
                "guard": guard,
                "violation_message": violation_message
            }
        }

    if spec_type == "ERE":
        events = _extract_events(text)
        pattern = _extract_ere_pattern(text) or ""
        return {
            "id": spec_id,
            "category": "ERE",
            "ir": {
                "type": "ere",
                "events": events if events else [{"name": "UNKNOWN_EVENT", "timing": "before"}],
                "pattern": pattern,
                "violation_message": violation_message
            }
        }

    if spec_type == "LTL":
        events = _extract_events(text)
        formula = _extract_ltl_formula(text) or ""
        return {
            "id": spec_id,
            "category": "LTL",
            "ir": {
                "type": "ltl",
                "events": events if events else [{"name": "UNKNOWN_EVENT", "timing": "before"}],
                "formula": formula,
                "violation_message": violation_message
            }
        }

    # FSM
    states, start_state, transitions = _parse_fsm_block(text)
    return {
        "id": spec_id,
        "category": "FSM",
        "ir": {
            "type": "fsm",
            "states": states if states else ["start", "UNKNOWN_STATE"],
            "start_state": start_state or "start",
            "transitions": transitions,
            "violation_message": violation_message
        }
    }

def mop_file_to_ir(mop_path: str) -> Dict[str, Any]:
    p = Path(mop_path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    base = mop_text_to_ir(text, spec_id=p.stem)
    domain = _infer_domain_from_path(p)

    # reordena explicitamente os campos
    ordered_ir = {
        "id": base.get("id"),
        "category": base.get("category"),
        "domain": domain,
        "ir": base.get("ir"),
    }

    return ordered_ir



def convert_mop_dir_to_ir(
    mop_root: str,
    out_dir: str,
    keep_structure: bool = True
) -> int:
    """
    Convert all .mop files under mop_root into IR JSON files under out_dir.

    - keep_structure=True: preserves relative folder structure
    - returns number of converted files
    """
    mop_root_p = Path(mop_root)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    count = 0
    for mop_file in mop_root_p.rglob("*.mop"):
        ir = mop_file_to_ir(str(mop_file))

        if keep_structure:
            rel = mop_file.relative_to(mop_root_p)
            target_dir = out_dir_p / rel.parent
        else:
            target_dir = out_dir_p

        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / (mop_file.stem + ".json")
        target_file.write_text(
            __import__("json").dumps(ir, indent=2),
            encoding="utf-8"
        )
        count += 1

    return count
