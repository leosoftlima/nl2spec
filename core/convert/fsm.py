import re
from typing import Dict, List, Tuple, Optional


# ==========================================================
# MAIN
# ==========================================================

def extract_fsm_ir(mop_text: str, spec_id: str, domain: str) -> Dict:
    signature = extract_signature(mop_text)
    events = extract_events(mop_text)
    fsm_block = extract_fsm_block(mop_text)
    violation = extract_violation_block(mop_text)

    return {
        "id": spec_id,
        "formalism": "fsm",
        "domain": domain,
        "signature": signature,
        "ir": {
            "type": "fsm",
            "events": events,

            # BLOCO FIEL (para reconstrução)
            "fsm": {
                "raw_block": fsm_block["raw_block"],
                "raw_lines": fsm_block["raw_lines"],
            },

            # ANÁLISE ESTRUTURAL (para pesquisa)
            "analysis": {
                "states": fsm_block["states"],
                "initial": fsm_block["initial"],
                "transitions": fsm_block["transitions"],
            },

            # VIOLATION (tag + linhas)
            "violation": violation,
        },
    }


# ==========================================================
# SIGNATURE (robusta)
# ==========================================================

_PROP_HEADER_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*\(")

def extract_signature(text: str) -> Dict:
    """
    Extrai assinatura do cabeçalho:
        Name(type1 p1, type2 p2, ...)
    Suporta:
      - generics com vírgula (Map<String, Integer>)
      - arrays (byte[], char[])
      - qualified (java.io.OutputStream)
      - modifiers simples (final)
      - + (OutputStream+)
    """
    m = _PROP_HEADER_RE.search(text)
    if not m:
        return {"parameters": []}

    name = m.group(1)

    open_paren = text.find("(", m.start())
    if open_paren == -1:
        return {"parameters": []}

    params_raw, _close_pos = _extract_balanced_parens(text, open_paren)
    if params_raw is None:
        return {"parameters": []}

    params_raw = params_raw.strip()
    if not params_raw:
        return {"name": name, "parameters": []}

    parts = _split_commas_balanced(params_raw)
    params: List[Dict] = []

    for part in parts:
        ptype, pname = _split_type_name(part.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return {"name": name, "parameters": params}


def _extract_balanced_parens(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    depth = 0
    i = open_pos
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1 : i], i
        i += 1
    return None, None


def _split_commas_balanced(s: str) -> List[str]:
    """
    Divide por vírgula ignorando vírgulas dentro de:
      - <...> generics
      - (...) (raro em tipos)
      - [...] arrays
    """
    out: List[str] = []
    cur: List[str] = []

    angle = 0
    paren = 0
    bracket = 0

    for ch in s:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)

        if ch == "," and angle == 0 and paren == 0 and bracket == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)

    if cur:
        out.append("".join(cur).strip())

    return out


def _split_type_name(param: str) -> Tuple[str, str]:
    """
    Separa "Type name" usando o último token como name.
    Remove:
      - annotations simples @X ou @X(...)
      - 'final'
    """
    param = re.sub(r"@\w+(\([^)]*\))?\s*", "", param).strip()
    param = re.sub(r"^\s*final\s+", "", param).strip()

    tokens = param.split()
    if len(tokens) < 2:
        return "", ""

    name = tokens[-1]
    ptype = " ".join(tokens[:-1]).strip()
    return ptype, name


# ==========================================================
# EVENTS (suporta "creation event")
# ==========================================================

# Captura:
#   creation event NAME before/after(PARAMS) [returning(...)] : POINTCUT { BODY }
#
# Observação:
#   - O BODY pode ser vazio ou multilinha
#   - O POINTCUT pode estar quebrado em linhas
EVENT_FULL_RE = re.compile(
    r"(?is)"
    r"\b(creation\s+)?event\s+(\w+)\s+"
    r"(before|after)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:\s*"
    r"(.*?)\s*\{(.*?)\}",
)

def extract_events(text: str) -> List[Dict]:
    events: List[Dict] = []

    for creation_kw, name, timing, params, returning, pointcut, body in EVENT_FULL_RE.findall(text):
        kind = "creation" if (creation_kw and creation_kw.strip()) else "event"

        event: Dict = {
            "kind": kind,  # "creation" | "event"
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params),
            "pointcut_raw": normalize(pointcut.strip()),
        }

        if returning and returning.strip():
            rtype, rname = _split_type_name(returning.strip())
            if rtype and rname:
                event["returning"] = {"type": rtype, "name": rname}

        # opcional: preserva se o corpo do event tinha algo (mesmo vazio)
        # (mantém fidelidade; útil se houver código)
        body_lines = [ln.rstrip() for ln in body.splitlines() if ln.strip() != ""]
        if body_lines:
            event["body"] = {"raw_lines": body_lines}
        else:
            event["body"] = {"raw_lines": []}

        events.append(event)

    return events


def parse_parameters(param_text: str) -> List[Dict]:
    if not param_text.strip():
        return []

    parts = _split_commas_balanced(param_text)
    params: List[Dict] = []

    for p in parts:
        ptype, pname = _split_type_name(p.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return params


# ==========================================================
# FSM BLOCK (formato real: fsm : + blocos state [ ... ])
# ==========================================================

_FSM_LINE_RE = re.compile(r"(?im)^\s*fsm\s*:\s*$")

def extract_fsm_block(text: str) -> Dict:
    """
    Extrai o FSM do formato:
        fsm :
            initial [
                ev -> S1
            ]
            S1 [
                ...
            ]

    Captura do 'fsm :' até antes de @fail/@violation/@match ou até o '}' final.
    """
    m = _FSM_LINE_RE.search(text)
    if not m:
        return {"raw_block": "", "raw_lines": [], "states": [], "initial": None, "transitions": []}

    start = m.end()
    end = _find_next_directive_or_property_end(text, start)

    block_text = text[start:end].rstrip()

    raw_lines = [ln.rstrip("\n") for ln in block_text.splitlines() if ln.strip() != ""]
    raw_block = "\n".join(raw_lines)

    states, initial, transitions = _parse_fsm_state_blocks(raw_lines)

    return {
        "raw_block": raw_block,
        "raw_lines": raw_lines,
        "states": states,
        "initial": initial,
        "transitions": transitions,
    }


def _find_next_directive_or_property_end(text: str, start: int) -> int:
    m_dir = re.search(r"(?im)^\s*@(fail|violation|match)\b", text[start:])
    if m_dir:
        return start + m_dir.start()

    m_end = re.search(r"(?m)^\s*\}\s*$", text[start:])
    if m_end:
        return start + m_end.start()

    return len(text)


_STATE_OPEN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*$")
_TRANS_LINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*$")

def _parse_fsm_state_blocks(lines: List[str]) -> Tuple[List[str], Optional[str], List[Dict]]:
    """
    Lê blocos:
        stateName [
            event -> nextState
        ]
    """
    states_set = set()
    transitions: List[Dict] = []
    initial_state: Optional[str] = None

    current_state: Optional[str] = None
    inside = False

    for ln in lines:
        line = ln.rstrip()

        m_open = _STATE_OPEN_RE.match(line)
        if m_open:
            current_state = m_open.group(1)
            states_set.add(current_state)
            if initial_state is None:
                initial_state = current_state
            inside = True
            continue

        if inside and line.strip() == "]":
            inside = False
            current_state = None
            continue

        if inside and current_state:
            m_tr = _TRANS_LINE_RE.match(line)
            if m_tr:
                ev, dst = m_tr.group(1), m_tr.group(2)
                states_set.add(dst)
                transitions.append({"from": current_state, "event": ev, "to": dst})

    return sorted(states_set), initial_state, transitions


# ==========================================================
# FAIL / VIOLATION / MATCH
# ==========================================================

def extract_violation_block(text: str) -> Dict:
    m = re.search(
        r"@(fail|violation|match)\s*\{(.*?)\}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not m:
        return {"tag": None, "raw_block": []}

    tag = m.group(1).lower()
    block = m.group(2).strip()

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return {"tag": tag, "raw_block": lines}


# ==========================================================
# UTILS
# ==========================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()