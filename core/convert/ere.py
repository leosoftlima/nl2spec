import re
from typing import Dict, List, Tuple, Optional


# ==========================================================
# MAIN
# ==========================================================

def extract_ere_ir(mop_text: str, spec_id: str, domain: str) -> Dict:
    signature = extract_signature(mop_text)
    events = extract_events(mop_text)
    formula = extract_ere_formula(mop_text)
    violation = extract_violation_block(mop_text)

    return {
        "id": spec_id,
        "formalism": "ere",
        "domain": domain,
        "signature": signature,
        "ir": {
            "type": "ere",
            "events": events,
            "formula": formula,
            "violation": violation,
        },
    }


# ==========================================================
# SIGNATURE
# ==========================================================

_HEADER_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*\(")

def extract_signature(text: str) -> Dict:
    m = _HEADER_RE.search(text)
    if not m:
        return {"parameters": []}

    name = m.group(1)
    open_pos = text.find("(", m.start())
    params_raw, _ = extract_balanced(text, open_pos)

    if not params_raw:
        return {"name": name, "parameters": []}

    parts = split_commas_balanced(params_raw)
    params = []

    for p in parts:
        ptype, pname = split_type_name(p.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return {"name": name, "parameters": params}


# ==========================================================
# EVENTS
# ==========================================================

EVENT_RE = re.compile(
    r"(?is)"
    r"\b(creation\s+)?event\s+(\w+)\s+"
    r"(before|after)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:\s*"
    r"(.*?)\{(.*?)\}"
)

def extract_events(text: str) -> List[Dict]:
    events = []

    for creation_kw, name, timing, params, returning, pointcut_raw, body in EVENT_RE.findall(text):

        kind = "creation" if creation_kw else "event"

        event = {
            "kind": kind,
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params),
            "pointcut": parse_pointcut(pointcut_raw.strip()),
        }

        if returning:
            rtype, rname = split_type_name(returning.strip())
            if rtype and rname:
                event["returning"] = {"type": rtype, "name": rname}

        body_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if body_lines:
            event["body"] = {"raw_lines": body_lines}

        events.append(event)

    return events


# ==========================================================
# POINTCUT (ROBUSTO, SEM TRUNCAR)
# ==========================================================

def parse_pointcut(raw: str) -> Dict:
    calls = []
    i = 0

    while True:
        idx = raw.find("call(", i)
        if idx == -1:
            break

        content, end_pos = extract_balanced(raw, idx + 4)
        if content:
            calls.append(content.strip())
            i = end_pos
        else:
            break

    return {
        "raw": normalize(raw),
        "calls": calls
    }


# ==========================================================
# ERE FORMULA
# ==========================================================

ERE_RE = re.compile(r"(?im)^\s*ere\s*:\s*(.*)$")

def extract_ere_formula(text: str) -> Dict:
    m = ERE_RE.search(text)
    if not m:
        return {"raw": "", "raw_lines": []}

    first_line = m.group(1).strip()
    start = m.end()

    lines = [first_line] if first_line else []

    for ln in text[start:].splitlines():
        if re.match(r"^\s*@(?:fail|violation|match)", ln, re.I):
            break
        if re.match(r"^\s*\}", ln):
            break
        if ln.strip():
            lines.append(ln.strip())

    return {
        "raw": " ".join(lines).strip(),
        "raw_lines": lines
    }


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

    return {
        "tag": tag,
        "raw_block": lines
    }


# ==========================================================
# PARAMS
# ==========================================================

def parse_parameters(param_text: str) -> List[Dict]:
    if not param_text.strip():
        return []

    parts = split_commas_balanced(param_text)
    params = []

    for p in parts:
        ptype, pname = split_type_name(p.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return params


# ==========================================================
# HELPERS
# ==========================================================

def extract_balanced(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    """
    Extrai conteúdo entre parênteses balanceados.
    open_pos deve apontar para '('
    """
    if s[open_pos] != "(":
        return None, None

    depth = 0
    i = open_pos

    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1:i], i + 1
        i += 1

    return None, None


def split_commas_balanced(s: str) -> List[str]:
    result = []
    cur = []
    angle = paren = bracket = 0

    for ch in s:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1

        if ch == "," and angle == 0 and paren == 0 and bracket == 0:
            result.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)

    if cur:
        result.append("".join(cur).strip())

    return result


def split_type_name(param: str) -> Tuple[str, str]:
    param = re.sub(r"@\w+(\([^)]*\))?\s*", "", param)
    param = re.sub(r"^\s*final\s+", "", param)

    tokens = param.split()
    if len(tokens) < 2:
        return "", ""

    name = tokens[-1]
    ptype = " ".join(tokens[:-1])
    return ptype.strip(), name.strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()