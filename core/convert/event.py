import re
from typing import Dict, List, Tuple, Optional


# ==========================================================
# MAIN
# ==========================================================

def extract_event_ir(mop_text: str, spec_id: str, domain: str) -> Dict:
    signature = extract_signature(mop_text)
    events = extract_events(mop_text)
    violation = extract_violation_block(mop_text)

    return {
        "id": spec_id,
        "formalism": "event",
        "domain": domain,
        "signature": signature,
        "ir": {
            "type": "event",
            "events": events,
            "violation": violation,
        },
    }


# ==========================================================
# SIGNATURE
# ==========================================================

_PROP_HEADER_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*\(")

def extract_signature(text: str) -> Dict:
    m = _PROP_HEADER_RE.search(text)
    if not m:
        return {"parameters": []}

    name = m.group(1)
    open_paren = text.find("(", m.start())
    params_raw, _ = _extract_balanced_parens(text, open_paren)

    params = []
    if params_raw:
        for part in _split_commas_balanced(params_raw):
            tokens = part.strip().split()
            if len(tokens) >= 2:
                params.append({
                    "type": " ".join(tokens[:-1]),
                    "name": tokens[-1]
                })

    return {"name": name, "parameters": params}


def _extract_balanced_parens(s: str, open_pos: int):
    depth = 0
    i = open_pos
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return s[open_pos+1:i], i
        i += 1
    return None, None


def _split_commas_balanced(s: str) -> List[str]:
    out, cur = [], []
    depth = 0
    for ch in s:
        if ch in "<([": depth += 1
        elif ch in ">)]": depth -= 1

        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)

    if cur:
        out.append("".join(cur))
    return out


# ==========================================================
# EVENTS — CORRIGIDO
# ==========================================================

EVENT_HEADER_RE = re.compile(
    r"\bevent\s+(\w+)\s+"
    r"(before|after)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:",
    re.DOTALL
)


def extract_events(text: str) -> List[Dict]:
    events = []

    for match in EVENT_HEADER_RE.finditer(text):
        name, timing, params, returning = match.groups()

        header_end = match.end()

        # --------------------------------------------------
        # CAPTURAR POINTCUT MULTILINHA ATÉ '{'
        # --------------------------------------------------
        brace_pos = text.find("{", header_end)
        pointcut_raw = text[header_end:brace_pos].strip()

        # normalizar quebras de linha
        pointcut_raw = normalize(pointcut_raw)

        # --------------------------------------------------
        # CAPTURAR CORPO COM CHAVES BALANCEADAS
        # --------------------------------------------------
        body_text, body_end = _extract_balanced_block(text, brace_pos)

        body_lines = [
            ln.rstrip()
            for ln in body_text.splitlines()
            if ln.strip()
        ]

        event = {
            "kind": "event",
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params),
            "pointcut_raw": pointcut_raw,
            "body": {
                "raw_lines": body_lines
            }
        }

        if returning and returning.strip():
            tokens = returning.strip().split()
            if len(tokens) >= 2:
                event["returning"] = {
                    "type": " ".join(tokens[:-1]),
                    "name": tokens[-1]
                }

        events.append(event)

    return events


def _extract_balanced_block(text: str, open_brace_pos: int):
    depth = 0
    i = open_brace_pos
    start = open_brace_pos + 1

    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i
        i += 1

    return "", None


def parse_parameters(param_text: str) -> List[Dict]:
    if not param_text.strip():
        return []

    params = []
    for part in _split_commas_balanced(param_text):
        tokens = part.strip().split()
        if len(tokens) >= 2:
            params.append({
                "type": " ".join(tokens[:-1]),
                "name": tokens[-1]
            })
    return params


# ==========================================================
# VIOLATION
# ==========================================================

def extract_violation_block(text: str) -> Dict:
    m = re.search(
        r"@(fail|violation|match)\s*\{(.*?)\}",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not m:
        return {"tag": None, "raw_block": []}

    tag = m.group(1).lower()
    block = m.group(2)

    lines = [
        ln.strip()
        for ln in block.splitlines()
        if ln.strip()
    ]

    return {
        "tag": tag,
        "raw_block": lines
    }


# ==========================================================
# UTILS
# ==========================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()