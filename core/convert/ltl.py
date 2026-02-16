import re
from typing import Dict, List


# ==========================================================
# MAIN
# ==========================================================

def extract_ltl_ir(mop_text: str, spec_id: str, domain: str) -> Dict:

    signature = extract_signature(mop_text)
    events = extract_events(mop_text)
    formula = extract_formula(mop_text)
    violation = extract_violation_block(mop_text)

    return {
        "id": spec_id,
        "formalism": "ltl",
        "domain": domain,
        "signature": signature,
        "ir": {
            "type": "ltl",
            "events": events,
            "formula": formula,
            "violation": violation
        }
    }


# ==========================================================
# EVENTS
# ==========================================================

EVENT_RE = re.compile(
    r"event\s+(\w+)\s+"
    r"(before|after)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:\s*"
    r"(.*?)\s*\{\s*\}",
    re.DOTALL
)

def extract_events(text: str) -> List[Dict]:
    events = []

    for name, timing, params, returning, pointcut in EVENT_RE.findall(text):
        event = {
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params),
            "pointcut": parse_pointcut(pointcut.strip())
        }

        if returning:
            tokens = returning.strip().split()
            if len(tokens) == 2:
                event["returning"] = {
                    "type": tokens[0],
                    "name": tokens[1]
                }

        events.append(event)

    return events


def parse_parameters(param_text: str) -> List[Dict]:
    if not param_text.strip():
        return []

    params = []
    parts = [p.strip() for p in param_text.split(",")]

    for p in parts:
        tokens = p.split()
        if len(tokens) == 2:
            params.append({
                "type": tokens[0],
                "name": tokens[1]
            })

    return params


# ==========================================================
# POINTCUT
# ==========================================================

def parse_pointcut(text: str) -> Dict:
    return {
        "raw": normalize(text),
        "joinpoints": extract_calls(text),
        "bindings": extract_targets(text),
        "exclusions": extract_cflow(text)
    }


def extract_calls(text: str) -> List[Dict]:
    """
    Extracts call(...) joinpoints using balanced-parentheses scanning.
    This avoids truncation when the call body contains parentheses, e.g. readPassword(..).
    """
    calls = []

    for body in _extract_balanced_call_bodies(text):
        body = normalize(body)

        calls.append({
            "kind": "call",
            "pattern": body,  # agora vem completo: "char[] Console+.readPassword(..)"
            "simple": extract_simple_method(body),
            "declaring_type": extract_declaring_type(body),
        })

    return calls

def _extract_balanced_call_bodies(text: str) -> List[str]:
    """
    Returns the inside of each call( ... ) with correct balancing.
    Example: call(char[] Console+.readPassword(..)) -> "char[] Console+.readPassword(..)"
    """
    out: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        m = re.search(r"\bcall\s*\(", text[i:])
        if not m:
            break

        start = i + m.start()
        # position at the '(' after 'call'
        j = start + m.group(0).rfind("(")

        depth = 0
        k = j
        while k < n:
            ch = text[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    # body is inside outer call(...)
                    body = text[j + 1 : k]
                    out.append(body)
                    i = k + 1
                    break
            k += 1
        else:
            # unbalanced: stop scanning
            break

    return out

def extract_simple_method(body: str) -> str:
    m = re.search(r"\.(\w+\(\.\.\))", body)
    return m.group(1) if m else ""


def extract_declaring_type(body: str) -> str:
    
    # remove leading return type if exists
    # split by space and take last token before '.'
    m = re.search(r"\b([A-Za-z_]\w*\+?)\s*\.\s*\w+\s*\(", body)

    if m:
        return m.group(1)

    return ""


def extract_targets(text: str) -> List[Dict]:
    matches = re.findall(r"target\((\w+)\)", text)
    return [{"type": "target", "variable": m} for m in matches]


def extract_cflow(text: str) -> List[Dict]:
    matches = re.findall(r"!cflow\((.*?)\)", text, re.DOTALL)

    return [{
        "type": "cflow",
        "pattern": normalize(m)
    } for m in matches]


# ==========================================================
# FORMULA
# ==========================================================

def extract_formula(text: str) -> Dict:
    m = re.search(r"(?im)^\s*(ptltl|ltl)\s*:\s*(.*)", text)

    raw = m.group(2).strip() if m else ""

    return {
        "raw": raw,
    }

def extract_signature(text: str) -> Dict:
    m = re.search(r"^\s*(\w+)\s*\((.*?)\)\s*\{", text, re.MULTILINE)
    if not m:
        return {"parameters": []}

    params_raw = m.group(2).strip()
    if not params_raw:
        return {"parameters": []}

    params = []
    for p in params_raw.split(","):
        tokens = p.strip().split()
        if len(tokens) == 2:
            params.append({
                "type": tokens[0],
                "name": tokens[1]
            })

    return {"parameters": params}

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
        return {"raw_block": []}

    block = m.group(2).strip()

    # dividir por linhas preservando conteúdo
    lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip()
    ]

    return {"raw_block": lines}

def extract_violation(text: str) -> str:
    """
    Extract violation message from @fail / @violation / @match blocks.

    Priority:
    1) First quoted string inside the block
    2) Argument of println(...)
    3) Safe default message
    """

    # Capture block inside @fail / @violation / @match
    m = re.search(
        r"@(fail|violation|match)\s*\{(.*?)\}",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not m:
        return "Violation detected."

    block = m.group(2)

    # 1️⃣ Try first quoted string
    s = re.search(r'"([^"]+)"', block)
    if s:
        return s.group(1).strip()

    # 2️⃣ Try argument of println(...)
    p = re.search(r'println\s*\([^,]+,\s*([^)]+)\)', block)
    if p:
        return p.group(1).strip()

    # 3️⃣ Fallback
    return "Violation detected."


# ==========================================================
# UTILS
# ==========================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()