import re
from typing import Dict, List, Optional, Tuple


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
            "events": events,
            "ltl": formula,
            "violation": violation,
        },
    }


# ==========================================================
# SIGNATURE
# ==========================================================

def extract_signature(text: str) -> Dict:
    m = re.search(r"^\s*(\w+)\s*\((.*?)\)\s*\{", text, re.MULTILINE | re.DOTALL)
    if not m:
        return {"name": "", "parameters": []}

    name = m.group(1).strip()
    params_raw = m.group(2).strip()

    return {
        "name": name,
        "parameters": parse_parameters(params_raw),
    }


# ==========================================================
# EVENTS
# ==========================================================

EVENT_RE = re.compile(
    r"event\s+(\w+)\s+"
    r"(before|after)"
    r"(?:\((.*?)\))?\s*"
    r"(?:returning\((.*?)\))?\s*:\s*"
    r"(.*?)\{\s*\}",
    re.DOTALL,
)


def extract_events(text: str) -> List[Dict]:
    events: List[Dict] = []

    for name, timing, params, returning, pointcut in EVENT_RE.findall(text):
        method = {
            "action": "event",
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params or ""),
            "returning": parse_returning(returning),
            "procediments": ":",
            "function": extract_pointcut_functions(pointcut.strip()),
            "operation": extract_pointcut_operations(pointcut.strip()),
        }

        events.append({"body": {"methods": [method]}})

    return events


def parse_parameters(param_text: str) -> List[Dict]:
    param_text = (param_text or "").strip()
    if not param_text:
        return []

    params = []
    parts = [p.strip() for p in param_text.split(",") if p.strip()]

    for part in parts:
        tokens = part.split()
        if len(tokens) >= 2:
            params.append({
                "type": " ".join(tokens[:-1]),
                "name": tokens[-1],
            })

    return params


def parse_returning(returning_text: Optional[str]) -> Optional[Dict]:
    returning_text = (returning_text or "").strip()
    if not returning_text:
        return None

    tokens = returning_text.split()
    if len(tokens) >= 2:
        return {
            "type": " ".join(tokens[:-1]),
            "name": tokens[-1],
        }

    return None


# ==========================================================
# POINTCUT
# ==========================================================

def extract_pointcut_functions(text: str) -> List[Dict]:
    functions: List[Dict] = []
    i = 0
    n = len(text)

    while i < n:
        match = re.search(r"\b(call|target|args|condition|cflow|endProgram)\b", text[i:])
        if not match:
            break

        start = i + match.start()
        fname = match.group(1)
        j = start + len(fname)

        while j < n and text[j].isspace():
            j += 1

        if j >= n or text[j] != "(":
            i = j
            continue

        body, end_idx = extract_balanced_parenthesized(text, j)
        body = normalize(body)

        functions.append({
            "name": fname,
            "parameters": build_function_parameters(fname, body),
        })

        i = end_idx

    return functions


def extract_pointcut_operations(text: str) -> List[str]:
    ops: List[str] = []
    i = 0
    depth = 0
    n = len(text)

    while i < n - 1:
        ch = text[i]
        nxt = text[i + 1]

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and ch == "&" and nxt == "&":
            ops.append("&&")
            i += 1
        elif depth == 0 and ch == "|" and nxt == "|":
            ops.append("||")
            i += 1

        i += 1

    return ops


def extract_balanced_parenthesized(text: str, open_paren_idx: int) -> Tuple[str, int]:
    depth = 0
    i = open_paren_idx
    n = len(text)

    while i < n:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : i], i + 1
        i += 1

    return text[open_paren_idx + 1 :], n


def build_function_parameters(fname: str, body: str) -> List[Dict]:
    if fname in {"call", "cflow"}:
        return [{"return": "*" if fname == "call" else "", "name": body}]

    if fname == "endProgram":
        return []

    if fname == "target":
        return [{"return": "", "name": body}]

    if fname == "condition":
        return [{"return": "", "name": body}]

    if fname == "args":
        args = split_top_level_args(body)
        return [{"return": "", "name": arg} for arg in args]

    return [{"return": "", "name": body}]


def split_top_level_args(text: str) -> List[str]:
    args = []
    current = []
    depth = 0

    for ch in text:
        if ch == "," and depth == 0:
            arg = normalize("".join(current))
            if arg:
                args.append(arg)
            current = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        current.append(ch)

    arg = normalize("".join(current))
    if arg:
        args.append(arg)

    return args


# ==========================================================
# FORMULA
# ==========================================================

def extract_formula(text: str) -> Dict:
    m = re.search(r"(?im)^\s*(ptltl|ltl)\s*:\s*(.*)$", text, re.MULTILINE)
    if not m:
        return {
            "name": "",
            "value": ""
        }

    return {
        "name": m.group(1).strip(),
        "value": normalize(m.group(2))
    }


TOKEN_RE = re.compile(r"\[\]|\(\*\)|<>|=>|\|\||&&|\(|\)|!|\bo\b|\bor\b|\band\b|\b[A-Za-z_]\w*\b")


class FormulaParser:
    def __init__(self, text: str):
        self.tokens = TOKEN_RE.findall(text)
        self.pos = 0

    def current(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, token: str) -> None:
        if self.current() != token:
            raise ValueError(f"Expected '{token}', found '{self.current()}'")
        self.pos += 1

    def parse(self) -> Dict:
        expr = self.parse_implies()
        if self.current() is not None:
            raise ValueError(f"Unexpected token: {self.current()}")
        return expr

    def parse_implies(self) -> Dict:
        left = self.parse_or()
        while self.current() == "=>":
            self.eat("=>")
            right = self.parse_implies()
            left = {"name": "implies", "value": [left, right]}
        return left

    def parse_or(self) -> Dict:
        left = self.parse_and()
        while self.current() in {"or", "||"}:
            self.pos += 1
            right = self.parse_and()
            left = {"name": "or", "value": [left, right]}
        return left

    def parse_and(self) -> Dict:
        left = self.parse_unary()
        while self.current() in {"and", "&&"}:
            self.pos += 1
            right = self.parse_unary()
            left = {"name": "and", "value": [left, right]}
        return left

    def parse_unary(self) -> Dict:
        tok = self.current()
        if tok == "[]":
            self.eat("[]")
            return {"name": "globally", "value": self.parse_unary()}
        if tok == "<>":
            self.eat("<>")
            return {"name": "eventually", "value": self.parse_unary()}
        if tok == "o":
            self.eat("o")
            return {"name": "next", "value": self.parse_unary()}
        if tok == "(*)":
            self.eat("(*)")
            return {"name": "previous", "value": self.parse_unary()}
        if tok == "!":
            self.eat("!")
            return {"name": "not", "value": self.parse_unary()}
        return self.parse_primary()

    def parse_primary(self) -> Dict:
        tok = self.current()
        if tok == "(":
            self.eat("(")
            expr = self.parse_implies()
            self.eat(")")
            return expr
        if tok is None:
            raise ValueError("Unexpected end of formula")
        self.pos += 1
        return {"name": "event", "value": tok}


def parse_ltl_formula(raw: str) -> Dict:
    parser = FormulaParser(raw)
    return parser.parse()


# ==========================================================
# VIOLATION
# ==========================================================

def extract_violation_block(text: str) -> Dict:
    m = re.search(
        r"@(fail|violation|match)\s*\{(.*?)\}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not m:
        return {"tag": "", "statements": []}

    tag = m.group(1).strip().lower()
    block = m.group(2).strip()

    lines = [normalize(line) for line in block.splitlines() if normalize(line)]
    statements = [classify_violation_statement(line) for line in lines]

    return {
        "tag": tag,
        "statements": statements,
    }


def classify_violation_statement(line: str) -> Dict:
    if "println" in line:
        return {
            "type": "log",
            "value": extract_log_value(line),
        }
    return {
        "type": "raw",
        "value": line,
    }


def extract_log_value(line: str) -> str:
    default_match = re.search(r"__DEFAULT_MESSAGE", line)
    if default_match:
        return "__DEFAULT_MESSAGE"

    quoted_strings = re.findall(r'"([^"]*)"', line)
    if quoted_strings:
        return f'"{quoted_strings[-1]}"'

    m = re.search(r"println\s*\((.*)\)\s*;?", line)
    if m:
        return normalize(m.group(1))

    return line


# ==========================================================
# UTILS
# ==========================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()