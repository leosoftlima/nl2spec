import re
from typing import Dict, List, Tuple, Optional, Any, Set


# ==========================================================
# MAIN
# ==========================================================

def extract_ere_ir(mop_text: str, spec_id: str, domain: str) -> Dict[str, Any]:
    signature = extract_signature(mop_text)
    methods = extract_events(mop_text)
    declared_events = {m["name"] for m in methods}
    ere_expression = extract_ere_expression(mop_text, declared_events=declared_events)
    violation = extract_violation_block(mop_text)

    return {
        "id": spec_id,
        "formalism": "ere",
        "domain": domain,
        "signature": signature,
        "ir": {
            "events": [
                {
                    "body": {
                        "methods": methods
                    }
                }
            ],
            "ere": {
                "expression": ere_expression
            },
            "violation": violation
        }
    }


# ==========================================================
# SIGNATURE
# ==========================================================

_SPEC_HEADER_RE = re.compile(
    r"(?m)^\s*(?!package\b|import\b|event\b|creation\b|@|if\b|for\b|while\b|switch\b|catch\b|return\b)"
    r"([A-Za-z_]\w*)\s*\("
)

def extract_signature(text: str) -> Dict[str, Any]:
    m = _SPEC_HEADER_RE.search(text)
    if not m:
        return {"name": "", "parameters": []}

    name = m.group(1)
    open_pos = text.find("(", m.start())
    params_raw, _ = extract_balanced_round(text, open_pos)

    if params_raw is None:
        return {"name": name, "parameters": []}

    params = parse_parameters(params_raw)
    return {"name": name, "parameters": params}


# ==========================================================
# EVENTS
# ==========================================================

_EVENT_START_RE = re.compile(
    r"(?im)^\s*(creation\s+event|event)\s+([A-Za-z_]\w*)\s+(before|after)\b"
)

def extract_events(text: str) -> List[Dict[str, Any]]:
    methods = []

    for m in _EVENT_START_RE.finditer(text):
        action = m.group(1).strip().lower()   # "creation event" or "event"
        name = m.group(2).strip()
        timing = m.group(3).strip()

        cursor = skip_ws(text, m.end())

        # parâmetros do evento são opcionais
        params_raw = ""
        if cursor < len(text) and text[cursor] == "(":
            params_raw, cursor = extract_balanced_round(text, cursor)
            cursor = skip_ws(text, cursor)

        # returning(...) é opcional
        returning_obj = None
        if text[cursor:cursor + 9].lower() == "returning":
            ret_open = text.find("(", cursor)
            ret_raw, cursor = extract_balanced_round(text, ret_open)
            cursor = skip_ws(text, cursor)

            if ret_raw:
                rtype, rname = split_type_name(ret_raw.strip())
                if rtype and rname:
                    returning_obj = {"type": rtype, "name": rname}

        if cursor >= len(text) or text[cursor] != ":":
            continue

        cursor += 1
        cursor = skip_ws(text, cursor)

        pointcut_raw, _, _ = extract_event_pointcut_and_body(text, cursor)
        if pointcut_raw is None:
            continue

        pointcut_struct = parse_pointcut(pointcut_raw)

        method = {
            "action": action,
            "name": name,
            "timing": timing,
            "parameters": parse_parameters(params_raw or "")
        }

        # returning deve entrar aqui, antes de procediments
        if returning_obj is not None:
            method["returning"] = returning_obj

        method["procediments"] = ":"
        method["function"] = pointcut_struct["function"]
        method["operation"] = pointcut_struct["operation"]

        methods.append(method)

    return methods


def extract_event_pointcut_and_body(text: str, start_pos: int) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Lê o pointcut até a primeira chave { no nível externo,
    depois extrai o corpo balanceado do evento.
    O body é ignorado no JSON final, mas ainda é usado para
    localizar corretamente o fim do bloco.
    """
    i = start_pos
    paren = 0

    while i < len(text):
        ch = text[i]

        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "{" and paren == 0:
            pointcut_raw = text[start_pos:i].strip()
            body_raw, end_pos = extract_balanced_curly(text, i)
            return pointcut_raw, body_raw, end_pos

        i += 1

    return None, None, None


# ==========================================================
# POINTCUT STRUCTURE
# ==========================================================

def parse_pointcut(raw: str) -> Dict[str, Any]:
    """
    Estrutura simplificada do pointcut:
    - function: lista de primitivas como call(...), target(...), args(...), if(...), condition(...)
    - operation: sequência dos operadores lógicos encontrados (&&, ||)

    Negação atômica:
    - !call(...)
    - !target(...)
    - !cflow(...)

    Observação:
    não reconstrói AST completa com agrupamento.
    """
    function = []
    operation = []

    i = 0
    pending_negation = False

    while i < len(raw):
        if raw[i].isspace():
            i += 1
            continue

        if raw.startswith("&&", i):
            operation.append("&&")
            i += 2
            continue

        if raw.startswith("||", i):
            operation.append("||")
            i += 2
            continue

        if raw[i] == "!":
            pending_negation = True
            i += 1
            continue

        if raw[i] in "()":
            i += 1
            continue

        if raw[i].isalpha() or raw[i] == "_":
            j = i
            while j < len(raw) and (raw[j].isalnum() or raw[j] == "_"):
                j += 1

            fname = raw[i:j]
            j = skip_ws(raw, j)

            if j < len(raw) and raw[j] == "(":
                inner, end_pos = extract_balanced_round(raw, j)
                arguments = []

                if inner is not None:
                    for p in split_commas_balanced(inner):
                        p = p.strip()
                        if p:
                            arguments.append({"value": p})

                fn_obj = {
                    "name": fname,
                    "arguments": arguments
                }

                if pending_negation:
                    fn_obj["negated"] = True
                    pending_negation = False

                function.append(fn_obj)
                i = end_pos if end_pos is not None else j + 1
                continue

        i += 1

    return {
        "function": function,
        "operation": operation
    }

# ==========================================================
# ERE EXPRESSION (STRING ONLY)
# ==========================================================

_ERE_RE = re.compile(r"(?im)^\s*ere\s*:\s*(.*)$")

def extract_ere_expression(text: str, declared_events: Optional[Set[str]] = None) -> str:
    expression = extract_ere_formula_text(text)
    if not expression:
        return ""

    expression = normalize_ere_expression(expression)

    if declared_events is not None:
        validate_ere_expression(expression, declared_events)

    return expression


def extract_ere_formula_text(text: str) -> str:
    m = _ERE_RE.search(text)
    if not m:
        return ""

    first_line = m.group(1).strip()
    start = m.end()

    lines = [first_line] if first_line else []

    for ln in text[start:].splitlines():
        stripped = ln.strip()

        if not stripped:
            continue

        if re.match(r"^@(fail|match|violation)\b", stripped, re.I):
            break

        if stripped == "}":
            break

        lines.append(stripped)

    return " ".join(lines).strip()


def normalize_ere_expression(expr: str) -> str:
    """
    Normaliza espaços sem transformar a expressão em AST.
    Ex.: múltiplos espaços viram um só.
    """
    return re.sub(r"\s+", " ", expr).strip()


def validate_ere_expression(expr: str, declared_events: Set[str]) -> None:
    tokens = tokenize_ere(expr)

    for kind, value in tokens:
        if kind == "IDENT" and value not in declared_events:
            raise ValueError(
                f"ERE references event '{value}', but this event was not declared in events."
            )


def tokenize_ere(expr: str) -> List[Tuple[str, str]]:
    token_spec = [
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("OR", r"\|"),
        ("STAR", r"\*"),
        ("PLUS", r"\+"),
        ("QMARK", r"\?"),
        ("NOT", r"[!~]"),
        ("EPSILON", r"\bepsilon\b"),
        ("EMPTY", r"\bempty\b"),
        ("IDENT", r"[A-Za-z_]\w*"),
        ("WS", r"\s+"),
    ]

    regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
    pos = 0
    tokens = []

    while pos < len(expr):
        m = re.match(regex, expr[pos:])
        if not m:
            raise ValueError(f"Unexpected token in ERE near: {expr[pos:pos+40]!r}")

        kind = m.lastgroup
        value = m.group(kind)

        if kind != "WS":
            tokens.append((kind, value))

        pos += len(m.group(0))

    return tokens


# ==========================================================
# VIOLATION
# ==========================================================

_VIOLATION_RE = re.compile(r"(?im)^\s*@(fail|match|violation)\s*\{")

def extract_violation_block(text: str) -> Dict[str, Any]:
    m = _VIOLATION_RE.search(text)
    if not m:
        return {
            "tag": "",
            "body": {
                "statements": [],
                "has_reset": False
            }
        }

    tag = m.group(1).lower()
    open_brace = text.find("{", m.start())
    block_raw, _ = extract_balanced_curly(text, open_brace)

    lines = [ln.strip() for ln in (block_raw or "").splitlines() if ln.strip()]
    statements = []
    has_reset = False

    for ln in lines:
        stmt = parse_violation_statement(ln)

        if stmt["type"] == "command" and stmt.get("name") == "reset":
            has_reset = True

        if stmt["type"] == "raw" and "reset(" in stmt.get("value", ""):
            has_reset = True

        statements.append(stmt)

    return {
        "tag": tag,
        "body": {
            "statements": statements,
            "has_reset": has_reset
        }
    }


def parse_violation_statement(line: str) -> Dict[str, Any]:
    log_m = re.search(
        r"RVMLogging\.out\.println\s*\(\s*Level\.(\w+)\s*,\s*(.*?)\s*\)\s*;?$",
        line
    )
    if log_m:
        return {
            "type": "log",
            "level": log_m.group(1),
            "message": log_m.group(2).strip()
        }

    if re.search(r"\breset\s*\(", line):
        return {
            "type": "command",
            "name": "reset"
        }

    return {
        "type": "raw",
        "value": line
    }


# ==========================================================
# PARAMS
# ==========================================================

def parse_parameters(param_text: str) -> List[Dict[str, str]]:
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

def extract_balanced_round(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    if open_pos is None or open_pos >= len(s) or s[open_pos] != "(":
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


def extract_balanced_curly(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    if open_pos is None or open_pos >= len(s) or s[open_pos] != "{":
        return None, None

    depth = 0
    i = open_pos

    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1:i], i + 1
        i += 1

    return None, None


def split_commas_balanced(s: str) -> List[str]:
    result = []
    cur = []
    angle = paren = bracket = curly = 0

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
        elif ch == "{":
            curly += 1
        elif ch == "}":
            curly -= 1

        if ch == "," and angle == 0 and paren == 0 and bracket == 0 and curly == 0:
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
    param = param.strip()

    tokens = param.split()
    if len(tokens) < 2:
        return "", ""

    name = tokens[-1]
    ptype = " ".join(tokens[:-1])
    return ptype.strip(), name.strip()


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos