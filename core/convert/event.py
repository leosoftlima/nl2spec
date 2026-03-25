import re
from typing import Dict, List, Optional, Tuple


# ==========================================================
# MAIN
# ==========================================================

def extract_event_ir(mop_text: str, spec_id: str, domain: str) -> Dict:
    signature = extract_signature(mop_text)
    events, event_bodies = extract_events(mop_text)
    violation = extract_violation_block(mop_text, event_bodies)

    return {
        "id": spec_id,
        "formalism": "event",
        "domain": domain,
        "signature": signature,
        "ir": {
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
        return {"name": "", "parameters": []}

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


def _extract_balanced_parens(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    depth = 0
    i = open_pos
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1:i], i
        i += 1
    return None, None


def _split_commas_balanced(s: str) -> List[str]:
    out, cur = [], []
    depth = 0
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1

        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)

    if cur:
        out.append("".join(cur))
    return out


# ==========================================================
# EVENTS
# ==========================================================

EVENT_HEADER_RE = re.compile(
    r"\bevent\s+(\w+)\s+"
    r"(before|after)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:",
    re.DOTALL
)

POINTCUT_TOKEN_RE = re.compile(
    r"""
    (?P<func>
        [A-Za-z_]\w*
        \(
            (?:
                [^()]+
                |
                \([^()]*\)
            )*
        \)
    )
    |
    (?P<op>\&\&|\|\|)
    """,
    re.VERBOSE | re.DOTALL
)


def extract_events(text: str) -> Tuple[List[Dict], List[str]]:
    events = []
    event_bodies = []

    for match in EVENT_HEADER_RE.finditer(text):
        event_name, timing, params_raw, returning_raw = match.groups()
        header_end = match.end()

        brace_pos = text.find("{", header_end)
        if brace_pos == -1:
            continue

        pointcut_raw = text[header_end:brace_pos].strip()
        pointcut_raw = normalize(pointcut_raw)

        body_text, _ = _extract_balanced_block(text, brace_pos)
        event_bodies.append(body_text)

        method = {
            "action": "event",
            "name": event_name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params_raw),
            "returning": parse_returning(returning_raw),
            "procediments": parse_pointcut_to_ast(pointcut_raw),
        }

        event = {
            "name": event_name.strip(),
            "body": {
                "methods": [method]
            }
        }

        events.append(event)

    return events, event_bodies


def _extract_balanced_block(text: str, open_brace_pos: int) -> Tuple[str, Optional[int]]:
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
    if not param_text or not param_text.strip():
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


def parse_returning(returning_text: Optional[str]) -> Optional[Dict]:
    if not returning_text or not returning_text.strip():
        return None

    tokens = returning_text.strip().split()
    if len(tokens) >= 2:
        return {
            "type": " ".join(tokens[:-1]),
            "name": tokens[-1]
        }

    return None


def parse_pointcut_to_ast(pointcut_raw: str) -> Dict:
    functions = []
    operations = []

    for m in POINTCUT_TOKEN_RE.finditer(pointcut_raw):
        if m.group("op"):
            operations.append(m.group("op"))
        elif m.group("func"):
            fn_name, fn_args = split_function_call(m.group("func"))
            functions.append({
                "name": fn_name,
                "parameters": [{"value": arg.strip()} for arg in _split_commas_balanced(fn_args)] if fn_args.strip() else []
            })

    return {
        "function": functions,
        "operation": operations
    }


def split_function_call(token: str) -> Tuple[str, str]:
    token = token.strip()
    open_pos = token.find("(")
    if open_pos == -1 or not token.endswith(")"):
        return token, ""

    name = token[:open_pos].strip()
    inner = token[open_pos + 1:-1].strip()
    return name, inner


# ==========================================================
# VIOLATION
# ==========================================================

def extract_violation_block(text: str, event_bodies: Optional[List[str]] = None) -> Dict:
    """
    Estratégia:
    1. tenta @fail/@violation/@match
    2. se não achar, tenta extrair ações de violação dos corpos dos eventos
    """
    formal_block = extract_formal_violation_block(text)
    if formal_block["tag"] is not None or formal_block["body"]["statements"]:
        return formal_block

    if event_bodies:
        statements = []
        for body in event_bodies:
            statements.extend(parse_violation_statements_from_event_body(body))

        statements = deduplicate_statements(statements)

        return {
            "tag": "fail" if statements else None,
            "body": {
                "statements": statements,
                #"has_reset": any(
                #    stmt.get("type") == "command" and stmt.get("name") == "__RESET"
                #    for stmt in statements
                #)
            }
        }

    return {
        "tag": None,
        "body": {
            "statements": [],
            #"has_reset": False
        }
    }


def extract_formal_violation_block(text: str) -> Dict:
    m = re.search(
        r"@(fail|violation|match)\s*\{(.*?)\}",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not m:
        return {
            "tag": None,
            "body": {
                "statements": [],
                "has_reset": False
            }
        }

    tag = m.group(1).lower()
    block = m.group(2)
    statements = parse_violation_statements(block)

    return {
        "tag": tag,
        "body": {
            "statements": statements,
            "has_reset": any(
                stmt.get("type") == "command" and stmt.get("name") == "__RESET"
                for stmt in statements
            )
        }
    }


def parse_violation_statements(block: str) -> List[Dict]:
    """
    Parser genérico para blocos formais @fail { ... }
    """
    statements = []
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    for line in lines:
        stmt = parse_single_violation_line(line)
        if stmt is not None:
            if isinstance(stmt, list):
                statements.extend(stmt)
            else:
                statements.append(stmt)

    return statements


def parse_violation_statements_from_event_body(body: str) -> List[Dict]:
    """
    Parser específico para corpo do event { ... }
    Ignora guardas imperativas e mantém apenas ações de violação.
    """
    statements = []

    # 1. Captura logs RVMLogging.out.println(...)
    log_pattern = re.compile(
        r'RVMLogging\.out\.println\s*\(\s*Level\.([A-Z_]+)\s*,\s*(.+?)\s*\)\s*;',
        flags=re.DOTALL
    )
    for m in log_pattern.finditer(body):
        level = m.group(1)
        message = normalize_java_expression(m.group(2).strip())
        statements.append({
            "type": "log",
            "level": level,
            "message": message
        })

    # 2. Captura __RESET
    if "__RESET" in body:
        statements.append({
            "type": "command",
            "name": "__RESET"
        })

    # 3. Aqui você pode adicionar outros padrões no futuro,
    # como throw new ..., System.err.println(...), etc.

    return statements


def parse_single_violation_line(line: str):
    log_re = re.compile(
        r'RVMLogging\.out\.println\s*\(\s*Level\.([A-Z_]+)\s*,\s*(.+?)\s*\)\s*;'
    )

    log_match = log_re.search(line)
    if log_match:
        level = log_match.group(1)
        message = normalize_java_expression(log_match.group(2).strip())
        return {
            "type": "log",
            "level": level,
            "message": message
        }

    if "__RESET" in line:
        return {
            "type": "command",
            "name": "__RESET"
        }

    # Se quiser preservar outras linhas de @fail:
    return {
        "type": "raw",
        "value": line
    }


def normalize_java_expression(expr: str) -> str:
    """
    Não tenta interpretar Java profundamente.
    Só limpa o suficiente para o IR ficar menos barulhento.
    """
    expr = normalize(expr)

    # Remove aspas simples de string literal inteira
    if expr.startswith('"') and expr.endswith('"'):
        return expr[1:-1]

    return expr


def deduplicate_statements(statements: List[Dict]) -> List[Dict]:
    seen = set()
    result = []

    for stmt in statements:
        key = repr(stmt)
        if key not in seen:
            seen.add(key)
            result.append(stmt)

    return result


# ==========================================================
# UTILS
# ==========================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()