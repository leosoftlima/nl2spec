import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set

from nl2spec.core.convert.ere import (
    extract_events,
    extract_signature,
    extract_violation_block,
)


# ==========================================================
# ERE extraction for NL only
# ==========================================================

_ERE_LINE_RE = re.compile(r"(?im)^\s*ere\s*:\s*(.*)$")

_ERE_TOKEN_RE = re.compile(
    r"""\s*(
        [A-Za-z_][A-Za-z0-9_]*   |   # event names and special identifiers
        \(|\)|\||\+|\*|\?            # operators
    )\s*""",
    flags=re.VERBOSE,
)


def extract_ere_expression_nl(
    mop_text: str,
    declared_events: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    declared_events = declared_events or set()

    raw = _extract_ere_formula_text_nl(mop_text)
    if not raw:
        return {"raw": "", "tokens": [], "ast": None}

    raw = _normalize_ere_expression_nl(raw).rstrip(";").strip()
    if not raw:
        return {"raw": "", "tokens": [], "ast": None}

    tokens = _tokenize_ere_nl(raw)
    ast = _parse_ere_nl(tokens, declared_events)

    return {
        "raw": raw,
        "tokens": tokens,
        "ast": ast,
    }


def _extract_ere_formula_text_nl(text: str) -> str:
    m = _ERE_LINE_RE.search(text)
    if not m:
        return ""

    first_line = m.group(1).strip()
    start = m.end()

    lines = [first_line] if first_line else []

    for ln in text[start:].splitlines():
        stripped = ln.strip()

        if not stripped:
            continue

        if stripped.startswith("@"):
            break

        if stripped == "}":
            break

        lines.append(stripped)

    return " ".join(lines).strip()


def _normalize_ere_expression_nl(expr: str) -> str:
    return re.sub(r"\s+", " ", expr).strip()


def _tokenize_ere_nl(expr: str) -> List[str]:
    tokens: List[str] = []
    pos = 0

    while pos < len(expr):
        match = _ERE_TOKEN_RE.match(expr, pos)
        if not match:
            raise ValueError(f"Invalid ERE token near: {expr[pos:pos+30]!r}")
        tokens.append(match.group(1))
        pos = match.end()

    return tokens


class _ERENLParser:
    def __init__(self, tokens: List[str], declared_events: Set[str]):
        self.tokens = tokens
        self.pos = 0
        self.declared_events = declared_events

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of ERE expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def parse(self) -> Dict[str, Any]:
        node = self.parse_union()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token at end of ERE: {self.peek()!r}")
        return node

    def parse_union(self) -> Dict[str, Any]:
        left = self.parse_concat()
        options = [left]

        while self.peek() == "|":
            self.consume("|")
            options.append(self.parse_concat())

        if len(options) == 1:
            return left

        return {
            "type": "union",
            "options": options,
        }

    def parse_concat(self) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []

        while True:
            tok = self.peek()
            if tok is None or tok in {")", "|"}:
                break
            parts.append(self.parse_postfix())

        if not parts:
            raise ValueError("Empty concatenation in ERE")

        if len(parts) == 1:
            return parts[0]

        return {
            "type": "concat",
            "parts": parts,
        }

    def parse_postfix(self) -> Dict[str, Any]:
        node = self.parse_primary()

        while self.peek() in {"+", "*", "?"}:
            op = self.consume()
            if op == "+":
                node = {"type": "plus", "expr": node}
            elif op == "*":
                node = {"type": "star", "expr": node}
            elif op == "?":
                node = {"type": "optional", "expr": node}

        return node

    def parse_primary(self) -> Dict[str, Any]:
        tok = self.peek()

        if tok == "(":
            self.consume("(")
            inner = self.parse_union()
            self.consume(")")
            return inner

        if tok is None:
            raise ValueError("Unexpected end of ERE expression")

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            name = self.consume()

            if name in {"epsilon", "empty"}:
                return {"type": name}

            if self.declared_events and name not in self.declared_events:
                raise ValueError(f"ERE references undeclared event: {name}")

            return {
                "type": "event",
                "name": name,
            }

        raise ValueError(f"Unexpected token in ERE: {tok!r}")


def _parse_ere_nl(tokens: List[str], declared_events: Set[str]) -> Dict[str, Any]:
    parser = _ERENLParser(tokens, declared_events)
    return parser.parse()


class ERENL:
    """
    MOP -> NL renderer for ERE specifications.

    Reads the .mop source directly and fills the ERE task template
    without converting the specification to JSON IR first.
    """

    def extract_context(
        self,
        source: Union[str, Path],
        domain: Optional[str] = None,
        spec_id: Optional[str] = None,
    ) -> dict:
        mop_text, resolved_spec_id, resolved_domain = self._read_mop_source(
            source=source,
            domain=domain,
            spec_id=spec_id,
        )

        signature = extract_signature(mop_text)
        methods = extract_events(mop_text)
        declared_events = {m["name"] for m in methods}
        ere_expression = extract_ere_expression_nl(
            mop_text,
            declared_events=declared_events,
        )
        violation = extract_violation_block(mop_text)

        return {
            "SPEC_ID": resolved_spec_id,
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(methods),
            "ERE_BLOCK": self._render_ere_block(ere_expression),
            "VIOLATION_TAG": self._render_violation_tag(violation),
            "VIOLATION_STATEMENTS": self._render_violation_statements(violation),
            "HAS_RESET": self._render_has_reset(violation),
            "DOMAIN": resolved_domain,
        }

    # ==========================================================
    # SOURCE
    # ==========================================================

    def _read_mop_source(
        self,
        source: Union[str, Path],
        domain: Optional[str],
        spec_id: Optional[str],
    ) -> tuple:
        if isinstance(source, Path):
            mop_path = source
        elif isinstance(source, str):
            mop_path = Path(source)
        else:
            raise TypeError("ERENL.extract_context expected str or Path.")

        if not mop_path.exists() or not mop_path.is_file():
            raise FileNotFoundError(f"MOP file not found: {mop_path}")

        mop_text = mop_path.read_text(encoding="utf-8", errors="replace")
        resolved_spec_id = spec_id or mop_path.stem
        resolved_domain = domain or self._detect_domain_from_path(mop_path)

        return mop_text, resolved_spec_id, resolved_domain

    def _detect_domain_from_path(self, path: Path) -> str:
        for part in path.parts:
            if part in {"io", "lang", "util", "net"}:
                return part
        return "unknown"

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature_parameters(self, signature: Dict[str, Any]) -> str:
        params = signature.get("parameters", []) if isinstance(signature, dict) else []

        if not isinstance(params, list) or not params:
            return "none"

        rendered = []
        for p in params:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type", "<?>")
            pname = p.get("name", "<?>")
            rendered.append(f"{ptype} {pname}")

        return ", ".join(rendered) if rendered else "none"

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _render_events_block(self, methods: List[Dict[str, Any]]) -> str:
        if not methods:
            return "No monitored events were provided."

        lines: List[str] = []

        for idx, ev in enumerate(methods, 1):
            name = ev.get("name", f"event_{idx}")
            action = (ev.get("action") or "event").strip()
            timing = (ev.get("timing") or "").strip()
            params = self._format_parameters(ev.get("parameters"))
            returning = self._format_returning(ev.get("returning"))

            lines.append(
                f"{idx}) {name} is observed {timing} the call as {self._article_for(action)} {action}."
            )

            if params and params != "none":
                lines.append(f"   Parameters: {params}.")
            else:
                lines.append("   Parameters: none.")

            if returning:
                lines.append(f"   Returning: {returning}.")

            pointcut_lines = self._render_pointcut_semantic(ev)
            if pointcut_lines:
                lines.extend(pointcut_lines)
            else:
                lines.append("   No pointcut atoms were provided.")

            lines.append("")

        return "\n".join(lines).strip()

    def _render_pointcut_semantic(self, ev: Dict[str, Any]) -> List[str]:
        funcs = ev.get("function", []) or []
        ops = ev.get("operation", []) or []

        if not funcs:
            return []

        lines: List[str] = []

        if not ops:
            lines.append("   It matches the following top-level pointcut atom:")
            lines.append(f"   - {self._describe_pointcut_atom(funcs[0])}")
            return lines

        connectors = [str(op).strip() for op in ops if str(op).strip()]
        unique_connectors = sorted(set(connectors))

        if len(unique_connectors) == 1:
            connector = unique_connectors[0]
            connector_text = self._connector_to_text(connector)
            if connector == "&&":
                lines.append(
                    f"   It matches when all the following top-level pointcut atoms are combined using {connector_text}, in this order:"
                )
            else:
                lines.append(
                    f"   It matches when the following top-level pointcut atoms are combined using {connector_text}, in this order:"
                )
        else:
            rendered_ops = ", ".join(connectors) if connectors else "none"
            lines.append(
                "   It matches the following top-level pointcut atoms, in order. "
                f"The connectors between consecutive atoms are: {rendered_ops}."
            )

        for fn in funcs:
            lines.append(f"   - {self._describe_pointcut_atom(fn)}")

        return lines

    def _describe_pointcut_atom(self, fn: Dict[str, Any]) -> str:
        if not isinstance(fn, dict):
            return "an unspecified pointcut atom"

        fname = (fn.get("name") or "unknown").strip()
        args = fn.get("arguments", []) or []
        negated = bool(fn.get("negated", False))

        values = []
        for a in args:
            if isinstance(a, dict):
                value = (a.get("value") or "").strip()
                if value:
                    values.append(value)

        text = self._describe_pointcut_atom_by_name(fname, values)

        if negated:
            text = self._negate_pointcut_description(fname, values, text)

        return text

    def _describe_pointcut_atom_by_name(self, fname: str, values: List[str]) -> str:
        joined = ", ".join(values) if values else ""

        if fname == "call":
            return f"a call matching {joined}"

        if fname == "target":
            return f"the target is {joined}"

        if fname == "args":
            return f"the arguments are {joined}"

        if fname == "thread":
            return f"the executing thread is {joined}"

        if fname == "condition":
            return f"the condition {joined} holds"

        if fname == "cflow":
            return f"the control flow includes {joined}"

        if fname == "within":
            return f"the execution is within {joined}"

        if fname == "this":
            return f"the current object is {joined}"

        if fname == "if":
            return f"the guard {joined} holds"

        if joined:
            return f"{fname}({joined})"

        return fname

    def _negate_pointcut_description(self, fname: str, values: List[str], default_text: str) -> str:
        joined = ", ".join(values) if values else ""

        if fname == "target":
            return f"the target is not {joined}"

        if fname == "thread":
            return f"the executing thread is not {joined}"

        if fname == "condition":
            return f"the condition {joined} does not hold"

        if fname == "cflow":
            return f"the control flow does not include {joined}"

        if fname == "call":
            return f"no call matching {joined} occurs"

        return f"it is not the case that {default_text}"

    def _connector_to_text(self, op: str) -> str:
        mapping = {
            "&&": "logical AND",
            "||": "logical OR",
        }
        return mapping.get(op, op)

    def _format_parameters(self, params) -> str:
        if not params:
            return "none"

        rendered = []
        for p in params:
            if not isinstance(p, dict):
                continue
            rendered.append(f'{p.get("type", "<?>")} {p.get("name", "<?>")}')

        return ", ".join(rendered) if rendered else "none"

    def _format_returning(self, returning):
        if not returning or not isinstance(returning, dict):
            return None

        rtype = (returning.get("type") or "").strip()
        rname = (returning.get("name") or "").strip()

        if not rtype and not rname:
            return None

        return f"{rtype} {rname}".strip()

    def _article_for(self, text: str) -> str:
        if not text:
            return "a"
        return "an" if text[0].lower() in {"a", "e", "i", "o", "u"} else "a"

    # ==========================================================
    # ERE
    # ==========================================================

    def _render_ere_block(self, ere_expression: Dict[str, Any]) -> str:
        if not isinstance(ere_expression, dict):
            return "No ERE expression was provided."

        ast = ere_expression.get("ast")
        if not ast:
            return "No ERE expression was provided."

        if self._is_complex_ere(ast):
            description = self._describe_ere_hierarchical(ast)
        else:
            description = self._describe_ere(ast)

        declared = self._collect_declared_names_for_text(ast)
        declared_line = (
            "Reconstruct the ERE expression from this behavioral description using only the declared event names"
            + (f": {', '.join(declared)}." if declared else ".")
        )

        lines = [
            declared_line,
            "",
            "Behavioral pattern to encode:",
            description,
            "",
            "Preserve the exact ordering, repetition, alternatives, and grouping implied by the described behavior.",
            "When a grouped sequence is said to repeat, the repetition applies to the entire group.",
        ]

        return "\n".join(lines).strip()

    def _collect_declared_names_for_text(self, node: Dict[str, Any]) -> List[str]:
        found: List[str] = []

        def visit(n: Any) -> None:
            if not isinstance(n, dict):
                return

            ntype = n.get("type")

            if ntype == "event":
                name = n.get("name")
                if name and name not in found:
                    found.append(name)
                return

            if ntype in {"epsilon", "empty"}:
                if ntype not in found:
                    found.append(ntype)
                return

            if ntype in {"star", "plus", "optional"}:
                visit(n.get("expr"))
                return

            if ntype == "concat":
                for part in n.get("parts", []) or []:
                    visit(part)
                return

            if ntype == "union":
                for opt in n.get("options", []) or []:
                    visit(opt)
                return

        visit(node)
        return found

    def _is_complex_ere(self, node: Dict[str, Any]) -> bool:
        metrics = self._collect_ere_metrics(node)

        if metrics["group_repeat_count"] >= 1:
            return True

        if metrics["depth"] >= 4:
            return True

        if metrics["union_count"] >= 2:
            return True

        if metrics["has_epsilon"] and metrics["union_count"] >= 1:
            return True

        if metrics["event_count"] >= 5 and metrics["depth"] >= 3:
            return True

        return False

    def _collect_ere_metrics(self, node: Dict[str, Any], depth: int = 1) -> Dict[str, Any]:
        metrics = {
            "depth": depth,
            "event_count": 0,
            "union_count": 0,
            "group_repeat_count": 0,
            "has_epsilon": False,
        }

        if not isinstance(node, dict):
            return metrics

        ntype = node.get("type")

        if ntype == "event":
            metrics["event_count"] += 1
            return metrics

        if ntype == "epsilon":
            metrics["has_epsilon"] = True
            return metrics

        if ntype == "empty":
            return metrics

        if ntype == "union":
            metrics["union_count"] += 1
            for child in node.get("options", []) or []:
                child_metrics = self._collect_ere_metrics(child, depth + 1)
                metrics = self._merge_ere_metrics(metrics, child_metrics)
            return metrics

        if ntype in {"star", "plus"}:
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                metrics["group_repeat_count"] += 1

            child_metrics = self._collect_ere_metrics(expr, depth + 1)
            metrics = self._merge_ere_metrics(metrics, child_metrics)
            return metrics

        if ntype == "optional":
            child_metrics = self._collect_ere_metrics(node.get("expr"), depth + 1)
            metrics = self._merge_ere_metrics(metrics, child_metrics)
            return metrics

        if ntype == "concat":
            for child in node.get("parts", []) or []:
                child_metrics = self._collect_ere_metrics(child, depth + 1)
                metrics = self._merge_ere_metrics(metrics, child_metrics)
            return metrics

        return metrics

    def _merge_ere_metrics(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "depth": max(a["depth"], b["depth"]),
            "event_count": a["event_count"] + b["event_count"],
            "union_count": a["union_count"] + b["union_count"],
            "group_repeat_count": a["group_repeat_count"] + b["group_repeat_count"],
            "has_epsilon": a["has_epsilon"] or b["has_epsilon"],
        }

    def _describe_ere(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "The forbidden behavior follows the declared monitored events."

        ntype = node.get("type")

        if ntype == "event":
            name = node.get("name", "unknown")
            return (
                f"The forbidden behavior consists of a single occurrence "
                f"of the {name} event."
            )

        if ntype == "epsilon":
            return (
                "This part of the forbidden behavior may use the epsilon alternative, "
                "meaning that no event is required at this point."
            )

        if ntype == "empty":
            return (
                "This part of the forbidden behavior denotes the empty language "
                "and does not correspond to an observable event sequence."
            )

        if ntype == "concat":
            parts = node.get("parts", []) or []

            special = self._try_special_concat_patterns(parts)
            if special:
                return special

            rendered = [self._describe_ere_part(p) for p in parts]
            lines = ["The forbidden behavior follows this order:"]
            for idx, item in enumerate(rendered, 1):
                lines.append(f"{idx}) {item}.")
            return "\n".join(lines)

        if ntype == "union":
            options = node.get("options", []) or []
            rendered = [self._describe_ere_option(o) for o in options]
            lines = ["The forbidden behavior allows one of the following alternatives:"]
            for item in rendered:
                lines.append(f"- {item}.")
            return "\n".join(lines)

        if ntype == "plus":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                inner = self._describe_grouped_subexpression(expr)
                return (
                    "The following grouped behavior must repeat one or more times:\n"
                    f"{inner}"
                )

            inner = self._describe_ere_atom(expr)
            return f"{inner} must occur one or more times."

        if ntype == "star":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                inner = self._describe_grouped_subexpression(expr)
                return (
                    "The following grouped behavior may repeat any number of times, including zero:\n"
                    f"{inner}"
                )

            inner = self._describe_ere_atom(expr)
            return f"{inner} may occur any number of times, including zero."

        if ntype == "optional":
            inner = self._describe_ere_atom(node.get("expr"))
            return f"{inner} may occur optionally."

        return "The forbidden behavior follows the declared monitored events."

    def _describe_grouped_subexpression(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "the described grouped behavior"

        ntype = node.get("type")

        if ntype == "concat":
            parts = node.get("parts", []) or []
            lines = []
            for idx, part in enumerate(parts, 1):
                lines.append(f"{idx}) {self._describe_ere_part(part)}.")
            return "\n".join(lines)

        if ntype == "union":
            options = node.get("options", []) or []
            lines = ["One of the following alternatives is allowed within the group:"]
            for opt in options:
                lines.append(f"- {self._describe_ere_option(opt)}.")
            return "\n".join(lines)

        return self._describe_ere_part(node)

    def _describe_ere_part(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "the described behavior occurs"

        ntype = node.get("type")

        if ntype == "event":
            return f"the {node.get('name', 'unknown')} event occurs"

        if ntype == "epsilon":
            return "the epsilon alternative is allowed at this point, meaning that no event is required"

        if ntype == "empty":
            return "the empty language is specified at this point"

        if ntype == "plus":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                inner = self._describe_grouped_subexpression(expr)
                return (
                    "the following grouped behavior must repeat one or more times:\n"
                    f"{inner}"
                )

            inner = self._describe_ere_atom(expr)
            return f"{inner} occurs one or more times"

        if ntype == "star":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                inner = self._describe_grouped_subexpression(expr)
                return (
                    "the following grouped behavior may repeat any number of times, including zero:\n"
                    f"{inner}"
                )

            inner = self._describe_ere_atom(expr)
            return f"{inner} may repeat any number of times, including zero"

        if ntype == "optional":
            inner = self._describe_ere_atom(node.get("expr"))
            return f"{inner} may occur optionally"

        if ntype == "union":
            options = [self._describe_ere_option(o) for o in node.get("options", []) or []]
            return "one of the following alternatives occurs: " + "; ".join(options)

        if ntype == "concat":
            parts = [self._describe_ere_part(p) for p in node.get("parts", []) or []]
            return " then ".join(parts)

        return "the described behavior occurs"

    def _describe_ere_atom(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "the event"

        ntype = node.get("type")

        if ntype == "event":
            return f"the {node.get('name', 'unknown')} event"

        if ntype == "epsilon":
            return "the epsilon alternative"

        if ntype == "empty":
            return "the empty language"

        if ntype == "union":
            options = [self._describe_ere_option(o) for o in node.get("options", []) or []]
            if not options:
                return "one of the declared alternatives"
            if len(options) == 1:
                return options[0]
            return "either " + " or ".join(options)

        if ntype == "concat":
            return "the grouped sequence in which " + self._describe_ere_part(node)

        if ntype == "plus":
            inner = self._describe_ere_atom(node.get("expr"))
            return f"{inner}, repeated one or more times"

        if ntype == "star":
            inner = self._describe_ere_atom(node.get("expr"))
            return f"{inner}, repeated zero or more times"

        if ntype == "optional":
            inner = self._describe_ere_atom(node.get("expr"))
            return f"{inner}, optionally"

        return "the described behavior"

    def _describe_ere_option(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "the described behavior"

        ntype = node.get("type")

        if ntype == "event":
            return f"the {node.get('name', 'unknown')} event"

        if ntype == "epsilon":
            return "the epsilon alternative, meaning that no event is required"

        if ntype == "empty":
            return "the empty language"

        return self._describe_ere_part(node)

    def _describe_ere_hierarchical(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "The forbidden behavior follows the declared monitored events."

        if node.get("type") != "concat":
            return self._describe_hierarchical_node_as_text(node)

        lines = ["The forbidden behavior follows this order:"]
        parts = node.get("parts", []) or []

        for idx, part in enumerate(parts, 1):
            lines.extend(self._describe_hierarchical_lines(part, level=0, index=str(idx)))

        return "\n".join(lines)

    def _describe_hierarchical_node_as_text(self, node: Dict[str, Any]) -> str:
        lines = self._describe_hierarchical_lines(node, level=0, index="1")
        return "\n".join(lines)

    def _describe_hierarchical_lines(self, node: Dict[str, Any], level: int, index: str) -> List[str]:
        indent = "   " * level
        lines: List[str] = []

        if not isinstance(node, dict):
            lines.append(f"{indent}{index}) The described behavior occurs.")
            return lines

        ntype = node.get("type")

        if ntype == "event":
            lines.append(f"{indent}{index}) The {node.get('name', 'unknown')} event occurs.")
            return lines

        if ntype == "epsilon":
            lines.append(
                f"{indent}{index}) The epsilon alternative may be chosen, meaning that no event is required at this point."
            )
            return lines

        if ntype == "empty":
            lines.append(
                f"{indent}{index}) The empty-language alternative is specified at this point."
            )
            return lines

        if ntype == "star":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                lines.append(
                    f"{indent}{index}) A grouped main sequence may repeat any number of times, including zero. This grouped sequence consists of:"
                )
                lines.extend(self._describe_group_body(expr, level + 1, f"{index}"))
                return lines

            atom = self._describe_ere_atom(expr)
            lines.append(f"{indent}{index}) {atom.capitalize()} may repeat any number of times, including zero.")
            return lines

        if ntype == "plus":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") in {"concat", "union"}:
                lines.append(
                    f"{indent}{index}) A grouped sequence must repeat one or more times. This grouped sequence consists of:"
                )
                lines.extend(self._describe_group_body(expr, level + 1, f"{index}"))
                return lines

            atom = self._describe_ere_atom(expr)
            lines.append(f"{indent}{index}) {atom.capitalize()} occurs one or more times.")
            return lines

        if ntype == "optional":
            atom = self._describe_ere_atom(node.get("expr"))
            lines.append(f"{indent}{index}) {atom.capitalize()} may occur optionally.")
            return lines

        if ntype == "union":
            lines.append(f"{indent}{index}) One of the following alternatives is allowed:")
            options = node.get("options", []) or []
            for opt in options:
                opt_indent = "   " * (level + 1)
                lines.append(f"{opt_indent}- {self._describe_union_option_sentence(opt)}")
            return lines

        if ntype == "concat":
            lines.append(f"{indent}{index}) A grouped sequence occurs in this order:")
            parts = node.get("parts", []) or []
            for sub_idx, part in enumerate(parts, 1):
                lines.extend(self._describe_hierarchical_lines(part, level + 1, f"{index}.{sub_idx}"))
            return lines

        lines.append(f"{indent}{index}) The described behavior occurs.")
        return lines

    def _describe_group_body(self, node: Dict[str, Any], level: int, parent_index: str) -> List[str]:
        lines: List[str] = []

        if not isinstance(node, dict):
            return lines

        ntype = node.get("type")

        if ntype == "concat":
            parts = node.get("parts", []) or []
            for idx, part in enumerate(parts, 1):
                lines.extend(self._describe_hierarchical_lines(part, level, f"{parent_index}.{idx}"))
            return lines

        if ntype == "union":
            indent = "   " * level
            lines.append(f"{indent}{parent_index}.1) One of the following alternatives is allowed:")
            for opt in node.get("options", []) or []:
                opt_indent = "   " * (level + 1)
                lines.append(f"{opt_indent}- {self._describe_union_option_sentence(opt)}")
            return lines

        lines.extend(self._describe_hierarchical_lines(node, level, f"{parent_index}.1"))
        return lines

    def _describe_union_option_sentence(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "the described behavior occurs."

        ntype = node.get("type")

        if ntype == "event":
            return f"the {node.get('name', 'unknown')} event occurs."

        if ntype == "epsilon":
            return "the epsilon alternative is chosen, meaning that no event is required at this point."

        if ntype == "empty":
            return "the empty-language alternative is chosen."

        if ntype == "plus":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") == "event":
                return f"the {expr.get('name', 'unknown')} event occurs one or more times."
            return f"{self._describe_ere_atom(expr).capitalize()} occurs one or more times."

        if ntype == "star":
            expr = node.get("expr")
            if isinstance(expr, dict) and expr.get("type") == "event":
                return f"the {expr.get('name', 'unknown')} event may repeat any number of times, including zero."
            return f"{self._describe_ere_atom(expr).capitalize()} may repeat any number of times, including zero."

        if ntype == "optional":
            return f"{self._describe_ere_atom(node.get('expr')).capitalize()} may occur optionally."

        if ntype == "concat":
            return f"the grouped sequence occurs in this order: {self._describe_ere_part(node)}."

        if ntype == "union":
            return f"one of the following alternatives occurs: {self._describe_ere_part(node)}."

        return "the described behavior occurs."

    def _try_special_concat_patterns(self, parts: List[Dict[str, Any]]) -> Optional[str]:
        if len(parts) != 2:
            return None

        first, second = parts

        if (
            first.get("type") == "event"
            and second.get("type") == "plus"
            and isinstance(second.get("expr"), dict)
            and second["expr"].get("type") == "event"
            and first.get("name") == second["expr"].get("name")
        ):
            name = first.get("name", "unknown")
            return (
                f"The forbidden behavior starts with one {name} event. "
                f"After that initial {name}, one or more additional {name} events must occur."
            )

        if first.get("type") == "event" and second.get("type") == "plus":
            first_name = first.get("name", "unknown")
            second_expr = second.get("expr", {})
            second_name = second_expr.get("name", "unknown") if isinstance(second_expr, dict) else "unknown"
            return (
                f"The forbidden behavior starts when the {first_name} event occurs. "
                f"After that, the {second_name} event must occur one or more times."
            )

        if first.get("type") == "plus" and second.get("type") == "event":
            first_expr = first.get("expr", {})
            first_name = first_expr.get("name", "unknown") if isinstance(first_expr, dict) else "unknown"
            second_name = second.get("name", "unknown")
            return (
                f"The forbidden behavior begins with one or more occurrences of the "
                f"{first_name} event. After that, the {second_name} event occurs."
            )

        if first.get("type") == "event" and second.get("type") == "event":
            first_name = first.get("name", "unknown")
            second_name = second.get("name", "unknown")
            return (
                f"The forbidden behavior occurs when the {first_name} event is "
                f"followed by the {second_name} event."
            )

        return None

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation_tag(self, violation: Dict[str, Any]) -> str:
        return (violation.get("tag") or "").strip()

    def _render_violation_statements(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {}) or {}
        statements = body.get("statements", []) or []

        if not statements:
            return "none"

        lines = []

        for idx, stmt in enumerate(statements, 1):
            if not isinstance(stmt, dict):
                continue

            stype = stmt.get("type", "")

            if stype == "log":
                level = stmt.get("level", "")
                message = stmt.get("message", "")
                lines.append(f"{idx}) type=log; level={level}; message={message}")

            elif stype == "command":
                name = stmt.get("name", "")
                lines.append(f"{idx}) type=command; name={name}")

            elif stype == "raw":
                value = stmt.get("value", "")
                lines.append(f"{idx}) type=raw; value={value}")

            else:
                lines.append(f"{idx}) type={stype}")

        return "\n".join(lines) if lines else "none"

    def _render_has_reset(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {}) or {}
        return str(bool(body.get("has_reset", False))).lower()