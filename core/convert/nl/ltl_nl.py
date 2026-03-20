from typing import Dict, List, Optional
import re


class LTLNL:


    def extract_context(self, spec: Dict, domain: str) -> Dict[str, str]:
        signature = spec.get("signature", {}) or {}
        ir = spec.get("ir", {}) or {}

        return {
            "SPEC_NAME": signature.get("name") or spec.get("id", ""),
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(ir.get("events", []) or []),
            "LTL_BLOCK": self._render_ltl_block(ir.get("ltl", {}) or {}),
            "LTL_SEMANTICS": self._render_ltl_semantics(ir.get("ltl", {}) or {}),
            "VIOLATION_SEMANTICS": self._render_violation_semantics(ir.get("violation", {}) or {}),
            "DOMAIN": domain or spec.get("domain", ""),
        }

    # ======================================================
    # SIGNATURE
    # ======================================================

    def _render_signature_parameters(self, signature: Dict) -> str:
        params = signature.get("parameters", []) or []
        if not params:
            return "none"

        lines = []
        for p in params:
            ptype = p.get("type", "<?>")
            pname = p.get("name", "<?>")
            lines.append(f"- {ptype} {pname}")
        return "\n".join(lines)

    # ======================================================
    # EVENTS
    # ======================================================

    def _render_events_block(self, events: List[Dict]) -> str:
        lines: List[str] = []

        for event in events:
            body = event.get("body", {}) or {}
            methods = body.get("methods", []) or []
            for method in methods:
                lines.append(self._render_single_event(method))

        return "\n\n".join(lines) if lines else "No monitored events were provided."

    def _render_single_event(self, method: Dict) -> str:
        name = method.get("name", "<?>")
        timing = method.get("timing", "")
        params = method.get("parameters", []) or []
        returning = method.get("returning")
        pointcut = self._render_pointcut(method)

        pieces: List[str] = []

        if timing:
            pieces.append(f"{name} is an event observed {timing} the call.")
        else:
            pieces.append(f"{name} is a monitored event.")

        if params:
            pieces.append(f"It takes {self._format_parameters_inline(params)}.")
        else:
            pieces.append("It does not take explicit event parameters.")

        if returning:
            rtype = returning.get("type", "<?>")
            rname = returning.get("name", "<?>")
            pieces.append(f"It captures the returned value as {rtype} {rname}.")

        if pointcut:
            pieces.append(f"It matches: {pointcut}")

        return " ".join(pieces)

    def _render_pointcut(self, method: Dict) -> str:
        functions = method.get("function", []) or []
        operations = method.get("operation", []) or []
        if not functions:
            return ""

        rendered_functions = [self._render_function(fn) for fn in functions]
        if len(rendered_functions) == 1:
            return rendered_functions[0]

        out = [rendered_functions[0]]
        for i, fn_text in enumerate(rendered_functions[1:], start=0):
            op = operations[i] if i < len(operations) else "&&"
            out.append(op)
            out.append(fn_text)
        return " ".join(out)

    def _render_function(self, fn: Dict) -> str:
        name = fn.get("name", "")
        params = fn.get("parameters", []) or []
        if not params:
            return f"{name}()"

        values = [p.get("name", "") for p in params]
        return f"{name}({', '.join(values)})"

    def _format_parameters_inline(self, params: List[Dict]) -> str:
        items = [f"{p.get('type', '<?>')} {p.get('name', '<?>')}" for p in params]
        if len(items) == 1:
            return f"{items[0]} as parameter"
        if len(items) == 2:
            return f"{items[0]} and {items[1]} as parameters"
        return f"{', '.join(items[:-1])}, and {items[-1]} as parameters"

    # ======================================================
    # LTL
    # ======================================================

    def _render_ltl_block(self, ltl: Dict) -> str:
        formula_name = (ltl.get("name") or "ltl").strip()
        formula_value = (ltl.get("value") or "").strip()

        if not formula_value:
            return "No temporal formula was provided."

        return f"{formula_name}: {formula_value}"

    def _render_ltl_semantics(self, ltl: Dict) -> str:
        formula_value = (ltl.get("value") or "").strip()
        if not formula_value:
            return "No temporal semantics were provided."

        formula_value = self._strip_outer_parentheses(formula_value)

        # common patterns first
        m = re.fullmatch(r"\[\]\((.+?)\s*=>\s*o\s+(.+?)\)", formula_value)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return (
                f"it must always hold that whenever {left} occurs, then {right} occurs in the next step"
            )

        m = re.fullmatch(r"\[\]\((.+?)\s*=>\s*\(\*\)\s+(.+?)\)", formula_value)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return (
                f"it must always hold that whenever {left} occurs, then {right} must have occurred in the previous step"
            )

        m = re.fullmatch(r"\[\]\((.+?)\s*=>\s*<>\s+(.+?)\)", formula_value)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return (
                f"it must always hold that whenever {left} occurs, then {right} eventually occurs"
            )

        if formula_value.startswith("[]"):
            inner = self._strip_outer_parentheses(formula_value[2:].strip())
            return f"it must always hold that {self._humanize_expression(inner)}"

        if formula_value.startswith("<>"):
            inner = self._strip_outer_parentheses(formula_value[2:].strip())
            return f"at some point, {self._humanize_expression(inner)}"

        return f"the execution must satisfy the temporal rule {formula_value}"

    def _humanize_expression(self, expr: str) -> str:
        expr = self._strip_outer_parentheses(expr)

        m = re.fullmatch(r"(.+?)\s*=>\s*o\s+(.+)", expr)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"whenever {left} occurs, then {right} occurs in the next step"

        m = re.fullmatch(r"(.+?)\s*=>\s*\(\*\)\s+(.+)", expr)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"whenever {left} occurs, then {right} must have occurred in the previous step"

        m = re.fullmatch(r"(.+?)\s*=>\s*<>\s+(.+)", expr)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"whenever {left} occurs, then {right} eventually occurs"

        m = re.fullmatch(r"(.+?)\s+or\s+(.+)", expr)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"either {left} or {right} occurs"

        m = re.fullmatch(r"(.+?)\s+and\s+(.+)", expr)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"both {left} and {right} occur"

        return f"the formula {expr} holds"

    def _humanize_fragment(self, fragment: str) -> str:
        fragment = self._strip_outer_parentheses(fragment.strip())

        if fragment.startswith("o "):
            inner = fragment[2:].strip()
            return f"{self._humanize_fragment(inner)} in the next step"

        if fragment.startswith("(*) "):
            inner = fragment[4:].strip()
            return f"{self._humanize_fragment(inner)} in the previous step"

        if fragment.startswith("<>"):
            inner = fragment[2:].strip()
            return f"{self._humanize_fragment(inner)} eventually"

        if fragment.startswith("[]"):
            inner = fragment[2:].strip()
            return f"{self._humanize_expression(inner)} always"

        m = re.fullmatch(r"(.+?)\s+or\s+(.+)", fragment)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"either {left} or {right}"

        m = re.fullmatch(r"(.+?)\s+and\s+(.+)", fragment)
        if m:
            left = self._humanize_fragment(m.group(1))
            right = self._humanize_fragment(m.group(2))
            return f"both {left} and {right}"

        return fragment

    def _strip_outer_parentheses(self, text: str) -> str:
        text = text.strip()
        while text.startswith("(") and text.endswith(")") and self._is_balanced(text[1:-1]):
            text = text[1:-1].strip()
        return text

    def _is_balanced(self, text: str) -> bool:
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    # ======================================================
    # VIOLATION
    # ======================================================

    def _render_violation_semantics(self, violation: Dict) -> str:
        if not violation:
            return "No violation handler was provided."

        tag = (violation.get("tag") or "violation").strip()
        statements = violation.get("statements", []) or []

        if not statements:
            return f"When the property is {tag}, no explicit handler action is provided"

        actions = [self._render_violation_statement(stmt) for stmt in statements]
        if len(actions) == 1:
            return f"When the property is {tag}, the monitor {actions[0]}"
        if len(actions) == 2:
            return f"When the property is {tag}, the monitor {actions[0]} and {actions[1]}"

        return (
            f"When the property is {tag}, the monitor "
            + ", ".join(actions[:-1])
            + f", and {actions[-1]}"
        )

    def _render_violation_statement(self, stmt: Dict) -> str:
        stype = (stmt.get("type") or "raw").strip().lower()
        value = (stmt.get("value") or "").strip()

        if stype == "log":
            return f"logs {value}"
        return f"executes `{value}`"
