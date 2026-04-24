from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re

from nl2spec.core.convert.event import extract_signature
from nl2spec.core.convert.ltl import (
    extract_formula,
    extract_violation_block,
    extract_events,
)


class LTLNL:
    """
    MOP -> NL renderer for LTL specifications.

    Direct .mop -> NL (NO IR).
    """

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def extract_context(
        self,
        source: Union[str, Path],
        domain: Optional[str] = None,
        spec_id: Optional[str] = None,
    ) -> dict:

        mop_text, resolved_id, resolved_domain = self._read_mop_source(
            source, domain, spec_id
        )

        signature = extract_signature(mop_text)

        # ⚠️ CORREÇÃO
        events = extract_events(mop_text)
        methods = self._unwrap_methods(events)

        ltl = extract_formula(mop_text)
        violation = extract_violation_block(mop_text)

        return {
            "SPEC_NAME": resolved_id,
            "SIGNATURE_PARAMETERS": self._render_signature(signature),
            "EVENT_BLOCK": self._render_events(methods),
            "LTL_BLOCK": self._render_ltl_block(ltl),
            "LTL_SEMANTICS": self._render_ltl_semantics(ltl),
            "VIOLATION_SEMANTICS": self._render_violation(violation),
            "DOMAIN": resolved_domain,
        }

    # ==========================================================
    # SOURCE
    # ==========================================================

    def _read_mop_source(self, source, domain, spec_id):
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"MOP file not found: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")

        resolved_id = spec_id or path.stem
        resolved_domain = domain or self._detect_domain(path)

        return text, resolved_id, resolved_domain

    def _detect_domain(self, path: Path):
        for p in path.parts:
            if p in {"io", "net", "util", "lang"}:
                return p
        return "unknown"

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature(self, signature):
        params = signature.get("parameters", []) or []

        if not params:
            return "none"

        return "\n".join(
            f"- {p.get('type','<?>')} {p.get('name','<?>')}"
            for p in params
        )

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _unwrap_methods(self, events: List[Dict]) -> List[Dict]:
        methods = []

        for ev in events:
            body = ev.get("body", {})
            methods.extend(body.get("methods", []))

        return methods

    def _render_events(self, methods):
        if not methods:
            return "No monitored events were provided."

        lines = []

        for m in methods:
            name = m.get("name", "")
            timing = m.get("timing", "")
            params = self._render_params(m.get("parameters", []))

            lines.append(
                f"{name} is observed {timing} the call and receives {params}."
            )

        return "\n\n".join(lines)

    def _render_params(self, params):
        if not params:
            return "no parameters"

        items = [
            f"{p.get('type','<?>')} {p.get('name','<?>')}"
            for p in params
        ]

        return self._join(items)

    # ==========================================================
    # LTL
    # ==========================================================

    def _render_ltl_block(self, ltl):
        name = ltl.get("name", "ltl")
        value = ltl.get("value", "")

        if not value:
            return "No temporal formula was provided."

        return f"{name}: {value}"

    def _render_ltl_semantics(self, ltl):
        formula = (ltl.get("value") or "").strip()

        if not formula:
            return "No temporal semantics were provided."

        formula = self._strip_outer(formula)

        m = re.fullmatch(r"\[\]\((.+?) => o (.+?)\)", formula)
        if m:
            return f"whenever {m.group(1)} occurs, {m.group(2)} happens next"

        m = re.fullmatch(r"\[\]\((.+?) => <> (.+?)\)", formula)
        if m:
            return f"whenever {m.group(1)} occurs, {m.group(2)} eventually happens"

        if formula.startswith("[]"):
            inner = self._strip_outer(formula[2:].strip())
            return f"it must always hold that {inner}"

        if formula.startswith("<>"):
            inner = self._strip_outer(formula[2:].strip())
            return f"eventually {inner}"

        return f"the execution must satisfy {formula}"

    def _strip_outer(self, text):
        text = text.strip()
        while text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        return text

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation(self, violation):
        stmts = violation.get("statements", []) or []

        for st in stmts:
            if st.get("type") == "log":
                val = st.get("value", "")
                return self._clean(val)

        return "A violation must be reported."

    def _clean(self, msg):
        msg = msg.strip('"')
        msg = msg.replace("+ __LOC", "")
        return msg.strip()

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _join(self, items):
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"