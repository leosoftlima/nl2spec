from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from nl2spec.core.convert.event import (
    extract_events,
    extract_signature,
    extract_violation_block,
)


class EventNL:
    """
    MOP -> NL renderer for EVENT specifications.

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

        mop_text, resolved_spec_id, resolved_domain = self._read_mop_source(
            source, domain, spec_id
        )

        signature = extract_signature(mop_text)

        # ⚠️ CORREÇÃO CRÍTICA
        events, event_bodies = extract_events(mop_text)

        methods = self._unwrap_methods(events)

        violation = extract_violation_block(mop_text, event_bodies)

        return {
            "SPEC_NAME": resolved_spec_id,
            "SIGNATURE_PARAMETERS_TEXT": self._render_signature(signature),
            "EVENT_BLOCK": self._render_events(methods),
            "VIOLATION_SEMANTICS": self._render_violation(violation),
            "DOMAIN": resolved_domain,
        }

    # ==========================================================
    # SOURCE
    # ==========================================================

    def _read_mop_source(self, source, domain, spec_id):
        mop_path = Path(source)

        if not mop_path.exists():
            raise FileNotFoundError(f"MOP file not found: {mop_path}")

        text = mop_path.read_text(encoding="utf-8", errors="replace")

        resolved_id = spec_id or mop_path.stem
        resolved_domain = domain or self._detect_domain(mop_path)

        return text, resolved_id, resolved_domain

    def _detect_domain(self, path: Path) -> str:
        for p in path.parts:
            if p in {"io", "net", "util", "lang"}:
                return p
        return "unknown"

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature(self, signature: dict) -> str:
        params = signature.get("parameters", [])

        if not params:
            return ""

        rendered = []
        for p in params:
            rendered.append(f'{p.get("type")} {p.get("name")}')

        return f"It should use these parameters: [{', '.join(rendered)}]."

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _unwrap_methods(self, events: List[Dict]) -> List[Dict]:
        methods = []

        for ev in events:
            body = ev.get("body", {})
            methods.extend(body.get("methods", []))

        return methods

    def _render_events(self, methods: List[Dict]) -> str:
        if not methods:
            return "No monitored events were provided."

        blocks = []

        for m in methods:
            blocks.append(self._render_method(m))

        return "\n\n".join(blocks)

    def _render_method(self, m: Dict) -> str:
        name = m.get("name", "")
        timing = m.get("timing", "")
        action = m.get("action", "event")

        params = self._render_params(m.get("parameters", []))
        returning = self._render_returning(m.get("returning"))
        pointcut = self._render_pointcut(m)

        parts = []

        parts.append(f"{name} is observed {timing} the call as a {action}.")

        if params:
            parts.append(f"It takes {params}.")

        if returning:
            parts.append(f"It captures the returned value as {returning}.")

        if pointcut:
            parts.append(f"It matches: {pointcut}.")

        return " ".join(parts)

    def _render_params(self, params):
        if not params:
            return None

        items = [f'{p["type"]} {p["name"]}' for p in params]

        return self._join(items)

    def _render_returning(self, r):
        if not r:
            return None

        return f'{r.get("type")} {r.get("name")}'

    def _render_pointcut(self, m):
        funcs = m.get("procediments", {}).get("function", [])
        ops = m.get("procediments", {}).get("operation", [])

        if not funcs:
            return ""

        parts = []

        for i, f in enumerate(funcs):
            name = f.get("name")
            params = f.get("parameters", [])

            rendered = []
            for p in params:
                val = p.get("value") or p.get("name") or ""
                if val:
                    rendered.append(val)

            atom = f"{name}({', '.join(rendered)})"
            parts.append(atom)

            if i < len(ops):
                parts.append(ops[i])

        return " ".join(parts)

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation(self, violation: dict) -> str:
        stmts = violation.get("body", {}).get("statements", [])

        for st in stmts:
            if st.get("type") == "log":
                msg = st.get("message", "")
                return self._clean(msg)

        return "__DEFAULT_MESSAGE"

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