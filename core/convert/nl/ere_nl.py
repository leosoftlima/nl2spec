from typing import List, Dict, Any


class ERENL:
    """
    IR -> NL renderer for ERE specifications using only the canonical new schema.

    Expected placeholders:
      - SPEC_ID
      - SIGNATURE_PARAMETERS
      - EVENT_BLOCK
      - ERE_BLOCK
      - VIOLATION_TAG
      - VIOLATION_STATEMENTS
      - HAS_RESET
      - DOMAIN
    """

    def extract_context(self, spec: dict, domain: str) -> dict:
        ir = spec.get("ir", {}) or {}
        signature = spec.get("signature", {}) or {}

        return {
            "SPEC_ID": (spec.get("id") or "").strip(),
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(ir),
            "ERE_BLOCK": self._render_ere_block(ir),
            "VIOLATION_TAG": self._render_violation_tag(ir),
            "VIOLATION_STATEMENTS": self._render_violation_statements(ir),
            "HAS_RESET": self._render_has_reset(ir),
            "DOMAIN": domain,
        }

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature_parameters(self, signature: dict) -> str:
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

    def _get_methods(self, ir: dict) -> List[dict]:
        events = ir.get("events", [])

        if not isinstance(events, list):
            return []

        methods: List[dict] = []

        for item in events:
            if not isinstance(item, dict):
                continue

            body = item.get("body", {})
            if not isinstance(body, dict):
                continue

            inner = body.get("methods", [])
            if isinstance(inner, list):
                methods.extend([m for m in inner if isinstance(m, dict)])

        return methods

    def _render_events_block(self, ir: dict) -> str:
        methods = self._get_methods(ir)

        if not methods:
            return "No monitored events were provided."

        lines = []

        for idx, ev in enumerate(methods, 1):
            name = ev.get("name", f"event_{idx}")
            action = (ev.get("action") or "event").strip()
            timing = (ev.get("timing") or "").strip()
            params = self._format_parameters(ev.get("parameters"))
            returning = self._format_returning(ev.get("returning"))
            pointcut = self._render_pointcut(ev)

            article = self._article_for(action)

            if returning:
                lines.append(
                    f"{idx}) {name} is observed {timing} the call as {article} {action}. "
                    f"It receives {params}, returns {returning}, and matches:"
                )
                
            else:
                lines.append(
                    f"{idx}) {name} is observed {timing} the call as {article} {action}. "
                    f"It receives {params} and matches:"
                )
            lines.append(f"   Top-level operators between consecutive pointcut atoms: {self._render_operations_inline(ev)}")
            lines.append(f"   {pointcut}")
            lines.append("")

        return "\n".join(lines).strip()
    
    def _render_operations_inline(self, ev: dict) -> str:
        ops = ev.get("operation", []) or []
        return ", ".join(ops) if ops else "none"

    def _render_pointcut(self, ev: dict) -> str:
        funcs = ev.get("function", []) or []
        ops = ev.get("operation", []) or []

        if not funcs:
            return "<missing pointcut>"

        pieces = []

        for i, fn in enumerate(funcs):
            if not isinstance(fn, dict):
                continue

            fname = fn.get("name", "unknown")
            args = fn.get("arguments", []) or []
            negated = fn.get("negated", False)

            inner_parts = []
            for a in args:
                if not isinstance(a, dict):
                    continue
                value = a.get("value", "")
                if value:
                    inner_parts.append(value)

            inner = ", ".join(inner_parts)

            atom = f"{fname}({inner})" if fname != "unknown" else inner
            if negated:
                atom = f"!{atom}"

            pieces.append(atom)

            if i < len(ops):
                pieces.append(ops[i])

        return " ".join(pieces).strip()

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

    def _render_ere_block(self, ir: dict) -> str:
        ere = ir.get("ere", {}) or {}
        expression = (ere.get("expression") or "").strip()

        if not expression:
            return "No ERE expression was provided."

        return expression

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation_tag(self, ir: dict) -> str:
        viol = ir.get("violation", {}) or {}
        return (viol.get("tag") or "").strip()

    def _render_violation_statements(self, ir: dict) -> str:
        viol = ir.get("violation", {}) or {}
        body = viol.get("body", {}) or {}
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
                lines.append(f'{idx}) type=log; level={level}; message={message}')

            elif stype == "command":
                name = stmt.get("name", "")
                lines.append(f'{idx}) type=command; name={name}')

            elif stype == "raw":
                value = stmt.get("value", "")
                lines.append(f'{idx}) type=raw; value={value}')

            else:
                lines.append(f'{idx}) type={stype}')

        return "\n".join(lines) if lines else "none"

    def _render_has_reset(self, ir: dict) -> str:
        viol = ir.get("violation", {}) or {}
        body = viol.get("body", {}) or {}
        return str(bool(body.get("has_reset", False))).lower()