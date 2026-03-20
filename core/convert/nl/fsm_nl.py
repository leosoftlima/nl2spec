from typing import List


class FSMNL:
    """
    IR -> NL renderer for FSM specifications using only the canonical new schema.
    """

    def extract_context(self, spec: dict, domain: str) -> dict:
        ir = spec.get("ir", {}) or {}
        signature = spec.get("signature", {}) or {}

        return {
            "SPEC_NAME": signature.get("name", ""),
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(ir),
            "FSM_BLOCK": self._render_fsm_block(ir),
            "FSM_SEMANTICS": self._extract_fsm_semantics(ir),
            "VIOLATION_SEMANTICS": self._extract_violation(ir),
            "DOMAIN": domain,
        }

    def _render_signature_parameters(self, signature: dict) -> str:
        params = signature.get("parameters", []) or []

        if not params:
            return "none"

        return ", ".join(
            f'{p.get("type", "<?>")} {p.get("name", "<?>")}'
            for p in params
        )

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
            action = ev.get("action", "event")
            timing = ev.get("timing", "")
            params = self._format_parameters(ev.get("parameters"))
            returning = self._format_returning(ev.get("returning"))
            pointcut = self._render_pointcut(ev)

            if returning:
                lines.append(
                    f"{idx}) {name} is a {action} observed {timing} the call. "
                    f"It takes {params} as parameter, returns {returning}, and matches:"
                )
            else:
                lines.append(
                    f"{idx}) {name} is a {action} observed {timing} the call. "
                    f"It takes {params} as parameter and matches:"
                )

            lines.append(f"   {pointcut}")
            lines.append("")

        return "\n".join(lines).strip()

    def _render_pointcut(self, ev: dict) -> str:
        funcs = ev.get("function", []) or []
        ops = ev.get("operation", []) or []

        if not funcs:
            return "<missing pointcut>"

        pieces = []

        for i, fn in enumerate(funcs):
            fname = fn.get("name", "unknown")
            params = fn.get("parameters", []) or []

            inner_parts = []
            for p in params:
                ret = p.get("return", "")
                name = p.get("name", "")
                inner_parts.append(f"{ret} {name}".strip() if ret else name)

            inner = ", ".join(inner_parts)

            if fname == "unknown":
                pieces.append(inner)
            else:
                pieces.append(f"{fname}({inner})")

            if i < len(ops):
                pieces.append(ops[i])

        return " ".join(pieces).strip()

    def _format_parameters(self, params):
        if not params:
            return "none"
        return ", ".join(
            f'{p.get("type", "<?>")} {p.get("name", "<?>")}'
            for p in params
        )

    def _format_returning(self, returning):
        if not returning:
            return None
        return f'{returning.get("type", "<?>")} {returning.get("name", "<?>")}'

    # ==========================================================
    # FSM
    # ==========================================================
    def _render_state_description(self, state_name: str, transitions: list, initial_state: str) -> str:
        parts = []

        intro = (
            f"When the monitor is in {state_name}, "
            if state_name != initial_state
            else f"From {state_name}, "
        )

        rendered_transitions = []
        for tr in transitions:
            ev = tr.get("event", "<?event?>")
            tgt = tr.get("target", "<?target?>")

            if tgt == state_name:
                rendered_transitions.append(f"{ev} keeps it in {state_name}")
            else:
                rendered_transitions.append(f"{ev} moves it to {tgt}")

        sentence = intro + self._join_phrases(rendered_transitions) + "."
        return sentence
    
    def _join_phrases(self, items: list) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"
    
    def _render_fsm_block(self, ir: dict) -> str:
        fsm = ir.get("fsm", {}) or {}

        initial = fsm.get("initial_state", "")
        states = fsm.get("states", []) or []

        if not states:
            return "No FSM states were provided."

        lines = []

        if initial:
            lines.append(f"The execution begins in the state {initial}.")
            lines.append("")

        for state in states:
            sname = state.get("name", "<?state?>")
            transitions = state.get("transitions", []) or []

            if not transitions:
                lines.append(
                    f"When the monitor is in {sname}, no further transitions are defined."
            )
                lines.append("")
                continue

            lines.append(self._render_state_description(sname, transitions, initial))
            lines.append("")

        return "\n".join(lines).strip()

    def _extract_fsm_semantics(self, ir: dict) -> str:
        violation = ir.get("violation", {}) or {}
        fsm = ir.get("fsm", {}) or {}

        methods = self._get_methods(ir)
        event_names = [m.get("name", "") for m in methods if m.get("name")]

        initial = fsm.get("initial_state")
        states = [s.get("name", "") for s in fsm.get("states", []) or []]

        tag = (violation.get("tag") or "").lower()
        violation_text = self._extract_violation(ir)

        semantic_parts = []

        if initial:
            semantic_parts.append(
                f"The monitored execution starts in the state '{initial}'."
            )

        if states:
            semantic_parts.append(
                f"The property distinguishes the following states: {', '.join(states)}."
            )

        if event_names:
            semantic_parts.append(
                f"The behavior evolves through the occurrence of these events: {', '.join(event_names)}."
            )

        if tag == "unsafe":
            semantic_parts.append(
                "The property describes unsafe behavior that must not be reached during execution."
            )
        elif tag == "fail":
            semantic_parts.append(
                "The property describes a protocol that must be respected; violating the expected transition behavior should trigger a failure."
            )
        elif tag == "err":
            semantic_parts.append(
                "The property describes an erroneous usage pattern that should be reported as an error."
            )
        elif tag == "violation":
            semantic_parts.append(
                "The property defines execution constraints whose violation must be reported."
            )

        if violation_text:
            semantic_parts.append(
                f"Semantically, the monitored rule is about the following condition: {violation_text}"
            )

        return " ".join(semantic_parts).strip()

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _extract_violation(self, ir: dict) -> str:
        viol = ir.get("violation", {}) or {}
        body = viol.get("body", {}) or {}
        statements = body.get("statements", []) or []

        for stmt in statements:
            if stmt.get("type") == "log":
                msg = stmt.get("message", "")
                if msg and msg != "__DEFAULT_MESSAGE":
                    msg = msg.strip('"')
                    msg = msg.replace('" + __LOC', "")
                    msg = msg.replace("+ __LOC", "")
                    return msg.strip()

        return "A violation must be reported when the execution reaches an invalid or unsafe behavior."