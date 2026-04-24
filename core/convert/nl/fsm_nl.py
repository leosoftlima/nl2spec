from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from nl2spec.core.convert.fsm import (
    extract_signature,
    extract_events,
    extract_fsm_block,
    extract_violation_block,
)


class FSMNL:
    """
    MOP -> NL renderer for FSM specifications.

    Reads the .mop source directly and fills the FSM task template
    without converting the specification to JSON IR first.
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
            source=source,
            domain=domain,
            spec_id=spec_id,
        )

        signature = extract_signature(mop_text)
        events_wrapped = extract_events(mop_text)
        methods = self._unwrap_methods(events_wrapped)
        fsm = extract_fsm_block(mop_text)
        violation = extract_violation_block(mop_text)

        return {
            "SPEC_NAME": resolved_spec_id,
            "SIGNATURE_PARAMETERS": self._render_signature_parameters(signature),
            "EVENT_BLOCK": self._render_events_block(methods),
            "FSM_BLOCK": self._render_fsm_block(fsm),
            "FSM_SEMANTICS": self._render_fsm_semantics(fsm, methods, violation),
            "VIOLATION_SEMANTICS": self._render_violation(violation),
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
            raise TypeError("FSMNL.extract_context expected str or Path.")

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

    def _unwrap_methods(self, events_wrapped: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        methods: List[Dict[str, Any]] = []

        if not isinstance(events_wrapped, list):
            return methods

        for item in events_wrapped:
            if not isinstance(item, dict):
                continue
            body = item.get("body", {}) or {}
            inner = body.get("methods", []) or []
            for method in inner:
                if isinstance(method, dict):
                    methods.append(method)

        return methods

    def _render_events_block(self, methods: List[Dict[str, Any]]) -> str:
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

            intro = f"{idx}) {name} is observed {timing} the call as a {action}."

            if params != "none":
                intro += f" It takes {params}."
            else:
                intro += " It does not take explicit event parameters."

            if returning:
                intro += f" It captures the returned value as {returning}."

            if pointcut:
                intro += f" It matches: {pointcut}."

            lines.append(intro)

        return "\n\n".join(lines)

    def _render_pointcut(self, ev: Dict[str, Any]) -> str:
        functions = ev.get("function", []) or []
        operations = ev.get("operation", []) or []

        if not functions:
            return ""

        pieces = []

        for i, fn in enumerate(functions):
            if not isinstance(fn, dict):
                continue

            fname = fn.get("name", "").strip()
            params = fn.get("parameters", []) or []

            rendered_params = []
            for p in params:
                if not isinstance(p, dict):
                    continue

                ret = (p.get("return") or "").strip()
                name = (p.get("name") or "").strip()

                if ret and name:
                    rendered_params.append(f"{ret} {name}")
                elif name:
                    rendered_params.append(name)
                elif ret:
                    rendered_params.append(ret)

            atom = f"{fname}({', '.join(rendered_params)})" if fname else f"({', '.join(rendered_params)})"
            pieces.append(atom)

            if i < len(operations):
                pieces.append(operations[i])

        return " ".join(pieces).strip()

    def _format_parameters(self, params: Any) -> str:
        if not params:
            return "none"

        rendered = []
        for p in params:
            if not isinstance(p, dict):
                continue
            rendered.append(f'{p.get("type", "<?>")} {p.get("name", "<?>")}')

        if not rendered:
            return "none"

        if len(rendered) == 1:
            return f"{rendered[0]} as parameter"
        if len(rendered) == 2:
            return f"{rendered[0]} and {rendered[1]} as parameters"
        return f"{', '.join(rendered[:-1])}, and {rendered[-1]} as parameters"

    def _format_returning(self, returning: Any) -> Optional[str]:
        if not returning or not isinstance(returning, dict):
            return None

        rtype = (returning.get("type") or "").strip()
        rname = (returning.get("name") or "").strip()

        if not rtype and not rname:
            return None

        return f"{rtype} {rname}".strip()

    # ==========================================================
    # FSM
    # ==========================================================

    def _render_fsm_block(self, fsm: Dict[str, Any]) -> str:
        initial = fsm.get("initial_state", "") if isinstance(fsm, dict) else ""
        states = fsm.get("states", []) if isinstance(fsm, dict) else []

        if not isinstance(states, list) or not states:
            return "No FSM states were provided."

        lines = []

        if initial:
            lines.append(f"The execution begins in the state {initial}.")
            lines.append("")

        for state in states:
            if not isinstance(state, dict):
                continue

            sname = state.get("name", "<?state?>")
            transitions = state.get("transitions", []) or []

            if not transitions:
                lines.append(f"When the monitor is in {sname}, no further transitions are defined.")
                lines.append("")
                continue

            rendered_transitions = []
            for tr in transitions:
                if not isinstance(tr, dict):
                    continue

                ev = tr.get("event", "<?event?>")
                tgt = tr.get("target", "<?target?>")

                if tgt == sname:
                    rendered_transitions.append(f"{ev} keeps it in {sname}")
                else:
                    rendered_transitions.append(f"{ev} moves it to {tgt}")

            if sname == initial:
                lines.append(f"From {sname}, {self._join_phrases(rendered_transitions)}.")
            else:
                lines.append(f"When the monitor is in {sname}, {self._join_phrases(rendered_transitions)}.")
            lines.append("")

        return "\n".join(lines).strip()

    def _render_fsm_semantics(
        self,
        fsm: Dict[str, Any],
        methods: List[Dict[str, Any]],
        violation: Dict[str, Any],
    ) -> str:
        parts = []

        initial_state = fsm.get("initial_state") if isinstance(fsm, dict) else None
        states = fsm.get("states", []) if isinstance(fsm, dict) else []

        if initial_state:
            parts.append(f"The monitored execution starts in the state '{initial_state}'.")

        if states:
            state_names = [s.get("name", "") for s in states if isinstance(s, dict) and s.get("name")]
            if state_names:
                parts.append(f"The property distinguishes the following states: {', '.join(state_names)}.")

        event_names = [m.get("name", "") for m in methods if isinstance(m, dict) and m.get("name")]
        if event_names:
            parts.append(f"The behavior evolves through the occurrence of these events: {', '.join(event_names)}.")

        violation_text = self._render_violation(violation)
        if violation_text:
            parts.append(f"Any execution that deviates from the expected transition behavior must be treated as a violation because: {violation_text}")

        return " ".join(parts).strip() if parts else "No FSM semantics were provided."

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {}) if isinstance(violation, dict) else {}
        statements = body.get("statements", []) if isinstance(body, dict) else []

        for st in statements:
            if not isinstance(st, dict):
                continue

            if st.get("type") == "log":
                msg = (st.get("message") or "").strip()
                if msg:
                    return self._clean_message(msg)

        return "A violation must be reported."

    def _clean_message(self, msg: str) -> str:
        msg = msg.strip()
        msg = msg.strip('"')
        msg = msg.replace('" + __LOC', "")
        msg = msg.replace("+ __LOC", "")
        return msg.strip()

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _join_phrases(self, items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"