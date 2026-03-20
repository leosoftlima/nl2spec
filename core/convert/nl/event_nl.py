from typing import Dict, List, Any


class EventNL:
    """
    Natural-language renderer for specs in the EVENT formalism.
    It prepares textual blocks used by the prompt builder.
    """

    def render(self, spec: Dict[str, Any]) -> Dict[str, str]:
        signature = spec.get("signature", {})
        ir = spec.get("ir", {})

        return {
            "SPEC_NAME": signature.get("name", spec.get("id", "")),
            "SIGNATURE_PARAMETERS_TEXT": self._render_signature_parameters_text(
                signature.get("parameters", [])
            ),
            "EVENT_BLOCK": self._render_events_block(ir.get("events", [])),
            "VIOLATION_SEMANTICS": self._render_violation(ir.get("violation", {})),
            "DOMAIN": spec.get("domain", ""),
        }

    # ==========================================================
    # SIGNATURE
    # ==========================================================

    def _render_signature_parameters(self, parameters: List[Dict[str, str]]) -> str:
        if not parameters:
            return "[]"

        rendered = []
        for param in parameters:
            ptype = param.get("type", "").strip()
            pname = param.get("name", "").strip()

            if ptype and pname:
                rendered.append(f"{ptype} {pname}")
            elif pname:
                rendered.append(pname)
            elif ptype:
                rendered.append(ptype)

        return "[" + ", ".join(rendered) + "]"

    # ==========================================================
    # EVENTS
    # ==========================================================

    def _render_events_block(self, events: List[Dict[str, Any]]) -> str:
        if not events:
            return "No monitored events were provided."

        descriptions = []

        for event in events:
            methods = event.get("body", {}).get("methods", [])
            if not methods:
                event_name = event.get("name", "").strip()
                if event_name:
                    descriptions.append(
                        f"{event_name} is declared as a monitored event."
                    )
                continue

            for method in methods:
                descriptions.append(self._render_method(method))

        return "\n\n".join(descriptions)

    def _render_method(self, method: Dict[str, Any]) -> str:
        name = method.get("name", "").strip()
        timing = method.get("timing", "").strip()

        parameters = method.get("parameters", [])
        returning = method.get("returning")
        procediments = method.get("procediments", {})

        param_text = self._render_method_parameters(parameters)
        returning_text = self._render_returning(returning)
        pointcut_text = self._render_pointcut(procediments)

        sentence_parts = []

        base = f"{name} is an event observed {timing} the call."
        sentence_parts.append(base)

        if param_text:
            plural = "s" if len(parameters) > 1 else ""
            sentence_parts.append(
                f"It takes {param_text} as parameter{plural}."
            )

        if returning_text:
            sentence_parts.append(
                f"It captures the returned value as {returning_text}."
            )

        if pointcut_text:
            sentence_parts.append(
                f"It matches: {pointcut_text}."
            )

        return " ".join(sentence_parts)

    def _render_method_parameters(self, parameters: List[Dict[str, str]]) -> str:
        if not parameters:
            return ""

        rendered = []
        for param in parameters:
            ptype = param.get("type", "").strip()
            pname = param.get("name", "").strip()

            if ptype and pname:
                rendered.append(f"{ptype} {pname}")
            elif pname:
                rendered.append(pname)
            elif ptype:
                rendered.append(ptype)

        return self._join_phrases(rendered)

    def _render_returning(self, returning: Any) -> str:
        if not returning:
            return ""

        rtype = returning.get("type", "").strip()
        rname = returning.get("name", "").strip()

        if rtype and rname:
            return f"{rtype} {rname}"
        return rname or rtype

    # ==========================================================
    # POINTCUT
    # ==========================================================
    def _render_signature_parameters_text(self, parameters: List[Dict[str, str]]) -> str:
        rendered = self._render_signature_parameters(parameters)

        if rendered == "[]":
            return ""

        return f"It should use these parameters: {rendered}."

    def _render_pointcut(self, procediments: Dict[str, Any]) -> str:
        functions = procediments.get("function", [])
        operations = procediments.get("operation", [])

        if not functions:
            return ""

        rendered_functions = [self._render_function(fn) for fn in functions]

        if not operations:
            return " ".join(rendered_functions)

        pieces = [rendered_functions[0]]

        for index, op in enumerate(operations):
            next_index = index + 1
            if next_index < len(rendered_functions):
                pieces.append(op)
                pieces.append(rendered_functions[next_index])

        return " ".join(pieces)

    def _render_function(self, function: Dict[str, Any]) -> str:
        name = function.get("name", "").strip()
        parameters = function.get("parameters", [])

        args = []
        for param in parameters:
            value = param.get("value", "").strip()
            if value:
                args.append(value)

        return f"{name}({', '.join(args)})"

    # ==========================================================
    # VIOLATION
    # ==========================================================

    def _render_violation(self, violation: Dict[str, Any]) -> str:
        body = violation.get("body", {})
        statements = body.get("statements", [])

        messages = []

        for statement in statements:
            if statement.get("type") == "log":
                message = statement.get("message", "").strip()
                if message:
                    messages.append(message)

        if not messages:
            return "__DEFAULT_MESSAGE"

        non_default = [msg for msg in messages if msg != "__DEFAULT_MESSAGE"]
        if non_default:
            return non_default[-1]

        return messages[-1]

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