# nl2spec/core/ir_to_nl.py

from pathlib import Path
import re


class IRToNL:
    """
    Deterministic IR → NL converter.
    Uses external task templates.
    No paraphrasing. No inference.
    Pure structural projection.
    """

    def __init__(self, template_dir: Path):
        self.template_dir = template_dir

    # ==========================================================
    # BATCH GENERATION
    # ==========================================================

    def generate_from_directory(
        self,
        ir_root: Path,
        nl_root: Path,
    ):
        """
        Convert all IR JSON files from ir_root into NL files inside nl_root.
        Always deletes nl_root if it exists.
        """

        import shutil
        import json

        if not ir_root.exists():
            raise FileNotFoundError(f"IR directory not found: {ir_root}")

        # Always delete automatically
        if nl_root.exists():
            shutil.rmtree(nl_root)

        nl_root.mkdir(parents=True, exist_ok=True)

        total = 0

        for ir_file in ir_root.rglob("*.json"):
            spec = json.loads(ir_file.read_text(encoding="utf-8"))

            domain = (spec.get("domain") or "other").lower()
            out_dir = nl_root / domain
            out_dir.mkdir(parents=True, exist_ok=True)

            task = self.build_task(spec)

            spec_id = spec.get("id") or ir_file.stem
            out_file = out_dir / f"{spec_id}.txt"

            out_file.write_text(task, encoding="utf-8")

            total += 1

        return total

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build_task(self, spec: dict) -> str:
        formalism = (spec.get("formalism") or "").lower()
        ir = spec.get("ir", {})

        if formalism == "event":
            context = self._extract_event(ir)

        elif formalism == "ere":
            context = self._extract_ere(ir)

        elif formalism == "fsm":
            context = self._extract_fsm(ir)

        elif formalism == "ltl":
            context = self._extract_ltl(ir)

        else:
            raise ValueError(f"Unknown formalism: {formalism}")

        return self._render_template(formalism, context)

    # ==========================================================
    # EVENT
    # ==========================================================

    def _extract_event(self, ir: dict) -> dict:
        return {
            "EVENT_BLOCK": self._render_events_block(ir),
            "VIOLATION_SEMANTICS": self._extract_violation(ir),
        }

    # ==========================================================
    # ERE
    # ==========================================================

    def _extract_ere(self, ir: dict) -> dict:
        return {
            "EVENT_BLOCK": self._render_events_block(ir),
            "FORMULA": ir.get("formula", {}).get("raw", ""),
            "VIOLATION_SEMANTICS": self._extract_violation(ir),
        }

    # ==========================================================
    # FSM
    # ==========================================================

    def _extract_fsm(self, ir: dict) -> dict:
        return {
            "EVENT_BLOCK": self._render_events_block(ir),
            "FSM_BLOCK": ir.get("fsm", {}).get("raw_block", ""),
            "VIOLATION_SEMANTICS": self._extract_violation(ir),
        }

    # ==========================================================
    # LTL
    # ==========================================================

    def _extract_ltl(self, ir: dict) -> dict:
        return {
            "EVENT_BLOCK": self._render_events_block(ir),
            "FORMULA": ir.get("formula", {}).get("raw", ""),
            "VIOLATION_SEMANTICS": self._extract_violation(ir),
        }

    # ==========================================================
    # SHARED HELPERS
    # ==========================================================

    def _render_events_block(self, ir: dict) -> str:
        events = ir.get("events", [])
        lines = []

        for idx, ev in enumerate(events, 1):
            name = ev.get("name", f"event_{idx}")
            timing = ev.get("timing", "")
            kind = ev.get("kind", "")
            params = self._format_parameters(ev.get("parameters"))
            returning = self._format_returning(ev.get("returning"))
            pointcut = self._get_pointcut(ev)

            lines.append(f"{idx}) Event name: {name}")
            if kind:
                lines.append(f"   Kind: {kind}")
            if timing:
                lines.append(f"   Timing: {timing}")
            lines.append(f"   Parameters: {params}")
            if returning:
                lines.append(f"   Returning: {returning}")
            lines.append(f"   Pointcut: {pointcut}")
            lines.append("")

        return "\n".join(lines).strip()

    def _format_parameters(self, params):
        if not params:
            return "none"
        parts = []
        for p in params:
            t = p.get("type", "<?>")
            n = p.get("name", "<?>")
            parts.append(f"{t} {n}")
        return ", ".join(parts) if parts else "none"

    def _format_returning(self, returning):
        if not returning:
            return None
        t = returning.get("type", "<?>")
        n = returning.get("name", "<?>")
        return f"{t} {n}"

    def _get_pointcut(self, ev: dict) -> str:
        if "pointcut_raw" in ev:
            return ev["pointcut_raw"]
        if "pointcut" in ev and isinstance(ev["pointcut"], dict):
            return ev["pointcut"].get("raw", "")
        return "<missing pointcut>"

    def _extract_violation(self, ir: dict) -> str:
        viol = ir.get("violation", {})
        raw_block = viol.get("raw_block")

        if isinstance(raw_block, list):
            messages = self._extract_strings(raw_block)
            if messages:
                return messages[0]

        return "A violation must be reported when this condition occurs."

    def _extract_strings(self, lines):
        pattern = re.compile(r'"([^"]+)"')
        msgs = []
        for line in lines:
            msgs.extend(pattern.findall(line))
        return msgs

    def _render_template(self, formalism: str, context: dict) -> str:
        template_path = (
            self.template_dir
            / formalism
            / f"task_{formalism}.txt"
        )

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}"
            )

        template = template_path.read_text(encoding="utf-8")

        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", value or "")

        return template.strip() + "\n"