from pathlib import Path

from nl2spec.core.convert.nl.fsm_nl import FSMNL
from nl2spec.core.convert.nl.ere_nl import ERENL
from nl2spec.core.convert.nl.ltl_nl import LTLNL
from nl2spec.core.convert.nl.event_nl import EventNL
from nl2spec.logging_utils import get_logger


log = get_logger(__name__)


class IRToNL:

    def __init__(self, template_dir: Path):
        self.template_dir = template_dir
        self.fsm_nl = FSMNL()
        self.ere_nl = ERENL()
        self.ltl_nl = LTLNL()
        self.event_nl = EventNL()

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

        if nl_root.exists():
            shutil.rmtree(nl_root)

        nl_root.mkdir(parents=True, exist_ok=True)

        total = 0

        for ir_file in ir_root.rglob("*.json"):
            spec = json.loads(ir_file.read_text(encoding="utf-8"))

            domain = (spec.get("domain") or "other").lower()
            out_dir = nl_root / domain
            out_dir.mkdir(parents=True, exist_ok=True)

            task = self.build_task(spec, domain)

            spec_id = spec.get("id") or ir_file.stem
            out_file = out_dir / f"{spec_id}.txt"
            out_file.write_text(task, encoding="utf-8")

            total += 1

        return total

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build_task(self, spec: dict, domain: str) -> str:
        formalism = (spec.get("formalism") or "").lower()

        # garante domínio consistente no texto renderizado
        if not spec.get("domain"):
            spec["domain"] = domain

        if formalism == "event":
            context = self.event_nl.render(spec)

        elif formalism == "ere":
            context = self.ere_nl.extract_context(spec, domain)

        elif formalism == "fsm":
            context = self.fsm_nl.extract_context(spec, domain)

        elif formalism == "ltl":
            context = self.ltl_nl.extract_context(spec, domain)

        else:
            raise ValueError(f"Unknown formalism: {formalism}")

        return self._render_template(formalism, context)

    # ==========================================================
    # TEMPLATE RENDERING
    # ==========================================================

    def _render_template(self, formalism: str, context: dict) -> str:
        template_path = self.template_dir / formalism / f"task_{formalism}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")

        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", value or "")

        return template.strip() + "\n"