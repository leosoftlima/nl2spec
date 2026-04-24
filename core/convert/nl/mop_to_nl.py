from pathlib import Path
from typing import Optional
import shutil

from nl2spec.core.convert.mop_to_ir import detect_domain, detect_formalism
from nl2spec.core.convert.nl.fsm_nl import FSMNL
from nl2spec.core.convert.nl.ere_nl import ERENL
from nl2spec.core.convert.nl.ltl_nl import LTLNL
from nl2spec.core.convert.nl.event_nl import EventNL
from nl2spec.logging_utils import get_logger


log = get_logger(__name__)


class MOPToNL:
    """
    Convert .mop specifications directly to NL prompt files.

    This class only dispatches to the extractor/renderer of each language:
    - ere
    - fsm
    - ltl
    - event

    Expected extractor contract for all four language files:
        extract_context(source: Path, domain: Optional[str] = None, spec_id: Optional[str] = None) -> dict
    """

    def __init__(self, template_dir: Path):
        self.template_dir = Path(template_dir)

        self.fsm_nl = FSMNL()
        self.ere_nl = ERENL()
        self.ltl_nl = LTLNL()
        self.event_nl = EventNL()

    # ==========================================================
    # BATCH GENERATION
    # ==========================================================

    def generate_from_directory(
        self,
        mop_root: Path,
        nl_root: Path,
    ) -> int:
        mop_root = Path(mop_root)
        nl_root = Path(nl_root)

        if not mop_root.exists():
            raise FileNotFoundError(f"MOP directory not found: {mop_root}")

        if nl_root.exists():
            shutil.rmtree(nl_root)

        nl_root.mkdir(parents=True, exist_ok=True)

        total = 0
        skipped = 0

        for mop_file in mop_root.rglob("*.mop"):
            try:
                text = mop_file.read_text(encoding="utf-8", errors="replace")
                formalism = (detect_formalism(text) or "").lower()

                if formalism not in {"ere", "fsm", "ltl", "event"}:
                    skipped += 1
                    log.info("[SKIP] file=%s | formalism=%s", mop_file.name, formalism)
                    continue

                domain = (detect_domain(mop_file) or "other").lower()
                out_dir = nl_root / domain
                out_dir.mkdir(parents=True, exist_ok=True)

                task = self.build_task(
                    mop_file=mop_file,
                    domain=domain,
                    formalism=formalism,
                )

                out_file = out_dir / f"{mop_file.stem}.txt"
                out_file.write_text(task, encoding="utf-8")

                total += 1
                log.info("[OK] generated=%s", out_file)

            except Exception as exc:
                skipped += 1
                log.exception("[ERROR] file=%s | reason=%s", mop_file, exc)

        log.info(
            "MOP -> NL finished | generated=%d | skipped=%d",
            total,
            skipped,
        )

        return total

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build_task(
        self,
        mop_file: Path,
        domain: Optional[str] = None,
        formalism: Optional[str] = None,
    ) -> str:
        mop_file = Path(mop_file)

        if not mop_file.exists():
            raise FileNotFoundError(f"MOP file not found: {mop_file}")

        text = mop_file.read_text(encoding="utf-8", errors="replace")

        resolved_formalism = (formalism or detect_formalism(text) or "").lower()
        resolved_domain = (domain or detect_domain(mop_file) or "other").lower()

        context = self._extract_context(
            mop_file=mop_file,
            domain=resolved_domain,
            formalism=resolved_formalism,
        )

        return self._render_template(resolved_formalism, context)

    # ==========================================================
    # DISPATCH
    # ==========================================================

    def _extract_context(
        self,
        mop_file: Path,
        domain: str,
        formalism: str,
    ) -> dict:
        if formalism == "ere":
            return self.ere_nl.extract_context(
                source=mop_file,
                domain=domain,
                spec_id=mop_file.stem,
            )

        if formalism == "fsm":
            return self.fsm_nl.extract_context(
                source=mop_file,
                domain=domain,
                spec_id=mop_file.stem,
            )

        if formalism == "ltl":
            return self.ltl_nl.extract_context(
                source=mop_file,
                domain=domain,
                spec_id=mop_file.stem,
            )

        if formalism == "event":
            return self.event_nl.extract_context(
                source=mop_file,
                domain=domain,
                spec_id=mop_file.stem,
            )

        raise ValueError(f"Unknown formalism: {formalism}")

    # ==========================================================
    # TEMPLATE RENDERING
    # ==========================================================

    def _render_template(self, formalism: str, context: dict) -> str:
        template_path = self.template_dir / formalism / f"task_{formalism}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")

        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", str(value or ""))

        return template.strip() + "\n"