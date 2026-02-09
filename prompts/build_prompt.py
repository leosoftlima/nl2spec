from pathlib import Path
import json
from typing import List, Optional
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

PROMPT_DIR = Path(__file__).parent
GENERATED_DIR = PROMPT_DIR / "generated"

SUPPORTED_IR_TYPES = {"fsm", "ere", "event", "ltl"}


def _load(path: Path) -> str:
    log.debug("Loading prompt file: %s", path)
    return path.read_text(encoding="utf-8")


def _save_prompt(
    prompt: str,
    *,
    scenario_id: str,
    ir_type: str,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Persist the generated prompt for inspection and reproducibility.
    """

    base_dir = output_dir or GENERATED_DIR
    target_dir = base_dir / ir_type.upper()
    target_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = target_dir / f"{scenario_id}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    log.info("Prompt saved to: %s", prompt_path)


def build_prompt(
    ir_type: str,
    nl_text: str,
    fewshot_files: List[Path],
    *,
    scenario_id: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Build a prompt for a given IR type (fsm, ere, event, ltl).

    The prompt is composed of:
      1) header.txt (global instructions)
      2) template_<ir_type>.txt (IR-specific guidance)
      3) few-shot examples (if any)
      4) explicit task with the NL description

    If save=True and scenario_id is provided, the prompt is saved to disk.
    """

    ir_type = ir_type.lower()
    if ir_type not in SUPPORTED_IR_TYPES:
        raise ValueError(
            f"Unsupported ir_type '{ir_type}'. "
            f"Supported types: {SUPPORTED_IR_TYPES}"
        )

    log.info("Building prompt for IR type: %s", ir_type)

    # ---------- fixed prompt parts ----------
    header_path = PROMPT_DIR / "base" / "header.txt"
    template_path = PROMPT_DIR / ir_type / "template.txt"

    if not header_path.exists():
        raise FileNotFoundError(f"Prompt header not found: {header_path}")

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    header = _load(header_path)
    template = _load(template_path)

    # ---------- few-shot examples ----------
    examples = []
    for fs in fewshot_files:
        log.debug("Loading few-shot example: %s", fs)
        data = json.loads(fs.read_text(encoding="utf-8"))
        examples.append(json.dumps(data, indent=2))

    if examples:
        fewshot_block = "\n\n".join(
            f"Example {i + 1}:\n{ex}"
            for i, ex in enumerate(examples)
        )
    else:
        fewshot_block = "None"

    # inject few-shot block into template
    template = template.replace(
        "{{FEW_SHOT_EXAMPLES}}",
        fewshot_block
    )

    # ---------- explicit task (GENERIC) ----------
    task = f"""
Task:
Generate a {ir_type.upper()} runtime verification IR in JSON format
that captures the behavior described below.

Natural language description:
\"\"\"
{nl_text}
\"\"\"
""".strip()

    log.info(
        "Prompt parts -> NL chars: %d | few-shot examples: %d",
        len(nl_text),
        len(examples),
    )

    # ---------- final prompt ----------
    prompt = "\n\n".join([
        header,
        template,
        task
    ])

    # ---------- optional persistence ----------
    if save and scenario_id:
        _save_prompt(
            prompt,
            scenario_id=scenario_id,
            ir_type=ir_type,
            output_dir=output_dir,
        )

    return prompt