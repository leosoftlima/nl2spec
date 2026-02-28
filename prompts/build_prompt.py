from pathlib import Path
import json
from typing import List, Optional
from nl2spec.logging_utils import get_logger

log = get_logger(__name__)

PROMPT_DIR = Path(__file__).parent

# project_root/nl2spec/output/prompt
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_ROOT / "nl2spec" / "output" / "prompt"

SUPPORTED_IR_TYPES = {"fsm", "ere", "event", "ltl"}


def _load(path: Path) -> str:
    log.debug("Loading prompt file: %s", path)
    return path.read_text(encoding="utf-8")


def _save_prompt(
    prompt: str,
    *,
    scenario_id: str,
    ir_type: str,
    shot_mode: str,
    selection: str,
    output_dir: Optional[Path] = None,
) -> None:

    base_dir = output_dir or GENERATED_DIR
    if shot_mode.lower() == "zero":
        target_dir = base_dir / "zero" / ir_type.lower()
    else:
        target_dir = (
            base_dir
            / selection.lower()
            / shot_mode.lower()
            / ir_type.lower()
        )
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
    shot_mode: str,
    selection: str,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> str:

    ir_type = ir_type.lower()
    if ir_type not in SUPPORTED_IR_TYPES:
        raise ValueError(
            f"Unsupported ir_type '{ir_type}'. "
            f"Supported types: {SUPPORTED_IR_TYPES}"
        )

    log.info("Building prompt for IR type: %s | shot_mode: %s", ir_type, shot_mode)

    # ---------- fixed prompt parts ----------
    header_path = PROMPT_DIR / "templates" / "base" / "header.txt"
    context_path = PROMPT_DIR / "templates" / ir_type / f"context_{ir_type}.txt"

    if not header_path.exists():
        raise FileNotFoundError(f"Prompt header not found: {header_path}")

    if not context_path.exists():
        raise FileNotFoundError(f"Context template not found: {context_path}")

    header = _load(header_path)
    context_template = _load(context_path)

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

    context_template = context_template.replace(
        "{{FEW_SHOT_EXAMPLES}}",
        fewshot_block
    )

    # ---------- NL task ----------
    task_block = f"""
Natural Language Specification:
\"\"\"
{nl_text}
\"\"\"
""".strip()

    log.info(
        "Prompt parts -> NL chars: %d | few-shot examples: %d",
        len(nl_text),
        len(examples),
    )

    prompt = "\n\n".join([
        header,
        context_template,
        task_block
    ])

    if save and scenario_id:
        _save_prompt(
           prompt,
           scenario_id=scenario_id,
           ir_type=ir_type,
           shot_mode=shot_mode,
           selection=selection,
           output_dir=output_dir,
        )

    return prompt