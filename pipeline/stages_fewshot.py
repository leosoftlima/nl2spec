from pathlib import Path
import json
import csv

from nl2spec.logging_utils import get_logger
from nl2spec.core.handlers.fewshot_loader import FewShotLoader

log = get_logger(__name__)


def stage_fewshot(ctx, flags):

    log.info("Stage: fewshot selection experiment")

    cfg = ctx.config

    baseline_ir_root = Path(cfg["paths"]["baseline_ir"])
    fewshot_dir = Path(cfg["prompting"]["fewshot"]["dataset_dir"])

    shot_mode = cfg["prompting"]["shot_mode"].lower()
    
    if shot_mode == "zero":
        log.info("Shot mode is ZERO. No few-shot selection experiment required.")
        return
    
    k = int(cfg["prompting"]["k"])
    selection = cfg["prompting"]["fewshot"]["selection"].lower()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    results_root = (
         PROJECT_ROOT
           / "nl2spec"
           / "output"
          # / "evaluation"
            / "fewshot"
            / selection
            / shot_mode
    )

    results_root.mkdir(parents=True, exist_ok=True)

    loader = FewShotLoader(str(fewshot_dir))

    rows_by_type = {}

    for domain_dir in baseline_ir_root.iterdir():
        if not domain_dir.is_dir():
            continue

        for file_path in domain_dir.glob("*.json"):

            spec_json = json.loads(file_path.read_text(encoding="utf-8"))
            ir_type = spec_json.get("ir", {}).get("type", "").lower()

            rows_by_type.setdefault(ir_type, [])

            try:
                selected = loader.get(
                    ir_type=ir_type,
                    shot_mode=shot_mode,
                    k=k,
                    selection=selection,
                    ir_base=spec_json,
                    return_scores=True,
                )

                if not selected:
                    rows_by_type[ir_type].append({
                        "dataset": file_path.stem,
                        "formalism": ir_type,
                        "rank": 0,
                        "selected_fewshot": "",
                        "distance": "",
                    })
                else:
                    for rank, (path, dist) in enumerate(selected, start=1):
                        rows_by_type[ir_type].append({
                            "dataset": file_path.stem,
                            "formalism": ir_type,
                            "rank": rank,
                            "selected_fewshot": path.name,
                            "distance": f"{dist:.6f}",
                        })

            except Exception as e:
                rows_by_type[ir_type].append({
                    "dataset": file_path.stem,
                    "formalism": ir_type,
                    "rank": -1,
                    "selected_fewshot": f"ERROR: {str(e)}",
                    "distance": "",
                })

    # salvar separado por formalismo
    for ir_type, rows in rows_by_type.items():

        ir_dir = results_root / ir_type
        ir_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{selection}_k{k}.csv"
        output_file = ir_dir / filename

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "dataset",
                    "formalism",
                    "rank",
                    "selected_fewshot",
                    "distance",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        log.info("Fewshot CSV saved to: %s", output_file)