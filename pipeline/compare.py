import json
from pathlib import Path
from typing import Dict, Any, List

from core.comparator.compare_ir import compare_ir


class BatchCompareError(Exception):
    pass


def _index_ir_dir(root: Path) -> Dict[str, Path]:
    """
    Index IR files by relative path (without extension).
    """
    index = {}
    for f in root.rglob("*.json"):
        key = str(f.relative_to(root).with_suffix(""))
        index[key] = f
    return index


def compare_dirs(
    baseline_dir: str,
    generated_dir: str,
    out_dir: str
) -> List[Dict[str, Any]]:
    """
    Compare baseline IR directory against generated IR directory.

    - baseline_dir: datasets/baseline_ir
    - generated_dir: datasets/generated_ir/<llm>/<run>
    - out_dir: datasets/results/<llm>/<run>

    Returns a list of comparison results.
    """
    baseline_root = Path(baseline_dir)
    generated_root = Path(generated_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_index = _index_ir_dir(baseline_root)
    generated_index = _index_ir_dir(generated_root)

    results = []

    for key, base_path in baseline_index.items():
        if key not in generated_index:
            results.append({
                "id": key,
                "status": "missing_generated"
            })
            continue

        with open(base_path, "r", encoding="utf-8") as f:
            baseline_ir = json.load(f)

        with open(generated_index[key], "r", encoding="utf-8") as f:
            generated_ir = json.load(f)

        diff = compare_ir(baseline_ir, generated_ir)

        result = {
            "id": key,
            "equal": diff.is_equal,
            "errors": diff.errors,
            "warnings": diff.warnings
        }

        results.append(result)

        # save per-spec diff
        out_file = out_root / (key.replace("/", "__") + ".diff.json")
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # global summary
    summary = {
        "total": len(results),
        "equal": sum(1 for r in results if r.get("equal")),
        "with_errors": sum(1 for r in results if r.get("errors")),
        "missing_generated": sum(1 for r in results if r.get("status") == "missing_generated")
    }

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    return results
