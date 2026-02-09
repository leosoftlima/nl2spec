"""
Convert baseline IR JSON files into controlled NL text files, grouped by domain.

Key policy:
- Always prefer `violation_message` (if present) as the NL output.
- Only fallback to a minimal, safe template when violation_message is missing.
- Output is grouped into folders: io, lang, util, net, concurrent, other.
- If output directory exists, ask confirmation before deleting/regenerating.
"""

# executed
# python.exe -m nl2spec.scripts.convert_ir_to_nl
#

from __future__ import annotations

from pathlib import Path
import json
import sys
import shutil


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IR_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_ir"
NL_ROOT = PROJECT_ROOT / "nl2spec" / "datasets" / "baseline_nl"


# ==========================================================
# UTILS
# ==========================================================

def ask_overwrite(path: Path) -> bool:
    answer = input(
        f"[WARN] NL output directory already exists:\n"
        f"       {path}\n"
        f"Do you want to delete and regenerate it? [y/N]: "
    ).strip().lower()
    return answer in {"y", "yes"}


def infer_domain(data: dict, ir_file: Path) -> str:
    # 1) Prefer domain stored in JSON
    dom = data.get("domain")
    if isinstance(dom, str) and dom.strip():
        return dom.strip().lower()

    # 2) Fallback: infer from IR path (baseline_ir/<domain>/...)
    parts = [p.lower() for p in ir_file.parts]
    for d in ("io", "lang", "util", "net", "concurrent"):
        if d in parts:
            return d

    return "other"


def safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path}\nReason: {e}") from e


def unique_out_path(out_dir: Path, spec_id: str) -> Path:
    """
    Avoid overwriting when multiple IR files share the same id in the same domain.
    """
    out_file = out_dir / f"{spec_id}.txt"
    if not out_file.exists():
        return out_file

    i = 2
    while True:
        candidate = out_dir / f"{spec_id}__{i}.txt"
        if not candidate.exists():
            return candidate
        i += 1


# ==========================================================
# NL GENERATION (SAFE POLICY)
# ==========================================================

def pick_violation_message(data: dict) -> str | None:
    msg = data.get("violation_message")
    if isinstance(msg, str):
        msg = msg.strip()
        if msg:
            return msg

    # Some IRs store message inside ir
    ir = data.get("ir", {})
    msg2 = ir.get("violation_message")
    if isinstance(msg2, str):
        msg2 = msg2.strip()
        if msg2:
            return msg2

    return None


def fallback_nl(data: dict) -> str:
    """
    Minimal fallback when violation_message is missing.
    Keep it generic to avoid semantic hallucination.
    """
    cat = (data.get("category") or "UNKNOWN").upper()
    spec_id = data.get("id") or "unknown_id"

    ir = data.get("ir", {})
    t = (ir.get("type") or "").lower()

    if cat == "EVENT":
        events = ir.get("events", [])
        if events and isinstance(events, list):
            e0 = events[0] or {}
            name = e0.get("name", "some event")
            timing = e0.get("timing", "")
            if timing:
                return f"[{spec_id}] Forbidden event: {name} ({timing})."
            return f"[{spec_id}] Forbidden event: {name}."
        return f"[{spec_id}] Forbidden event specification."

    if cat == "ERE":
        # ✅ no schema/conversor é "pattern", não "expression"
        pattern = ir.get("pattern")
        if isinstance(pattern, str) and pattern.strip():
            return f"[{spec_id}] The execution must match the event pattern: {pattern.strip()}."
        return f"[{spec_id}] Event pattern constraint (ERE)."

    if cat == "FSM":
        events = set()
        for tr in ir.get("transitions", []) or []:
            if isinstance(tr, dict) and "event" in tr:
                events.add(str(tr["event"]).strip())
        if events:
            evs = ", ".join(sorted(e for e in events if e))
            return f"[{spec_id}] Finite-state rule over events: {evs}."
        return f"[{spec_id}] Finite-state usage rule (FSM)."

    if cat == "LTL":
        formula = ir.get("formula")
        if isinstance(formula, str) and formula.strip():
            return f"[{spec_id}] Temporal property (LTL): {formula.strip()}."
        return f"[{spec_id}] Temporal property (LTL)."

    if t:
        return f"[{spec_id}] Constraint of type: {t}."
    return f"[{spec_id}] Constraint (unknown category)."


def ir_to_nl(data: dict) -> str:
    msg = pick_violation_message(data)
    if msg:
        return msg
    return fallback_nl(data)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    print("=" * 70)
    print("[INFO] Converting baseline IR to NL (grouped by domain)")
    print("[INFO] IR source :", IR_ROOT)
    print("[INFO] NL output :", NL_ROOT)
    print("[INFO] Policy    : prefer violation_message; fallback minimal templates")
    print("=" * 70)

    if not IR_ROOT.exists():
        print(f"[ERROR] IR directory not found: {IR_ROOT}")
        sys.exit(1)

    if NL_ROOT.exists():
        if not ask_overwrite(NL_ROOT):
            print("[INFO] Conversion aborted by user.")
            return
        print("[INFO] Removing existing NL directory...")
        shutil.rmtree(NL_ROOT)

    NL_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    by_domain: dict[str, int] = {}

    for ir_file in IR_ROOT.rglob("*.json"):
        data = safe_read_json(ir_file)

        spec_id = data.get("id") or ir_file.stem
        domain = infer_domain(data, ir_file)

        out_dir = NL_ROOT / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        nl_text = ir_to_nl(data).strip() + "\n"

        out_file = unique_out_path(out_dir, spec_id)
        out_file.write_text(nl_text, encoding="utf-8")

        total += 1
        by_domain[domain] = by_domain.get(domain, 0) + 1

    print("=" * 70)
    print(f"[OK] NL specifications generated: {total}")
    for d in sorted(by_domain):
        print(f"[OK]  - {d:11s}: {by_domain[d]}")
    print("=" * 70)


if __name__ == "__main__":
    main()