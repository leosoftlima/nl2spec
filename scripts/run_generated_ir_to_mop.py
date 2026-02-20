from pathlib import Path
import json
import sys
import shutil

# run # run: python -m nl2spec.scripts.run_generated_ir_to_mop

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
IR_INPUT_DIR = PROJECT_ROOT / "datasets" / "baseline_ir_temp"
MOP_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "reconstructed_mop"


# ==========================================================
# UTILS
# ==========================================================

def ask_overwrite(path: Path) -> bool:
    answer = input(
        f"[WARN] Output directory exists:\n"
        f"       {path}\n"
        f"Delete and regenerate? [y/N]: "
    ).strip().lower()
    return answer in {"y", "yes"}


def _format_signature(signature: dict) -> str:
    params = signature.get("parameters", [])
    return ", ".join(f"{p['type']} {p['name']}" for p in params)


def _format_event_prefix(event: dict) -> str:
    # Para preservar "creation event" quando existir no IR
    kind = event.get("kind", "event")
    return "creation event" if kind == "creation" else "event"


def _emit_violation_if_present(lines: list, vio: dict):
    """
    Só emite @fail/@violation/@match se:
      - tag existe
      - e/ou raw_block não é vazio
    """
    if not vio:
        return

    tag = vio.get("tag")
    raw_block = vio.get("raw_block", [])

    # Se não tem tag e não tem conteúdo, não emite nada.
    if (tag is None or str(tag).strip() == "") and not raw_block:
        return

    tag = (tag or "violation").lower()

    lines.append(f"    @{tag} {{")
    for ln in raw_block:
        lines.append(f"        {ln}")
    lines.append("    }")


def _emit_event_block(lines: list, pointcut: str, body_lines: list):
    """
    Emite o bloco do event com chaves exatamente 1x.
    """
    lines.append(f"        {pointcut} {{")
    for bl in body_lines:
        lines.append(f"        {bl}")
    lines.append("        }\n")


# ==========================================================
# RECONSTRUCTION — LTL
# ==========================================================

def reconstruct_ltl(ir_json: dict) -> str:
    spec_id = ir_json["id"]
    signature = ir_json.get("signature", {})
    ir = ir_json["ir"]

    lines = []
    lines.append(f"{spec_id}({_format_signature(signature)}) {{\n")

    for event in ir.get("events", []):
        name = event["name"]
        timing = event["timing"]

        params = ", ".join(f"{p['type']} {p['name']}" for p in event.get("parameters", []))
        header = f"    event {name} {timing}({params})"

        if "returning" in event:
            r = event["returning"]
            header += f" returning({r['type']} {r['name']})"

        header += " :"
        lines.append(header)

        # LTL: pointcut estruturado
        pointcut = event.get("pointcut", {}).get("raw", "")
        _emit_event_block(lines, pointcut, body_lines=[])

    formula = ir.get("formula", {}).get("raw", "")
    lines.append(f"    ltl : {formula}\n")

    _emit_violation_if_present(lines, ir.get("violation", {}))

    lines.append("}")
    return "\n".join(lines)


# ==========================================================
# RECONSTRUCTION — ERE
# ==========================================================

def reconstruct_ere(ir_json: dict) -> str:
    spec_id = ir_json["id"]
    signature = ir_json.get("signature", {})
    ir = ir_json["ir"]

    lines = []
    lines.append(f"{spec_id}({_format_signature(signature)}) {{\n")

    for event in ir.get("events", []):
        prefix = _format_event_prefix(event)
        name = event["name"]
        timing = event["timing"]

        params = ", ".join(f"{p['type']} {p['name']}" for p in event.get("parameters", []))
        header = f"    {prefix} {name} {timing}({params})"

        if "returning" in event:
            r = event["returning"]
            header += f" returning({r['type']} {r['name']})"

        header += " :"
        lines.append(header)

        # ERE: pointcut estruturado (igual LTL)
        pointcut = event.get("pointcut", {}).get("raw", "")
        # ERE normalmente não precisa de body; mas se vier, emitimos fielmente
        body_lines = event.get("body", {}).get("raw_lines", [])
        _emit_event_block(lines, pointcut, body_lines)

    formula = ir.get("formula", {}).get("raw", "")
    lines.append(f"    ere : {formula}\n")

    _emit_violation_if_present(lines, ir.get("violation", {}))

    lines.append("}")
    return "\n".join(lines)


# ==========================================================
# RECONSTRUCTION — FSM
# ==========================================================

def reconstruct_fsm(ir_json: dict) -> str:
    spec_id = ir_json["id"]
    signature = ir_json.get("signature", {})
    ir = ir_json["ir"]

    lines = []
    lines.append(f"{spec_id}({_format_signature(signature)}) {{\n")

    for event in ir.get("events", []):
        prefix = _format_event_prefix(event)
        name = event["name"]
        timing = event["timing"]

        params = ", ".join(f"{p['type']} {p['name']}" for p in event.get("parameters", []))
        header = f"    {prefix} {name} {timing}({params})"

        if "returning" in event:
            r = event["returning"]
            header += f" returning({r['type']} {r['name']})"

        header += " :"
        lines.append(header)

        # FSM: pointcut_raw (não estruturado)
        pointcut = event.get("pointcut_raw", "")
        body_lines = event.get("body", {}).get("raw_lines", [])
        _emit_event_block(lines, pointcut, body_lines)

    # FSM block
    lines.append("    fsm :")
    raw_lines = ir.get("fsm", {}).get("raw_lines", [])
    for ln in raw_lines:
        lines.append(ln)
    lines.append("")

    _emit_violation_if_present(lines, ir.get("violation", {}))

    lines.append("}")
    return "\n".join(lines)


# ==========================================================
# RECONSTRUCTION — EVENT
# ==========================================================

def reconstruct_event(ir_json: dict) -> str:
    """
    EVENT é especial:
    - Pode NÃO ter @fail/@violation/@match no final.
    - A “violação” pode estar embutida no corpo do evento (try/catch, prints, etc.).
    """
    spec_id = ir_json["id"]
    signature = ir_json.get("signature", {})
    ir = ir_json["ir"]

    lines = []
    lines.append(f"{spec_id}({_format_signature(signature)}) {{\n")

    for event in ir.get("events", []):
        # EVENT não tem creation keyword no corpus que você mostrou,
        # mas deixo compatível caso apareça.
        prefix = _format_event_prefix(event)
        name = event["name"]
        timing = event["timing"]

        params = ", ".join(f"{p['type']} {p['name']}" for p in event.get("parameters", []))
        header = f"    {prefix} {name} {timing}({params})"

        if "returning" in event:
            r = event["returning"]
            header += f" returning({r['type']} {r['name']})"

        header += " :"
        lines.append(header)

        # EVENT: pointcut_raw
        pointcut = event.get("pointcut_raw", "")
        body_lines = event.get("body", {}).get("raw_lines", [])
        _emit_event_block(lines, pointcut, body_lines)

    # EVENT: só emite bloco @... se de fato existir no JSON
    _emit_violation_if_present(lines, ir.get("violation", {}))

    lines.append("}")
    return "\n".join(lines)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("[INFO] Reconstructing MOP from IR JSON (LTL + FSM + ERE + EVENT)")
    print("[INFO] Source :", IR_INPUT_DIR)
    print("[INFO] Output :", MOP_OUTPUT_DIR)
    print("=" * 70)

    if not IR_INPUT_DIR.exists():
        print("[ERROR] baseline_ir_temp not found.")
        sys.exit(1)

    if MOP_OUTPUT_DIR.exists():
        if not ask_overwrite(MOP_OUTPUT_DIR):
            print("[INFO] Aborted.")
            return
        shutil.rmtree(MOP_OUTPUT_DIR)

    MOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count_ltl = 0
    count_fsm = 0
    count_ere = 0
    count_event = 0

    for json_file in IR_INPUT_DIR.rglob("*.json"):
        print("Processing:", json_file)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ir_type = data["ir"]["type"]

        if ir_type == "ltl":
            mop_code = reconstruct_ltl(data)
            count_ltl += 1
        elif ir_type == "fsm":
            mop_code = reconstruct_fsm(data)
            count_fsm += 1
        elif ir_type == "ere":
            mop_code = reconstruct_ere(data)
            count_ere += 1
        elif ir_type == "event":
            mop_code = reconstruct_event(data)
            count_event += 1
        else:
            # desconhecido: ignora silenciosamente
            continue

        relative = json_file.relative_to(IR_INPUT_DIR)
        output_file = MOP_OUTPUT_DIR / relative.with_suffix(".mop")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(mop_code, encoding="utf-8")

    total = count_ltl + count_fsm + count_ere + count_event

    print("=" * 70)
    print("[SUMMARY]")
    print(f"  Reconstructed LTL   : {count_ltl}")
    print(f"  Reconstructed FSM   : {count_fsm}")
    print(f"  Reconstructed ERE   : {count_ere}")
    print(f"  Reconstructed EVENT : {count_event}")
    print("-" * 70)
    print(f"  Total Reconstructed : {total}")
    print("=" * 70)


if __name__ == "__main__":
    main()