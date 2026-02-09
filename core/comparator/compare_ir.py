from typing import Dict, Any, List


class IRDiff:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_equal(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equal": self.is_equal,
            "errors": self.errors,
            "warnings": self.warnings
        }

    def __repr__(self):
        return (
            f"IRDiff(equal={self.is_equal}, "
            f"errors={self.errors}, "
            f"warnings={self.warnings})"
        )


# ----------------------------------------------------------------------
# Main comparison entry point
# ----------------------------------------------------------------------
def compare_ir(reference: Dict[str, Any], generated: Dict[str, Any]) -> IRDiff:
    diff = IRDiff()

    # 1. Category
    if reference.get("category") != generated.get("category"):
        diff.add_error(
            f"Category mismatch: expected '{reference.get('category')}', "
            f"got '{generated.get('category')}'"
        )

    ref_ir = reference.get("ir", {})
    gen_ir = generated.get("ir", {})

    # 2. IR type
    if ref_ir.get("type") != gen_ir.get("type"):
        diff.add_error(
            f"IR type mismatch: expected '{ref_ir.get('type')}', "
            f"got '{gen_ir.get('type')}'"
        )

    ir_type = ref_ir.get("type")

    # 3. Dispatch by IR type
    if ir_type == "event":
        _compare_event(ref_ir, gen_ir, diff)

    elif ir_type == "ere":
        _compare_ere(ref_ir, gen_ir, diff)

    elif ir_type == "fsm":
        _compare_fsm(ref_ir, gen_ir, diff)

    elif ir_type == "ltl":
        _compare_ltl(ref_ir, gen_ir, diff)

    else:
        diff.add_warning(f"Unknown IR type '{ir_type}'")

    return diff


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _compare_events(ref_events, gen_events, diff: IRDiff):
    ref_set = {(e["name"], e["timing"]) for e in ref_events}
    gen_set = {(e["name"], e["timing"]) for e in gen_events}

    missing = ref_set - gen_set
    extra = gen_set - ref_set

    for name, timing in missing:
        diff.add_error(f"Missing event: {name}@{timing}")

    for name, timing in extra:
        diff.add_warning(f"Extra event: {name}@{timing}")


def _compare_event(ref_ir, gen_ir, diff: IRDiff):
    _compare_events(
        ref_ir.get("events", []),
        gen_ir.get("events", []),
        diff
    )

    if ref_ir.get("guard") != gen_ir.get("guard"):
        diff.add_warning(
            f"Guard mismatch: expected '{ref_ir.get('guard')}', "
            f"got '{gen_ir.get('guard')}'"
        )


def _compare_ere(ref_ir, gen_ir, diff: IRDiff):
    _compare_events(
        ref_ir.get("events", []),
        gen_ir.get("events", []),
        diff
    )

    if ref_ir.get("pattern") != gen_ir.get("pattern"):
        diff.add_error(
            f"ERE pattern mismatch: expected '{ref_ir.get('pattern')}', "
            f"got '{gen_ir.get('pattern')}'"
        )


def _compare_fsm(ref_ir, gen_ir, diff: IRDiff):
    ref_states = set(ref_ir.get("states", []))
    gen_states = set(gen_ir.get("states", []))

    for s in ref_states - gen_states:
        diff.add_error(f"Missing FSM state: {s}")

    for s in gen_states - ref_states:
        diff.add_warning(f"Extra FSM state: {s}")


def _compare_ltl(ref_ir, gen_ir, diff: IRDiff):
    _compare_events(
        ref_ir.get("events", []),
        gen_ir.get("events", []),
        diff
    )

    if ref_ir.get("formula") != gen_ir.get("formula"):
        diff.add_error(
            f"LTL formula mismatch: expected '{ref_ir.get('formula')}', "
            f"got '{gen_ir.get('formula')}'"
        )
