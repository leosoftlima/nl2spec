from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random
import json
import re

from nl2spec.logging_utils import get_logger
from nl2spec.exceptions import FewShotNotAvailableError, NL2SpecException

log = get_logger(__name__)

ALLOWED_K = {0, 1, 3}
ALLOWED_SELECTION = {"random", "structural"}


class FewShotLoader:
    """
    Few-shot selector supporting:
      - random selection
      - structural selection (IR-SP) based on feature vectors + weighted Manhattan distance
    """

    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.root = Path(fewshot_dir)
        self.seed = seed
        random.seed(seed)
        log.info("Few-shot root directory: %s", self.root)

    # =========================================================
    # Configuration Validation
    # =========================================================

    def _validate_configuration(self, shot_mode: str, k: int, selection: str):
        if k not in ALLOWED_K:
            raise NL2SpecException(f"Invalid k={k}. Allowed values are {ALLOWED_K}.")

        if shot_mode == "zero" and k != 0:
            raise NL2SpecException("shot_mode='zero' requires k=0.")

        if shot_mode == "one" and k != 1:
            raise NL2SpecException("shot_mode='one' requires k=1.")

        if shot_mode == "few" and k < 2:
            raise NL2SpecException("shot_mode='few' requires k >= 2.")

        if selection not in ALLOWED_SELECTION:
            raise NL2SpecException(
                f"Invalid selection='{selection}'. Allowed values: {ALLOWED_SELECTION}"
            )

    # =========================================================
    # Public API
    # =========================================================

    def get(
        self,
        ir_type: str,
        shot_mode: str,
        k: int,
        selection: str,
        ir_base: Optional[dict] = None,
        return_scores: bool = False,
    ):

        self._validate_configuration(shot_mode, k, selection)

        if k == 0:
            return []

        ir_type = ir_type.lower()
        ir_dir = self.root / ir_type

        if not ir_dir.exists():
            raise FewShotNotAvailableError(str(ir_dir))

        files = sorted(ir_dir.glob("*.json"))
        if not files:
            raise FewShotNotAvailableError(str(ir_dir))

        if selection == "random":
            chosen = self._select_random(files, k)
            return [(p, 0.0) for p in chosen] if return_scores else chosen

        if selection == "structural":
            if ir_base is None:
                raise NL2SpecException("Structural selection requires ir_base.")
            return self._select_structural(files, k, ir_base, ir_type, return_scores)

        raise NL2SpecException("Unexpected selection mode.")

    # =========================================================
    # Random Selection
    # =========================================================

    def _select_random(self, files: List[Path], k: int) -> List[Path]:
        total = len(files)
        if k >= total:
            return files
        return random.sample(files, k)

    # =========================================================
    # Structural Selection
    # =========================================================

    def _select_structural(
        self,
        files: List[Path],
        k: int,
        ir_base: dict,
        ir_type: str,
        return_scores: bool = False,
    ):

        base_vector = self._extract_vector_by_type(ir_base, ir_type)
        scored: List[Dict[str, Any]] = []

        for path in files:

            with open(path, "r", encoding="utf-8") as f:
                template_json = json.load(f)

            template_ir_type = template_json.get("ir", {}).get("type", "").lower()
            if template_ir_type != ir_type:
                continue

            template_vector = self._extract_vector_by_type(template_json, ir_type)
            dist = self._distance_by_type(template_vector, base_vector, ir_type)

            scored.append({"path": path, "distance": dist})

        scored.sort(key=lambda x: (x["distance"], x["path"].name))
        top = scored[:k]

        if return_scores:
            return [(item["path"], float(item["distance"])) for item in top]

        return [item["path"] for item in top]

    # =========================================================
    # VECTOR DISPATCHER
    # =========================================================

    def _extract_vector_by_type(self, spec_json: dict, ir_type: str) -> dict:

        if ir_type == "ltl":
            return self._extract_vector_ltl(spec_json)

        if ir_type == "fsm":
            return self._extract_vector_fsm(spec_json)

        if ir_type == "event":
            return self._extract_vector_event(spec_json)

        if ir_type == "ere":
            return self._extract_vector_ere(spec_json)

        raise NL2SpecException(
            f"Structural extractor not implemented for ir_type '{ir_type}'."
        )

    # =========================================================
    # ERE EXTRACTOR
    # =========================================================

    def _extract_vector_ere(self, spec_json: dict) -> dict:

        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {})
        events = ir.get("events", [])
        formula_raw = ir.get("formula", {}).get("raw", "") or ""
        raw = formula_raw.strip()

        vector["num_events"] = len(events)
        vector["formula_length"] = len(raw)
        vector["num_tokens"] = len(raw.split())

        vector["uses_kleene_star"] = raw.count("*")
        vector["uses_plus"] = raw.count("+")
        vector["uses_optional"] = raw.count("?")
        vector["uses_union"] = raw.count("|")
        vector["uses_negation"] = raw.count("~") + raw.count("!")
        vector["uses_epsilon"] = 1 if "epsilon" in raw.lower() else 0

        vector["num_groups"] = raw.count("(")
        vector["num_alternations"] = raw.count("|")
        vector["num_repetitions"] = raw.count("*") + raw.count("+") + raw.count("?")

        depth = 0
        max_depth = 0
        for char in raw:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth -= 1

        vector["nesting_depth"] = max_depth

        vector["is_union_pattern"] = 1 if "|" in raw else 0
        vector["is_simple_star_pattern"] = 1 if raw.endswith("*") and "|" not in raw else 0
        vector["is_single_event_pattern"] = 1 if len(events) == 1 and "*" not in raw else 0

        return vector

    # =========================================================
    # LTL Extractor (Pattern-Aware + Composite Features)
    # =========================================================

    def _extract_vector_ltl(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {})
        events = ir.get("events", [])
        formula_raw = ir.get("formula", {}).get("raw", "")

        vector["num_events"] = len(events)

        has_returning = 0
        has_boolean_returning = 0
        has_condition_in_pointcut = 0
        has_end_program_event = 0
        saw_true_variant = 0
        saw_false_variant = 0

        for e in events:
            name = e.get("name", "")

            if name == "endProg":
                has_end_program_event = 1

            if "returning" in e:
                has_returning = 1
                ret_type = e.get("returning", {}).get("type", "")
                if ret_type == "boolean":
                    has_boolean_returning = 1

            pointcut_raw = e.get("pointcut", {}).get("raw", "")
            if "condition(" in pointcut_raw:
                has_condition_in_pointcut = 1

            if "true" in name:
                saw_true_variant = 1
            if "false" in name:
                saw_false_variant = 1

        has_true_false_split = 1 if (saw_true_variant and saw_false_variant) else 0

        vector["has_returning_event"] = has_returning
        vector["has_boolean_returning"] = has_boolean_returning
        vector["has_condition_in_pointcut"] = has_condition_in_pointcut
        vector["has_end_program_event"] = has_end_program_event
        vector["has_true_false_split"] = has_true_false_split

        uses_always = 1 if "[]" in formula_raw else 0
        uses_eventually = 1 if "<>" in formula_raw else 0
        uses_past_o = 1 if "=> o" in formula_raw or " o " in formula_raw else 0
        uses_star_operator = 1 if "(*)" in formula_raw else 0
        uses_or = 1 if " or " in formula_raw else 0

        pattern_precedence = 1 if "=> o" in formula_raw else 0
        pattern_response = 1 if "=> <>" in formula_raw else 0

        vector["uses_always"] = uses_always
        vector["uses_eventually"] = uses_eventually
        vector["uses_past_o"] = uses_past_o
        vector["uses_star_operator"] = uses_star_operator
        vector["uses_or"] = uses_or

        vector["pattern_precedence"] = pattern_precedence
        vector["pattern_response"] = pattern_response

        vector["pattern_endprog_obligation"] = 1 if (
            has_end_program_event and uses_always and pattern_precedence
        ) else 0

        vector["pattern_boolean_split"] = 1 if (
            has_boolean_returning
            and has_condition_in_pointcut
            and has_true_false_split
            and uses_star_operator
        ) else 0

        vector["pattern_precedence_only"] = 1 if (
            pattern_precedence and not has_end_program_event and not uses_eventually
        ) else 0

        vector["pattern_response_only"] = 1 if (pattern_response and uses_eventually) else 0

        return vector

    # =========================================================
    # FSM Extractor (Robust Graph + Semantics)
    # =========================================================

    def _extract_vector_fsm(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {})
        states = ir.get("states", [])
        transitions = ir.get("transitions", [])
        events = ir.get("events", [])

        vector["num_states"] = len(states)
        vector["num_transitions"] = len(transitions)

        initial_state = None
        final_states: List[str] = []
        error_states: List[str] = []

        for s in states:
            name = s.get("name", "")
            if s.get("initial", False):
                initial_state = name
            if s.get("final", False):
                final_states.append(name)
            if "error" in name.lower() or "violation" in name.lower():
                error_states.append(name)

        vector["num_final_states"] = len(final_states)
        vector["num_error_states"] = len(error_states)

        intermediate = [
            s.get("name")
            for s in states
            if s.get("name") not in final_states
            and s.get("name") not in error_states
            and s.get("name") != initial_state
        ]
        vector["num_intermediate_states"] = len(intermediate)

        adjacency: Dict[str, List[str]] = {}
        outgoing_count: Dict[str, int] = {}
        has_cycle = 0

        for t in transitions:
            src = t.get("source")
            tgt = t.get("target")

            adjacency.setdefault(src, []).append(tgt)
            outgoing_count[src] = outgoing_count.get(src, 0) + 1

            if src == tgt:
                has_cycle = 1

        max_out_degree = 0
        for c in outgoing_count.values():
            if c > max_out_degree:
                max_out_degree = c

        vector["max_out_degree"] = max_out_degree
        vector["has_cycle"] = has_cycle
        vector["has_branching"] = 1 if max_out_degree > 1 else 0
        vector["is_linear_chain"] = 1 if max_out_degree <= 1 else 0

        max_depth = 0
        if initial_state:
            visited = set()
            queue = [(initial_state, 0)]

            while queue:
                node, depth = queue.pop(0)
                max_depth = max(max_depth, depth)

                if node in visited:
                    continue
                visited.add(node)

                for neighbor in adjacency.get(node, []):
                    queue.append((neighbor, depth + 1))

        vector["max_depth"] = max_depth

        def count_paths(start: str, target: str, visited: set, limit: int = 20) -> int:
            if start == target:
                return 1
            if len(visited) > limit:
                return 0
            total = 0
            for nxt in adjacency.get(start, []):
                if nxt not in visited:
                    total += count_paths(nxt, target, visited | {nxt}, limit)
            return total

        num_paths = 0
        if initial_state:
            for err in error_states:
                num_paths += count_paths(initial_state, err, {initial_state})

        vector["num_paths_to_error"] = num_paths

        error_is_terminal = 1
        for err in error_states:
            if len(adjacency.get(err, [])) > 0:
                error_is_terminal = 0
                break
        vector["error_is_terminal"] = error_is_terminal

        has_creation_event = 0
        has_boolean_returning = 0
        has_condition_in_pointcut = 0

        for e in events:
            name = e.get("name", "")
            if "create" in name.lower():
                has_creation_event = 1

            if "returning" in e:
                ret_type = e.get("returning", {}).get("type", "")
                if ret_type == "boolean":
                    has_boolean_returning = 1

            pointcut_raw = e.get("pointcut", {}).get("raw", "")
            if "condition(" in pointcut_raw:
                has_condition_in_pointcut = 1

        vector["has_creation_event"] = has_creation_event
        vector["has_boolean_returning"] = has_boolean_returning
        vector["has_condition_in_pointcut"] = has_condition_in_pointcut

        vector["pattern_precedence_like"] = 1 if (num_paths == 1 and max_depth <= 2) else 0
        vector["pattern_response_like"] = 1 if (num_paths > 1) else 0

        return vector

    # =========================================================
    # EVENT Extractor (Robust Structural + Light Semantic Signals)
    # =========================================================

    _NUMERIC_CLASSES = {"byte", "short", "integer", "long", "float", "double", "boolean", "character"}
    _COLLECTION_TOKENS = {"collection", "collections", "list", "set", "map", "sortedset", "sortedmap",
                          "treemap", "treeset", "hashmap", "hashset", "arraylist", "vector", "deque",
                          "priorityqueue", "enumset", "enummap", "dictionary", "arrays"}
    _NETWORK_TOKENS = {"socket", "serversocket", "datagramsocket", "datagrampacket", "multicastsocket",
                       "inetaddress", "inetsocketaddress", "urlconnection", "urlencoder", "urldecoder",
                       "httpcookie", "idn"}
    _PERMISSION_TOKENS = {"permission", "socketpermission", "runtimepermission", "netpermission"}
    _IO_TOKENS = {"closeable", "inputstream", "reader", "file", "classloader"}

    def _extract_vector_event(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {})
        events = ir.get("events", []) or []
        constraints = ir.get("constraints", []) or []
        violation = ir.get("violation_message", "") or ""

        vector["num_events"] = len(events)
        vector["num_constraints"] = len(constraints)
        vector["has_violation_message"] = 1 if violation.strip() else 0
        vector["multi_event_constraints"] = 1 if len(events) > 1 else 0

        total_params = 0
        max_params = 0
        has_returning = 0
        has_boolean_returning = 0
        has_condition = 0
        has_static = 0
        has_constructor = 0

        raw_parts: List[str] = []
        event_names: List[str] = []

        for e in events:
            event_names.append((e.get("name") or "").lower())

            params = e.get("parameters", None)
            if params is None:
                params = (e.get("signature", {}) or {}).get("parameters", []) or []
            param_count = len(params)

            total_params += param_count
            max_params = max(max_params, param_count)

            if "returning" in e:
                has_returning = 1
                if (e.get("returning", {}) or {}).get("type", "") == "boolean":
                    has_boolean_returning = 1

            pointcut_raw = (e.get("pointcut", {}) or {}).get("raw", "") or ""
            raw_parts.append(pointcut_raw)

            if " static " in f" {pointcut_raw.lower()} " or "static(" in pointcut_raw.lower():
                has_static = 1

            if "<init>" in pointcut_raw or ".<init>" in pointcut_raw:
                has_constructor = 1

            if "condition(" in pointcut_raw:
                has_condition = 1

        vector["total_event_parameters"] = total_params
        vector["max_event_parameters"] = max_params
        vector["has_returning_event"] = has_returning
        vector["has_boolean_returning"] = has_boolean_returning
        vector["has_condition_in_pointcut"] = has_condition
        vector["has_static_method"] = has_static
        vector["has_constructor_event"] = has_constructor

        for c in constraints:
            raw_parts.append((c.get("raw") or ""))

        raw_parts.append(violation)
        raw_blob = " ".join(raw_parts).lower()

        vector["uses_equality"] = 1 if "==" in raw_blob else 0
        vector["uses_inequality"] = 1 if "!=" in raw_blob else 0
        vector["uses_comparison"] = 1 if any(op in raw_blob for op in ["<=", ">=", "<", ">"]) else 0
        vector["uses_not"] = raw_blob.count("!")
        vector["uses_and"] = raw_blob.count("&&")
        vector["uses_or"] = raw_blob.count("||")

        vector["uses_before"] = 1 if "before" in raw_blob else 0
        vector["uses_after"] = 1 if "after" in raw_blob else 0

        vector["compares_to_null"] = 1 if "null" in raw_blob else 0

        has_number = bool(re.search(r"\b\d+\b", raw_blob)) or bool(re.search(r"\b0x[0-9a-f]+\b", raw_blob))
        has_string_literal = '"' in raw_blob or "'" in raw_blob
        vector["compares_to_constant"] = 1 if (has_number or has_string_literal) else 0

        param_like = re.findall(r"\b[a-zA-Z_]\w*\b", raw_blob)
        vector["compares_two_idents"] = 1 if (
            len(set(param_like)) >= 8 and (
                vector["uses_equality"] or vector["uses_inequality"] or vector["uses_comparison"]
            )
        ) else 0

        name_blob = " ".join(event_names) + " " + raw_blob

        def has_any(tokens: set) -> int:
            return 1 if any(t in name_blob for t in tokens) else 0

        vector["domain_numeric"] = has_any(self._NUMERIC_CLASSES)
        vector["domain_collection"] = has_any(self._COLLECTION_TOKENS)
        vector["domain_network"] = has_any(self._NETWORK_TOKENS)
        vector["domain_permission"] = has_any(self._PERMISSION_TOKENS)
        vector["domain_io"] = has_any(self._IO_TOKENS)
        vector["domain_comparable"] = 1 if ("compareto" in name_blob or "comparable" in name_blob) else 0

        vector["pattern_null"] = 1 if "null" in raw_blob else 0
        vector["pattern_decode"] = 1 if "decode" in raw_blob else 0
        vector["pattern_parse"] = 1 if "parse" in raw_blob else 0
        vector["pattern_factory"] = 1 if "static" in raw_blob else 0
        vector["pattern_comparable"] = 1 if "compareto" in raw_blob else 0
        vector["pattern_permission"] = 1 if "permission" in raw_blob else 0
        vector["pattern_timeout"] = 1 if "timeout" in raw_blob else 0
        vector["pattern_port"] = 1 if "port" in raw_blob else 0

        return vector

    # =========================================================
    # DISTANCE DISPATCHER
    # =========================================================

    def _distance_by_type(self, v1: dict, v2: dict, ir_type: str) -> float:

        if ir_type == "ltl":
            return self._distance_ltl(v1, v2)

        if ir_type == "fsm":
            return self._distance_fsm(v1, v2)

        if ir_type == "event":
            return self._distance_event(v1, v2)

        if ir_type == "ere":
            return self._distance_ere(v1, v2)

        raise NL2SpecException(
            f"Distance not implemented for ir_type '{ir_type}'."
        )

    # =========================================================
    # DISTANCE ERE
    # =========================================================

    def _distance_ere(self, v1: dict, v2: dict) -> float:

        weights = {
            "num_events": 2.0,
            "formula_length": 1.0,
            "num_tokens": 1.0,
            "uses_kleene_star": 3.0,
            "uses_plus": 2.0,
            "uses_optional": 2.0,
            "uses_union": 4.0,
            "uses_negation": 4.0,
            "uses_epsilon": 3.0,
            "num_groups": 2.0,
            "nesting_depth": 5.0,
            "num_alternations": 4.0,
            "num_repetitions": 2.0,
            "is_union_pattern": 6.0,
            "is_simple_star_pattern": 5.0,
            "is_single_event_pattern": 4.0,
        }

        return self._weighted_manhattan(v1, v2, weights)

    # =========================================================
    # Distance: LTL
    # =========================================================

    def _distance_ltl(self, v1: dict, v2: dict) -> float:
        weights = {
            "num_events": 1.0,

            "has_returning_event": 1.0,
            "has_boolean_returning": 2.0,
            "has_condition_in_pointcut": 2.0,
            "has_end_program_event": 2.0,
            "has_true_false_split": 3.0,

            "uses_always": 1.0,
            "uses_eventually": 1.5,
            "uses_past_o": 2.0,
            "uses_star_operator": 2.0,
            "uses_or": 1.5,

            "pattern_precedence": 3.0,
            "pattern_response": 3.0,

            "pattern_endprog_obligation": 6.0,
            "pattern_boolean_split": 8.0,
            "pattern_precedence_only": 2.0,
            "pattern_response_only": 2.0,
        }
        return self._weighted_manhattan(v1, v2, weights)

    # =========================================================
    # Distance: FSM
    # =========================================================

    def _distance_fsm(self, v1: dict, v2: dict) -> float:
        weights = {
            "num_states": 2.0,
            "num_transitions": 1.5,

            "num_final_states": 2.5,
            "num_intermediate_states": 2.0,
            "num_error_states": 5.0,

            "max_out_degree": 2.0,
            "has_cycle": 3.0,
            "has_branching": 3.0,
            "is_linear_chain": 3.0,

            "max_depth": 4.0,
            "num_paths_to_error": 6.0,
            "error_is_terminal": 4.0,

            "has_creation_event": 2.0,
            "has_boolean_returning": 3.0,
            "has_condition_in_pointcut": 3.0,

            "pattern_precedence_like": 5.0,
            "pattern_response_like": 5.0,
        }
        return self._weighted_manhattan(v1, v2, weights)

    # =========================================================
    # Distance: EVENT
    # =========================================================

    def _distance_event(self, v1: dict, v2: dict) -> float:
        weights = {
            "num_events": 2.0,
            "num_constraints": 3.0,

            "total_event_parameters": 1.5,
            "max_event_parameters": 2.0,

            "has_returning_event": 2.5,
            "has_boolean_returning": 3.0,
            "has_condition_in_pointcut": 2.5,

            "has_static_method": 2.0,
            "has_constructor_event": 2.0,

            "multi_event_constraints": 4.0,

            "uses_equality": 2.0,
            "uses_inequality": 2.0,
            "uses_comparison": 2.0,
            "uses_not": 1.0,
            "uses_and": 0.8,
            "uses_or": 0.8,
            "uses_before": 2.0,
            "uses_after": 2.0,

            "compares_to_null": 2.0,
            "compares_to_constant": 2.0,
            "compares_two_idents": 1.5,

            "domain_numeric": 3.0,
            "domain_collection": 3.0,
            "domain_network": 3.0,
            "domain_permission": 3.0,
            "domain_io": 2.5,
            "domain_comparable": 3.0,

            "pattern_decode": 1.5,
            "pattern_parse": 1.5,
            "pattern_factory": 1.2,
            "pattern_permission": 1.5,
            "pattern_timeout": 1.5,
            "pattern_port": 1.5,
            "pattern_comparable": 1.5,

            "has_violation_message": 1.0,
        }
        return self._weighted_manhattan(v1, v2, weights)

    # =========================================================
    # Generic Weighted Manhattan
    # =========================================================

    def _weighted_manhattan(self, v1: dict, v2: dict, weights: dict) -> float:
        distance = 0.0
        keys = set(weights.keys()) | set(v1.keys()) | set(v2.keys())
        for key in keys:
            diff = abs(v1.get(key, 0) - v2.get(key, 0))
            distance += weights.get(key, 1.0) * diff
        return distance