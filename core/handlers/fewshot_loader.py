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

        # ===============================
        # RANDOM SELECTION (WITH DISTANCE)
        # ===============================
        if selection == "random":

            chosen = self._select_random(files, k)

            if not return_scores:
                return chosen

            if ir_base is None:
                raise NL2SpecException(
                    "Random selection with scoring requires ir_base."
                )

            base_vector = self._extract_vector_by_type(ir_base, ir_type)
            results: List[Tuple[Path, float]] = []

            for path in chosen:
                with open(path, "r", encoding="utf-8") as f:
                    template_json = json.load(f)

                template_ir_type = (
                    template_json.get("ir", {}).get("type", "").lower()
                )
                if template_ir_type != ir_type:
                    continue

                template_vector = self._extract_vector_by_type(
                    template_json, ir_type
                )

                dist = self._distance_by_type(
                    template_vector, base_vector, ir_type
                )

                results.append((path, float(dist)))

            return results

        # ===============================
        # STRUCTURAL SELECTION
        # ===============================
        if selection == "structural":
            if ir_base is None:
                raise NL2SpecException(
                    "Structural selection requires ir_base."
                )

            return self._select_structural(
                files, k, ir_base, ir_type, return_scores
            )

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
        events = ir.get("events", []) or []
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
        formula_raw = ir.get("formula", {}).get("raw", "") or ""

        vector["len_formula"] = len(formula_raw)

        vector["count_always"] = formula_raw.count("[]")
        vector["count_eventually"] = formula_raw.count("<>")
        vector["count_next"] = formula_raw.count(" X ") + formula_raw.count("X(")
        vector["count_until"] = formula_raw.count(" U ")
        vector["count_negation"] = formula_raw.count("!")
        vector["count_implication"] = formula_raw.count("=>")
        vector["count_or"] = formula_raw.count(" or ")

        vector["paren_depth"] = formula_raw.count("(")

        props = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula_raw)
        vector["num_atomic_props"] = len(set(props))

        return vector

    # =========================================================
    # FSM Extractor (CORRIGIDO PARA analysis STRUCTURE)
    # =========================================================

    def _extract_vector_fsm(self, spec_json: dict) -> dict:
        """FSM extractor ajustado para o formato do dataset (ir.analysis.*).

        Estrutura esperada:
          - ir.analysis.states: List[str]
          - ir.analysis.initial: str
          - ir.analysis.transitions: List[dict] com chaves from/to (ou source/target)
          - ir.events: List[dict] com pointcut_raw
          - ir.violation.tag (opcional)

        Objetivo: gerar features suficientes para diferenciar templates (fsm_01/03/05 etc.)
        sem deixar features binárias dominarem a distância.
        """
        import math

        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {}) or {}
        analysis = ir.get("analysis", {}) or {}

        states: List[str] = analysis.get("states", []) or []
        transitions: List[dict] = analysis.get("transitions", []) or []
        events: List[dict] = ir.get("events", []) or []

        initial_state = analysis.get("initial")

        # --- Classificação de estados (heurística por nome)
        final_states = [
            s for s in states
            if str(s).lower() in {"final", "finished", "closed", "end"}
        ]
        error_states = [
            s for s in states
            if any(k in str(s).lower() for k in ("error", "fail", "violation"))
        ]
        intermediate = [
            s for s in states
            if s not in final_states and s not in error_states and s != initial_state
        ]

        vector["num_states"] = len(states)
        vector["num_transitions"] = len(transitions)
        vector["num_final_states"] = len(final_states)
        vector["num_error_states"] = len(error_states)
        vector["num_intermediate_states"] = len(intermediate)

        # --- Grafo
        adjacency: Dict[str, List[str]] = {}
        outgoing_count: Dict[str, int] = {}
        incoming_count: Dict[str, int] = {}
        num_self_loops = 0

        for t in transitions:
            src = t.get("from") or t.get("source")
            tgt = t.get("to") or t.get("target")
            if src is None or tgt is None:
                continue

            adjacency.setdefault(src, []).append(tgt)
            outgoing_count[src] = outgoing_count.get(src, 0) + 1
            incoming_count[tgt] = incoming_count.get(tgt, 0) + 1

            if src == tgt:
                num_self_loops += 1

        vector["num_self_loops"] = num_self_loops

        max_out = max(outgoing_count.values()) if outgoing_count else 0
        max_in = max(incoming_count.values()) if incoming_count else 0
        vector["max_out_degree"] = max_out
        vector["max_in_degree"] = max_in

        # Média por estado (inclui estados sem saída como 0 implicitamente)
        vector["avg_out_degree"] = (sum(outgoing_count.values()) / len(states)) if states else 0.0

        # Sinks (sem saída)
        sinks = [s for s in states if len(adjacency.get(s, [])) == 0]
        vector["num_sink_states"] = len(sinks)

        # --- Ciclo (melhor que só self-loop): DFS com stack
        def has_any_cycle() -> bool:
            visited = set()
            stack = set()

            def dfs(u: str) -> bool:
                visited.add(u)
                stack.add(u)
                for v in adjacency.get(u, []):
                    if v not in visited:
                        if dfs(v):
                            return True
                    elif v in stack:
                        return True
                stack.remove(u)
                return False

            for n in states:
                if n not in visited:
                    if dfs(n):
                        return True
            return False

        vector["has_cycle"] = 1 if (num_self_loops > 0 or has_any_cycle()) else 0
        vector["has_branching"] = 1 if max_out > 1 else 0
        vector["is_linear_chain"] = 1 if max_out <= 1 else 0

        # --- Profundidade (BFS) a partir do inicial
        max_depth = 0
        if initial_state:
            visited = set()
            queue: List[Tuple[str, int]] = [(initial_state, 0)]
            while queue:
                node, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in adjacency.get(node, []):
                    queue.append((neighbor, depth + 1))
        vector["max_depth"] = max_depth

        # --- Caminhos para erro + erro terminal
        def count_paths(start: str, target: str, visited: set, limit: int = 30) -> int:
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

        error_out_degree = 0
        error_is_terminal = 1
        for err in error_states:
            outd = len(adjacency.get(err, []))
            error_out_degree += outd
            if outd > 0:
                error_is_terminal = 0
        vector["error_out_degree"] = error_out_degree
        vector["error_is_terminal"] = error_is_terminal

        # --- Violação (se existir no IR)
        violation = ir.get("violation", {}) or {}
        vector["violation_is_fail"] = 1 if violation.get("tag") == "fail" else 0

        # --- Eventos
        has_creation_event = 0
        has_boolean_returning = 0
        has_condition_in_pointcut = 0

        for e in events:
            name = (e.get("name", "") or "")
            if "create" in name.lower():
                has_creation_event = 1

            if "returning" in e:
                ret_type = (e.get("returning", {}) or {}).get("type", "")
                if ret_type == "boolean":
                    has_boolean_returning = 1

            pointcut_raw = e.get("pointcut_raw", "") or ""
            if "condition(" in pointcut_raw:
                has_condition_in_pointcut = 1

        vector["has_creation_event"] = has_creation_event
        vector["has_boolean_returning"] = has_boolean_returning
        vector["has_condition_in_pointcut"] = has_condition_in_pointcut

        # Heurísticas de padrão
        vector["pattern_precedence_like"] = 1 if (num_paths == 1 and max_depth <= 2) else 0
        vector["pattern_response_like"] = 1 if (num_paths > 1 or max_out > 1) else 0

        # --- Normalização log1p SOMENTE para features de escala (não-bool)
        for k in (
            "num_states",
            "num_transitions",
            "num_final_states",
            "num_error_states",
            "num_intermediate_states",
            "num_self_loops",
            "num_sink_states",
            "max_out_degree",
            "max_in_degree",
            "max_depth",
            "num_paths_to_error",
            "error_out_degree",
            "avg_out_degree",
        ):
            val = vector.get(k, 0)
            vector[k] = math.log1p(val) if val >= 0 else 0.0

        return vector
    # =========================================================
    # EVENT Extractor (Robust Structural + Light Semantic Signals)
    # =========================================================

    _NUMERIC_CLASSES = {
        "byte", "short", "integer", "long", "float", "double", "boolean", "character"
    }
    _COLLECTION_TOKENS = {
        "collection", "collections", "list", "set", "map", "sortedset", "sortedmap",
        "treemap", "treeset", "hashmap", "hashset", "arraylist", "vector", "deque",
        "priorityqueue", "enumset", "enummap", "dictionary", "arrays"
    }
    _NETWORK_TOKENS = {
        "socket", "serversocket", "datagramsocket", "datagrampacket", "multicastsocket",
        "inetaddress", "inetsocketaddress", "urlconnection", "urlencoder", "urldecoder",
        "httpcookie", "idn"
    }
    _PERMISSION_TOKENS = {"permission", "socketpermission", "runtimepermission", "netpermission"}
    _IO_TOKENS = {"closeable", "inputstream", "reader", "file", "classloader"}

    def _extract_vector_event(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {})
        events = ir.get("events", []) or []
        constraints = ir.get("constraints", []) or []

        vector["num_events"] = len(events)
        vector["num_constraints"] = len(constraints)

        total_params = 0
        has_return = 0
        has_after = 0
        has_before = 0

        for e in events:
            params = e.get("parameters", []) or []
            total_params += len(params)

            if any(p.get("name") == "ret" for p in params):
                has_return = 1

            if e.get("timing") == "after":
                has_after = 1
            if e.get("timing") == "before":
                has_before = 1

        vector["num_parameters"] = total_params
        vector["has_return"] = has_return
        vector["has_after"] = has_after
        vector["has_before"] = has_before

        full_expr = " ".join(c.get("expression", "") or "" for c in constraints)

        vector["num_and"] = full_expr.count("&&")
        vector["num_or"] = full_expr.count("||")
        vector["num_eq"] = full_expr.count("==")
        vector["num_neq"] = full_expr.count("!=")
        vector["num_gt"] = full_expr.count(">")
        vector["num_lt"] = full_expr.count("<")
        vector["num_ge"] = full_expr.count(">=")
        vector["num_le"] = full_expr.count("<=")

        vector["expr_length"] = len(full_expr)

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

        raise NL2SpecException(f"Distance not implemented for ir_type '{ir_type}'.")

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
        # Rebalanceado para:
        # - destacar assinatura estrutural (sinks, loops, graus, profundidade)
        # - reduzir dominância de binárias (branching/linear/pattern)
        # - permitir que fsm_01/03/05 “ganhem” quando forem o melhor fit
        weights = {
            "num_states": 2.5,
            "num_transitions": 2.5,
            "num_final_states": 2.0,
            "num_intermediate_states": 2.0,
            "num_error_states": 2.0,

            "max_out_degree": 2.0,
            "max_in_degree": 1.5,
            "avg_out_degree": 3.0,

            "max_depth": 3.0,

            "num_sink_states": 3.0,
            "num_self_loops": 3.5,
            "has_cycle": 2.0,

            "num_paths_to_error": 3.0,
            "error_out_degree": 4.0,
            "error_is_terminal": 2.0,

            "violation_is_fail": 4.0,

            "has_creation_event": 2.0,
            "has_boolean_returning": 3.0,
            "has_condition_in_pointcut": 3.0,

            # binárias/padrões: peso baixo (evita “sempre o mesmo template”)
            "has_branching": 0.3,
            "is_linear_chain": 0.3,
            "pattern_precedence_like": 0.5,
            "pattern_response_like": 0.5,
        }
        return self._weighted_manhattan(v1, v2, weights)
    # =========================================================
    # Distance: EVENT
    # =========================================================

    def _distance_event(self, v1: dict, v2: dict) -> float:
        weights = {
            "num_events": 4.0,
            "num_constraints": 5.0,
            "num_parameters": 3.0,
            "has_return": 6.0,
            "has_after": 2.0,
            "has_before": 2.0,
            "num_and": 4.0,
            "num_or": 4.0,
            "num_eq": 2.0,
            "num_neq": 3.0,
            "num_gt": 2.0,
            "num_lt": 2.0,
            "num_ge": 2.0,
            "num_le": 2.0,
            "expr_length": 1.0,
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