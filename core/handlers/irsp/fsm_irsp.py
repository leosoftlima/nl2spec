from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import math
import re


class FSMFewShotSelector:
    """
    Structure-aware FSM few-shot selector.

    Combines:
    - graph/topology features
    - event/pointcut features
    - semantic token features
    - composite pattern features tailored to the dataset

    Distance:
    - weighted Manhattan over numeric/binary features
    """

    # ======================================================
    # PUBLIC API
    # ======================================================

    def select(
        self,
        files: List[Path],
        k: int,
        ir_base: dict,
        return_scores: bool = False,
    ):
        base_vector = self.extract_vector(ir_base)
        scored: List[Dict[str, Any]] = []

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                template_json = json.load(f)

            template_formalism = (template_json.get("formalism", "") or "").lower()
            if template_formalism != "fsm":
                continue

            template_vector = self.extract_vector(template_json)
            dist = self.distance(template_vector, base_vector)
            scored.append({"path": path, "distance": float(dist)})

        scored.sort(key=lambda x: (x["distance"], x["path"].name))
        top = scored[:k]

        if return_scores:
            return [(item["path"], float(item["distance"])) for item in top]

        return [item["path"] for item in top]

    def score_candidates(self, chosen: List[Path], ir_base: dict):
        base_vector = self.extract_vector(ir_base)
        results: List[Tuple[Path, float]] = []

        for path in chosen:
            with open(path, "r", encoding="utf-8") as f:
                template_json = json.load(f)

            template_formalism = (template_json.get("formalism", "") or "").lower()
            if template_formalism != "fsm":
                continue

            template_vector = self.extract_vector(template_json)
            dist = self.distance(template_vector, base_vector)
            results.append((path, float(dist)))

        results.sort(key=lambda x: (x[1], x[0].name))
        return results

    # ======================================================
    # HELPERS
    # ======================================================

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _split_tokens(self, text: str) -> List[str]:
        if not text:
            return []

        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        text = re.sub(r"(hasnext)", r" has next ", text, flags=re.IGNORECASE)
        text = re.sub(r"(hasprevious)", r" has previous ", text, flags=re.IGNORECASE)
        text = re.sub(r"(hasmoreelements)", r" has more elements ", text, flags=re.IGNORECASE)
        text = re.sub(r"(deleteonexit)", r" delete on exit ", text, flags=re.IGNORECASE)
        text = re.sub(r"(readnonproxy)", r" read non proxy ", text, flags=re.IGNORECASE)
        text = re.sub(r"(initnonproxy)", r" init non proxy ", text, flags=re.IGNORECASE)
        text = re.sub(r"(initproxy)", r" init proxy ", text, flags=re.IGNORECASE)
        text = re.sub(r"(readline)", r" read line ", text, flags=re.IGNORECASE)
        text = re.sub(r"(bsearch)", r" binary search ", text, flags=re.IGNORECASE)
        text = re.sub(r"(createtempfile)", r" create temp file ", text, flags=re.IGNORECASE)
        text = re.sub(r"(nexttoken)", r" next token ", text, flags=re.IGNORECASE)
        text = re.sub(r"(sval)", r" s val ", text, flags=re.IGNORECASE)
        text = re.sub(r"(nval)", r" n val ", text, flags=re.IGNORECASE)
        text = re.sub(r"(awtcall)", r" awt call ", text, flags=re.IGNORECASE)
        text = re.sub(r"(swingcall)", r" swing call ", text, flags=re.IGNORECASE)

        text = re.sub(r"[^a-zA-Z0-9]", " ", text)

        return text.lower().split()

    # ======================================================
    # VECTOR EXTRACTION
    # ======================================================

    def extract_vector(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        ir = spec_json.get("ir", {}) or {}
        fsm = ir.get("fsm", {}) or {}
        signature = spec_json.get("signature", {}) or {}

        # ======================================================
        # EVENTS
        # ======================================================
        events_ast = self._as_list(ir.get("events", []))
        methods: List[dict] = []

        for ev in events_ast:
            if not isinstance(ev, dict):
                continue

            body = ev.get("body", {}) or {}
            if not isinstance(body, dict):
                continue

            methods_raw = self._as_list(body.get("methods", []))
            for m in methods_raw:
                if isinstance(m, dict):
                    methods.append(m)

        # ======================================================
        # FSM STATES / TRANSITIONS
        # ======================================================
        states_ast_raw = self._as_list(fsm.get("states", []))
        states_ast: List[dict] = [s for s in states_ast_raw if isinstance(s, dict)]
        initial_state = fsm.get("initial_state")

        states = [s.get("name") for s in states_ast if s.get("name")]
        transitions: List[dict] = []

        for state in states_ast:
            src = state.get("name")
            transitions_raw = self._as_list(state.get("transitions", []))

            for tr in transitions_raw:
                if not isinstance(tr, dict):
                    continue

                tgt = tr.get("target")
                ev = tr.get("event")

                if src is not None and tgt is not None and ev is not None:
                    transitions.append({
                        "from": src,
                        "event": ev,
                        "to": tgt,
                    })

        final_states = [
            s for s in states
            if str(s).lower() in {"final", "finished", "closed", "end", "done", "terminated"}
        ]
        error_states = [
            s for s in states
            if any(k in str(s).lower() for k in ("error", "fail", "violation", "unsafe", "err"))
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

        adjacency: Dict[str, List[str]] = {}
        outgoing_count: Dict[str, int] = {}
        incoming_count: Dict[str, int] = {}
        num_self_loops = 0

        for t in transitions:
            src = t.get("from")
            tgt = t.get("to")
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
        vector["avg_out_degree"] = (sum(outgoing_count.values()) / len(states)) if states else 0.0

        sinks = [s for s in states if len(adjacency.get(s, [])) == 0]
        vector["num_sink_states"] = len(sinks)

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
        vector["is_linear_chain"] = 1 if (max_out <= 1 and num_self_loops == 0) else 0

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

        num_paths_to_error = 0
        if initial_state:
            for err in error_states:
                num_paths_to_error += count_paths(initial_state, err, {initial_state})

        vector["num_paths_to_error"] = num_paths_to_error

        error_out_degree = 0
        error_is_terminal = 1
        for err in error_states:
            outd = len(adjacency.get(err, []))
            error_out_degree += outd
            if outd > 0:
                error_is_terminal = 0

        vector["error_out_degree"] = error_out_degree
        vector["error_is_terminal"] = error_is_terminal

        # ======================================================
        # VIOLATION
        # ======================================================
        violation = ir.get("violation", {}) or {}
        tag = (violation.get("tag") or "").lower()
        vector["violation_is_fail"] = 1 if tag == "fail" else 0
        vector["violation_is_violation"] = 1 if tag == "violation" else 0

        # ======================================================
        # EVENT / POINTCUT FEATURES
        # ======================================================
        has_creation_event = 0
        has_boolean_returning = 0
        has_condition_in_pointcut = 0
        has_target_in_pointcut = 0
        has_call_in_pointcut = 0
        has_args_in_pointcut = 0

        tokens = set()
        transition_event_tokens = set()

        for tr in transitions:
            transition_event_tokens.update(self._split_tokens(tr.get("event", "")))

        tokens |= transition_event_tokens

        # signature tokens
        tokens.update(self._split_tokens(signature.get("name", "")))
        for p in self._as_list(signature.get("parameters", [])):
            if isinstance(p, dict):
                tokens.update(self._split_tokens(p.get("type", "")))
                tokens.update(self._split_tokens(p.get("name", "")))

        # state name tokens also help
        for s in states:
            tokens.update(self._split_tokens(s))

        for method in methods:
            name = (method.get("name", "") or "")
            tokens.update(self._split_tokens(name))

            if (method.get("name", "") or "").lower() in {"create", "init", "constructor", "connect", "open"}:
                has_creation_event = 1

            returning = method.get("returning") or {}
            if isinstance(returning, dict):
                if (returning.get("type") or "").lower() == "boolean":
                    has_boolean_returning = 1
                tokens.update(self._split_tokens(returning.get("type", "")))
                tokens.update(self._split_tokens(returning.get("name", "")))

            for p in self._as_list(method.get("parameters", [])):
                if isinstance(p, dict):
                    tokens.update(self._split_tokens(p.get("type", "")))
                    tokens.update(self._split_tokens(p.get("name", "")))

            functions_raw = self._as_list(method.get("function", []))
            functions = [fn for fn in functions_raw if isinstance(fn, dict)]

            for fn in functions:
                fn_name = (fn.get("name", "") or "").lower()
                tokens.update(self._split_tokens(fn_name))

                if fn_name == "condition":
                    has_condition_in_pointcut = 1
                elif fn_name == "target":
                    has_target_in_pointcut = 1
                elif fn_name == "call":
                    has_call_in_pointcut = 1
                elif fn_name == "args":
                    has_args_in_pointcut = 1

                params_raw = self._as_list(fn.get("parameters", []))
                params = [p for p in params_raw if isinstance(p, dict)]

                for p in params:
                    pname = p.get("name", "") or ""
                    pret = p.get("return", "") or ""
                    ptype = p.get("type", "") or ""

                    tokens.update(self._split_tokens(pname))
                    tokens.update(self._split_tokens(pret))
                    tokens.update(self._split_tokens(ptype))

                    if "new(" in pname.lower():
                        has_creation_event = 1

        vector["has_creation_event"] = has_creation_event
        vector["has_boolean_returning"] = has_boolean_returning
        vector["has_condition_in_pointcut"] = has_condition_in_pointcut
        vector["has_target_in_pointcut"] = has_target_in_pointcut
        vector["has_call_in_pointcut"] = has_call_in_pointcut
        vector["has_args_in_pointcut"] = has_args_in_pointcut

        # ======================================================
        # SEMANTIC TOKEN FEATURES
        # ======================================================
        vector["has_open"] = 1 if any(t in tokens for t in {"open", "create", "init", "constructor", "new"}) else 0
        vector["has_close"] = 1 if "close" in tokens else 0
        vector["has_connect"] = 1 if "connect" in tokens else 0
        vector["has_disconnect"] = 1 if "disconnect" in tokens else 0
        vector["has_shutdown"] = 1 if "shutdown" in tokens else 0
        vector["has_register"] = 1 if "register" in tokens else 0
        vector["has_unregister"] = 1 if "unregister" in tokens else 0
        vector["has_start"] = 1 if "start" in tokens else 0
        vector["has_interrupt"] = 1 if "interrupt" in tokens else 0
        vector["has_exit"] = 1 if "exit" in tokens else 0
        vector["has_awt"] = 1 if "awt" in tokens else 0
        vector["has_swing"] = 1 if "swing" in tokens else 0

        vector["has_read"] = 1 if "read" in tokens else 0
        vector["has_write"] = 1 if "write" in tokens else 0
        vector["has_flush"] = 1 if "flush" in tokens else 0
        vector["has_search"] = 1 if "search" in tokens else 0
        vector["has_binary"] = 1 if "binary" in tokens else 0
        vector["has_sort"] = 1 if "sort" in tokens else 0
        vector["has_modify"] = 1 if "modify" in tokens else 0
        vector["has_timeout"] = 1 if "timeout" in tokens else 0
        vector["has_enter"] = 1 if "enter" in tokens else 0
        vector["has_leave"] = 1 if "leave" in tokens else 0
        vector["has_input"] = 1 if "input" in tokens else 0
        vector["has_output"] = 1 if "output" in tokens else 0

        vector["has_iterator"] = 1 if "iterator" in tokens else 0
        vector["has_listiterator"] = 1 if "list" in tokens and "iterator" in tokens else 0
        vector["has_next"] = 1 if "next" in tokens else 0
        vector["has_previous"] = 1 if "previous" in tokens else 0
        vector["has_hasnext"] = 1 if ("has" in tokens and "next" in tokens) else 0
        vector["has_hasprevious"] = 1 if ("has" in tokens and "previous" in tokens) else 0
        vector["has_more"] = 1 if "more" in tokens else 0
        vector["has_elements"] = 1 if "elements" in tokens else 0
        vector["has_tokenizer"] = 1 if "tokenizer" in tokens else 0

        vector["has_word"] = 1 if "word" in tokens else 0
        vector["has_num"] = 1 if "num" in tokens else 0
        vector["has_sval"] = 1 if ("s" in tokens and "val" in tokens) or "sval" in tokens else 0
        vector["has_nval"] = 1 if ("n" in tokens and "val" in tokens) or "nval" in tokens else 0
        vector["has_field_access"] = 1 if vector["has_sval"] or vector["has_nval"] else 0

        vector["is_stream"] = 1 if "stream" in tokens else 0
        vector["is_socket"] = 1 if "socket" in tokens else 0
        vector["is_file"] = 1 if "file" in tokens else 0
        vector["is_reader"] = 1 if "reader" in tokens else 0
        vector["is_writer"] = 1 if "writer" in tokens else 0
        vector["is_piped"] = 1 if "piped" in tokens else 0

        # ======================================================
        # COMPOSITE SEMANTIC PATTERNS
        # ======================================================
        vector["pattern_precedence_like"] = 1 if (num_paths_to_error == 1 and max_depth <= 2) else 0
        vector["pattern_response_like"] = 1 if (num_paths_to_error > 1 or max_out > 1) else 0

        vector["pattern_resource_lifecycle"] = 1 if (
            vector["has_open"] and vector["has_close"]
        ) else 0

        vector["pattern_connect_use_close"] = 1 if (
            vector["has_connect"] and (vector["has_read"] or vector["has_write"]) and vector["has_close"]
        ) else 0

        vector["pattern_iterator_guard"] = 1 if (
            vector["has_iterator"] and vector["has_next"] and has_boolean_returning
        ) else 0

        vector["pattern_bidirectional_iterator"] = 1 if (
            vector["has_listiterator"] and vector["has_next"] and vector["has_previous"]
        ) else 0

        vector["pattern_tokenizer_guard"] = 1 if (
            vector["has_tokenizer"] and (vector["has_more"] or vector["has_elements"]) and vector["has_next"]
        ) else 0

        vector["pattern_timeout_socket"] = 1 if (
            vector["is_socket"] and vector["has_timeout"] and vector["has_enter"] and vector["has_leave"]
        ) else 0

        vector["pattern_sort_then_search"] = 1 if (
            vector["has_sort"] and (vector["has_search"] or vector["has_binary"])
        ) else 0

        vector["pattern_sort_modify_search"] = 1 if (
            vector["pattern_sort_then_search"] and vector["has_modify"]
        ) else 0

        vector["pattern_piped_unconnected_io"] = 1 if (
            vector["is_piped"] and vector["has_connect"] and (vector["has_read"] or vector["has_write"])
        ) else 0

        vector["pattern_shutdown_hook"] = 1 if (
            vector["has_register"] and vector["has_start"] and ("unsafe" in {str(s).lower() for s in states} or vector["num_error_states"] > 0)
        ) else 0

        vector["pattern_ui_unsafe_shutdown"] = 1 if (
            vector["pattern_shutdown_hook"] and (vector["has_awt"] or vector["has_swing"])
        ) else 0

        vector["pattern_streamtokenizer_field_access"] = 1 if (
            vector["has_tokenizer"] and vector["has_field_access"] and (vector["has_word"] or vector["has_num"])
        ) else 0

        vector["pattern_flush_before_retrieve"] = 1 if (
            vector["has_flush"] and (("byte" in tokens) or ("string" in tokens) or ("array" in tokens))
        ) else 0

        # ======================================================
        # COMPRESS SELECTED NUMERIC FEATURES
        # ======================================================
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

    # ======================================================
    # DISTANCE
    # ======================================================

    def distance(self, v1: dict, v2: dict) -> float:
        weights = {
            # topology
            "num_states": 2.0,
            "num_transitions": 2.0,
            "num_final_states": 1.5,
            "num_intermediate_states": 1.5,
            "num_error_states": 2.0,
            "max_out_degree": 1.5,
            "max_in_degree": 1.0,
            "avg_out_degree": 1.5,
            "max_depth": 3.0,
            "num_sink_states": 3.0,
            "num_self_loops": 3.5,
            "has_cycle": 2.0,
            "num_paths_to_error": 2.5,
            "error_out_degree": 2.5,
            "error_is_terminal": 2.0,

            # violation / pointcut
            "violation_is_fail": 1.5,
            "violation_is_violation": 1.0,
            "has_creation_event": 2.5,
            "has_boolean_returning": 3.5,
            "has_condition_in_pointcut": 3.5,
            "has_target_in_pointcut": 2.0,
            "has_call_in_pointcut": 1.5,
            "has_args_in_pointcut": 1.5,

            # topology-style patterns
            "has_branching": 1.5,
            "is_linear_chain": 1.5,
            "pattern_precedence_like": 1.5,
            "pattern_response_like": 1.5,

            # semantic features
            "has_open": 5.0,
            "has_close": 6.0,
            "has_connect": 7.0,
            "has_disconnect": 6.0,
            "has_shutdown": 7.0,
            "has_register": 7.0,
            "has_unregister": 5.0,
            "has_start": 6.0,
            "has_interrupt": 7.0,
            "has_exit": 7.0,
            "has_awt": 8.0,
            "has_swing": 8.0,
            "has_read": 5.0,
            "has_write": 5.0,
            "has_flush": 6.0,
            "has_search": 6.0,
            "has_binary": 6.0,
            "has_sort": 6.0,
            "has_modify": 5.0,
            "has_timeout": 7.0,
            "has_enter": 5.0,
            "has_leave": 5.0,
            "has_input": 5.0,
            "has_output": 5.0,
            "has_iterator": 6.0,
            "has_listiterator": 7.0,
            "has_next": 5.0,
            "has_previous": 6.0,
            "has_hasnext": 6.0,
            "has_hasprevious": 7.0,
            "has_more": 5.0,
            "has_elements": 5.0,
            "has_tokenizer": 7.0,
            "has_word": 6.0,
            "has_num": 6.0,
            "has_sval": 7.0,
            "has_nval": 7.0,
            "has_field_access": 7.0,
            "is_stream": 5.0,
            "is_socket": 7.0,
            "is_file": 6.0,
            "is_reader": 5.0,
            "is_writer": 5.0,
            "is_piped": 8.0,

            # composite semantic patterns
            "pattern_resource_lifecycle": 7.0,
            "pattern_connect_use_close": 8.0,
            "pattern_iterator_guard": 8.0,
            "pattern_bidirectional_iterator": 9.0,
            "pattern_tokenizer_guard": 8.0,
            "pattern_timeout_socket": 9.0,
            "pattern_sort_then_search": 9.0,
            "pattern_sort_modify_search": 10.0,
            "pattern_piped_unconnected_io": 10.0,
            "pattern_shutdown_hook": 10.0,
            "pattern_ui_unsafe_shutdown": 10.0,
            "pattern_streamtokenizer_field_access": 10.0,
            "pattern_flush_before_retrieve": 9.0,
        }
        return self.weighted_manhattan(v1, v2, weights)

    def weighted_manhattan(self, v1: dict, v2: dict, weights: dict) -> float:
        distance = 0.0
        for key, weight in weights.items():
            diff = abs(v1.get(key, 0) - v2.get(key, 0))
            distance += weight * diff
        return distance