from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import json
import math
import re


class EREFewShotSelector:
 

    _ROLE_KEYWORDS = {
        "create": {"create", "init", "new", "open", "iterator", "descendingiterator", "register", "load"},
        "modify": {"add", "remove", "set", "clear", "offer", "push", "pop", "retain", "put", "delete", "write", "unread"},
        "use": {"use", "next", "previous", "iter", "iterator", "search", "read", "write", "access", "manipulate"},
        "query": {"get", "has", "contains", "available", "more", "peek", "tostring", "tobytearray"},
        "close": {"close", "shutdown", "disconnect"},
        "config": {"set", "timeout", "reuse", "connect", "performance", "preference", "daemon"},
        "thread": {"thread", "synchronized", "concurrent", "contended", "singlethread"},
        "permission": {"permission", "checkpermission"},
    }

    _TYPE_FAMILIES = {
        "iterator": {"iterator", "listiterator", "enumeration"},
        "collection": {"collection", "list", "set", "deque", "queue", "arraydeque"},
        "map": {"map", "hashmap", "treemap", "dictionary", "properties", "navigablemap"},
        "stream": {"stream", "inputstream", "outputstream", "objectinput", "objectoutput", "randomaccessfile"},
        "reader": {"reader", "writer", "scanner", "appendable", "bufferedinputstream", "pushbackinputstream"},
        "socket": {"socket", "serversocket", "socketimpl", "urlconnection", "httpurlconnection"},
        "thread": {"thread", "threadgroup", "shutdownhook"},
        "process": {"processbuilder", "process"},
        "permission": {"permission", "securitymanager"},
        "math": {"math", "strictmath"},
    }

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
            if template_formalism != "ere":
                continue

            template_vector = self.extract_vector(template_json)
            dist = self.distance(template_vector, base_vector)
            scored.append({"path": path, "distance": dist})

        scored.sort(key=lambda x: (x["distance"], x["path"].name))
        top = scored[:k]

        if return_scores:
            return [(item["path"], float(item["distance"])) for item in top]

        return [item["path"] for item in top]

    def score_random(self, chosen: List[Path], ir_base: dict):
        base_vector = self.extract_vector(ir_base)
        results: List[Tuple[Path, float]] = []

        for path in chosen:
            with open(path, "r", encoding="utf-8") as f:
                template_json = json.load(f)

            template_formalism = (template_json.get("formalism", "") or "").lower()
            if template_formalism != "ere":
                continue

            template_vector = self.extract_vector(template_json)
            dist = self.distance(template_vector, base_vector)
            results.append((path, float(dist)))

        return results

    # ======================================================
    # VECTOR EXTRACTION
    # ======================================================

    def extract_vector(self, spec_json: dict) -> dict:
        vector: Dict[str, Any] = {}

        signature = spec_json.get("signature", {}) or {}
        ir = spec_json.get("ir", {}) or {}

        methods = self._extract_methods(ir)
        expression = ((ir.get("ere", {}) or {}).get("expression", "") or "").strip()

        ast = self.parse_expression(expression) if expression else {"type": "empty"}
        ast_stats = self._extract_ast_stats(ast)

        # ----------------------------
        # ERE structural features
        # ----------------------------
        vector["num_nodes"] = ast_stats["num_nodes"]
        vector["num_expr_event_refs"] = ast_stats["num_expr_event_refs"]
        vector["num_unique_expr_events"] = len(ast_stats["event_names"])
        vector["num_concat"] = ast_stats["num_concat"]
        vector["num_or"] = ast_stats["num_or"]
        vector["num_star"] = ast_stats["num_star"]
        vector["num_plus"] = ast_stats["num_plus"]
        vector["num_optional"] = ast_stats["num_optional"]
        vector["num_not"] = ast_stats["num_not"]
        vector["num_epsilon"] = ast_stats["num_epsilon"]
        vector["num_empty"] = ast_stats["num_empty"]
        vector["max_depth"] = ast_stats["max_depth"]
        vector["max_branching"] = ast_stats["max_branching"]

        vector["has_alternation"] = 1 if ast_stats["num_or"] > 0 else 0
        vector["has_repetition"] = 1 if (ast_stats["num_star"] + ast_stats["num_plus"] + ast_stats["num_optional"]) > 0 else 0
        vector["has_negation"] = 1 if ast_stats["num_not"] > 0 else 0
        vector["has_nested_repetition"] = 1 if self._has_nested_repetition(ast) else 0
        vector["repetition_on_group"] = 1 if self._repetition_on_group(ast) else 0

        vector["is_single_event"] = 1 if ast.get("type") == "event" else 0
        vector["is_pure_sequence"] = 1 if self._is_pure_sequence(ast) else 0
        vector["sequence_length"] = self._sequence_length(ast)

        left_event = self._leftmost_event(ast)
        right_event = self._rightmost_event(ast)
        vector["starts_with_create_role"] = 1 if self._event_role_name(left_event) == "create" else 0
        vector["ends_with_use_role"] = 1 if self._event_role_name(right_event) == "use" else 0
        vector["ends_with_query_role"] = 1 if self._event_role_name(right_event) == "query" else 0
        vector["ends_with_close_role"] = 1 if self._event_role_name(right_event) == "close" else 0

        expr_tokens = self.tokenize_ere(expression) if expression else []
        vector["expr_token_count"] = len([t for t in expr_tokens if t[0] != "EOF"])

        # ----------------------------
        # Signature features
        # ----------------------------
        sig_params = signature.get("parameters", []) or []
        sig_types = [str(p.get("type", "")).lower() for p in sig_params if isinstance(p, dict)]
        type_families = self._map_types_to_families(sig_types)

        vector["num_signature_params"] = len(sig_params)
        vector["has_iterator_param"] = 1 if "iterator" in type_families else 0
        vector["has_collection_param"] = 1 if "collection" in type_families else 0
        vector["has_map_param"] = 1 if "map" in type_families else 0
        vector["has_stream_param"] = 1 if ("stream" in type_families or "reader" in type_families) else 0
        vector["has_socket_param"] = 1 if "socket" in type_families else 0
        vector["has_thread_param"] = 1 if "thread" in type_families else 0
        vector["has_permission_param"] = 1 if "permission" in type_families else 0

        # ----------------------------
        # Event / pointcut features
        # ----------------------------
        vector["num_declared_events"] = len(methods)
        vector["num_before_events"] = 0
        vector["num_after_events"] = 0
        vector["num_creation_events"] = 0
        vector["num_returning_events"] = 0

        vector["num_pointcut_atoms"] = 0
        vector["num_pointcut_and"] = 0
        vector["num_pointcut_or"] = 0
        vector["num_negated_pointcuts"] = 0

        vector["num_call_atoms"] = 0
        vector["num_target_atoms"] = 0
        vector["num_args_atoms"] = 0
        vector["num_cflow_atoms"] = 0
        vector["num_if_atoms"] = 0
        vector["num_condition_atoms"] = 0
        vector["num_thread_atoms"] = 0
        vector["num_thread_role_events"] = 0
        vector["num_permission_role_events"] = 0

        vector["num_negated_call"] = 0
        vector["num_negated_target"] = 0
        vector["num_negated_cflow"] = 0

        vector["num_create_role_events"] = 0
        vector["num_modify_role_events"] = 0
        vector["num_use_role_events"] = 0
        vector["num_query_role_events"] = 0
        vector["num_close_role_events"] = 0
        vector["num_config_role_events"] = 0

        pointcut_fn_names: Set[str] = set()
        event_role_set: Set[str] = set()
        lexical_tokens: Set[str] = set()
        distinct_bound_vars: Set[str] = set()

        for e in methods:
            action = (e.get("action", "") or "").lower()
            timing = (e.get("timing", "") or "").lower()
            event_name = (e.get("name", "") or "").lower()

            lexical_tokens |= self._tokenize_text(event_name)
            role = self._event_role(e)
            if role:
                event_role_set.add(role)
                vector[f"num_{role}_role_events"] += 1

            if action == "creation event":
                vector["num_creation_events"] += 1
            if timing == "before":
                vector["num_before_events"] += 1
            elif timing == "after":
                vector["num_after_events"] += 1

            if self._has_real_returning(e):
                vector["num_returning_events"] += 1

            for p in e.get("parameters", []) or []:
                if isinstance(p, dict):
                    lexical_tokens |= self._tokenize_text(p.get("type", ""))
                    lexical_tokens |= self._tokenize_text(p.get("name", ""))
                    if p.get("name"):
                        distinct_bound_vars.add(str(p["name"]).lower())

            ret = e.get("returning", {}) or {}
            if isinstance(ret, dict):
                lexical_tokens |= self._tokenize_text(ret.get("type", ""))
                lexical_tokens |= self._tokenize_text(ret.get("name", ""))
                if ret.get("name"):
                    distinct_bound_vars.add(str(ret["name"]).lower())

            functions = e.get("function", []) or []
            operations = e.get("operation", []) or []

            vector["num_pointcut_atoms"] += len(functions)
            vector["num_pointcut_and"] += sum(1 for op in operations if op == "&&")
            vector["num_pointcut_or"] += sum(1 for op in operations if op == "||")

            for fn in functions:
                fn_name = (fn.get("name", "") or "").lower()
                pointcut_fn_names.add(fn_name)
                lexical_tokens |= self._tokenize_text(fn_name)

                neg = fn.get("negated") is True
                if neg:
                    vector["num_negated_pointcuts"] += 1

                if fn_name == "call":
                    vector["num_call_atoms"] += 1
                    if neg:
                        vector["num_negated_call"] += 1
                elif fn_name == "target":
                    vector["num_target_atoms"] += 1
                    if neg:
                        vector["num_negated_target"] += 1
                elif fn_name == "args":
                    vector["num_args_atoms"] += 1
                elif fn_name == "if":
                    vector["num_if_atoms"] += 1
                elif fn_name == "condition":
                    vector["num_condition_atoms"] += 1
                elif fn_name == "cflow":
                    vector["num_cflow_atoms"] += 1
                    if neg:
                        vector["num_negated_cflow"] += 1
                elif fn_name == "thread":
                    vector["num_thread_atoms"] += 1

                for arg in fn.get("arguments", []) or []:
                    if isinstance(arg, dict):
                        value = str(arg.get("value", ""))
                        lexical_tokens |= self._tokenize_text(value)

        vector["num_distinct_bound_vars"] = len(distinct_bound_vars)
        vector["pointcut_has_disjunction"] = 1 if vector["num_pointcut_or"] > 0 else 0
        vector["pointcut_has_conjunction"] = 1 if vector["num_pointcut_and"] > 0 else 0
        vector["has_iterator_usage_pattern"] = 1 if ("iterator" in type_families or "iterator" in lexical_tokens) else 0
        vector["has_close_after_use_pattern"] = 1 if ("close" in event_role_set and "use" in event_role_set) else 0
        vector["has_modify_after_use_pattern"] = 1 if ("modify" in event_role_set and "use" in event_role_set) else 0

        # ----------------------------
        # Violation features
        # ----------------------------
        violation = ir.get("violation", {}) or {}
        tag = (violation.get("tag") or "").lower()
        body = violation.get("body", {}) or {}

        vector["violation_is_match"] = 1 if tag == "match" else 0
        vector["violation_is_fail"] = 1 if tag == "fail" else 0
        vector["violation_has_reset"] = 1 if body.get("has_reset", False) else 0
        vector["num_violation_statements"] = len(body.get("statements", []) or [])

        # ----------------------------
        # categorical sets for hybrid distance
        # ----------------------------
        vector["_expr_event_set"] = set(ast_stats["event_names"])
        vector["_event_role_set"] = event_role_set
        vector["_pointcut_fn_set"] = pointcut_fn_names
        vector["_type_family_set"] = type_families
        vector["_lexical_token_set"] = lexical_tokens

        # ----------------------------
        # Compress selected numeric features
        # ----------------------------
        for key in (
            "num_nodes",
            "num_expr_event_refs",
            "num_unique_expr_events",
            "num_concat",
            "num_or",
            "num_star",
            "num_plus",
            "num_optional",
            "num_not",
            "num_epsilon",
            "num_empty",
            "max_depth",
            "max_branching",
            "sequence_length",
            "expr_token_count",
            "num_signature_params",
            "num_declared_events",
            "num_before_events",
            "num_after_events",
            "num_creation_events",
            "num_returning_events",
            "num_pointcut_atoms",
            "num_pointcut_and",
            "num_pointcut_or",
            "num_negated_pointcuts",
            "num_call_atoms",
            "num_target_atoms",
            "num_args_atoms",
            "num_cflow_atoms",
            "num_if_atoms",
            "num_condition_atoms",
            "num_thread_atoms",
            "num_thread_role_events",
            "num_permission_role_events",
            "num_negated_call",
            "num_negated_target",
            "num_negated_cflow",
            "num_create_role_events",
            "num_modify_role_events",
            "num_use_role_events",
            "num_query_role_events",
            "num_close_role_events",
            "num_config_role_events",
            "num_distinct_bound_vars",
            "num_violation_statements",
        ):
            val = vector.get(key, 0)
            vector[key] = math.log1p(val) if val >= 0 else 0.0

        return vector

    # ======================================================
    # DISTANCE
    # ======================================================

    def distance(self, v1: dict, v2: dict) -> float:
        numeric_weights = {
            # expression
            "num_nodes": 1.4,
            "num_expr_event_refs": 1.5,
            "num_unique_expr_events": 1.8,
            "num_concat": 1.5,
            "num_or": 2.8,
            "num_star": 2.7,
            "num_plus": 2.7,
            "num_optional": 2.0,
            "num_not": 2.8,
            "num_epsilon": 2.0,
            "num_empty": 2.0,
            "max_depth": 3.0,
            "max_branching": 2.0,
            "has_alternation": 1.8,
            "has_repetition": 1.6,
            "has_negation": 1.8,
            "has_nested_repetition": 3.2,
            "repetition_on_group": 2.7,
            "is_single_event": 1.0,
            "is_pure_sequence": 1.0,
            "sequence_length": 1.3,
            "starts_with_create_role": 2.2,
            "ends_with_use_role": 2.0,
            "ends_with_query_role": 1.8,
            "ends_with_close_role": 1.8,
            "expr_token_count": 1.0,

            # signature
            "num_signature_params": 1.8,
            "has_iterator_param": 3.0,
            "has_collection_param": 2.4,
            "has_map_param": 2.4,
            "has_stream_param": 2.4,
            "has_socket_param": 2.8,
            "has_thread_param": 2.8,
            "has_permission_param": 2.5,

            # events / pointcuts
            "num_declared_events": 2.2,
            "num_before_events": 1.2,
            "num_after_events": 1.2,
            "num_creation_events": 2.8,
            "num_returning_events": 3.2,
            "num_pointcut_atoms": 1.7,
            "num_pointcut_and": 1.3,
            "num_pointcut_or": 1.6,
            "num_negated_pointcuts": 2.7,
            "num_call_atoms": 1.4,
            "num_target_atoms": 1.4,
            "num_args_atoms": 1.6,
            "num_cflow_atoms": 2.8,
            "num_if_atoms": 2.0,
            "num_condition_atoms": 2.0,
            "num_thread_atoms": 2.2,
            "num_negated_call": 2.0,
            "num_negated_target": 2.2,
            "num_negated_cflow": 3.0,
            "num_create_role_events": 2.4,
            "num_modify_role_events": 2.6,
            "num_use_role_events": 2.4,
            "num_query_role_events": 2.0,
            "num_close_role_events": 2.4,
            "num_config_role_events": 2.2,
            "num_distinct_bound_vars": 1.7,
            "pointcut_has_disjunction": 1.5,
            "pointcut_has_conjunction": 1.3,
            "has_iterator_usage_pattern": 2.6,
            "has_close_after_use_pattern": 2.5,
            "has_modify_after_use_pattern": 3.0,

            # violation
            "violation_is_match": 1.6,
            "violation_is_fail": 1.6,
            "violation_has_reset": 2.0,
            "num_violation_statements": 1.0,
        }

        numeric = self.weighted_manhattan(v1, v2, numeric_weights)

        set_distance = 0.0
        set_distance += 3.2 * self.jaccard_distance(v1.get("_event_role_set", set()), v2.get("_event_role_set", set()))
        set_distance += 2.8 * self.jaccard_distance(v1.get("_type_family_set", set()), v2.get("_type_family_set", set()))
        set_distance += 2.5 * self.jaccard_distance(v1.get("_pointcut_fn_set", set()), v2.get("_pointcut_fn_set", set()))
        set_distance += 2.8 * self.jaccard_distance(v1.get("_expr_event_set", set()), v2.get("_expr_event_set", set()))
        set_distance += 1.6 * self.jaccard_distance(v1.get("_lexical_token_set", set()), v2.get("_lexical_token_set", set()))

        return numeric + set_distance

    def weighted_manhattan(self, v1: dict, v2: dict, weights: dict) -> float:
        distance = 0.0
        keys = set(weights.keys())
        for key in keys:
            diff = abs(v1.get(key, 0) - v2.get(key, 0))
            distance += weights.get(key, 1.0) * diff
        return distance

    def jaccard_distance(self, s1: Set[str], s2: Set[str]) -> float:
        s1 = s1 or set()
        s2 = s2 or set()

        if not s1 and not s2:
            return 0.0

        inter = len(s1 & s2)
        union = len(s1 | s2)
        return 1.0 - (inter / union if union > 0 else 0.0)

    # ======================================================
    # JSON HELPERS
    # ======================================================

    def _extract_methods(self, ir: dict) -> List[dict]:
        events_block = ir.get("events", []) or []

        if isinstance(events_block, list):
            methods: List[dict] = []
            for item in events_block:
                if isinstance(item, dict):
                    methods.extend(item.get("body", {}).get("methods", []) or [])
            return methods

        if isinstance(events_block, dict):
            return events_block.get("body", {}).get("methods", []) or []

        return []

    def _has_real_returning(self, event: dict) -> bool:
        ret = event.get("returning")
        if not isinstance(ret, dict):
            return False
        rtype = (ret.get("type") or "").strip()
        rname = (ret.get("name") or "").strip()
        return bool(rtype or rname)

    def _map_types_to_families(self, types_: List[str]) -> Set[str]:
        families: Set[str] = set()
        for t in types_:
            tokens = self._tokenize_text(t)
            for fam, fam_tokens in self._TYPE_FAMILIES.items():
                if tokens & fam_tokens:
                    families.add(fam)
        return families

    def _event_role_name(self, event_name: str) -> str:
        if not event_name:
            return ""
        tokens = self._tokenize_text(event_name)
        for role, kws in self._ROLE_KEYWORDS.items():
            if tokens & kws:
                return role
        return ""

    def _event_role(self, event: dict) -> str:
        name = (event.get("name", "") or "").lower()
        tokens = self._tokenize_text(name)

        for fn in event.get("function", []) or []:
            tokens |= self._tokenize_text(fn.get("name", ""))
            for arg in fn.get("arguments", []) or []:
                if isinstance(arg, dict):
                    tokens |= self._tokenize_text(arg.get("value", ""))

        for role, kws in self._ROLE_KEYWORDS.items():
            if tokens & kws:
                return role

        return ""

    def _tokenize_text(self, text: str) -> Set[str]:
        if not text:
            return set()

        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
        text = text.replace("+", " ").replace("*", " ").replace(".", " ")
        text = text.replace("(", " ").replace(")", " ").replace(",", " ")
        text = text.replace("..", " ")
        tokens = re.findall(r"[A-Za-z_]\w*", text.lower())

        stop = {"call", "target", "args", "condition", "if", "value", "type", "name"}
        return {t for t in tokens if t not in stop}

    # ======================================================
    # INTERNAL ERE PARSER
    # ======================================================

    def tokenize_ere(self, expr: str) -> List[Tuple[str, str]]:
        token_spec = [
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("OR", r"\|"),
            ("STAR", r"\*"),
            ("PLUS", r"\+"),
            ("QMARK", r"\?"),
            ("NOT", r"[!~]"),
            ("EPSILON", r"\bepsilon\b"),
            ("EMPTY", r"\bempty\b"),
            ("IDENT", r"[A-Za-z_]\w*"),
            ("WS", r"\s+"),
        ]

        regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
        pos = 0
        tokens = []

        while pos < len(expr):
            m = re.match(regex, expr[pos:])
            if not m:
                raise ValueError(f"Unexpected token in ERE near: {expr[pos:pos+40]!r}")

            kind = m.lastgroup
            value = m.group(kind)

            if kind != "WS":
                tokens.append((kind, value))

            pos += len(m.group(0))

        tokens.append(("EOF", "EOF"))
        return tokens

    def parse_expression(self, expr: str) -> dict:
        tokens = self.tokenize_ere(expr)
        parser = _EREParser(tokens)
        return parser.parse()

    # ======================================================
    # AST FEATURES
    # ======================================================

    def _extract_ast_stats(self, node: dict) -> dict:
        stats = {
            "num_nodes": 0,
            "num_expr_event_refs": 0,
            "num_concat": 0,
            "num_or": 0,
            "num_star": 0,
            "num_plus": 0,
            "num_optional": 0,
            "num_not": 0,
            "num_epsilon": 0,
            "num_empty": 0,
            "max_depth": 0,
            "max_branching": 0,
            "event_names": set(),
        }

        def walk(n: dict, depth: int):
            ntype = n.get("type")
            stats["num_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            if ntype == "event":
                stats["num_expr_event_refs"] += 1
                if n.get("name"):
                    stats["event_names"].add(n["name"])

            elif ntype == "concat":
                stats["num_concat"] += 1
                children = n.get("children", []) or []
                stats["max_branching"] = max(stats["max_branching"], len(children))
                for ch in children:
                    walk(ch, depth + 1)

            elif ntype == "or":
                stats["num_or"] += 1
                children = n.get("children", []) or []
                stats["max_branching"] = max(stats["max_branching"], len(children))
                for ch in children:
                    walk(ch, depth + 1)

            elif ntype == "star":
                stats["num_star"] += 1
                walk(n.get("child", {}), depth + 1)

            elif ntype == "plus":
                stats["num_plus"] += 1
                walk(n.get("child", {}), depth + 1)

            elif ntype == "optional":
                stats["num_optional"] += 1
                walk(n.get("child", {}), depth + 1)

            elif ntype == "not":
                stats["num_not"] += 1
                walk(n.get("child", {}), depth + 1)

            elif ntype == "epsilon":
                stats["num_epsilon"] += 1

            elif ntype == "empty":
                stats["num_empty"] += 1

        walk(node, 1)
        return stats

    def _has_nested_repetition(self, node: dict, inside_repeat: bool = False) -> bool:
        ntype = node.get("type")
        is_repeat = ntype in {"star", "plus", "optional"}

        if is_repeat and inside_repeat:
            return True

        if "child" in node:
            return self._has_nested_repetition(node["child"], inside_repeat or is_repeat)

        if "children" in node:
            return any(self._has_nested_repetition(ch, inside_repeat or is_repeat) for ch in node["children"])

        return False

    def _repetition_on_group(self, node: dict) -> bool:
        ntype = node.get("type")

        if ntype in {"star", "plus", "optional"}:
            child_type = (node.get("child", {}) or {}).get("type")
            if child_type not in {"event", "epsilon", "empty"}:
                return True
            return self._repetition_on_group(node.get("child", {}))

        if "child" in node:
            return self._repetition_on_group(node["child"])

        if "children" in node:
            return any(self._repetition_on_group(ch) for ch in node["children"])

        return False

    def _is_pure_sequence(self, node: dict) -> bool:
        ntype = node.get("type")

        if ntype == "event":
            return True

        if ntype == "not":
            child = node.get("child", {}) or {}
            return child.get("type") == "event"

        if ntype == "concat":
            return all(self._is_pure_sequence_atom(ch) for ch in node.get("children", []) or [])

        return False

    def _is_pure_sequence_atom(self, node: dict) -> bool:
        ntype = node.get("type")
        if ntype == "event":
            return True
        if ntype == "not":
            child = node.get("child", {}) or {}
            return child.get("type") == "event"
        return False

    def _sequence_length(self, node: dict) -> int:
        if node.get("type") == "concat":
            return len(node.get("children", []) or [])
        return 1

    def _leftmost_event(self, node: dict):
        ntype = node.get("type")

        if ntype == "event":
            return node.get("name")

        if "child" in node:
            return self._leftmost_event(node["child"])

        if "children" in node and node["children"]:
            return self._leftmost_event(node["children"][0])

        return None

    def _rightmost_event(self, node: dict):
        ntype = node.get("type")

        if ntype == "event":
            return node.get("name")

        if "child" in node:
            return self._rightmost_event(node["child"])

        if "children" in node and node["children"]:
            return self._rightmost_event(node["children"][-1])

        return None


class _EREParser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Tuple[str, str]:
        return self.tokens[self.pos]

    def eat(self, expected_kind: str) -> Tuple[str, str]:
        tok = self.current()
        if tok[0] != expected_kind:
            raise ValueError(f"Expected {expected_kind}, found {tok}")
        self.pos += 1
        return tok

    def parse(self) -> dict:
        node = self.parse_alternation()
        if self.current()[0] != "EOF":
            raise ValueError(f"Unexpected trailing token: {self.current()}")
        return node

    def parse_alternation(self) -> dict:
        parts = [self.parse_concatenation()]

        while self.current()[0] == "OR":
            self.eat("OR")
            parts.append(self.parse_concatenation())

        return self._make_nary("or", parts)

    def parse_concatenation(self) -> dict:
        parts = [self.parse_prefix()]

        while self.current()[0] in {"IDENT", "EPSILON", "EMPTY", "LPAREN", "NOT"}:
            parts.append(self.parse_prefix())

        return self._make_nary("concat", parts)

    def parse_prefix(self) -> dict:
        if self.current()[0] == "NOT":
            self.eat("NOT")
            child = self.parse_prefix()
            return {"type": "not", "child": child}

        return self.parse_postfix()

    def parse_postfix(self) -> dict:
        node = self.parse_primary()

        while self.current()[0] in {"STAR", "PLUS", "QMARK"}:
            kind = self.current()[0]

            if kind == "STAR":
                self.eat("STAR")
                node = {"type": "star", "child": node}
            elif kind == "PLUS":
                self.eat("PLUS")
                node = {"type": "plus", "child": node}
            elif kind == "QMARK":
                self.eat("QMARK")
                node = {"type": "optional", "child": node}

        return node

    def parse_primary(self) -> dict:
        kind, value = self.current()

        if kind == "IDENT":
            self.eat("IDENT")
            return {"type": "event", "name": value}

        if kind == "EPSILON":
            self.eat("EPSILON")
            return {"type": "epsilon"}

        if kind == "EMPTY":
            self.eat("EMPTY")
            return {"type": "empty"}

        if kind == "LPAREN":
            self.eat("LPAREN")
            node = self.parse_alternation()
            self.eat("RPAREN")
            return node

        raise ValueError(f"Unexpected token in primary: {(kind, value)}")

    def _make_nary(self, kind: str, children: List[dict]) -> dict:
        flat = []
        for ch in children:
            if ch.get("type") == kind and "children" in ch:
                flat.extend(ch["children"])
            else:
                flat.append(ch)

        if len(flat) == 1:
            return flat[0]

        return {"type": kind, "children": flat}