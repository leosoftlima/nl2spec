from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import json
import math
import re


class EventFewShotSelector:
    """
    Structural selector specialized for EVENT few-shots in the new schema.

    Expected JSON format:
      - formalism = "event"
      - ir.events[*].body.methods[*]
      - optional ir.violation.body.statements
      - optional methods[*].violation.body.statements

    Distance:
      - weighted Manhattan only
    """

    _EVENT_ROLE_KEYWORDS = {
        "create": {"create", "init", "new", "construct", "constructor", "staticinit", "staticinitialization"},
        "read": {"read", "decode", "parse", "get", "load", "available"},
        "write": {"write", "append", "print", "flush", "store", "put", "set"},
        "close": {"close", "shutdown", "disconnect"},
        "modify": {"add", "remove", "clear", "retain", "delete", "insert", "offer", "push", "pop"},
        "compare": {"compare", "comparable", "compareto", "equals", "hashcode"},
        "copy": {"copy", "copyof", "arraycopy"},
        "permission": {"permission", "checkpermission"},
        "timeout": {"timeout", "ttl", "trafficclass"},
        "null": {"null"},
    }

    _TYPE_FAMILIES = {
        "iterator": {"iterator", "listiterator", "enumeration"},
        "collection": {"collection", "list", "set", "queue", "deque", "vector", "arraydeque", "priorityqueue"},
        "map": {"map", "hashmap", "treemap", "sortedmap", "navigablemap", "dictionary", "properties", "enummap"},
        "stream": {
            "stream", "inputstream", "outputstream", "reader", "writer",
            "filereader", "bufferedreader", "stringwriter", "chararraywriter"
        },
        "socket": {
            "socket", "serversocket", "datagramsocket", "multicastsocket",
            "inetaddress", "inetsocketaddress", "urlconnection"
        },
        "thread": {"thread", "runnable"},
        "permission": {"permission", "securitymanager", "runtimepermission", "netpermission", "socketpermission"},
        "array": {"array", "arrays"},
        "enum": {"enum", "enummap", "enumset"},
    }

    _TOKEN_FEATURES = {
        "tok_read": {"read"},
        "tok_write": {"write"},
        "tok_close": {"close"},
        "tok_flush": {"flush"},
        "tok_put": {"put"},
        "tok_add": {"add"},
        "tok_remove": {"remove"},
        "tok_compare": {"compare", "comparable", "compareto"},
        "tok_copy": {"copy", "copyof", "arraycopy"},
        "tok_decode": {"decode", "parse"},
        "tok_encode": {"encode"},
        "tok_timeout": {"timeout", "ttl", "trafficclass"},
        "tok_permission": {"permission"},
        "tok_constructor": {"construct", "constructor", "new"},
        "tok_staticinit": {"staticinit", "staticinitialization"},
        "tok_null": {"null"},
        "tok_hashcode": {"hashcode"},
        "tok_clone": {"clone"},
        "tok_serializable": {"serializable"},
        "tok_factory": {"factory", "valueof"},
        "tok_enum": {"enum"},
    }

    _CALL_TOKEN_FEATURES = {
        "call_tok_map": {"map", "hashmap", "treemap", "sortedmap", "navigablemap", "dictionary", "properties"},
        "call_tok_set": {"set", "hashset", "treeset", "sortedset", "enumset"},
        "call_tok_list": {"list", "arraylist", "linkedlist", "vector"},
        "call_tok_collection": {"collection", "collections"},
        "call_tok_queue": {"queue", "priorityqueue", "deque", "arraydeque"},
        "call_tok_stream": {"stream", "inputstream", "outputstream", "objectinputstream", "objectoutputstream"},
        "call_tok_reader": {"reader", "filereader", "bufferedreader"},
        "call_tok_writer": {"writer", "stringwriter", "chararraywriter"},
        "call_tok_socket": {"socket", "serversocket", "datagramsocket", "multicastsocket", "inetsocketaddress"},
        "call_tok_url": {"url", "urlconnection", "httpcookie", "urldecoder", "urlencoder", "idn"},
        "call_tok_system": {"system"},
        "call_tok_arrays": {"arrays", "arraycopy"},
        "call_tok_enum": {"enum", "enummap", "enumset"},
        "call_tok_thread": {"thread", "runnable"},
        "call_tok_permission": {"permission", "runtimepermission", "netpermission", "socketpermission"},
        "call_tok_classloader": {"classloader"},
        "call_tok_constructor": {"new", "constructor", "construct"},
        "call_tok_staticinit": {"staticinitialization", "staticinit"},
        "call_tok_compare": {"compare", "comparable", "compareto"},
        "call_tok_hashcode": {"hashcode"},
        "call_tok_clone": {"clone"},
        "call_tok_timeout": {"timeout", "ttl", "trafficclass"},
        "call_tok_read": {"read"},
        "call_tok_write": {"write"},
        "call_tok_close": {"close"},
        "call_tok_flush": {"flush"},
        "call_tok_add": {"add"},
        "call_tok_put": {"put"},
        "call_tok_remove": {"remove"},
        "call_tok_get": {"get"},
        "call_tok_decode": {"decode", "parse", "valueof"},
        "call_tok_encode": {"encode"},
        "call_tok_copy": {"copy", "copyof", "arraycopy"},
        "call_tok_range_api": {"length", "index", "offset", "size"},
    }

    _COND_TOKEN_FEATURES = {
        "cond_tok_null": {"null"},
        "cond_tok_minus_one": {"-1"},
        "cond_tok_zero": {"0"},
        "cond_tok_range": {"range", "bounds", "offset", "length", "index", "size"},
        "cond_tok_self": {"self", "itself"},
        "cond_tok_same_ref": {"=="},
        "cond_tok_neq": {"!="},
        "cond_tok_gt": {">"},
        "cond_tok_lt": {"<"},
        "cond_tok_ge": {">="},
        "cond_tok_le": {"<="},
        "cond_tok_comparable": {"comparable", "compareto"},
        "cond_tok_permission": {"permission"},
        "cond_tok_timeout": {"timeout", "ttl", "trafficclass"},
        "cond_tok_serializable": {"serializable"},
        "cond_tok_clone": {"clone"},
        "cond_tok_hashcode": {"hashcode"},
        "cond_tok_constructor": {"constructor", "new"},
        "cond_tok_type": {"instanceof", "class", "type"},
        "cond_tok_empty": {"empty"},
    }

    _MESSAGE_TOKEN_FEATURES = {
        "msg_tok_null": {"null"},
        "msg_tok_close": {"close", "closed"},
        "msg_tok_timeout": {"timeout", "ttl", "trafficclass"},
        "msg_tok_permission": {"permission"},
        "msg_tok_constructor": {"constructor"},
        "msg_tok_comparable": {"comparable", "compareto"},
        "msg_tok_hashcode": {"hashcode"},
        "msg_tok_clone": {"clone"},
        "msg_tok_serializable": {"serializable"},
        "msg_tok_range": {"range", "bounds", "index", "length", "offset", "size"},
        "msg_tok_self": {"self", "itself"},
        "msg_tok_read": {"read"},
        "msg_tok_write": {"write"},
        "msg_tok_flush": {"flush"},
        "msg_tok_put": {"put"},
        "msg_tok_add": {"add"},
        "msg_tok_remove": {"remove"},
        "msg_tok_copy": {"copy", "arraycopy"},
        "msg_tok_decode": {"decode", "parse"},
        "msg_tok_encode": {"encode"},
        "msg_tok_factory": {"factory"},
        "msg_tok_enum": {"enum"},
        "msg_tok_socket": {"socket"},
        "msg_tok_stream": {"stream"},
    }

    _PATTERN_FEATURES = {
        "pattern_null_argument": {"null"},
        "pattern_self_reference": {"self", "itself"},
        "pattern_constructor_validation": {"constructor", "construct", "new"},
        "pattern_static_factory": {"factory", "valueof", "decode", "parse"},
        "pattern_staticinit": {"staticinitialization", "staticinit"},
        "pattern_timeout_validation": {"timeout", "ttl", "trafficclass"},
        "pattern_permission_validation": {"permission"},
        "pattern_comparable_validation": {"comparable", "compareto"},
        "pattern_range_validation": {"range", "bounds", "index", "length", "offset", "size"},
        "pattern_serialization_validation": {"serializable", "serialization"},
        "pattern_clone_validation": {"clone"},
        "pattern_hashcode_validation": {"hashcode"},
        "pattern_close_misuse": {"close", "closed", "flush"},
        "pattern_encoding_validation": {"encode", "decode", "ascii", "utf8", "parse"},
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
            if template_formalism != "event":
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
            if template_formalism != "event":
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

        global_violation = (ir.get("violation", {}) or {})
        global_violation_body = (global_violation.get("body", {}) or {})
        global_statements = global_violation_body.get("statements", []) or []

        sig_params = signature.get("parameters", []) or []
        sig_types = [
            str(p.get("type", "")).lower()
            for p in sig_params
            if isinstance(p, dict)
        ]
        type_families = self._map_types_to_families(sig_types)

        # ----------------------------
        # Signature features
        # ----------------------------
        vector["num_signature_params"] = len(sig_params)
        vector["has_iterator_param"] = 1 if "iterator" in type_families else 0
        vector["has_collection_param"] = 1 if "collection" in type_families else 0
        vector["has_map_param"] = 1 if "map" in type_families else 0
        vector["has_stream_param"] = 1 if "stream" in type_families else 0
        vector["has_socket_param"] = 1 if "socket" in type_families else 0
        vector["has_thread_param"] = 1 if "thread" in type_families else 0
        vector["has_permission_param"] = 1 if "permission" in type_families else 0
        vector["has_array_param"] = 1 if any("[]" in t for t in sig_types) else 0
        vector["has_int_param"] = 1 if any(self._is_int_type(t) for t in sig_types) else 0
        vector["has_string_param"] = 1 if any("string" in t for t in sig_types) else 0
        vector["has_object_param"] = 1 if any(self._is_object_like_type(t) for t in sig_types) else 0

        # ----------------------------
        # Event / pointcut features
        # ----------------------------
        vector["num_declared_events"] = len(methods)
        vector["num_before_events"] = 0
        vector["num_after_events"] = 0
        vector["num_returning_events"] = 0
        vector["has_mixed_timing"] = 0

        vector["num_pointcut_atoms"] = 0
        vector["num_pointcut_and"] = 0
        vector["num_pointcut_or"] = 0

        vector["num_call_atoms"] = 0
        vector["num_target_atoms"] = 0
        vector["num_args_atoms"] = 0
        vector["num_condition_atoms"] = 0
        vector["num_if_atoms"] = 0

        vector["has_constructor_call"] = 0
        vector["has_staticinitialization_call"] = 0
        vector["has_void_call"] = 0
        vector["has_primitive_return_call"] = 0
        vector["has_wildcard_call"] = 0

        vector["num_read_role_events"] = 0
        vector["num_write_role_events"] = 0
        vector["num_close_role_events"] = 0
        vector["num_modify_role_events"] = 0
        vector["num_compare_role_events"] = 0
        vector["num_copy_role_events"] = 0
        vector["num_permission_role_events"] = 0
        vector["num_timeout_role_events"] = 0
        vector["num_create_role_events"] = 0

        lexical_tokens: Set[str] = set()
        method_name_tokens: Set[str] = set()
        call_tokens: Set[str] = set()
        condition_tokens: Set[str] = set()
        message_tokens: Set[str] = set()

        timing_seen: Set[str] = set()

        # ----------------------------
        # Violation features (global + local)
        # ----------------------------
        vector["has_local_violation"] = 0
        vector["has_global_violation"] = 1 if global_violation else 0
        vector["num_local_violation_methods"] = 0
        vector["num_global_violation_statements"] = len(global_statements)
        vector["num_local_violation_statements"] = 0

        violation_tags = set()
        if global_violation:
            global_tag = (global_violation.get("tag") or "").lower()
            if global_tag:
                violation_tags.add(global_tag)

        all_violation_statements = list(global_statements)

        for method in methods:
            event_name = (method.get("name", "") or "").lower()
            timing = (method.get("timing", "") or "").lower()
            params = method.get("parameters", []) or []
            returning = method.get("returning", None)
            procediments = method.get("procediments", {}) or {}

            lexical_tokens |= self._tokenize_text(event_name)
            method_name_tokens |= self._tokenize_text(event_name)

            role = self._event_role(method)
            if role:
                role_key = f"num_{role}_role_events"
                if role_key in vector:
                    vector[role_key] += 1

            if timing == "before":
                vector["num_before_events"] += 1
                timing_seen.add("before")
            elif timing == "after":
                vector["num_after_events"] += 1
                timing_seen.add("after")

            if self._has_real_returning(method):
                vector["num_returning_events"] += 1
                if isinstance(returning, dict):
                    lexical_tokens |= self._tokenize_text(returning.get("type", ""))
                    lexical_tokens |= self._tokenize_text(returning.get("name", ""))

            for p in params:
                if isinstance(p, dict):
                    lexical_tokens |= self._tokenize_text(p.get("type", ""))
                    lexical_tokens |= self._tokenize_text(p.get("name", ""))

            functions = procediments.get("function", []) or []
            operations = procediments.get("operation", []) or []

            vector["num_pointcut_atoms"] += len(functions)
            vector["num_pointcut_and"] += sum(1 for op in operations if op == "&&")
            vector["num_pointcut_or"] += sum(1 for op in operations if op == "||")

            for fn in functions:
                fn_name = (fn.get("name", "") or "").lower()
                lexical_tokens |= self._tokenize_text(fn_name)

                param_values = []
                for param in fn.get("parameters", []) or []:
                    if isinstance(param, dict):
                        value = str(param.get("value", ""))
                        lexical_tokens |= self._tokenize_text(value)
                        param_values.append(value)

                joined_value = " ".join(param_values)

                if fn_name == "call":
                    vector["num_call_atoms"] += 1
                    call_sig = self._extract_call_signature(fn)
                    call_tokens |= self._tokenize_text(call_sig)

                    if self._is_constructor_call(call_sig):
                        vector["has_constructor_call"] = 1
                    if self._is_staticinitialization_call(call_sig):
                        vector["has_staticinitialization_call"] = 1
                    if self._has_void_prefix(call_sig):
                        vector["has_void_call"] = 1
                    if self._has_primitive_return(call_sig):
                        vector["has_primitive_return_call"] = 1
                    if "*" in call_sig:
                        vector["has_wildcard_call"] = 1

                elif fn_name == "target":
                    vector["num_target_atoms"] += 1

                elif fn_name == "args":
                    vector["num_args_atoms"] += 1
                    condition_tokens |= self._tokenize_text(joined_value)

                elif fn_name == "condition":
                    vector["num_condition_atoms"] += 1
                    condition_tokens |= self._tokenize_text(joined_value)

                elif fn_name == "if":
                    vector["num_if_atoms"] += 1
                    condition_tokens |= self._tokenize_text(joined_value)

            local_violation = method.get("violation")
            if isinstance(local_violation, dict) and local_violation:
                vector["has_local_violation"] = 1
                vector["num_local_violation_methods"] += 1

                local_tag = (local_violation.get("tag") or "").lower()
                if local_tag:
                    violation_tags.add(local_tag)

                local_body = (local_violation.get("body", {}) or {})
                local_statements = local_body.get("statements", []) or []
                vector["num_local_violation_statements"] += len(local_statements)
                all_violation_statements.extend(local_statements)

        vector["has_mixed_timing"] = 1 if len(timing_seen) > 1 else 0
        vector["pointcut_has_disjunction"] = 1 if vector["num_pointcut_or"] > 0 else 0
        vector["pointcut_has_conjunction"] = 1 if vector["num_pointcut_and"] > 0 else 0
        vector["is_single_event"] = 1 if len(methods) == 1 else 0
        vector["is_multi_event"] = 1 if len(methods) > 1 else 0

        # ----------------------------
        # Aggregated violation features
        # ----------------------------
        vector["violation_is_fail"] = 1 if "fail" in violation_tags else 0
        vector["violation_is_match"] = 1 if "match" in violation_tags else 0
        vector["violation_has_reset"] = 1 if global_violation_body.get("has_reset", False) else 0
        vector["num_violation_statements"] = len(all_violation_statements)
        vector["has_log_violation"] = 1 if any(
            isinstance(stmt, dict) and stmt.get("type") == "log"
            for stmt in all_violation_statements
        ) else 0
        vector["has_raw_violation"] = 1 if any(
            isinstance(stmt, dict) and stmt.get("type") == "raw"
            for stmt in all_violation_statements
        ) else 0

        log_messages = []
        for stmt in all_violation_statements:
            if isinstance(stmt, dict) and stmt.get("type") == "log":
                msg = str(stmt.get("message", ""))
                log_messages.append(msg)
                message_tokens |= self._tokenize_text(msg)

        vector["has_default_message_only"] = (
            1 if log_messages and all(m == "__DEFAULT_MESSAGE" for m in log_messages) else 0
        )

        # ----------------------------
        # Generic token indicator features
        # ----------------------------
        all_tokens = lexical_tokens | method_name_tokens | call_tokens | condition_tokens | message_tokens
        for feat_name, feat_tokens in self._TOKEN_FEATURES.items():
            vector[feat_name] = 1 if all_tokens & feat_tokens else 0

        # ----------------------------
        # Call-token features
        # ----------------------------
        for feat_name, feat_tokens in self._CALL_TOKEN_FEATURES.items():
            vector[feat_name] = 1 if call_tokens & feat_tokens else 0

        # ----------------------------
        # Condition-token features
        # ----------------------------
        for feat_name, feat_tokens in self._COND_TOKEN_FEATURES.items():
            vector[feat_name] = 1 if condition_tokens & feat_tokens else 0

        # ----------------------------
        # Message-token features
        # ----------------------------
        for feat_name, feat_tokens in self._MESSAGE_TOKEN_FEATURES.items():
            vector[feat_name] = 1 if message_tokens & feat_tokens else 0

        # ----------------------------
        # Pattern-level features
        # ----------------------------
        for feat_name, feat_tokens in self._PATTERN_FEATURES.items():
            vector[feat_name] = 1 if all_tokens & feat_tokens else 0

        # ----------------------------
        # Compress selected numeric features only
        # ----------------------------
        for key in (
            "num_signature_params",
            "num_declared_events",
            "num_before_events",
            "num_after_events",
            "num_returning_events",
            "num_pointcut_atoms",
            "num_pointcut_and",
            "num_pointcut_or",
            "num_call_atoms",
            "num_target_atoms",
            "num_args_atoms",
            "num_condition_atoms",
            "num_if_atoms",
            "num_read_role_events",
            "num_write_role_events",
            "num_close_role_events",
            "num_modify_role_events",
            "num_compare_role_events",
            "num_copy_role_events",
            "num_permission_role_events",
            "num_timeout_role_events",
            "num_create_role_events",
            "num_violation_statements",
            "num_local_violation_methods",
            "num_global_violation_statements",
            "num_local_violation_statements",
        ):
            val = vector.get(key, 0)
            vector[key] = math.log1p(val) if val >= 0 else 0.0

        return vector

    # ======================================================
    # DISTANCE
    # ======================================================

    def distance(self, v1: dict, v2: dict) -> float:
        weights = {
            # signature
            "num_signature_params": 2.0,
            "has_iterator_param": 2.5,
            "has_collection_param": 2.0,
            "has_map_param": 2.0,
            "has_stream_param": 2.0,
            "has_socket_param": 2.2,
            "has_thread_param": 2.2,
            "has_permission_param": 2.2,
            "has_array_param": 1.8,
            "has_int_param": 2.0,
            "has_string_param": 1.8,
            "has_object_param": 1.4,

            # events / pointcuts
            "num_declared_events": 3.0,
            "num_before_events": 1.4,
            "num_after_events": 2.8,
            "num_returning_events": 4.8,
            "has_mixed_timing": 4.2,
            "num_pointcut_atoms": 2.0,
            "num_pointcut_and": 1.4,
            "num_pointcut_or": 1.8,
            "num_call_atoms": 1.6,
            "num_target_atoms": 1.6,
            "num_args_atoms": 1.5,
            "num_condition_atoms": 2.6,
            "num_if_atoms": 2.4,
            "has_constructor_call": 2.0,
            "has_staticinitialization_call": 3.2,
            "has_void_call": 1.5,
            "has_primitive_return_call": 3.8,
            "has_wildcard_call": 1.0,
            "pointcut_has_disjunction": 1.8,
            "pointcut_has_conjunction": 1.0,
            "is_single_event": 1.0,
            "is_multi_event": 1.8,

            # roles
            "num_read_role_events": 2.0,
            "num_write_role_events": 2.0,
            "num_close_role_events": 3.6,
            "num_modify_role_events": 2.2,
            "num_compare_role_events": 2.2,
            "num_copy_role_events": 2.2,
            "num_permission_role_events": 2.0,
            "num_timeout_role_events": 2.0,
            "num_create_role_events": 2.2,

            # violation
            "violation_is_fail": 1.6,
            "violation_is_match": 1.6,
            "violation_has_reset": 2.2,
            "num_violation_statements": 1.2,
            "has_log_violation": 1.8,
            "has_raw_violation": 1.2,
            "has_default_message_only": 1.0,
            "has_local_violation": 3.2,
            "has_global_violation": 0.8,
            "num_local_violation_methods": 2.6,
            "num_global_violation_statements": 0.6,
            "num_local_violation_statements": 1.4,

            # generic token indicators
            "tok_read": 1.8,
            "tok_write": 2.0,
            "tok_close": 2.8,
            "tok_flush": 2.0,
            "tok_put": 2.0,
            "tok_add": 2.0,
            "tok_remove": 2.0,
            "tok_compare": 2.0,
            "tok_copy": 2.0,
            "tok_decode": 3.2,
            "tok_encode": 2.0,
            "tok_timeout": 2.0,
            "tok_permission": 2.0,
            "tok_constructor": 2.2,
            "tok_staticinit": 2.4,
            "tok_null": 1.8,
            "tok_hashcode": 1.8,
            "tok_clone": 1.8,
            "tok_serializable": 1.8,
            "tok_factory": 1.8,
            "tok_enum": 1.8,

            # call token features
            "call_tok_map": 2.0,
            "call_tok_set": 1.6,
            "call_tok_list": 3.2,
            "call_tok_collection": 1.8,
            "call_tok_queue": 1.8,
            "call_tok_stream": 2.0,
            "call_tok_reader": 1.2,
            "call_tok_writer": 2.0,
            "call_tok_socket": 2.2,
            "call_tok_url": 2.0,
            "call_tok_system": 1.8,
            "call_tok_arrays": 2.0,
            "call_tok_enum": 2.0,
            "call_tok_thread": 2.0,
            "call_tok_permission": 2.2,
            "call_tok_classloader": 2.2,
            "call_tok_constructor": 2.4,
            "call_tok_staticinit": 2.6,
            "call_tok_compare": 2.2,
            "call_tok_hashcode": 2.0,
            "call_tok_clone": 2.0,
            "call_tok_timeout": 2.2,
            "call_tok_read": 2.2,
            "call_tok_write": 1.8,
            "call_tok_close": 3.0,
            "call_tok_flush": 1.8,
            "call_tok_add": 1.8,
            "call_tok_put": 1.8,
            "call_tok_remove": 1.8,
            "call_tok_get": 2.0,
            "call_tok_decode": 4.2,
            "call_tok_encode": 2.2,
            "call_tok_copy": 2.0,
            "call_tok_range_api": 2.4,

            # condition token features
            "cond_tok_null": 2.6,
            "cond_tok_minus_one": 3.8,
            "cond_tok_zero": 1.2,
            "cond_tok_range": 2.2,
            "cond_tok_self": 2.4,
            "cond_tok_same_ref": 2.0,
            "cond_tok_neq": 1.8,
            "cond_tok_gt": 1.5,
            "cond_tok_lt": 1.5,
            "cond_tok_ge": 1.5,
            "cond_tok_le": 1.5,
            "cond_tok_comparable": 2.2,
            "cond_tok_permission": 2.2,
            "cond_tok_timeout": 2.2,
            "cond_tok_serializable": 2.0,
            "cond_tok_clone": 2.0,
            "cond_tok_hashcode": 2.0,
            "cond_tok_constructor": 2.0,
            "cond_tok_type": 1.8,
            "cond_tok_empty": 2.2,

            # message token features
            "msg_tok_null": 2.0,
            "msg_tok_close": 3.4,
            "msg_tok_timeout": 2.0,
            "msg_tok_permission": 2.2,
            "msg_tok_constructor": 2.2,
            "msg_tok_comparable": 2.2,
            "msg_tok_hashcode": 2.0,
            "msg_tok_clone": 2.0,
            "msg_tok_serializable": 2.0,
            "msg_tok_range": 1.8,
            "msg_tok_self": 2.0,
            "msg_tok_read": 1.6,
            "msg_tok_write": 1.6,
            "msg_tok_flush": 1.6,
            "msg_tok_put": 1.6,
            "msg_tok_add": 1.6,
            "msg_tok_remove": 1.6,
            "msg_tok_copy": 1.8,
            "msg_tok_decode": 2.8,
            "msg_tok_encode": 1.8,
            "msg_tok_factory": 1.8,
            "msg_tok_enum": 1.8,
            "msg_tok_socket": 1.8,
            "msg_tok_stream": 1.8,

            # pattern features
            "pattern_null_argument": 2.5,
            "pattern_self_reference": 2.8,
            "pattern_constructor_validation": 2.0,
            "pattern_static_factory": 3.0,
            "pattern_staticinit": 2.8,
            "pattern_timeout_validation": 2.5,
            "pattern_permission_validation": 2.6,
            "pattern_comparable_validation": 2.6,
            "pattern_range_validation": 2.2,
            "pattern_serialization_validation": 2.4,
            "pattern_clone_validation": 2.2,
            "pattern_hashcode_validation": 2.2,
            "pattern_close_misuse": 4.2,
            "pattern_encoding_validation": 3.4,
        }

        return self.weighted_manhattan(v1, v2, weights)

    def weighted_manhattan(self, v1: dict, v2: dict, weights: dict) -> float:
        distance = 0.0
        for key, weight in weights.items():
            diff = abs(v1.get(key, 0) - v2.get(key, 0))
            distance += weight * diff
        return distance

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

    def _event_role(self, event: dict) -> str:
        name = (event.get("name", "") or "").lower()
        tokens = self._tokenize_text(name)

        procediments = event.get("procediments", {}) or {}
        for fn in procediments.get("function", []) or []:
            tokens |= self._tokenize_text(fn.get("name", ""))
            for p in fn.get("parameters", []) or []:
                if isinstance(p, dict):
                    tokens |= self._tokenize_text(p.get("value", ""))

        for role, kws in self._EVENT_ROLE_KEYWORDS.items():
            if tokens & kws:
                return role

        return ""

    def _extract_call_signature(self, fn: dict) -> str:
        if not isinstance(fn, dict):
            return ""
        params = fn.get("parameters", []) or []
        values = []
        for p in params:
            if isinstance(p, dict):
                v = str(p.get("value", "")).strip()
                if v:
                    values.append(v)
        return " ".join(values)

    def _tokenize_text(self, text: str) -> Set[str]:
        if not text:
            return set()

        text = str(text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.replace("+", " ")
        text = text.replace("*", " ")
        text = text.replace(".", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace(",", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        text = text.replace("..", " ")

        words = re.findall(r"[A-Za-z_]\w*", text.lower())

        symbolic = set()
        for sym in ("==", "!=", ">=", "<=", ">", "<"):
            if sym in text:
                symbolic.add(sym)

        stop = {
            "call", "target", "args", "condition", "if",
            "value", "type", "name", "event"
        }

        return ({w for w in words if w not in stop}) | symbolic

    def _is_int_type(self, t: str) -> bool:
        return t.strip().lower() == "int"

    def _is_object_like_type(self, t: str) -> bool:
        primitive = {"int", "long", "double", "float", "short", "byte", "char", "boolean", "void"}
        t = t.strip().lower()
        return bool(t) and t not in primitive and "[]" not in t

    def _is_constructor_call(self, sig: str) -> bool:
        sig = sig.lower()
        return "new(" in sig or sig.startswith("new ")

    def _is_staticinitialization_call(self, sig: str) -> bool:
        return "staticinitialization(" in sig.lower()

    def _has_void_prefix(self, sig: str) -> bool:
        return sig.strip().lower().startswith("void ")

    def _has_primitive_return(self, sig: str) -> bool:
        sig = sig.strip().lower()
        return sig.startswith(("int ", "boolean ", "long ", "double ", "float ", "short ", "byte ", "char "))