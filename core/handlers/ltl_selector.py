from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import re

class LTLFewShotSelector:

    def select(self, files, k, ir_base, return_scores=False):
        base_vector = self.extract_vector(ir_base)
        scored = []

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                template_json = json.load(f)

            if (template_json.get("formalism") or "").lower() != "ltl":
                continue

            template_vector = self.extract_vector(template_json)
            dist = self.distance(template_vector, base_vector)

            scored.append({"path": path, "distance": float(dist)})

        scored.sort(key=lambda x: (x["distance"], x["path"].name))
        top = scored[:k]

        if return_scores:
            return [(item["path"], item["distance"]) for item in top]

        return [item["path"] for item in top]

    def extract_vector(self, spec_json: dict) -> Dict[str, Any]:
        vector = {}

        ir = spec_json.get("ir", {}) or {}
        events = ir.get("events", []) or []
        ltl = ir.get("ltl", {}) or {}

        vector["num_events"] = len(events)

        tokens = set()

        for e in events:
            for m in (e.get("body", {}) or {}).get("methods", []):

                name = (m.get("name") or "")
                tokens.update(self._split_tokens(name))

                for fn in m.get("function", []) or []:
                    for p in fn.get("parameters", []) or []:
                            tokens.update(self._split_tokens(p.get("name", "")))

        # TYPES
        vector["is_password"] = 1 if "password" in tokens else 0
        vector["is_file"] = 1 if "file" in tokens else 0
        vector["is_iterator"] = 1 if "iterator" in tokens else 0
        vector["is_socket"] = 1 if "socket" in tokens else 0

        # ACTIONS
        vector["action_close"] = 1 if "close" in tokens else 0
        vector["action_init"] = 1 if "init" in tokens else 0
        vector["action_delete"] = 1 if "delete" in tokens else 0
        vector["action_deleteonexit"] = 1 if ("delete" in tokens and "exit" in tokens) else 0
        vector["action_fill"] = 1 if any(t in tokens for t in ["fill", "zero", "clear"]) else 0
        vector["action_read"] = 1 if "read" in tokens else 0
        vector["action_connect"] = 1 if "connect" in tokens else 0
        vector["action_next"] = 1 if "next" in tokens else 0
        vector["action_hasnext"] = 1 if ("has" in tokens and "next" in tokens) else 0
        vector["action_hasmore"] = 1 if ("has" in tokens and "more" in tokens) else 0

        # PATTERNS (🔥 NOVO)
        vector["pattern_delete_or"] = 1 if (
            vector["action_delete"] and vector["action_deleteonexit"]
        ) else 0

        vector["pattern_init"] = 1 if (
            vector["action_init"] and not vector["action_close"]
        ) else 0

        vector["pattern_tokenizer"] = 1 if (
            vector["action_hasmore"] and vector["action_next"] and not vector["is_iterator"]
        ) else 0

        # FORMULA
        formula = (ltl.get("value") or "").lower()
        vector["uses_or"] = 1 if "or" in formula else 0

        return vector

    def distance(self, v1, v2):
        weights = {
            "is_password": 12,
            "is_file": 10,
            "is_iterator": 10,
            "is_socket": 8,

            "action_close": 6,
            "action_init": 8,
            "action_delete": 10,
            "action_deleteonexit": 10,
            "action_fill": 12,
            "action_next": 10,
            "action_hasnext": 10,
            "action_hasmore": 10,

            "pattern_delete_or": 15,
            "pattern_init": 12,
            "pattern_tokenizer": 12,

            "uses_or": 6,
        }

        return self._weighted_manhattan(v1, v2, weights)

    def _split_tokens(self, text: str):
        if not text:
         return []

    # quebra camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # quebra nomes compostos importantes
        text = re.sub(r'(password)', r' password ', text, flags=re.IGNORECASE)
        text = re.sub(r'(hasnext)', r' has next ', text, flags=re.IGNORECASE)
        text = re.sub(r'(hasmoreelements)', r' has more elements ', text, flags=re.IGNORECASE)
        text = re.sub(r'(deleteonexit)', r' delete on exit ', text, flags=re.IGNORECASE)

    # remove símbolos
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)

        tokens = text.lower().split()

        return tokens

    def _weighted_manhattan(self, v1, v2, weights):
        distance = 0.0
        for k, w in weights.items():
            distance += w * abs(v1.get(k, 0) - v2.get(k, 0))
        return distance