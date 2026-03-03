import json


def normalize_json(obj):

    if isinstance(obj, dict):
        return {k: normalize_json(obj[k]) for k in sorted(obj)}

    elif isinstance(obj, list):

        normalized_list = []

        for item in obj:
            normalized_item = normalize_json(item)
            normalized_list.append(normalized_item)

        # Transformamos cada elemento em string JSON ordenada
        # para permitir ordenação determinística
        sortable_list = []

        for item in normalized_list:
            sortable_representation = json.dumps(item, sort_keys=True)
            sortable_list.append((sortable_representation, item))

        sortable_list.sort()

        # Recuperamos os objetos já ordenados
        ordered_items = []

        for _, original_item in sortable_list:
            ordered_items.append(original_item)

        return ordered_items

    else:
        return obj


def canonical_json_equal(a, b) -> bool:
    return normalize_json(a) == normalize_json(b)


def string_exact_equal(a: str, b: str) -> bool:
    return (a or "").strip() == (b or "").strip()