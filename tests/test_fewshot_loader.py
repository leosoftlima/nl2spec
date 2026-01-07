from nl2spec.handlers.fewshot_loader import FewShotLoader

def test_load_all_fewshot():
    loader = FewShotLoader(
        fewshot_dir="datasets/fewshot",
        schema_path="nl2spec/schemas/ir.schema.json"
    )
    all_examples = loader.all()
    assert len(all_examples) > 0

def test_filter_by_category():
    loader = FewShotLoader(
        fewshot_dir="datasets/fewshot",
        schema_path="nl2spec/schemas/ir.schema.json"
    )
    fsm = loader.by_category("FSM")
    assert all(ex["category"] == "FSM" for ex in fsm)

def test_deterministic_sampling():
    loader = FewShotLoader(
        fewshot_dir="datasets/fewshot",
        schema_path="nl2spec/schemas/ir.schema.json",
        seed=123
    )
    s1 = loader.sample("EVENT", k=2)
    s2 = loader.sample("EVENT", k=2)
    assert s1 == s2
