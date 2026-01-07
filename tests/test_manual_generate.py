from pathlib import Path
from core.llms.mock_llm import MockLLM
from pipeline.generate import generate_one


def test_manual_generate():
    base_dir = Path(__file__).resolve().parents[1]  # nl2spec/
    schema_path = base_dir / "core" / "schemas" / "ir.schema.json"

    ir = generate_one(
        scenario_text="A file must be closed after being opened.",
        llm=MockLLM(),
        fewshot_examples=[],
        schema_path=str(schema_path)
    )

    assert ir is not None

