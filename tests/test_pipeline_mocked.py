from core.pipeline.orchestrator import Orchestrator
from core.llms.mock_llm import MockLLM

def test_pipeline_mocked():
    orch = Orchestrator(
        llm=MockLLM(),
        schema_path="core/schemas/ir.schema.json",
        fewshot_dir="datasets/fewshot"
    )

    irs = orch.run_generate(
        scenarios=["test scenario"],
        category="EVENT",
        k_fewshot=1
    )

    assert len(irs) == 1
