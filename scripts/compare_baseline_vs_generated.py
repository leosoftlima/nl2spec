from nl2spec.pipeline.compare import compare_dirs

BASELINE = "datasets/baseline_ir"
GENERATED = "datasets/generated_ir/openai/run_01"
OUT = "datasets/results/openai/run_01"

results = compare_dirs(
    baseline_dir=BASELINE,
    generated_dir=GENERATED,
    out_dir=OUT
)

print("Compared:", len(results))
