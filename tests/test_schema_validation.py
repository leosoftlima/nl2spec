from nl2spec.inspection.validate_ir import IRValidator
from pathlib import Path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

SCHEMA = "nl2spec/schemas/ir.schema.json"

validator = IRValidator(SCHEMA)


def test_valid_ir_files():
    valid_dir = Path("tests/data/valid")
    for json_file in valid_dir.glob("*.json"):
        result = validator.validate_file(json_file)
        assert result.valid, f"{json_file} should be valid"


def test_invalid_ir_files():
    invalid_dir = Path("tests/data/invalid")
    for json_file in invalid_dir.glob("*.json"):
        result = validator.validate_file(json_file)
        assert not result.valid, f"{json_file} should be invalid"
