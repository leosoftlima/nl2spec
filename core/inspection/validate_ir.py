import json
from jsonschema import Draft202012Validator
from pathlib import Path
from operator import attrgetter


class IRValidationResult:
    def __init__(self, valid: bool, errors=None):
        self.valid = valid
        self.errors = errors or []

    def __bool__(self):
        return self.valid


class IRValidator:
    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.validator = Draft202012Validator(self.schema)

    def validate_dict(self, ir: dict) -> IRValidationResult:
        """
        Validate an IR object already loaded as a dict.
        """
        errors = sorted(
            self.validator.iter_errors(ir),
            key=attrgetter("path")
        )

        if not errors:
            return IRValidationResult(valid=True)

        formatted = [self._format_error(e) for e in errors]
        return IRValidationResult(valid=False, errors=formatted)

    def validate_file(self, json_path: str) -> IRValidationResult:
        """
        Validate an IR JSON file.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return IRValidationResult(
                valid=False,
                errors=[f"Invalid JSON file: {e}"]
            )

        return self.validate_dict(data)

    @staticmethod
    def _format_error(error) -> str:
        """
        Format jsonschema.ValidationError into a readable string.
        """
        location = ".".join(str(p) for p in error.absolute_path)
        location = location if location else "<root>"

        return f"[{location}] {error.message}"
