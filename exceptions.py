class NL2SpecException(RuntimeError):
    """Base exception for all nl2spec errors."""
    pass


class FewShotNotAvailableError(NL2SpecException):
    """Raised when few-shot examples are required but not available."""
    def __init__(self, path: str):
        msg = (
            "\n[ERROR] Few-shot examples are not available.\n\n"
            f"Expected directory not found:\n  {path}\n\n"
            "Few-shot examples are NOT publicly distributed because they\n"
            "are part of an ongoing research artifact.\n\n"
            "If you intend to reproduce or extend the experiments,\n"
            "please contact the authors:\n\n"
            "  📧 lsm2@cin.ufpe.br\n\n"
            "This is an expected limitation, not a misconfiguration.\n"
        )
        super().__init__(msg)

