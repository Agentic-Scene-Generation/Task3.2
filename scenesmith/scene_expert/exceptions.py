"""Typed execution outcomes shared by SceneExpert and SceneSmith stages."""


class StageValidationError(RuntimeError):
    """A stage exhausted repair and still violates deterministic constraints."""

    def __init__(self, stage: str, reasons: list[str] | str):
        self.stage = stage
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__(
            f"{stage} stage failed deterministic validation: "
            + "; ".join(self.reasons)
        )

