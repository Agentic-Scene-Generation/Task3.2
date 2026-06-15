"""Shared error helpers for retrieval backends."""


class FatalRetrievalError(RuntimeError):
    """A deterministic retrieval setup error that should not be retried."""


FATAL_RETRIEVAL_ERROR_MARKERS = (
    "failed to download weights",
    "open_clip_pytorch_model.bin",
    "cannot find the requested files in the local cache",
    "sceneexpert_openclip_checkpoint",
    "openclip checkpoint",
    "dfn5b-clip-vit-h-14-378",
)


def is_fatal_retrieval_error(error: str | Exception) -> bool:
    """Return True for deterministic retrieval setup failures."""
    message = str(error).lower()
    return any(marker in message for marker in FATAL_RETRIEVAL_ERROR_MARKERS)
