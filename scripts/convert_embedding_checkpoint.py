"""Convert a trusted local PyTorch embedding checkpoint to safetensors.

This is an offline preparation step for BGE-M3. Transformers 4.55.4 blocks
loading legacy ``pytorch_model.bin`` with torch < 2.6 because of CVE-2025-32434;
using safetensors keeps the SceneSmith/vLLM torch constraint intact.
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path


def convert_checkpoint(model_dir: Path, overwrite: bool = False) -> Path:
    """Convert ``pytorch_model.bin`` in ``model_dir`` to ``model.safetensors``."""
    import torch
    from safetensors.torch import save_file

    source = model_dir / "pytorch_model.bin"
    target = model_dir / "model.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {source}")
    if target.exists() and not overwrite:
        return target

    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict, got {type(state).__name__}")
    tensors = {
        str(key): value.contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError(f"No tensor entries found in checkpoint: {source}")

    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary), metadata={"source": source.name})
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    target = convert_checkpoint(args.model_dir.expanduser(), args.overwrite)
    print(f"Safetensors checkpoint ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
