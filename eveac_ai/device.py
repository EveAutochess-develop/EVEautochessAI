"""Pick CUDA for match inference; fall back to CPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferDevice:
    kind: str  # cuda | cpu | cpu_fallback
    torch_device: object | None
    name: str


def resolve_device(prefer: str = "auto") -> InferDevice:
    try:
        import torch
    except ImportError:
        return InferDevice(kind="cpu", torch_device=None, name="no-torch")
    want_gpu = prefer in ("auto", "gpu", "cuda")
    if prefer == "cpu":
        return InferDevice(kind="cpu", torch_device=torch.device("cpu"), name="cpu")
    if want_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return InferDevice(kind="cuda", torch_device=torch.device("cuda"), name=name)
    return InferDevice(
        kind="cpu_fallback",
        torch_device=torch.device("cpu"),
        name="cuda-unavailable",
    )
