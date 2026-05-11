"""CUDA memory telemetry: PyTorch allocator vs whole-GPU use (JAX/XLA shares the device)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _mib(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


def gpu_mem_get_info_mib(device: torch.device) -> Optional[Tuple[float, float]]:
    """Return (free_mib, total_mib) from the driver, or None if unavailable."""

    if device.type != "cuda":
        return None
    free_b, total_b = torch.cuda.mem_get_info(device)
    return _mib(free_b), _mib(total_b)


def log_cuda_mem(tag: str, device: torch.device, *, sync: bool = True) -> None:
    """One line: PyTorch allocated / reserved / peak since reset, plus approx. all clients on the GPU."""

    if device.type != "cuda":
        print(f"[mem] {tag} device={device} (no CUDA)")
        return
    if sync:
        torch.cuda.synchronize(device)
    alloc = _mib(torch.cuda.memory_allocated(device))
    rsrv = _mib(torch.cuda.memory_reserved(device))
    peak = _mib(torch.cuda.max_memory_allocated(device))
    msg = (
        f"[mem] {tag}: torch_alloc={alloc:.1f}MiB torch_reserved={rsrv:.1f}MiB "
        f"torch_peak={peak:.1f}MiB"
    )
    info = gpu_mem_get_info_mib(device)
    if info is not None:
        free_mib, total_mib = info
        used_all = total_mib - free_mib
        msg += f" gpu_used_all≈{used_all:.1f}MiB/{total_mib:.1f}MiB"
    print(msg)


def reset_peak_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def print_cuda_memory_summary(device: torch.device, *, abbreviated: bool = True) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    print(torch.cuda.memory_summary(device=device, abbreviated=abbreviated))


def torch_param_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())
