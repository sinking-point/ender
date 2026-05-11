"""JAX device configuration so the env (`jit_step`) runs on GPU when installed."""

from __future__ import annotations

import os
import warnings


def configure_jax_for_training(*, prefer_gpu: bool = True, verbose: bool = True) -> str:
    """Call once at process start (before heavy JAX work).

    - Leaves allocator defaults sensible for sharing GPU with PyTorch (`PREALLOCATE=false`).
    - Prints detected devices; warns if GPU was requested but JAX is CPU-only.

    Install JAX with CUDA matching your driver, e.g.::

        pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    See https://jax.readthedocs.io/en/latest/installation.html
    """

    # Share GPU more smoothly with PyTorch (both use CUDA malloc).
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax

    backend = jax.default_backend()
    devs = jax.devices()

    if verbose:
        print(f"[orbit_wars_pt] JAX default backend: {backend}")
        print(f"[orbit_wars_pt] JAX devices ({len(devs)}): {devs}")

    if prefer_gpu:
        try:
            gpu = jax.devices("gpu")
            if not gpu:
                warnings.warn(
                    "JAX did not find any GPU devices. `jit_step` will run on CPU unless you "
                    "install the CUDA build of JAX (see jax_setup.configure_jax_for_training docstring).",
                    stacklevel=2,
                )
        except RuntimeError:
            warnings.warn(
                "JAX GPU backend is not available (RuntimeError from jax.devices('gpu')). "
                "Install jax[cuda*_pip] from the JAX CUDA wheel index.",
                stacklevel=2,
            )

    return backend
