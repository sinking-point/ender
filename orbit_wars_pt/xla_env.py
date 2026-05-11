"""XLA / JAX GPU client settings. Import this **before** any module that does ``import jax``.

JAX reads ``XLA_PYTHON_CLIENT_*`` the first time the XLA GPU backend initializes; setting these
in `configure_jax_for_training` is often **too late** if something else (e.g. `parallel_rollout`)
already imported JAX. Use::

    import orbit_wars_pt.xla_env  # noqa: F401

as the first ``orbit_wars_pt`` import in your entry script.

Shell exports still win: we use ``setdefault`` so you can override from the environment.
"""

from __future__ import annotations

import os


def configure_xla_client_env() -> None:
    """Apply default XLA client env vars (no JAX imports here)."""

    # Grow GPU memory as needed instead of grabbing a large slab up front (helps PyTorch coexistence).
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Optional cap; omit by default so only users who set this (shell or here) get a limit.
    # os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")


configure_xla_client_env()
