"""Tiny JAX eigvals microbenchmark for CPU vs CUDA.

This isolates ``jnp.linalg.eigvals`` from the tangent pipeline so we can tell
whether batched tiny eigensolves are intrinsically slow on a given backend.

Run once per backend, for example:

    JAX_PLATFORMS=cpu ./.venv/bin/python -m orbit_wars_pt.bench_eigvals_jax --platform cpu
    JAX_PLATFORMS=cuda,cpu ./.venv/bin/python -m orbit_wars_pt.bench_eigvals_jax --platform cuda
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch", type=int, default=1536)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return p.parse_args()


ARGS = parse_args()
if ARGS.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif ARGS.platform == "cuda":
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp
from jax import lax


DTYPE = jnp.float32 if ARGS.dtype == "float32" else jnp.float64


def _make_random_dense(key: jax.Array, batch: int, n: int) -> jnp.ndarray:
    return jax.random.normal(key, (batch, n, n), dtype=DTYPE)


def _make_companion_like(key: jax.Array, batch: int, n: int) -> jnp.ndarray:
    coeff = jax.random.normal(key, (batch, n), dtype=DTYPE)
    coeff = coeff / jnp.maximum(jnp.max(jnp.abs(coeff), axis=1, keepdims=True), 1e-6)
    mats = jnp.zeros((batch, n, n), dtype=DTYPE)
    mats = mats.at[:, 1:, :-1].set(jnp.eye(n - 1, dtype=DTYPE)[None, :, :])
    mats = mats.at[:, :, -1].set(-coeff[:, ::-1])
    return mats


@jax.jit
def _eigvals_only(mats: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.eigvals(mats)


@jax.jit
def _lax_eigvals_default(mats: jnp.ndarray) -> jnp.ndarray:
    return lax.linalg.eig(
        mats,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=False,
    )[0]


@jax.jit
def _lax_eigvals_lapack(mats: jnp.ndarray) -> jnp.ndarray:
    return lax.linalg.eig(
        mats,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=False,
        implementation=lax.linalg.EigImplementation.LAPACK,
    )[0]


@jax.jit
def _lax_eigvals_cusolver(mats: jnp.ndarray) -> jnp.ndarray:
    return lax.linalg.eig(
        mats,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=False,
        implementation=lax.linalg.EigImplementation.CUSOLVER,
    )[0]


@jax.jit
def _lax_eigvals_magma(mats: jnp.ndarray) -> jnp.ndarray:
    return lax.linalg.eig(
        mats,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=False,
        implementation=lax.linalg.EigImplementation.MAGMA,
    )[0]


def _format_row(name: str, compile_run_s: float, mean_run_s: float, min_run_s: float) -> str:
    per_matrix_us = 1e6 * mean_run_s / float(ARGS.batch)
    return (
        f"{name:>18} "
        f"{compile_run_s:>13.4f} "
        f"{mean_run_s:>10.4f} "
        f"{min_run_s:>9.4f} "
        f"{per_matrix_us:>14.3f}"
    )


def _print_row(name: str, row: dict[str, float]) -> None:
    print(_format_row(name, **row), flush=True)


def _bench(name: str, fn, mats: jnp.ndarray) -> dict[str, float] | None:
    print(f"[bench] starting {name}...", flush=True)
    try:
        t0 = time.perf_counter()
        out = fn(mats)
        jax.block_until_ready(out)
        compile_run_s = time.perf_counter() - t0

        times: list[float] = []
        for _ in range(ARGS.repeat):
            t0 = time.perf_counter()
            out = fn(mats)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
    except Exception as e:
        print(f"[bench] failed {name}: {type(e).__name__}: {e}", flush=True)
        return None

    row = {
        "compile_run_s": compile_run_s,
        "mean_run_s": float(np.mean(times)),
        "min_run_s": float(np.min(times)),
    }
    print("[bench] finished " + _format_row(name, **row), flush=True)
    return row


def main() -> None:
    print(f"devices {jax.devices()}", flush=True)
    print(
        f"eig microbench platform={ARGS.platform} batch={ARGS.batch} n={ARGS.n} "
        f"dtype={ARGS.dtype} repeat={ARGS.repeat}",
        flush=True,
    )
    print(
        "name               compile+run_s mean_run_s min_run_s  per_matrix_us",
        flush=True,
    )

    key = jax.random.key(ARGS.seed)
    key_dense, key_comp = jax.random.split(key)
    dense = _make_random_dense(key_dense, int(ARGS.batch), int(ARGS.n))
    companion = _make_companion_like(key_comp, int(ARGS.batch), int(ARGS.n))

    bench_fns: list[tuple[str, object]] = [
        ("jnp_eigvals", _eigvals_only),
        ("lax_default", _lax_eigvals_default),
    ]
    if ARGS.platform == "cpu":
        bench_fns.append(("lax_lapack", _lax_eigvals_lapack))
    else:
        bench_fns.extend(
            [
                ("lax_lapack", _lax_eigvals_lapack),
                ("lax_cusolver", _lax_eigvals_cusolver),
                ("lax_magma", _lax_eigvals_magma),
            ]
        )

    rows: list[tuple[str, str, dict[str, float] | None]] = []
    for matrix_name, mats in [("random_dense", dense), ("companion_like", companion)]:
        for fn_name, fn in bench_fns:
            row = _bench(f"{matrix_name}/{fn_name}", fn, mats)
            rows.append((matrix_name, fn_name, row))

    print("summary", flush=True)
    print(
        "matrix/method       compile+run_s mean_run_s min_run_s  per_matrix_us",
        flush=True,
    )
    for matrix_name, fn_name, row in rows:
        if row is not None:
            _print_row(f"{matrix_name}/{fn_name}", row)


if __name__ == "__main__":
    main()
