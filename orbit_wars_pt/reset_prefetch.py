"""Background CPU processes that precompute ``reset_from_reference`` (Kaggle map + comets).

The main training process overlaps policy / env-step work with subprocesses that
run the slow Kaggle-backed reset on JAX CPU, then ship NumPy trees back for a
cheap ``jnp.asarray`` scatter into the batched device state.

Uses ``spawn`` so workers do not inherit the parent CUDA context. Workers set
``JAX_PLATFORMS=cpu`` before importing JAX.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Dict, List, Optional, Set, Tuple



@dataclass
class PrefetchPopMeta:
    wait_s: float = 0.0
    immediate_bank_hit: bool = False
    drained_results: int = 0
    banked_other_results: int = 0
    fallback_used: bool = False


def _coerce_host_state_type(state: Any) -> Any:
    """Convert worker-produced NumPy namedtuples back to the exact JAX pytree type."""

    try:
        from jax_orbit_wars import OrbitWarsState
    except Exception:
        return state
    if isinstance(state, OrbitWarsState):
        return state
    fields = getattr(state, "_fields", None)
    if fields == OrbitWarsState._fields:
        return OrbitWarsState(*[getattr(state, name) for name in OrbitWarsState._fields])
    return state


@contextmanager
def _spawn_cpu_only_jax_env():
    """Temporarily force CPU-only JAX/CUDA visibility for spawned worker import."""

    keys = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_CUDA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "XLA_FLAGS",
    )
    prev = {k: os.environ.get(k) for k in keys}
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    extra_flags = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    if extra_flags not in xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} {extra_flags}".strip()
    try:
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _worker_loop(task_q: "mp.Queue[Optional[Tuple[int, int, int, int, Optional[int]]]]", result_q: "mp.Queue[Any]") -> None:
    from orbit_wars_pt.reset_numpy import (
        build_unified_exploiter_state_variant_numpy,
        reset_from_reference_numpy,
    )

    while True:
        msg = task_q.get()
        if msg is None:
            break
        gen, seed, num_agents, max_fleets, mode_code = msg
        if mode_code is None:
            np_st = reset_from_reference_numpy(int(seed), int(num_agents), max_fleets=int(max_fleets))
        else:
            np_st = build_unified_exploiter_state_variant_numpy(
                int(seed),
                active_seat_count=int(mode_code),
                max_fleets=int(max_fleets),
            )
        result_q.put((gen, int(seed), int(num_agents), int(max_fleets), mode_code, np_st))


class RolloutResetPrefetch:
    """Submit reset jobs ahead of time; block in ``pop_state`` until a match is ready."""

    def __init__(self, num_workers: int, lookahead: int) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        self._num_workers = int(num_workers)
        self._lookahead = int(lookahead)
        self._ctx: BaseContext = mp.get_context("spawn")
        self._task_q: mp.Queue = self._ctx.Queue()
        self._result_q: mp.Queue = self._ctx.Queue()
        self._procs: List[mp.Process] = []
        self._gen = 0
        self._mf = -1
        self._submitted: Set[Tuple[int, int, int, int, Optional[int]]] = set()
        self._outstanding: Set[Tuple[int, int, int, int, Optional[int]]] = set()
        self._bank: Dict[Tuple[int, int, int, int, Optional[int]], Any] = {}
        self._started = False

    @property
    def lookahead(self) -> int:
        return self._lookahead

    def start(self) -> None:
        if self._started:
            return
        with _spawn_cpu_only_jax_env():
            for _ in range(self._num_workers):
                p = self._ctx.Process(target=_worker_loop, args=(self._task_q, self._result_q), daemon=True)
                p.start()
                self._procs.append(p)
        self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        for _ in self._procs:
            self._task_q.put(None)
        for p in self._procs:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        self._started = False
        self._submitted.clear()
        self._outstanding.clear()
        self._bank.clear()

    def notify_max_fleets(self, max_fleets: int) -> None:
        mf = int(max_fleets)
        if self._mf < 0:
            self._mf = mf
            return
        if mf == self._mf:
            return
        self._mf = mf
        self._gen += 1
        self._submitted.clear()
        self._outstanding.clear()
        self._bank.clear()
        while True:
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                break

    def _submit(
        self,
        gen: int,
        seed: int,
        num_agents: int,
        max_fleets: int,
        mode_code: Optional[int] = None,
    ) -> None:
        key = (gen, int(seed), int(num_agents), int(max_fleets), None if mode_code is None else int(mode_code))
        if key in self._submitted or key in self._bank:
            return
        self._submitted.add(key)
        self._outstanding.add(key)
        self._task_q.put(key)

    def drain_ready(self, max_items: Optional[int] = None) -> int:
        """Move completed worker results into the local bank without blocking."""

        drained = 0
        limit = None if max_items is None else max(0, int(max_items))
        while limit is None or drained < limit:
            try:
                gen_r, seed_r, na_r, mf_r, mode_r, np_st = self._result_q.get_nowait()
            except queue.Empty:
                break
            rkey = (gen_r, seed_r, na_r, mf_r, mode_r)
            self._outstanding.discard(rkey)
            np_st = _coerce_host_state_type(np_st)
            self._bank[rkey] = np_st
            drained += 1
        return drained

    def ready_banked_count(
        self,
        num_agents: int,
        max_fleets: int,
        *,
        mode_code: Optional[int] = None,
    ) -> int:
        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        g = self._gen
        mf = int(max_fleets)
        na = int(num_agents)
        mode = None if mode_code is None else int(mode_code)
        return sum(
            1
            for key in self._bank
            if key[0] == g and key[2] == na and key[3] == mf and key[4] == mode
        )

    def outstanding_count(
        self,
        num_agents: int,
        max_fleets: int,
        *,
        mode_code: Optional[int] = None,
    ) -> int:
        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        g = self._gen
        mf = int(max_fleets)
        na = int(num_agents)
        mode = None if mode_code is None else int(mode_code)
        return sum(
            1
            for key in self._outstanding
            if key[0] == g and key[2] == na and key[3] == mf and key[4] == mode
        )

    def pop_any_banked_state(
        self,
        num_agents: int,
        max_fleets: int,
        *,
        mode_code: Optional[int] = None,
    ) -> Optional[tuple[int, Any]]:
        """Return any ready state matching this shape/mode, along with its seed."""

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        g = self._gen
        mf = int(max_fleets)
        na = int(num_agents)
        mode = None if mode_code is None else int(mode_code)
        for key, val in list(self._bank.items()):
            if key[0] == g and key[2] == na and key[3] == mf and key[4] == mode:
                self._bank.pop(key, None)
                return int(key[1]), val
        return None

    def wait_any_banked_state(
        self,
        num_agents: int,
        max_fleets: int,
        *,
        mode_code: Optional[int] = None,
        sync_timeout_s: float = 120.0,
    ) -> Optional[tuple[int, Any]]:
        """Block until any matching ready state arrives, then return ``(seed, state)``."""

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        ready = self.pop_any_banked_state(
            int(num_agents),
            int(max_fleets),
            mode_code=mode_code,
        )
        if ready is not None:
            return ready
        g = self._gen
        mf = int(max_fleets)
        na = int(num_agents)
        mode = None if mode_code is None else int(mode_code)
        deadline = time.perf_counter() + float(sync_timeout_s)
        while time.perf_counter() < deadline:
            try:
                gen_r, seed_r, na_r, mf_r, mode_r, np_st = self._result_q.get(timeout=0.5)
            except queue.Empty:
                continue
            rkey = (gen_r, seed_r, na_r, mf_r, mode_r)
            self._outstanding.discard(rkey)
            np_st = _coerce_host_state_type(np_st)
            if gen_r == g and na_r == na and mf_r == mf and mode_r == mode:
                return int(seed_r), np_st
            self._bank[rkey] = np_st
            ready = self.pop_any_banked_state(
                int(num_agents),
                int(max_fleets),
                mode_code=mode_code,
            )
            if ready is not None:
                return ready
        return None

    def pop_any_state(
        self,
        num_agents: int,
        max_fleets: int,
        *,
        fallback_seed: int,
        sync_timeout_s: float = 120.0,
        return_meta: bool = False,
    ) -> Any:
        """Return any matching state, waiting on the worker queue first, then falling back in-process."""

        import jax
        import jax_orbit_wars as jow

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        meta = PrefetchPopMeta()
        ready = self.pop_any_banked_state(
            int(num_agents),
            int(max_fleets),
        )
        if ready is not None:
            meta.immediate_bank_hit = True
            seed_r, out = ready
            payload = (int(seed_r), out, meta)
            return payload if return_meta else payload[:2]

        t_wait0 = time.perf_counter()
        ready = self.wait_any_banked_state(
            int(num_agents),
            int(max_fleets),
            sync_timeout_s=float(sync_timeout_s),
        )
        meta.wait_s = time.perf_counter() - t_wait0
        if ready is not None:
            seed_r, out = ready
            payload = (int(seed_r), out, meta)
            return payload if return_meta else payload[:2]

        st = jow.reset_from_reference(int(fallback_seed), int(num_agents), max_fleets=int(max_fleets))
        meta.fallback_used = True
        out = jax.device_get(st)
        payload = (int(fallback_seed), out, meta)
        return payload if return_meta else payload[:2]

    def prefetch_ahead(self, first_seed: int, num_agents: int, max_fleets: int) -> None:
        """Ensure seeds ``first_seed .. first_seed + lookahead - 1`` are queued for current gen."""

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        mf = int(max_fleets)
        if self._mf < 0:
            self._mf = mf
        g = self._gen
        for k in range(self._lookahead):
            self._submit(g, int(first_seed) + k, int(num_agents), mf)

    def submit_plain_range(self, first_seed: int, count: int, num_agents: int, max_fleets: int) -> None:
        """Queue a specific number of future plain reset seeds for the current generation."""

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        mf = int(max_fleets)
        if self._mf < 0:
            self._mf = mf
        g = self._gen
        for k in range(max(0, int(count))):
            self._submit(g, int(first_seed) + k, int(num_agents), mf)

    def prefetch_unified_exploiter_ahead(
        self,
        first_seed_2p: int,
        first_seed_4p: int,
        max_fleets: int,
    ) -> None:
        """Queue padded-2p even seeds and native-4p odd seeds for future unified exploiter resets."""

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        mf = int(max_fleets)
        if self._mf < 0:
            self._mf = mf
        g = self._gen
        for k in range(self._lookahead):
            seed_2p = 2 * (int(first_seed_2p) + k)
            seed_4p = 2 * (int(first_seed_4p) + k) + 1
            self._submit(g, seed_2p, 4, mf, 2)
            self._submit(g, seed_4p, 4, mf, 4)

    def pop_state(
        self,
        seed: int,
        num_agents: int,
        max_fleets: int,
        *,
        sync_timeout_s: float = 120.0,
        return_meta: bool = False,
    ) -> Any:
        """Return a NumPy ``OrbitWarsState`` (``jax.device_get`` shape) for this seed."""

        import jax
        import jax_orbit_wars as jow

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        mf = int(max_fleets)
        g = self._gen
        key = (g, int(seed), int(num_agents), mf, None)
        meta = PrefetchPopMeta()
        if key in self._bank:
            meta.immediate_bank_hit = True
            out = self._bank.pop(key)
            return (out, meta) if return_meta else out
        self._submit(g, int(seed), int(num_agents), mf)

        t_wait0 = time.perf_counter()
        deadline = t_wait0 + float(sync_timeout_s)
        while time.perf_counter() < deadline:
            try:
                gen_r, seed_r, na_r, mf_r, mode_r, np_st = self._result_q.get(timeout=0.5)
            except queue.Empty:
                continue
            meta.drained_results += 1
            rkey = (gen_r, seed_r, na_r, mf_r, mode_r)
            self._outstanding.discard(rkey)
            np_st = _coerce_host_state_type(np_st)
            if rkey == key:
                meta.wait_s = time.perf_counter() - t_wait0
                return (np_st, meta) if return_meta else np_st
            self._bank[rkey] = np_st
            meta.banked_other_results += 1
            if key in self._bank:
                meta.wait_s = time.perf_counter() - t_wait0
                out = self._bank.pop(key)
                return (out, meta) if return_meta else out

        print(
            "[orbit_wars_pt] reset prefetch timed out; falling back to in-process reset "
            f"(seed={seed}, num_agents={num_agents}, max_fleets={mf}).",
            file=sys.stderr,
            flush=True,
        )
        st = jow.reset_from_reference(int(seed), int(num_agents), max_fleets=mf)
        meta.wait_s = time.perf_counter() - t_wait0
        meta.fallback_used = True
        out = jax.device_get(st)
        return (out, meta) if return_meta else out

    def pop_unified_exploiter_state(
        self,
        seed: int,
        active_seat_count: int,
        max_fleets: int,
        *,
        sync_timeout_s: float = 120.0,
        return_meta: bool = False,
    ) -> Any:
        """Return a NumPy unified exploiter reset state for padded-2p or native-4p."""

        import jax

        if not self._started:
            raise RuntimeError("RolloutResetPrefetch.start() first")
        mf = int(max_fleets)
        active = int(active_seat_count)
        g = self._gen
        key = (g, int(seed), 4, mf, active)
        meta = PrefetchPopMeta()
        if key in self._bank:
            meta.immediate_bank_hit = True
            out = self._bank.pop(key)
            return (out, meta) if return_meta else out
        self._submit(g, int(seed), 4, mf, active)

        t_wait0 = time.perf_counter()
        deadline = t_wait0 + float(sync_timeout_s)
        while time.perf_counter() < deadline:
            try:
                gen_r, seed_r, na_r, mf_r, mode_r, np_st = self._result_q.get(timeout=0.5)
            except queue.Empty:
                continue
            meta.drained_results += 1
            rkey = (gen_r, seed_r, na_r, mf_r, mode_r)
            self._outstanding.discard(rkey)
            np_st = _coerce_host_state_type(np_st)
            if rkey == key:
                meta.wait_s = time.perf_counter() - t_wait0
                return (np_st, meta) if return_meta else np_st
            self._bank[rkey] = np_st
            meta.banked_other_results += 1
            if key in self._bank:
                meta.wait_s = time.perf_counter() - t_wait0
                out = self._bank.pop(key)
                return (out, meta) if return_meta else out

        print(
            "[orbit_wars_pt] unified exploiter reset prefetch timed out; falling back to in-process reset "
            f"(seed={seed}, active_seats={active}, max_fleets={mf}).",
            file=sys.stderr,
            flush=True,
        )
        st = build_unified_exploiter_state_variant(int(seed), active_seat_count=active, max_fleets=mf)
        meta.wait_s = time.perf_counter() - t_wait0
        meta.fallback_used = True
        out = jax.device_get(st)
        return (out, meta) if return_meta else out
