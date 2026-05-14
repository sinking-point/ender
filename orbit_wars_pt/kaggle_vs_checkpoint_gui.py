"""Tkinter GUI: play Orbit Wars (official Kaggle env) against a policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Mapping

import numpy as np

from orbit_wars_pt.kaggle_local_play import boost_local_overage_after_reset, orbit_wars_local_configuration

CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

BOARD = 100.0
CENTER = BOARD / 2.0
SUN_R = 10.0


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "toJSON"):
        return obj.toJSON()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


def _observation_dict_for_player(env: Any, player_id: int) -> dict[str, Any]:
    """Kaggle player 1 observations omit ``step``; mirror player 0 for the policy."""
    o0 = env.state[0].observation
    d = dict(o0)
    d["player"] = int(player_id)
    st = d.get("step", None)
    if st is None:
        st = o0.get("step", o0.get("step_count", 0))
    d["step"] = int(st) if st is not None else 0
    return d


def _apply_cpu_threads(n: int) -> None:
    if n > 0:
        for name in CPU_THREAD_ENV_VARS:
            os.environ[name] = str(int(n))
        os.environ["ORBIT_WARS_CPU_THREADS"] = str(int(n))
    else:
        os.environ["ORBIT_WARS_CPU_THREADS"] = "0"


class VsCheckpointApp(tk.Tk):
    def __init__(
        self,
        *,
        checkpoint: Path,
        human_player: int,
        device: str,
        greedy: bool,
        agent_seed: int,
        raycast_rays: int | None,
        max_micro_steps: int | None,
        max_human_launches: int | None,
        seed: int | None,
        debug: bool,
        canvas_size: int,
        act_timeout_s: float,
        run_timeout_s: float,
    ) -> None:
        super().__init__()
        self.title("Orbit Wars vs checkpoint")
        self.human_player = int(human_player)
        self.bot_player = 1 - self.human_player
        self.canvas_size = int(canvas_size)
        self.margin = 12
        self.scale = (self.canvas_size - 2 * self.margin) / BOARD

        from kaggle_environments import make

        from orbit_wars_pt.kaggle_adapter import KaggleOrbitWarsAgent

        self.bot = KaggleOrbitWarsAgent(
            checkpoint,
            device=device,
            greedy=bool(greedy),
            max_micro_steps=max_micro_steps,
            seed=int(agent_seed),
            raycast_rays=raycast_rays,
        )
        self.max_human = int(max_human_launches) if max_human_launches is not None else int(self.bot.max_micro_steps)

        cfg = orbit_wars_local_configuration(
            seed=seed,
            act_timeout_s=float(act_timeout_s),
            run_timeout_s=float(run_timeout_s),
        )
        self.env = make("orbit_wars", configuration=cfg, debug=bool(debug))
        # Kaggle Orbit Wars reuses env.info["seed"] on reset(); clear between games unless --seed is set.
        self._fixed_map_seed: int | None = int(seed) if seed is not None else None

        self._selected_pid: int | None = None
        self._queued: list[list[float]] = []
        self._avail: dict[int, float] = {}
        self._preview_angle: float | None = None
        self._ray_angles: np.ndarray | None = None
        self._ray_hit: np.ndarray | None = None

        self._build_ui()
        self._reset_episode()

    def _world_to_canvas(self, wx: float, wy: float) -> tuple[float, float]:
        m, sc = self.margin, self.scale
        return m + wx * sc, m + wy * sc

    def _canvas_to_world(self, cx: float, cy: float) -> tuple[float, float]:
        m, sc = self.margin, self.scale
        return (cx - m) / sc, (cy - m) / sc

    def _ship_speed_config(self) -> float:
        cfg = self.env.configuration
        if isinstance(cfg, dict):
            return float(cfg.get("shipSpeed", 6.0))
        return float(getattr(cfg, "shipSpeed", 6.0))

    def _clear_ray_preview(self) -> None:
        self._ray_angles = None
        self._ray_hit = None

    def _recompute_ray_preview(self, pid: int | None) -> None:
        self._clear_ray_preview()
        if pid is None or self.env.done:
            return
        try:
            from orbit_wars_pt.constants import FRACTIONS
            from orbit_wars_pt.kaggle_adapter import discrete_policy_rays_hit_planet_mask, observation_to_state

            obs = dict(self._obs())
            st = int(obs.get("step", 0) or 0)
            state = observation_to_state(
                obs,
                self.env.configuration,
                max_fleets=512,
                step_count_override=st,
            )
            planets_np = np.asarray(state.planets)
            active = np.asarray(state.planet_active)
            origin_idx: int | None = None
            for i in range(int(planets_np.shape[0])):
                if bool(active[i]) and int(planets_np[i, 0]) == int(pid):
                    origin_idx = i
                    break
            if origin_idx is None:
                return
            frac_idx = len(FRACTIONS) - 1
            angles, hits = discrete_policy_rays_hit_planet_mask(
                state,
                origin_idx,
                frac_idx,
                ship_speed=self._ship_speed_config(),
                n_rays=int(self.bot.raycast_rays),
            )
            self._ray_angles = angles
            self._ray_hit = hits
        except Exception:
            self._clear_ray_preview()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(root, text="")
        self.status.pack(anchor=tk.W)

        mid = ttk.Frame(root)
        mid.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(mid, width=220)
        side.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            side,
            text=(
                f"You: player {self.human_player}\n"
                f"Bot: player {self.bot_player}  (micro_steps≤{self.bot.max_micro_steps})\n"
                "1) Click your planet\n"
                "2) Click the map to aim (from that planet toward the click)\n"
                "   Faint dashed rays = policy discrete raycast (100% ships) that hit a planet\n"
                "3) Adjust ships if needed, Add launch\n"
                "4) Submit turn"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(side, text="Ships to send").pack(anchor=tk.W)
        self.ships_var = tk.StringVar(value="1")
        self.ships_spin = ttk.Spinbox(side, from_=1, to=999, width=8, textvariable=self.ships_var)
        self.ships_spin.pack(anchor=tk.W, pady=(0, 6))

        ttk.Button(side, text="Add launch", command=self._on_add_launch).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Clear queue", command=self._on_clear_queue).pack(fill=tk.X, pady=2)

        self.queue_text = tk.Text(side, height=8, width=26, wrap=tk.WORD, state=tk.DISABLED)
        self.queue_text.pack(fill=tk.BOTH, expand=False, pady=6)

        ttk.Button(side, text="Submit turn (vs bot)", command=self._on_submit).pack(fill=tk.X, pady=4)
        ttk.Button(side, text="New game", command=self._on_new_game).pack(fill=tk.X, pady=2)
        ttk.Button(side, text="Save record JSON…", command=self._on_save_record).pack(fill=tk.X, pady=2)

        self.canvas = tk.Canvas(
            mid,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#0f1419",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _obs(self) -> Mapping[str, Any]:
        return self.env.state[0].observation

    def _reset_episode(self) -> None:
        self._prepare_episode_seed_for_reset()
        self.env.reset(2)
        boost_local_overage_after_reset(self.env)
        self._selected_pid = None
        self._queued = []
        self._preview_angle = None
        self._clear_ray_preview()
        self.ships_var.set("1")
        self._rebuild_avail()
        self._redraw()
        self._set_status()

    def _prepare_episode_seed_for_reset(self) -> None:
        """Let the interpreter draw a new map when ``--seed`` was not passed (see env.info reuse)."""
        if not hasattr(self.env, "info") or self.env.info is None:
            self.env.info = {}
        info = self.env.info
        if self._fixed_map_seed is not None:
            info["seed"] = int(self._fixed_map_seed)
        else:
            info.pop("seed", None)

    def _rebuild_avail(self) -> None:
        self._avail.clear()
        for row in self._obs().get("planets", []) or []:
            if len(row) < 6:
                continue
            pid, owner = int(row[0]), int(row[1])
            if owner == self.human_player:
                self._avail[pid] = float(row[5])

    def _ships_after_queue(self) -> dict[int, float]:
        left = dict(self._avail)
        for mv in self._queued:
            pid = int(mv[0])
            n = int(mv[2])
            if pid in left:
                left[pid] = left.get(pid, 0.0) - float(n)
        return left

    def _ships_default_str(self, pid: int) -> str:
        """Ships available on ``pid`` after queued launches (spinbox default, 1–999)."""
        left = self._ships_after_queue()
        n = int(math.floor(float(left.get(pid, 0.0))))
        n = max(1, min(999, n))
        return str(n)

    def _apply_ships_default_for_planet(self, pid: int | None) -> None:
        if pid is None:
            return
        self.ships_var.set(self._ships_default_str(int(pid)))

    def _set_status(self) -> None:
        if self.env.done:
            r0 = self.env.state[0].reward
            r1 = self.env.state[1].reward
            self.status.config(text=f"Game over. Rewards: P0={r0}  P1={r1}")
            return
        o = self._obs()
        step = o.get("step", 0)
        self.status.config(
            text=f"Step {step}/500  |  queued launches: {len(self._queued)}/{self.max_human}  |  "
            f"selected planet: {self._selected_pid}"
        )

    def _on_canvas_click(self, ev: tk.Event) -> None:
        if self.env.done:
            return
        wx, wy = self._canvas_to_world(ev.x, ev.y)
        best: tuple[float, int] | None = None
        for row in self._obs().get("planets", []) or []:
            if len(row) < 5:
                continue
            pid, owner = int(row[0]), int(row[1])
            if owner != self.human_player:
                continue
            px, py, pr = float(row[2]), float(row[3]), float(row[4])
            d = math.hypot(wx - px, wy - py)
            if d <= pr + 1.5:
                if best is None or d < best[0]:
                    best = (d, pid)
        if best is not None:
            self._selected_pid = best[1]
            self._preview_angle = None
            self._apply_ships_default_for_planet(self._selected_pid)
            self._recompute_ray_preview(self._selected_pid)
            self._redraw()
            self._set_status()
            return
        if self._selected_pid is None:
            return
        for row in self._obs().get("planets", []) or []:
            if len(row) < 4 or int(row[0]) != self._selected_pid:
                continue
            px, py = float(row[2]), float(row[3])
            self._preview_angle = math.atan2(wy - py, wx - px)
            break
        self._redraw()
        self._set_status()

    def _on_add_launch(self) -> None:
        if self.env.done:
            return
        if self._selected_pid is None:
            messagebox.showinfo("Add launch", "Select one of your planets on the map first.")
            return
        if self._preview_angle is None:
            messagebox.showinfo("Add launch", "Click the map (outside your planet) to set firing angle from the selected planet.")
            return
        if len(self._queued) >= self.max_human:
            messagebox.showinfo("Add launch", f"Queue is full ({self.max_human}). Submit or clear.")
            return
        try:
            ships = int(self.ships_var.get())
        except ValueError:
            messagebox.showerror("Add launch", "Ships must be an integer.")
            return
        if ships < 1:
            messagebox.showerror("Add launch", "Ships must be at least 1.")
            return
        left = self._ships_after_queue()
        pid = self._selected_pid
        if pid not in left or ships > math.floor(left[pid]):
            messagebox.showerror("Add launch", f"Not enough ships on planet {pid} for this queue.")
            return
        ang = float(self._preview_angle) % (2.0 * math.pi)
        self._queued.append([float(pid), ang, int(ships)])
        self._refresh_queue_text()
        self._apply_ships_default_for_planet(self._selected_pid)
        self._recompute_ray_preview(self._selected_pid)
        self._redraw()
        self._set_status()

    def _on_clear_queue(self) -> None:
        self._queued.clear()
        self._refresh_queue_text()
        self._apply_ships_default_for_planet(self._selected_pid)
        self._recompute_ray_preview(self._selected_pid)
        self._set_status()

    def _refresh_queue_text(self) -> None:
        self.queue_text.config(state=tk.NORMAL)
        self.queue_text.delete("1.0", tk.END)
        for i, mv in enumerate(self._queued):
            pid, ang, sh = int(mv[0]), float(mv[1]), int(mv[2])
            deg = math.degrees(ang)
            self.queue_text.insert(tk.END, f"{i+1}. id={pid} angle={deg:.1f}° ships={sh}\n")
        if not self._queued:
            self.queue_text.insert(tk.END, "(empty)\n")
        self.queue_text.config(state=tk.DISABLED)

    def _on_submit(self) -> None:
        if self.env.done:
            return
        human_actions = [list(x) for x in self._queued]
        try:
            bot_obs = _observation_dict_for_player(self.env, self.bot_player)
            bot_actions = self.bot(bot_obs, self.env.configuration)
        except Exception as exc:
            messagebox.showerror("Bot error", str(exc))
            return
        if self.human_player == 0:
            pair = [human_actions, bot_actions]
        else:
            pair = [bot_actions, human_actions]
        try:
            self.env.step(pair)
        except Exception as exc:
            messagebox.showerror("Environment step", str(exc))
            return
        self._queued.clear()
        self._refresh_queue_text()
        self._selected_pid = None
        self._preview_angle = None
        self._clear_ray_preview()
        self.ships_var.set("1")
        if not self.env.done:
            self._rebuild_avail()
        self._redraw()
        self._set_status()

    def _on_new_game(self) -> None:
        self._reset_episode()

    def _on_save_record(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        record = {
            "name": "orbit_wars",
            "configuration": json.loads(json.dumps(self.env.configuration, default=_json_default)),
            "steps": json.loads(json.dumps(self.env.steps, default=_json_default)),
        }
        Path(path).write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        messagebox.showinfo("Saved", path)

    def _owner_color(self, owner: int) -> str:
        if owner == -1:
            return "#7a8496"
        if owner == self.human_player:
            return "#3d9cfd"
        if owner == self.bot_player:
            return "#ff6b4a"
        return "#cfd6e6"

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        m, sc = self.margin, self.scale
        # subtle grid
        for g in (25, 50, 75):
            x1, y1 = self._world_to_canvas(g, 0)
            x2, y2 = self._world_to_canvas(g, BOARD)
            c.create_line(x1, y1, x2, y2, fill="#1e2633", width=1)
            x1, y1 = self._world_to_canvas(0, g)
            x2, y2 = self._world_to_canvas(BOARD, g)
            c.create_line(x1, y1, x2, y2, fill="#1e2633", width=1)

        sx, sy = self._world_to_canvas(CENTER, CENTER)
        sr = SUN_R * sc
        c.create_oval(sx - sr, sy - sr, sx + sr, sy + sr, fill="#f5c542", outline="#ffe08a", width=2)

        obs = self._obs()
        planets = obs.get("planets", []) or []
        row_sel = None
        for row in planets:
            if len(row) >= 4 and self._selected_pid is not None and int(row[0]) == int(self._selected_pid):
                row_sel = row
                break
        if (
            row_sel is not None
            and self._ray_angles is not None
            and self._ray_hit is not None
            and bool(np.any(self._ray_hit))
        ):
            px, py, pr = float(row_sel[2]), float(row_sel[3]), float(row_sel[4])
            ray_len = 130.0
            for i in np.flatnonzero(self._ray_hit):
                ang = float(self._ray_angles[int(i)])
                ax = px + math.cos(ang) * (pr + 0.1)
                ay = py + math.sin(ang) * (pr + 0.1)
                bx = ax + math.cos(ang) * ray_len
                by = ay + math.sin(ang) * ray_len
                x0, y0 = self._world_to_canvas(ax, ay)
                x1, y1 = self._world_to_canvas(bx, by)
                c.create_line(x0, y0, x1, y1, fill="#3a4f63", width=1, dash=(2, 5))

        for row in sorted(planets, key=lambda r: float(r[4]) if len(r) > 4 else 0.0):
            if len(row) < 6:
                continue
            pid, owner = int(row[0]), int(row[1])
            px, py, pr, ships = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            cx, cy = self._world_to_canvas(px, py)
            r = max(3.0, pr * sc)
            fill = self._owner_color(owner)
            outline = "#e8f0ff" if pid == self._selected_pid else "#1a2230"
            w = 3 if pid == self._selected_pid else 1
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline, width=w)
            c.create_text(cx, cy, text=str(int(math.floor(ships))), fill="#0a0e12", font=("TkDefaultFont", 9, "bold"))

        if self._selected_pid is not None and self._preview_angle is not None:
            for row in planets:
                if len(row) < 4 or int(row[0]) != self._selected_pid:
                    continue
                px, py, pr = float(row[2]), float(row[3]), float(row[4])
                x0, y0 = self._world_to_canvas(px, py)
                x2 = px + math.cos(self._preview_angle) * 40
                y2 = py + math.sin(self._preview_angle) * 40
                x2, y2 = self._world_to_canvas(x2, y2)
                c.create_line(x0, y0, x2, y2, fill="#7dffcf", width=2, arrow=tk.LAST)
                break

        for f in obs.get("fleets", []) or []:
            if len(f) < 7:
                continue
            owner = int(f[1])
            fx, fy = float(f[2]), float(f[3])
            ang = float(f[4])
            cx, cy = self._world_to_canvas(fx, fy)
            col = self._owner_color(owner)
            s = 4.0
            c.create_polygon(
                cx + s * math.cos(ang),
                cy + s * math.sin(ang),
                cx + s * math.cos(ang + 2.4),
                cy + s * math.sin(ang + 2.4),
                cx + s * math.cos(ang - 2.4),
                cy + s * math.sin(ang - 2.4),
                fill=col,
                outline="#0a0e12",
            )


def run_gui(args: argparse.Namespace) -> None:
    from orbit_wars_pt.kaggle_local_play import DEFAULT_LOCAL_ACT_TIMEOUT_S, DEFAULT_LOCAL_RUN_TIMEOUT_S

    _apply_cpu_threads(int(args.cpu_threads))
    app = VsCheckpointApp(
        checkpoint=Path(args.checkpoint).expanduser(),
        human_player=int(args.human_player),
        device=str(args.device),
        greedy=bool(args.greedy),
        agent_seed=int(args.agent_seed),
        raycast_rays=args.raycast_rays,
        max_micro_steps=args.max_micro_steps,
        max_human_launches=args.max_human_launches,
        seed=args.seed,
        debug=bool(args.debug),
        canvas_size=int(getattr(args, "canvas_size", 720)),
        act_timeout_s=float(getattr(args, "act_timeout", DEFAULT_LOCAL_ACT_TIMEOUT_S)),
        run_timeout_s=float(getattr(args, "run_timeout", DEFAULT_LOCAL_RUN_TIMEOUT_S)),
    )
    app.mainloop()


def main() -> None:
    from orbit_wars_pt.kaggle_local_play import DEFAULT_LOCAL_ACT_TIMEOUT_S, DEFAULT_LOCAL_RUN_TIMEOUT_S

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--human-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Argmax bot; default is stochastic sampling (see --agent-seed).",
    )
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--raycast-rays", type=int, default=None)
    parser.add_argument("--max-micro-steps", type=int, default=None)
    parser.add_argument("--max-human-launches", type=int, default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--canvas-size", type=int, default=720, help="Square canvas side in pixels.")
    parser.add_argument(
        "--act-timeout",
        type=float,
        default=DEFAULT_LOCAL_ACT_TIMEOUT_S,
        help="Kaggle configuration.actTimeout (seconds). Thinks longer than this consume remainingOverageTime.",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=DEFAULT_LOCAL_RUN_TIMEOUT_S,
        help="Kaggle configuration.runTimeout wall-clock seconds for env.run (unused in GUI step loop).",
    )
    args = parser.parse_args()
    run_gui(args)


if __name__ == "__main__":
    main()
