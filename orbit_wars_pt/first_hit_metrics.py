"""Compare per-angle first-hit tuples and incoming TA (eta) bins."""

from __future__ import annotations

import math
from dataclasses import dataclass

Hit = tuple[str, int, int]  # (kind, slot, hit_tick)


def hit_incoming_ta(hit: Hit) -> int:
    """TA bin index for a planet hit (``kaggle_adapter``: ``floor(max(hit_tick - 1, 0))``)."""

    kind, _slot, tick = hit
    if kind != "planet" or tick < 0:
        return -1
    return int(math.floor(max(float(tick) - 1.0, 0.0)))


def hit_event(hit: Hit) -> tuple[str, int]:
    """Terminal event without tick (kind + planet slot or -1)."""

    return (hit[0], hit[1])


@dataclass
class HitCompareCounts:
    total: int = 0
    full_match: int = 0
    event_match_tick_diff: int = 0
    planet_slot_match_ta_diff: int = 0
    planet_slot_match_tick_diff: int = 0

    def record(self, ref: Hit, other: Hit) -> None:
        self.total += 1
        if ref == other:
            self.full_match += 1
            return
        if hit_event(ref) == hit_event(other) and ref[2] != other[2]:
            self.event_match_tick_diff += 1
        if ref[0] == "planet" and other[0] == "planet" and ref[1] == other[1]:
            if ref[2] != other[2]:
                self.planet_slot_match_tick_diff += 1
            if hit_incoming_ta(ref) != hit_incoming_ta(other):
                self.planet_slot_match_ta_diff += 1

    def mismatches(self) -> int:
        return self.total - self.full_match

    def format_lines(self, label: str) -> list[str]:
        mm = self.mismatches()
        lines = [
            f"  {label}: full_match={self.full_match}/{self.total}  mismatches={mm}",
        ]
        if mm:
            lines.append(
                f"    same event (kind+slot/sun/board), different hit_tick: "
                f"{self.event_match_tick_diff}"
            )
            lines.append(
                f"    planet same slot, different hit_tick: "
                f"{self.planet_slot_match_tick_diff}"
            )
            lines.append(
                f"    planet same slot, different incoming TA (eta bin): "
                f"{self.planet_slot_match_ta_diff}"
            )
        return lines
