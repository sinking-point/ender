"""Render a saved Kaggle Orbit Wars record to an HTML replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=str, help="Record JSON written by play_kaggle_selfplay.py.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML path. Defaults to the record path with .html suffix.",
    )
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    record_path = Path(args.record).expanduser()
    out_path = Path(args.out).expanduser() if args.out else record_path.with_suffix(".html")

    with record_path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    from kaggle_environments import make
    from kaggle_environments.utils import structify

    env = make(record.get("name", "orbit_wars"), configuration=record.get("configuration") or {})
    env.steps = structify(record["steps"])

    try:
        html = env.render(mode="html", width=int(args.width), height=int(args.height))
    except TypeError:
        html = None
    if not html:
        html = env.to_html(width=int(args.width), height=int(args.height))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
