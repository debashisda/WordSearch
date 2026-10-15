#!/usr/bin/env python3
"""Open the game and press through to a playable board.

    python scripts/launch_game.py
    python scripts/launch_game.py --restart          # force-stop first
    python scripts/launch_game.py --package com.other.game

Launching alone is not enough: the app shows a splash, then a loading screen
for several seconds, then a main menu that has to be pressed before a board
exists. This waits for each in turn and reports what it sees.

`run.py --launch` does the same thing before it starts playing, so this script
is mainly for checking the sequence works on your device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.adb import Device, DeviceError          # noqa: E402
from wsbot.config import Config                    # noqa: E402
from wsbot.launcher import DEFAULT_PACKAGE, launch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None,
                    help=f"app to start (default {DEFAULT_PACKAGE})")
    ap.add_argument("--restart", action="store_true",
                    help="force-stop the app first, for a clean start")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.serial:
        cfg.serial = args.serial
    package = args.package or cfg.package or DEFAULT_PACKAGE

    try:
        dev = Device(serial=cfg.serial, backend=cfg.backend)
        serial = dev.check()
    except DeviceError as e:
        print(f"device error: {e}", file=sys.stderr)
        return 2

    print(f"device {serial} via {dev.backend} backend")
    ok = launch(dev, package=package, force_stop=args.restart,
                timeout=args.timeout)
    if ok:
        print("\nready — run `python run.py --forever` to start playing")
        return 0
    print("\ncould not reach a board. Run scripts/debug_view.py to see what "
          "is on screen.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
