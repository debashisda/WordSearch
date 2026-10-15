#!/usr/bin/env python3
"""Offline end-to-end test. No phone needed.

    python scripts/selftest.py <frame.png> [<frame2.png> ...]

Runs capture -> geometry -> glyphs -> board -> solver -> gesture planning
against saved frames and prints what would be sent to the device. Use it to
check a new build of the game, a new screen size, or a change to the pipeline
without touching a device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.adb import ReplayDevice                      # noqa: E402
from wsbot.board import read_board, read_board_stable   # noqa: E402
from wsbot.config import Config                         # noqa: E402
from wsbot.glyphs import GlyphLibrary                   # noqa: E402
from wsbot.player import overshoot, path_for            # noqa: E402
from wsbot.solver import Solver                         # noqa: E402


def main() -> int:
    frames = sys.argv[1:]
    if not frames:
        print(__doc__)
        return 2

    cfg = Config.load()
    lib = GlyphLibrary(cfg.glyph_dir, min_score=cfg.min_score,
                       margin=cfg.match_margin, aspect_tol=cfg.aspect_tol,
                       use_structure=cfg.use_structure)
    print(f"glyph library: {len(lib)} samples, {len(lib.letters)}/26 letters"
          + (f" (missing {''.join(lib.missing)})" if lib.missing else ""))

    dev = ReplayDevice(frames)
    board = (read_board_stable(dev, lib, frames=min(3, len(frames)), delay=0)
             if len(frames) > 1 else read_board(cv2.imread(frames[0]), lib))

    print(f"\n{board.grid}")
    print(board)
    print(board.grid_text())
    print("\nwords: " + ", ".join(f"{w.text}{'*' if w.found else ''}"
                                  for w in board.words))
    if board.unknown_cells:
        print(f"unreadable cells: {board.unknown_cells}")

    todo = [w.symbols for w in board.words if not w.found]
    placements, unplaced = Solver(board.letters).solve(todo)

    print(f"\nplanned gestures ({len(placements)}):")
    for p in placements:
        pts = overshoot(path_for(p, board.grid), int(board.grid.pitch_x * 0.18))
        ms = max(320, 55 * len(pts))
        trail = " -> ".join(f"({x},{y})" for x, y in pts)
        print(f"  {p.word:<8} {p.direction_name:<2} {ms:>4}ms  {trail}")

    if unplaced:
        print(f"\nnot located: {', '.join(unplaced)}")
        return 1

    print("\nall words located and planned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
