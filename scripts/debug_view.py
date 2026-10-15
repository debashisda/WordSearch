#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.board import read_board                      # noqa: E402
from wsbot.config import Config                         # noqa: E402
from wsbot.glyphs import GlyphLibrary                   # noqa: E402
from wsbot.screens import (button_centre, describe,     # noqa: E402
                           find_continue_button)
from wsbot.solver import Solver                         # noqa: E402

GREEN = (80, 220, 80)
RED = (60, 60, 240)
BLUE = (240, 170, 60)
YELLOW = (60, 220, 240)
WHITE = (255, 255, 255)


def annotate(img: np.ndarray, board, placements=None,
             out: str = "debug/debug.png") -> str:
    vis = img.copy()
    g = board.grid
    px, py, pw, ph = g.panel

    cv2.rectangle(vis, (px, py), (px + pw, py + ph), BLUE, 4)
    cv2.putText(vis, f"{g.rows}x{g.cols} pitch {g.pitch_x:.0f} "
                     f"conf {g.confidence:.2f}",
                (px, max(28, py - 14)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLUE, 3)

    for r in range(g.rows):
        for c in range(g.cols):
            x, y, w, h = g.cell_box(r, c)
            ch = board.letters[r][c]
            tok = ch is not None and ch.startswith("#")
            colour = RED if ch is None else (YELLOW if tok else GREEN)
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
            cx, cy = g.centre(r, c)
            cv2.circle(vis, (cx, cy), 5, YELLOW, -1)
            label = "?" if ch is None else ("?" + ch[1:] if tok else ch)
            cv2.putText(vis, label, (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)

    for w in board.words:
        x, y, bw, bh = w.box
        colour = GREEN if w.found else WHITE
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), colour, 3)
        cv2.putText(vis, f"{w.text}{'*' if w.found else ''}",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colour, 2)

    for p in placements or []:
        pts = [g.centre(r, c) for (r, c) in p.cells]
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, RED, 6)
        cv2.circle(vis, pts[0], 18, RED, 4)
        cv2.putText(vis, p.word, (pts[0][0] - 20, pts[0][1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, vis)
    return out


def main() -> int:
    cfg = Config.load()
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"could not read {sys.argv[1]}")
            return 2
    else:
        from wsbot.adb import Device
        img = Device(serial=cfg.serial, backend=cfg.backend).screencap()

    print("screen:", describe(img))

    # Not a board? Then the useful thing to show is the continue button.
    box = find_continue_button(img)
    if box is not None:
        x, y, w, h = box
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), GREEN, 6)
        cx, cy = button_centre(box)
        cv2.drawMarker(vis, (cx, cy), RED, cv2.MARKER_CROSS, 60, 6)
        cv2.putText(vis, f"continue -> tap ({cx},{cy})", (x, max(30, y - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3)
        Path("debug").mkdir(parents=True, exist_ok=True)
        cv2.imwrite("debug/debug.png", vis)
        print(f"continue button at {box}, tap point {(cx, cy)}")
        print("wrote debug/debug.png")
        return 0

    lib = GlyphLibrary(cfg.glyph_dir, min_score=cfg.min_score,
                       margin=cfg.match_margin, aspect_tol=cfg.aspect_tol,
                       use_structure=cfg.use_structure)
    board = read_board(img, lib, grid=cfg.pinned_grid())
    placements, unplaced = Solver(board.letters).solve(
        [w.symbols for w in board.words if not w.found])

    print(board.grid)
    print(board)
    print(board.grid_text())
    print("words:", ", ".join(f"{w.text}{'*' if w.found else ''}"
                              for w in board.words))
    if board.tokens:
        print(f"unidentified glyphs: {len(board.tokens)} "
              f"({', '.join('?' + t[1:] for t in board.tokens)}) "
              f"— shown yellow; words still solve through them")
    for p in placements:
        print("  ", p)
    if unplaced:
        print("  not located:", ", ".join(unplaced))

    print("wrote", annotate(img, board, placements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
