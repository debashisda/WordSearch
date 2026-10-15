#!/usr/bin/env python3
"""Teach the glyph library new letters.

    python scripts/label_glyphs.py                 from the device
    python scripts/label_glyphs.py shot.png        from a saved frame
    python scripts/label_glyphs.py shot.png --only-unknown

Writes a contact sheet to debug/to_label.png showing the grid with row and
column numbers, then asks you to type the letters row by row. Each labelled
cell is stored as glyphs/<LETTER>/<hash>.png.

You will usually not need this. A word the bot successfully finds labels every
cell it passes through, so the alphabet fills itself in during normal play.
Reach for this script to bootstrap a brand-new install, or when a rare letter
keeps coming back as '?'.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.config import Config                                  # noqa: E402
from wsbot.geometry import detect_grid                           # noqa: E402
from wsbot.glyphs import GLYPH_SIZE, GlyphLibrary, cell_glyph    # noqa: E402


def contact_sheet(img, grid, lib, scale: int = 3) -> np.ndarray:
    """Grid of normalised glyphs with current guesses, for eyeballing."""
    tile = GLYPH_SIZE * scale
    pad = 22
    sheet = np.zeros(((tile + pad) * grid.rows, tile * grid.cols, 3), np.uint8)
    for r in range(grid.rows):
        for c in range(grid.cols):
            g = cell_glyph(img, grid, r, c)
            cellimg = (np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.uint8)
                       if g is None else g)
            big = cv2.resize(cellimg, (tile, tile),
                             interpolation=cv2.INTER_NEAREST)
            y0 = r * (tile + pad)
            sheet[y0:y0 + tile, c * tile:(c + 1) * tile] = \
                cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            guess, score = lib.match(g)
            cv2.putText(sheet, f"r{r}c{c} {guess or '?'} {score:.2f}",
                        (c * tile + 3, y0 + tile + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--only-unknown", action="store_true",
                    help="only ask about cells the library cannot read")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"could not read {args.image}")
            return 2
    else:
        from wsbot.adb import Device
        img = Device(serial=cfg.serial, backend=cfg.backend).screencap()

    lib = GlyphLibrary(cfg.glyph_dir, min_score=cfg.min_score,
                       margin=cfg.match_margin, aspect_tol=cfg.aspect_tol,
                       use_structure=cfg.use_structure)
    grid = cfg.pinned_grid() or detect_grid(img)
    print(grid)

    Path("debug").mkdir(exist_ok=True)
    sheet_path = "debug/to_label.png"
    cv2.imwrite(sheet_path, contact_sheet(img, grid, lib))
    print(f"wrote {sheet_path} — open it alongside this terminal\n")

    if args.only_unknown:
        todo = [(r, c) for r in range(grid.rows) for c in range(grid.cols)
                if lib.match(cell_glyph(img, grid, r, c))[0] is None]
        if not todo:
            print("every cell already reads cleanly; nothing to label")
            return 0
        print(f"{len(todo)} unreadable cell(s)\n")
        added = 0
        for (r, c) in todo:
            ans = input(f"  r{r}c{c} = ").strip().upper()
            if ans in ("", "SKIP"):
                continue
            if len(ans) != 1 or not ans.isalpha():
                print("    one letter please; skipped")
                continue
            if lib.add(ans, cell_glyph(img, grid, r, c)):
                added += 1
        print(f"\nadded {added} sample(s); library now {len(lib)} "
              f"covering {len(lib.letters)}/26")
        return 0

    print(f"Type each row as {grid.cols} letters (blank line skips a row, "
          f"'q' quits).\n")
    added = 0
    for r in range(grid.rows):
        current = "".join(
            (lib.match(cell_glyph(img, grid, r, c))[0] or "?")
            for c in range(grid.cols))
        ans = input(f"  row {r} [{current}]: ").strip().upper()
        if ans == "Q":
            break
        if not ans:
            continue
        if len(ans) != grid.cols:
            print(f"    expected {grid.cols} letters, got {len(ans)}; skipped")
            continue
        for c, ch in enumerate(ans):
            if ch.isalpha() and lib.add(ch, cell_glyph(img, grid, r, c)):
                added += 1

    print(f"\nadded {added} sample(s); library now {len(lib)} samples "
          f"covering {len(lib.letters)}/26 letters")
    if lib.missing:
        print(f"still missing: {''.join(lib.missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
