#!/usr/bin/env python3
"""One-time check that the device is reachable and the board reads correctly.

    python scripts/calibrate.py
    python scripts/calibrate.py shot.png --save

Detection is automatic and re-runs on every level, so this is a confidence
check rather than a setup step you must complete. It reports the panel, the
lattice, the letters and the word list, and with --save pins the detected
geometry into config.json for the rare case where auto-detection picks wrong.

If cells come back as '?', the glyph library has not seen those letters yet.
Either play a level (confirmed words label their own cells automatically) or
run scripts/label_glyphs.py to type them in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.board import read_board                      # noqa: E402
from wsbot.config import Config                         # noqa: E402
from wsbot.geometry import detect_grid, fit_lattice, find_panel, glyph_blobs  # noqa: E402
from wsbot.glyphs import GlyphLibrary, cell_glyph       # noqa: E402
from wsbot.lexicon import Lexicon, auto_resolve         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="a saved frame; omit to use the device")
    ap.add_argument("--save", action="store_true",
                    help="pin the detected geometry into config.json")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"could not read {args.image}")
            return 2
        print(f"frame: {args.image}")
    else:
        from wsbot.adb import Device, DeviceError
        try:
            dev = Device(serial=cfg.serial, backend=cfg.backend)
            print(f"device {dev.check()} via {dev.backend} backend")
            img = dev.screencap()
            if getattr(dev, "_capture_timings", None):
                bits = ", ".join(f"{k} {v * 1000:.0f}ms"
                                 for k, v in sorted(dev._capture_timings.items()))
                print(f"screen capture: using {dev._capture} ({bits})")
        except DeviceError as e:
            print(f"device error: {e}")
            return 2

    h, w = img.shape[:2]
    print(f"screen: {w}x{h}")

    panel = find_panel(img)
    print(f"panel:  x={panel[0]} y={panel[1]} w={panel[2]} h={panel[3]}")

    blobs = glyph_blobs(img[panel[1]:panel[1] + panel[3],
                            panel[0]:panel[0] + panel[2]])
    print(f"glyph blobs: {len(blobs)}")

    print("\ncandidate lattices (lower error is better):")
    for err, rows, cols, pxp, _pyp in fit_lattice(blobs, panel[2], panel[3])[:5]:
        print(f"   {rows:2d} x {cols:<2d}  pitch {pxp:6.1f}  error {err:.4f}")

    grid = detect_grid(img, panel=panel)
    print(f"\nchosen: {grid}")
    print(f"cell (0,0) centre {grid.centre(0, 0)}, "
          f"({grid.rows - 1},{grid.cols - 1}) centre "
          f"{grid.centre(grid.rows - 1, grid.cols - 1)}")

    lib = GlyphLibrary(cfg.glyph_dir, min_score=cfg.min_score,
                       margin=cfg.match_margin, aspect_tol=cfg.aspect_tol,
                       use_structure=cfg.use_structure)
    print(f"\nglyph library: {len(lib)} samples, {len(lib.letters)}/26 letters"
          + (f", missing {''.join(lib.missing)}" if lib.missing else ""))

    board = read_board(img, lib, grid=grid)

    lexicon = Lexicon(cfg.word_list)
    if board.tokens and lexicon.available:
        # Preview against a throwaway copy of the library: resolution needs to
        # learn as it iterates, but calibrate must not change anything on disk.
        preview = GlyphLibrary(cfg.glyph_dir, min_score=cfg.min_score,
                               margin=cfg.match_margin,
                               aspect_tol=cfg.aspect_tol,
                               use_structure=cfg.use_structure, readonly=True)
        board, got, _ = auto_resolve(img, board, preview, lexicon, learn=True)
        if got:
            print(f"\nlexicon identified {''.join(sorted(got))} with no help "
                  f"needed (run.py also learns these permanently)")

    print("\ngrid:")
    print("  " + board.grid_text().replace("\n", "\n  "))
    print("\nwords:", ", ".join(f"{x.text}{'*' if x.found else ''}"
                                for x in board.words) or "(none found)")
    if board.unknown_cells:
        print(f"blank cells: {board.unknown_cells}")

    if board.tokens:
        print(f"\n{len(board.tokens)} glyph(s) remain unidentified even after "
              f"the lexicon pass; they play fine as position tokens.")
        print("To pin them down, run once with the real word list:")
        print("   python run.py --words WORD1 WORD2 ...")
        for tok, cells in sorted(board.token_cells().items()):
            print(f"   {'?' + tok[1:]:<4} at {cells[:6]}"
                  + (" ..." if len(cells) > 6 else ""))

    # Report match scores, separating letters that matched from glyphs that
    # were refused. A refusal scoring far below the threshold is a letter the
    # library simply does not have, which is expected and harmless; a refusal
    # scoring just below it is the only case where the threshold is suspect.
    matched, refused = [], []
    for r in range(grid.rows):
        for c in range(grid.cols):
            glyph = cell_glyph(img, grid, r, c)
            letter, score = lib.match(glyph)
            (matched if letter else refused).append(score)

    import numpy as _np
    if matched:
        a = _np.array(matched)
        print(f"\nmatched {len(matched)} cell(s): min={a.min():.3f} "
              f"p5={_np.percentile(a, 5):.3f} median={_np.median(a):.3f}")
    if refused:
        a = _np.array(refused)
        borderline = int((a > lib.min_score - 0.06).sum())
        print(f"refused {len(refused)} cell(s): best score {a.max():.3f} "
              f"(threshold {lib.min_score})")
        if borderline:
            print(f"   {borderline} refusal(s) scored just under the threshold "
                  f"— those may be letters the library does have. If so, lower "
                  f"min_score in config.json.")
        else:
            print("   all refusals scored well below the threshold, i.e. these "
                  "are letters not in the library. Expected — they solve as "
                  "tokens, and --words or normal play will learn them.")

    if args.save:
        cfg.panel = list(panel)
        cfg.rows, cfg.cols = grid.rows, grid.cols
        cfg.save(args.config)
        print(f"\npinned geometry into {args.config}")
        print("note: this stops the bot adapting to a different grid size. "
              "Delete panel/rows/cols from the file to go back to auto-detect.")
    else:
        print("\nlooks right? Geometry is re-detected every level, so there is "
              "nothing to save unless you want to pin it (--save).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
