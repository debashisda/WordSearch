#!/usr/bin/env python3
"""Play Word Search over ADB
    python run.py                      solve the level on screen, then stop
    python run.py --levels 10          keep going for ten levels
    python run.py --forever            keep playing until you press Ctrl-C
    python run.py --launch --forever   open the game first, then play on
    python run.py --dry-run            read and solve, but send no gestures
    python run.py --once --debug       one level, saving annotated frames
"""

from __future__ import annotations

import argparse
import sys
import time

from wsbot.adb import Device, DeviceError
from wsbot.board import learn_pill, read_board_stable, unique_pill
from wsbot.config import Config
from wsbot.geometry import GeometryError
from wsbot.glyphs import GlyphLibrary, is_token
from wsbot.launcher import launch
from wsbot.lexicon import Lexicon, auto_resolve
from wsbot.player import play_board, wait_for_new_board
from wsbot.solver import Solver, repair_unplaced


def banner(msg: str) -> None:
    print(f"\n{msg}\n" + "-" * len(msg))


def teach(device, lib: GlyphLibrary, board, words) -> int:
    shot = device.screencap()
    solver = Solver(board.letters)
    placements, unplaced = Solver(board.letters).solve(words)
    learned = 0
    for p in placements:
        if len(solver.find_all(list(p.symbols))) != 1:
            print(f"  {p.word} sits in more than one place; not learning from it")
            continue
        learned += lib.learn_from_word(shot, board.grid, p.cells, p.symbols)
        w = unique_pill(board, p.symbols)
        if w is not None:
            learned += learn_pill(shot, w.box, p.symbols, lib)
    print(f"  learned {learned} new glyph sample(s) from {len(placements)} word(s)")
    if unplaced:
        print(f"  could not place from your list: {', '.join(unplaced)}")
    print(
        f"  library now {len(lib)} samples, {len(lib.letters)}/26 letters"
        + (f" (missing {''.join(lib.missing)})" if lib.missing else "")
    )
    return learned


def solve_level(
    device,
    lib: GlyphLibrary,
    cfg: Config,
    dry_run: bool,
    debug: bool = False,
    learn: bool = True,
    words: list[str] | None = None,
    lexicon: Lexicon | None = None,
    no_batch: bool = False,
):

    board = read_board_stable(
        device, lib, grid=cfg.pinned_grid(), frames=cfg.consensus_frames
    )
    print(f"  {board.grid}")
    print(f"  {board}")
    print("  " + board.grid_text().replace("\n", "\n  "))

    if words:
        print(f"  using the word list you supplied: {', '.join(words)}")
        teach(device, lib, board, words)
        board = read_board_stable(
            device, lib, grid=board.grid, frames=cfg.consensus_frames
        )
        print("  re-read after learning:")
        print("  " + board.grid_text().replace("\n", "\n  "))

    # Work out what any unknown glyphs actually are, using the word list plus
    # the fact that a refused glyph cannot be a letter the library holds.
    if board.tokens and lexicon is not None and lexicon.available:
        board, got, n = auto_resolve(
            device.screencap(), board, lib, lexicon, learn=learn, verbose=True
        )
        if got:
            print(
                f"  identified {''.join(sorted(got))} on its own"
                + (f"; learned {n} glyph sample(s)" if n else "")
            )
            print("  " + board.grid_text().replace("\n", "\n  "))

    if board.tokens:
        print(
            f"  {len(board.tokens)} glyph(s) still unidentified; playing "
            f"through them as position tokens"
        )
        if lib.missing:
            print(
                f"  library is missing: {''.join(lib.missing)}  "
                f"(these fill in as words are found, or pass --words once)"
            )

    todo = [w.symbols for w in board.words if not w.found]
    if not todo:
        print("  nothing left to find on this board")
        return True, board
    print("  to find: " + ", ".join(w.text for w in board.words if not w.found))

    placements, unplaced = Solver(board.letters).solve(todo)

    # A word the grid does not contain is a misread pill, not a missing word.
    # The grid can usually say what it should have been.
    if unplaced:
        still: list[str] = []
        for text in unplaced:
            entry = next(
                (w for w in board.words if w.text == text and not w.found), None
            )
            fixed = (
                repair_unplaced(list(entry.symbols), board.letters, lexicon)
                if entry is not None
                else None
            )
            if fixed is None:
                still.append(text)
                continue
            symbols, placement = fixed
            print(
                f"    {text} is not in the grid; the grid reads it as "
                f"{''.join(symbols)}"
            )
            # Correct the board's own reading so the pill is matched, verified
            # and learned from exactly like a word that read cleanly.
            entry.symbols = list(symbols)
            placements.append(placement)
        unplaced = still

    for p in placements:
        print(f"    {p}")
    if unplaced:
        print(f"  could not locate: {', '.join(unplaced)}")

    if debug:
        from scripts.debug_view import annotate

        annotate(device.screencap(), board, placements, "debug/level.png")
        print("  wrote debug/level.png")

    if dry_run:
        print("  dry run: no gestures sent")
        return (not unplaced), board

    results, final, complete = play_board(
        device,
        lib,
        board,
        placements,
        learn=learn,
        settle=cfg.settle,
        gap=cfg.gap,
        batch=cfg.batch and not no_batch,
    )
    failed = [r.placement.word for r in results if not r.ok]
    if failed:
        print(f"  did not register: {', '.join(failed)}")
    return complete, final


def main() -> int:
    ap = argparse.ArgumentParser(description="Word Search bot")
    ap.add_argument(
        "--levels",
        type=int,
        default=1,
        help="how many levels to play (default 1). Use 0 to keep "
        "going until you press Ctrl-C.",
    )
    ap.add_argument("--forever", action="store_true", help="same as --levels 0")
    ap.add_argument("--once", action="store_true", help="same as --levels 1")
    ap.add_argument(
        "--dry-run", action="store_true", help="read and solve but send no gestures"
    )
    ap.add_argument(
        "--debug", action="store_true", help="write annotated frames to debug/"
    )
    ap.add_argument("--serial", default=None, help="adb device serial")
    ap.add_argument("--backend", default=None, choices=["auto", "u2", "adb"])
    ap.add_argument(
        "--adb",
        default=None,
        metavar="PATH",
        help="path to adb binary (default: 'adb' on system PATH)",
    )
    ap.add_argument("--config", default="config.json")
    ap.add_argument(
        "--no-learn",
        action="store_true",
        help="do not add confirmed glyphs to the library",
    )
    ap.add_argument(
        "--launch",
        action="store_true",
        help="open the game first and press through its menu",
    )
    ap.add_argument(
        "--restart", action="store_true", help="with --launch, force-stop the app first"
    )
    ap.add_argument(
        "--no-batch",
        action="store_true",
        help="verify each word as it is played instead of dragging "
        "them all and checking once (slower, but isolates a "
        "misbehaving drag)",
    )
    ap.add_argument(
        "--no-lexicon",
        action="store_true",
        help="do not use the bundled word list to identify " "unknown glyphs",
    )
    ap.add_argument(
        "--words",
        nargs="+",
        metavar="WORD",
        help="the level's words, typed in. Resolves glyphs the "
        "library does not know and learns them permanently. "
        "Applies to the first level only.",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.serial:
        cfg.serial = args.serial
    if args.backend:
        cfg.backend = args.backend
    levels = 1 if args.once else (0 if args.forever else max(0, args.levels))

    try:
        adb_kw = {"adb_path": args.adb} if args.adb else {}
        device = Device(serial=cfg.serial, backend=cfg.backend, **adb_kw)
        serial = device.check()
    except DeviceError as e:
        print(f"device error: {e}", file=sys.stderr)
        return 2

    lib = GlyphLibrary(
        cfg.glyph_dir,
        min_score=cfg.min_score,
        margin=cfg.match_margin,
        aspect_tol=cfg.aspect_tol,
        use_structure=cfg.use_structure,
    )
    lexicon = None if args.no_lexicon else Lexicon(cfg.word_list)
    print(f"device {serial} via {device.backend} backend")
    try:
        device.screencap()  # benchmarks raw vs png once
        if getattr(device, "_capture_timings", None):
            bits = ", ".join(
                f"{k} {v * 1000:.0f}ms"
                for k, v in sorted(device._capture_timings.items())
            )
            print(f"screen capture: using {device._capture} ({bits})")
    except DeviceError as e:
        print(f"device error: {e}", file=sys.stderr)
        return 2
    print(
        f"glyph library: {len(lib)} samples covering "
        f"{len(lib.letters)}/26 letters"
        + (f" (missing {''.join(lib.missing)})" if lib.missing else "")
    )
    if lexicon is not None and lexicon.available:
        print(
            f"lexicon: {len(lexicon)} words — unknown glyphs will be "
            f"identified automatically"
        )

    forever = levels == 0
    if forever:
        print("running until you stop it — press Ctrl-C to finish")

    if args.launch or args.restart:
        if not launch(device, package=cfg.package, force_stop=args.restart):
            print("could not reach a board; stopping", file=sys.stderr)
            return 2

    cleared = 0
    n = 0  # levels started
    attempt = 0  # attempts at the current level
    try:
        while forever or n < levels:
            if attempt == 0:
                n += 1
            banner(
                f"Level {n}"
                + ("" if forever else f" of {levels}")
                + (f" (attempt {attempt + 1})" if attempt else "")
            )
            try:
                done, last = solve_level(
                    device,
                    lib,
                    cfg,
                    args.dry_run,
                    args.debug,
                    learn=not args.no_learn,
                    words=[w.upper() for w in args.words]
                    if (args.words and n == 1 and attempt == 0)
                    else None,
                    lexicon=lexicon,
                    no_batch=args.no_batch,
                )
            except GeometryError as e:
                print(f"  could not read the board: {e}")
                break
            except DeviceError as e:
                print(f"  device error: {e}")
                return 2

            if not done and not args.dry_run:
                attempt += 1
                if attempt >= 3:
                    print("  could not clear this level in 3 attempts; stopping")
                    break
                print("  some words are still missing; retrying this level")
                time.sleep(1.0)
                continue

            attempt = 0
            if done:
                cleared += 1

            if args.dry_run or (not forever and n >= levels):
                break

            print("  advancing to the next level...")
            if (
                wait_for_new_board(
                    device, lib, last, poll=cfg.poll, advance_tap=cfg.advance_tap
                )
                is None
            ):
                print("  the next level did not appear; stopping")
                break
    except KeyboardInterrupt:
        print("\n\nstopped by you.")

    attempted = n

    print(f"\ncleared {cleared} of {attempted} level(s) attempted")
    print(
        f"glyph library now {len(lib)} samples, {len(lib.letters)}/26 letters"
        + (f" (missing {''.join(lib.missing)})" if lib.missing else " — complete")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())