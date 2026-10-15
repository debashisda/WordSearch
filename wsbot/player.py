"""Turn a placement into a gesture, perform it, and check it worked.

The drag passes through the centre of every cell the word covers, not just the
two ends. With the uiautomator2 backend each of those becomes a real MOVE
sample, which is what stops the game dropping an intermediate letter on long
words. The pure-ADB fallback can only send two points and lets Android
interpolate; that tracks a straight line well enough, but it is why u2 is
preferred.

Every move is verified by re-reading the board. A word counts as played only
when its pill flips to the found state. Verification is what turns this from a
demo that works once into something that can run a level unattended: a missed
drag is retried, more slowly, rather than silently skipped.

Confirmed words are also fed back into the glyph library. A placed word reveals
the true letter of every cell it passes through, so the alphabet completes
itself as the bot plays instead of needing all 26 labelled up front.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .board import (Board, learn_pill, pill_found, read_board,
                    read_board_stable, unique_pill)
from .glyphs import GlyphLibrary
from .screens import button_centre, find_continue_button, looks_like_board
from .solver import Placement


@dataclass
class MoveResult:
    placement: Placement
    ok: bool
    attempts: int
    board: Board
    learned: int = 0
    frame: object = None        # the screencap that confirmed it, reused later

    def __str__(self) -> str:
        mark = "ok" if self.ok else "FAILED"
        extra = f", learned {self.learned} glyphs" if self.learned else ""
        tries = f" after {self.attempts} tries" if self.attempts > 1 else ""
        return f"{self.placement.word}: {mark}{tries}{extra}"


def path_for(placement: Placement, grid) -> list[tuple[int, int]]:
    """Screen points for the drag: the centre of every cell, in order."""
    return [grid.centre(r, c) for (r, c) in placement.cells]


def overshoot(points: list[tuple[int, int]], px: int = 0) -> list[tuple[int, int]]:
    """Extend the path slightly past the final cell centre.

    Some hit-tests register the last cell only once the finger is well inside
    it. A small overshoot along the same heading costs nothing and removes that
    failure mode. Disabled by default (px=0); the retry ladder turns it on.
    """
    if px <= 0 or len(points) < 2:
        return points
    (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    return points + [(int(x2 + dx / norm * px), int(y2 + dy / norm * px))]


def _is_found(board: Board, placement) -> bool:
    """Has the pill matching this placement flipped to the found state?

    Compared on the symbol list, so a word containing unidentified glyphs is
    tracked just as reliably as a fully-read one.
    """
    target = list(placement.symbols) if hasattr(placement, "symbols") else list(placement)
    for w in board.words:
        if list(w.symbols) == target:
            return w.found
    # Fall back to display text, in case tokens were renumbered between reads.
    text = placement.word if hasattr(placement, "word") else str(placement)
    for w in board.words:
        if w.text == text:
            return w.found
    return False


def wait_for_pill(device, box, budget: float = 1.6, poll: float = 0.10):
    """Poll one word pill until it flips to found. Returns the frame, or None.

    This replaces a fixed sleep followed by a full multi-frame board read. That
    cost three screencaps and 0.77 s of unconditional sleep per word, to learn
    a single bit; here one screencap answers it, and the loop returns the
    moment the game responds instead of always waiting for the worst case.

    The frame that proved the word landed is handed back, so learning reuses it
    rather than taking yet another screencap.
    """
    deadline = time.time() + budget
    while True:
        shot = device.screencap()
        if pill_found(shot, box):
            return shot
        if time.time() >= deadline:
            return None
        time.sleep(poll)


def play_move(device, lib: GlyphLibrary, board: Board, placement: Placement,
              max_attempts: int = 4, settle: float = 1.6,
              learn: bool = True, verbose: bool = True,
              pill_box=None) -> MoveResult:
    """Drag out one word, verify it registered, retry if it did not.

    `settle` is now a *budget*, not a sleep: verification returns as soon as
    the pill changes, so a responsive game costs one screencap per word.
    """
    grid = board.grid
    base = path_for(placement, grid)
    current = board

    if pill_box is None:
        w = unique_pill(board, placement.symbols)
        pill_box = w.box if w is not None else None

    # Retry ladder: slower, further, and — since the game accepts a word
    # dragged from either end — from the other end. A word placed backwards in
    # the grid is the same cells either way, so if the forward drag does not
    # register, reversing it is a free second chance rather than a different
    # move. (This is also why no separate reversed-word search is needed: the
    # eight-direction search already finds backwards placements, which is how
    # CAROL, RYAN and MARK are located on the Names board.)
    ladder = [
        dict(ms_per_point=55, over=0, rev=False),
        dict(ms_per_point=90, over=int(grid.pitch_x * 0.18), rev=False),
        dict(ms_per_point=90, over=int(grid.pitch_x * 0.18), rev=True),
        dict(ms_per_point=140, over=int(grid.pitch_x * 0.30), rev=False),
    ]

    for attempt in range(1, max_attempts + 1):
        cfg = ladder[min(attempt - 1, len(ladder) - 1)]
        route = list(reversed(base)) if cfg["rev"] else base
        points = overshoot(route, cfg["over"])
        device.swipe_path(points, ms_per_point=cfg["ms_per_point"])

        if pill_box is not None:
            shot = wait_for_pill(device, pill_box, budget=settle)
            ok = shot is not None
        else:
            # No pill to watch (a word we could not pair). Fall back to the
            # slower whole-board check.
            time.sleep(min(settle, 0.6))
            current = read_board_stable(device, lib, grid=grid, frames=2)
            ok = _is_found(current, placement)
            shot = device.screencap() if ok else None

        if ok:
            learned = 0
            if learn and shot is not None:
                # A confirmed word reveals the true letter of every cell it
                # crossed, and of every glyph in its own pill. Both render
                # sizes go into the library, so the alphabet completes itself.
                # The frame is the one verification already captured.
                learned = lib.learn_from_word(shot, grid, placement.cells,
                                              placement.symbols)
                if pill_box is not None:
                    learned += learn_pill(shot, pill_box, placement.symbols, lib)
            return MoveResult(placement, True, attempt, current, learned,
                              frame=shot)

        if verbose and attempt < max_attempts:
            nxt = ladder[min(attempt, len(ladder) - 1)]
            how = "from the other end" if nxt["rev"] else "slower"
            print(f"    {placement.word} did not register, retrying {how} "
                  f"({attempt}/{max_attempts})")

    return MoveResult(placement, False, max_attempts, current)


def play_board(device, lib: GlyphLibrary, board: Board, placements,
               learn: bool = True, settle: float = 1.6, gap: float = 0.12,
               batch: bool = True, verbose: bool = True):
    """Play every placement, then read the board once at the end.

    Earlier this re-read the whole board after every word, which meant
    re-detecting geometry and re-recognising every cell just to see whether one
    pill had changed. The grid is constant for the life of a level, so the
    per-word check watches only that word's pill.

    Returns (results, board, complete).
    """
    results: list[MoveResult] = []
    boxes = {}
    for p in placements:
        w = unique_pill(board, p.symbols)
        boxes[p] = w.box if w is not None else None

    if batch and all(boxes.get(p) is not None for p in placements) and len(placements) > 1:
        return play_board_batched(device, lib, board, placements, boxes,
                                  learn=learn, settle=settle,
                                  gap=gap, verbose=verbose)

    last_frame = None          # the frame that confirmed the previous word
    for p in placements:
        box = boxes.get(p)
        # A drag can complete a second word that shares its cells. The frame
        # that confirmed the previous word already shows that, so this costs
        # nothing — no extra screencap.
        if (box is not None and last_frame is not None
                and pill_found(last_frame, box)):
            if verbose:
                print(f"    {p.word}: already found")
            continue

        t0 = time.time()
        res = play_move(device, lib, board, p, learn=learn, settle=settle,
                        verbose=verbose, pill_box=box)
        results.append(res)
        if res.frame is not None:
            last_frame = res.frame
        if verbose:
            print(f"    {res}  [{time.time() - t0:.1f}s]")

    done = confirm_level_done(device, board, verbose=verbose)
    return results, board, done


def play_board_batched(device, lib: GlyphLibrary, board: Board, placements,
                       boxes, learn: bool = True, settle: float = 1.6,
                       gap: float = 0.12, verbose: bool = True):
    """Drag every word first, then verify them all with a single capture.

    Verification is the expensive half of a move: the gesture costs one adb
    round trip, and so does the screencap that checks it. Checking after each
    word therefore doubles the round trips for information that is nearly
    always "yes, it worked".

    So all the drags go out back to back, then one capture settles the whole
    board. On a nine-word level that is 9 gestures + 1 capture instead of
    9 + 9. Anything that did not land is retried individually, with per-word
    verification, so a missed drag costs no more than it did before.

    `gap` is a short pause between drags so a finger-down is not delivered
    while the previous word is still animating.
    """
    results: list[MoveResult] = []

    if verbose:
        print(f"    dragging {len(placements)} words, then verifying once")

    paths = [overshoot(path_for(p, board.grid), 0) for p in placements]
    sent = False
    if hasattr(device, "swipe_many"):
        sent = device.swipe_many(paths, gap_ms=int(gap * 1000))
    if not sent:
        for i, pts in enumerate(paths):
            device.swipe_path(pts, ms_per_point=55)
            if gap and i < len(paths) - 1:
                time.sleep(gap)

    shot = device.screencap()

    # If the board is already gone, every word landed and the level is over.
    if not looks_like_board(shot):
        if verbose:
            print("    all words landed (completion screen)")
        for p in placements:
            results.append(MoveResult(p, True, 1, board, 0, frame=None))
        return results, board, True

    landed, missed = [], []
    for p in placements:
        (landed if pill_found(shot, boxes[p]) else missed).append(p)

    learned = 0
    if learn and landed:
        # One frame labels every word that landed, at both render sizes.
        for p in landed:
            learned += lib.learn_from_word(shot, board.grid, p.cells, p.symbols)
            learned += learn_pill(shot, boxes[p], p.symbols, lib)
    for p in landed:
        results.append(MoveResult(p, True, 1, board, 0, frame=shot))

    if verbose:
        note = f", learned {learned} glyphs" if learned else ""
        print(f"    {len(landed)}/{len(placements)} landed on the first pass{note}")

    for p in missed:
        if verbose:
            print(f"    retrying {p.word}")
        t0 = time.time()
        res = play_move(device, lib, board, p, learn=learn, settle=settle,
                        verbose=verbose, pill_box=boxes[p])
        results.append(res)
        if verbose:
            print(f"    {res}  [{time.time() - t0:.1f}s]")

    done = confirm_level_done(device, board, verbose=verbose)
    return results, board, done


def confirm_level_done(device, board: Board, budget: float = 4.0,
                       poll: float = 0.25, verbose: bool = True) -> bool:
    """Did the level actually clear?

    Two signals, either of which settles it:

      * the board is gone — the COMPLETED screen has replaced it. This is the
        definitive one, and it is why the old approach failed: it re-read the
        board after the last word, by which time the completion screen was
        already up, so it saw zero word pills. `Board.complete` requires at
        least one pill, so nine words all landing still reported "not cleared"
        and the level was replayed from the start.

      * every pill is struck through. Checked by brightness at the boxes we
        already know, so it costs no OCR and cannot be confused by the
        strike-through animation.

    The level screen does not change the instant the last word lands, so this
    polls briefly rather than judging on a single frame.
    """
    boxes = [w.box for w in board.words]
    deadline = time.time() + budget

    while True:
        shot = device.screencap()

        if not looks_like_board(shot):
            if verbose:
                print("    level complete (board gone)")
            # This frame already shows the between-level screen, so its button
            # can be pressed now rather than after another capture. A capture
            # is the most expensive thing here, and tapping twice costs
            # nothing if the screen has not changed by the next poll.
            box = find_continue_button(shot)
            if box is not None:
                device.tap(*button_centre(box))
            return True

        if boxes and all(pill_found(shot, b) for b in boxes):
            if verbose:
                print("    level complete (every word struck through)")
            return True

        if time.time() >= deadline:
            missing = sum(1 for b in boxes if not pill_found(shot, b))
            if verbose and missing:
                print(f"    {missing} word(s) still not struck through")
            return False
        time.sleep(poll)


def wait_for_new_board(device, lib: GlyphLibrary, previous: Board,
                       timeout: float = 40.0, poll: float = 0.25,
                       advance_tap: list[int] | None = None,
                       verbose: bool = True) -> Board | None:
    """Clear the between-level screens and wait for the next level's board.

    Each poll decides what it is looking at rather than tapping blind:

      a board with unplayed words -> done
      a between-level screen      -> tap its button, whatever the colour
      anything else               -> wait (an ad or a transition is on screen)

    Two things keep this quick. The board is read from the frame already in
    hand rather than taking fresh captures — the caller re-reads properly with
    consensus anyway, so those captures were thrown away. And the new level is
    recognised by having *unplayed* words rather than by comparing grid text:
    the level just finished has every pill struck through, so one unfound pill
    is proof the next one has loaded, without needing to recognise a letter.

    `advance_tap` from config.json overrides detection if a build needs it.
    """
    deadline = time.time() + timeout
    taps = 0
    last_tap: tuple[int, int] | None = None

    while True:
        img = device.screencap()

        if looks_like_board(img):
            try:
                board = read_board(img, lib)
                if board.words and any(not w.found for w in board.words):
                    return board
            except Exception:
                pass                     # mid-transition; look again next poll
        else:
            box = find_continue_button(img)
            if box is not None:
                x, y = button_centre(box)
                # More than one screen can stand between levels — a photo card
                # with COLLECT, then COMPLETED — so each distinct button is
                # reported as it is reached.
                if verbose and (x, y) != last_tap:
                    print(f"    between-level screen; tapping ({x},{y})")
                last_tap = (x, y)
                device.tap(x, y)
                taps += 1
            elif advance_tap:
                device.tap(int(advance_tap[0]), int(advance_tap[1]))
                taps += 1
            elif verbose and taps == 0:
                print("    waiting for the next level (ad or transition on screen)")

        if time.time() >= deadline:
            return None
        time.sleep(poll)
