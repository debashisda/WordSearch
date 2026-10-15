"""Read a whole board: the letter grid, the target words, and which are done.

Two regions matter.

  The grid panel, handled by geometry.py + glyphs.py.

  The word list above it: dark rounded pills, one per target word. White text
  means still to find; grey text means already found. Measured on the sample
  device the brightest 3% of pixels in a pill sits at ~252 when unfound and
  ~129 when found, which is a wide enough gap to threshold at 190 without
  tuning.

Word-list text is read with the *same* glyph library as the grid — it is the
same typeface, and normalise() rescales every glyph to a fixed box, so the
smaller point size costs nothing.

Symbols, not characters
-----------------------
A cell or a word position holds a *symbol*, which is one of:

    'A'..'Z'   a letter the library recognised
    '#n'       an unidentified glyph, consistent across the whole board
    None       no readable ink at all

The '#n' tokens are what let the bot play a level containing letters it has
never seen. Both regions are read through one shared UnknownPool, so the glyph
at word position 0 of JACOB and the glyph in the grid cell it corresponds to
receive the same token. The solver then matches `#1 A C O #2` against the grid
exactly as it would match known letters — the drag is correct whether or not
anyone ever establishes that #1 is a J.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Grid, detect_grid
from .glyphs import (GlyphLibrary, UnknownPool, cell_glyph, is_token,
                     normalise, symbols_to_text)

FOUND_BRIGHTNESS = 190      # p97 inside a pill: above = unfound, below = found


@dataclass
class WordEntry:
    symbols: list[str | None]
    found: bool
    box: tuple[int, int, int, int]       # x, y, w, h on screen
    confidence: float = 1.0

    @property
    def text(self) -> str:
        return symbols_to_text(self.symbols)

    @property
    def resolved(self) -> bool:
        """True when every position is a known letter."""
        return all(isinstance(s, str) and not is_token(s) for s in self.symbols)

    def __str__(self) -> str:
        return f"{self.text}{' (found)' if self.found else ''}"


@dataclass
class Board:
    grid: Grid
    letters: list[list[str | None]]
    words: list[WordEntry] = field(default_factory=list)
    unknown_cells: list[tuple[int, int]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return self.grid.rows

    @property
    def cols(self) -> int:
        return self.grid.cols

    @property
    def complete(self) -> bool:
        return bool(self.words) and all(w.found for w in self.words)

    def pending(self) -> list[WordEntry]:
        return [w for w in self.words if not w.found]

    def grid_text(self) -> str:
        def show(ch):
            if ch is None:
                return "?"
            return "?" + ch[1:] if is_token(ch) else ch
        return "\n".join(" ".join(show(ch) for ch in row) for row in self.letters)

    def token_cells(self) -> dict[str, list[tuple[int, int]]]:
        """Where each '#n' token appears in the grid."""
        out: dict[str, list[tuple[int, int]]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = self.letters[r][c]
                if is_token(s):
                    out.setdefault(s, []).append((r, c))
        return out

    def __str__(self) -> str:
        done = sum(w.found for w in self.words)
        bits = [f"Board {self.grid.rows}x{self.grid.cols}",
                f"{done}/{len(self.words)} words found"]
        if self.tokens:
            bits.append(f"{len(self.tokens)} unidentified glyph(s): "
                        f"{', '.join('?' + t[1:] for t in self.tokens)}")
        if self.unknown_cells:
            bits.append(f"{len(self.unknown_cells)} blank cells")
        return ", ".join(bits)


# ------------------------------------------------------------------- grid

def read_grid(img: np.ndarray, grid: Grid, lib: GlyphLibrary,
              pool: UnknownPool | None = None):
    """Identify every cell. Unknown glyphs become '#n' tokens via `pool`."""
    letters: list[list[str | None]] = []
    unknown: list[tuple[int, int]] = []
    for r in range(grid.rows):
        row: list[str | None] = []
        for c in range(grid.cols):
            g = cell_glyph(img, grid, r, c)
            letter, _score = lib.match(g)
            if letter is None and pool is not None:
                letter = pool.token(g)
            if letter is None:
                unknown.append((r, c))
            row.append(letter)
        letters.append(row)
    return letters, unknown


# -------------------------------------------------------------- word list

def _surround_brightness(img: np.ndarray, x: int, y: int, w: int, h: int,
                         pad: int = 12) -> float | None:
    """Median brightness of a thin ring just outside a box."""
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    outer = img[y0:y1, x0:x1]
    if outer.size == 0:
        return None
    g = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)
    mask = np.ones(g.shape, bool)
    iy0, ix0 = y - y0, x - x0
    mask[max(0, iy0):iy0 + h, max(0, ix0):ix0 + w] = False
    vals = g[mask]
    return float(np.median(vals)) if vals.size else None


def find_word_pills(img: np.ndarray, panel_y: int,
                    top_frac: float = 0.05) -> list[tuple[int, int, int, int]]:
    """Locate the dark word pills between the header and the grid panel."""
    H, W = img.shape[:2]
    top = int(top_frac * H)
    bot = max(top + 10, panel_y - 4)
    roi = img[top:bot]
    if roi.size == 0:
        return []

    V = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = (V < 95).astype(np.uint8)
    # A small, square close only tidies anti-aliased edges.
    #
    # This used to be a 41px-wide horizontal close, on the theory that gaps
    # between letters needed bridging. They do not: the pill's dark background
    # is continuous and the white letters are holes inside it, so the blob is
    # already one piece. What the wide kernel did do was bridge the gaps
    # *between* pills. On a level with three long names side by side the gaps
    # were 34px, so TOM PETTY, MADONNA and LIL WAYNE merged into a single
    # 1002px blob reading "TOMPETTYMADONNALILWAYNE", which of course appears
    # nowhere in the grid, and the level could never complete.
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, _lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w < 70 or not (0.014 * H < h < 0.050 * H):
            continue
        if w / h < 1.1 or a < 0.5 * w * h:
            continue
        cands.append((int(y) + top, int(x), int(w), int(h)))

    if not cands:
        return []

    med = float(np.median([c[3] for c in cands]))
    tol = max(6.0, 0.18 * med)
    pills = [c for c in cands if abs(c[3] - med) <= tol]

    # A real word pill is dark on the photo background; the header's theme text
    # is dark on a *light* pill, so its surroundings are bright. Drop those.
    kept = []
    for (y, x, w, h) in pills:
        ring = _surround_brightness(img, x, y, w, h)
        if ring is not None and ring > 185:
            continue
        kept.append((y, x, w, h))

    # A pill spanning most of the screen means two or more ran together. That
    # is unrecoverable downstream — the merged text matches nothing in the grid
    # and the level can never finish — so say so rather than fail silently.
    for (y, x, w, h) in kept:
        if w > 0.80 * W:
            print(f"  warning: a word pill is {w}px wide, nearly the full "
                  f"screen — two or more have probably merged. The word list "
                  f"will be wrong for this level.")
            break

    return [(x, y, w, h) for (y, x, w, h) in sorted(kept)]


def read_pill(img: np.ndarray, box: tuple[int, int, int, int],
              lib: GlyphLibrary, pool: UnknownPool | None = None):
    """Read one word pill. Returns (symbols, found, mean_confidence)."""
    x, y, w, h = box
    pill = img[y:y + h, x:x + w]
    if pill.size == 0:
        return [], False, 0.0

    g = cv2.cvtColor(pill, cv2.COLOR_BGR2GRAY)
    found = float(np.percentile(g, 97)) < FOUND_BRIGHTNESS

    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.5:
        th = 255 - th

    n, _lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < max(40, 0.004 * w * h):
            continue
        if ch < 0.30 * h or ch > 0.95 * h:
            continue
        comps.append((int(cx), int(cy), int(cw), int(ch)))
    comps.sort()

    symbols: list[str | None] = []
    scores = []
    for (cx, cy, cw, ch) in comps:
        pad = 3
        sub = pill[max(0, cy - pad):cy + ch + pad, max(0, cx - pad):cx + cw + pad]
        glyph = normalise(sub, inset=0.0)
        letter, score = lib.match(glyph)
        if letter is None and pool is not None:
            letter = pool.token(glyph)
        symbols.append(letter)
        scores.append(score)

    conf = float(np.mean(scores)) if scores else 0.0
    return symbols, found, conf


def pill_glyphs(img: np.ndarray, box: tuple[int, int, int, int]
                ) -> list[np.ndarray]:
    """Normalised glyphs for one pill, left to right. Used for learning.

    Word pills render smaller than grid cells, so they are effectively a second
    typeface size. The library needs samples of both: seeded from the grid
    alone, a pill's N scored 0.879 and fell below the matching threshold.
    """
    x, y, w, h = box
    pill = img[y:y + h, x:x + w]
    if pill.size == 0:
        return []
    g = cv2.cvtColor(pill, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.5:
        th = 255 - th
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < max(40, 0.004 * w * h):
            continue
        if ch < 0.30 * h or ch > 0.95 * h:
            continue
        comps.append((int(cx), int(cy), int(cw), int(ch)))
    comps.sort()
    out = []
    for (cx, cy, cw, ch) in comps:
        pad = 3
        sub = pill[max(0, cy - pad):cy + ch + pad, max(0, cx - pad):cx + cw + pad]
        out.append(normalise(sub, inset=0.0))
    return out


def pill_found(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Has the word in this pill been struck through?

    The whole verification after a drag reduces to this one bit, and it costs
    0.075 ms against ~75 ms for a full board read — before counting the two
    extra screencaps a multi-frame read would need. The grid itself cannot
    change mid-level, so re-reading 48 cells to learn whether one pill flipped
    is pure overhead.

    Unfound text peaks at ~252 and found text at ~129, so the threshold has
    enormous margin and needs no per-device tuning.
    """
    x, y, w, h = box
    pill = img[max(0, y):y + h, max(0, x):x + w]
    if pill.size == 0:
        return False
    g = cv2.cvtColor(pill, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(g, 97)) < FOUND_BRIGHTNESS


def unique_pill(board, symbols):
    """The one word pill that can be read as `symbols`, or None if ambiguous.

    Uniqueness is the point. A pill that is mostly tokens is compatible with
    almost any word of the same length — once JACOB's pill read `?4A?2O?7`,
    CAROL matched it too, and "learning" from that wrote J as C, C as R and B
    as L straight into the library. Corrupting the reference set is far worse
    than learning nothing, so an ambiguous pairing is refused outright.
    """
    target = list(symbols)
    hits = []
    for w in board.words:
        if len(w.symbols) != len(target):
            continue
        if all(a == b or is_token(a) or a is None
               for a, b in zip(w.symbols, target)):
            hits.append(w)
    return hits[0] if len(hits) == 1 else None


def learn_pill(img: np.ndarray, box, symbols, lib: GlyphLibrary) -> int:
    """Store a pill's glyphs under known letters. Tokens are skipped."""
    glyphs = pill_glyphs(img, box)
    if len(glyphs) != len(symbols):
        return 0
    added = 0
    for g, s in zip(glyphs, symbols):
        if isinstance(s, str) and len(s) == 1 and s.isalpha():
            if g is not None and lib.add(s.upper(), g):
                added += 1
    return added


def read_words(img: np.ndarray, panel_y: int, lib: GlyphLibrary,
               pool: UnknownPool | None = None) -> list[WordEntry]:
    out = []
    for box in find_word_pills(img, panel_y):
        symbols, found, conf = read_pill(img, box, lib, pool)
        if symbols:
            out.append(WordEntry(symbols=symbols, found=found, box=box,
                                 confidence=conf))
    return out


# -------------------------------------------------------------- whole board

def read_board(img: np.ndarray, lib: GlyphLibrary,
               grid: Grid | None = None,
               pool: UnknownPool | None = None) -> Board:
    """Read grid, word list and found-state from one frame.

    Grid and word list share one UnknownPool, so an unidentified glyph gets the
    same token in both and the solver can still match them.
    """
    g = grid if grid is not None else detect_grid(img)
    pool = UnknownPool() if pool is None else pool
    letters, unknown = read_grid(img, g, lib, pool)
    words = read_words(img, g.panel[1], lib, pool)
    return Board(grid=g, letters=letters, words=words, unknown_cells=unknown,
                 tokens=pool.tokens)


def read_board_stable(device, lib: GlyphLibrary, grid: Grid | None = None,
                      frames: int = 3, delay: float = 0.10) -> Board:
    """Read the board from several frames and merge by majority vote.

    A finger still on the glass, or a selection pill mid-animation, can hide a
    letter for one frame. The board itself is static between moves, so voting
    clears those.

    One UnknownPool is shared across every frame, so a given unidentified glyph
    keeps the same token throughout — the board is static, so the same glyph
    clusters to the same prototype each time. That keeps grid tokens and word
    tokens comparable no matter which frame each came from.
    """
    import time
    from collections import Counter

    pool = UnknownPool()
    boards: list[Board] = []
    # Fitting the lattice costs ~90 ms and the grid cannot move between frames
    # of the same level, so it is found on the first frame and reused.
    fitted = grid
    for i in range(max(1, frames)):
        if i:
            time.sleep(delay)
        try:
            b = read_board(device.screencap(), lib, fitted, pool=pool)
        except Exception:
            continue
        if fitted is None:
            fitted = b.grid
        boards.append(b)
    if not boards:
        raise RuntimeError("could not read the board from any frame")

    def known_count(b: Board) -> int:
        return sum(1 for row in b.letters for ch in row
                   if ch is not None and not is_token(ch))

    base = max(boards, key=known_count)

    letters: list[list[str | None]] = []
    unknown: list[tuple[int, int]] = []
    for r in range(base.rows):
        row: list[str | None] = []
        for c in range(base.cols):
            seen = [b.letters[r][c] for b in boards
                    if r < b.rows and c < b.cols and b.letters[r][c] is not None]
            known = Counter(s for s in seen if not is_token(s))
            if known:
                row.append(known.most_common(1)[0][0])
            elif seen:
                row.append(Counter(seen).most_common(1)[0][0])   # agreed token
            else:
                row.append(None)
                unknown.append((r, c))
        letters.append(row)

    by_pos: dict[int, list[WordEntry]] = {}
    for b in boards:
        for w in b.words:
            by_pos.setdefault(round(w.box[1] / 10) * 1000 + round(w.box[0] / 10),
                              []).append(w)

    words: list[WordEntry] = []
    for _key, group in sorted(by_pos.items()):
        if len(group) * 2 < len(boards):
            continue                     # a transient artefact, not a real pill
        best = max(group, key=lambda w: (sum(1 for s in w.symbols
                                             if s and not is_token(s)),
                                         w.confidence))
        found = sum(w.found for w in group) * 2 >= len(group)
        words.append(WordEntry(symbols=best.symbols, found=found,
                               box=best.box, confidence=best.confidence))

    return Board(grid=base.grid, letters=letters, words=words,
                 unknown_cells=unknown, tokens=base.tokens)
