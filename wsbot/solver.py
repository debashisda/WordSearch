"""Find target words in the grid.

All eight directions. The level-7 recording only showed horizontal, vertical
and one diagonal (PEACE, going SE), but diagonals are standard in this genre,
so every direction is searched.

Words and grid cells are both sequences of *symbols* (see board.py): a letter
'A'..'Z', a token '#n' for a glyph the library could not name, or None for an
empty cell. Matching rules:

    letter vs same letter    match
    '#n'   vs same '#n'      match — the same unidentified glyph
    None   (grid)            wildcard, counted and reported
    letter vs different      no match
    '#n'   vs different '#m' no match — demonstrably different glyphs

The token rule is what lets a level be solved with an incomplete alphabet.
A word reading `#1 A C O #2` has exactly one place it can sit in the grid, and
that is enough to drag it correctly, whether or not anyone ever works out that
#1 is a J and #2 is a B.
"""

from __future__ import annotations

from dataclasses import dataclass

from .glyphs import is_token, symbols_to_text

# (dr, dc) for E, SE, S, SW, W, NW, N, NE
DIRECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

DIRECTION_NAMES = {
    (0, 1): "E", (1, 1): "SE", (1, 0): "S", (1, -1): "SW",
    (0, -1): "W", (-1, -1): "NW", (-1, 0): "N", (-1, 1): "NE",
}


@dataclass(frozen=True)
class Placement:
    symbols: tuple[str | None, ...]
    start: tuple[int, int]
    direction: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    wildcards: int = 0

    @property
    def word(self) -> str:
        return symbols_to_text(self.symbols)

    @property
    def end(self) -> tuple[int, int]:
        return self.cells[-1]

    @property
    def ambiguous(self) -> bool:
        return self.wildcards > 0

    @property
    def has_tokens(self) -> bool:
        return any(is_token(s) for s in self.symbols)

    @property
    def direction_name(self) -> str:
        return DIRECTION_NAMES.get(self.direction, "?")

    def __str__(self) -> str:
        r, c = self.start
        bits = []
        if self.wildcards:
            bits.append(f"{self.wildcards} wildcard")
        if self.has_tokens:
            bits.append("via unidentified glyph")
        tail = f" ({', '.join(bits)})" if bits else ""
        return (f"{self.word} @ r{r}c{c} {self.direction_name} "
                f"-> r{self.end[0]}c{self.end[1]}{tail}")


def _compatible(grid_sym, word_sym, binding: dict) -> tuple[bool, bool]:
    """(matches, used_wildcard) for one position, honouring token bindings.

    A token is "a glyph I could not name", not "a glyph that differs from every
    letter". The same glyph can read as a letter in the grid and as a token in
    the word list, because the two are rendered at different point sizes, so a
    token is allowed to stand for a letter. To stop that being a free pass,
    every token must bind to *one* symbol consistently across the word: in
    `#1 N #1 R G Y` both tokens have to resolve to the same letter, which is
    what keeps ENERGY from matching a grid run that merely looks similar.
    """
    if grid_sym is None:
        return True, True            # empty cell: could be anything
    if word_sym is None:
        return True, True            # unreadable word position: accept

    g_tok, w_tok = is_token(grid_sym), is_token(word_sym)

    if not g_tok and not w_tok:
        return grid_sym == word_sym, False

    if g_tok and w_tok:
        if grid_sym == word_sym:
            return True, False       # same unknown glyph: a strong match
        # Different tokens may still be the same letter. Clustering splits on
        # sub-pixel rendering: on a real level-8 board the six J glyphs formed
        # two groups whose bounding boxes differed by one pixel, correlating
        # 0.874-0.888 across the groups and 0.99+ within them — straddling any
        # threshold. Since genuinely different letters also correlate up to
        # ~0.88, no cluster threshold separates the two cases, so splitting is
        # allowed and tolerated here instead: a cross-token match is permitted
        # but costs a wildcard, so exact token agreement always wins.
        return True, True

    # One side is a token, the other a named letter: allow, but bind the token
    # so it means the same letter everywhere in this word.
    token, letter = (word_sym, grid_sym) if w_tok else (grid_sym, word_sym)
    prev = binding.get(token)
    if prev is not None and prev != letter:
        return False, False
    binding[token] = letter
    return True, True


def _binding_consistent(symbols, cells, letters) -> bool:
    """A token must stand for one letter everywhere it appears in a word.

    The constraint is against *letters* only. An earlier version demanded that
    a token always line up with the same opposing *symbol*, which quietly broke
    every word containing a doubled letter. In MUFFINS the two Fs read as `F`
    and `#1` — the same letter, recognised once and refused once — and the
    grid's two F cells were both `#1`. Lining `#1` up with `F` at one position
    and with `#1` at the next looked like a contradiction, so the word could
    not be placed anywhere and the level never finished. EIFFEL failed the same
    way.

    Pairing a token with another token says nothing about which letter either
    one is, so it imposes no constraint; only a token paired with a named
    letter does.
    """
    word_means: dict[str, str] = {}      # word token -> the grid letter it is
    grid_means: dict[str, str] = {}      # grid token -> the word letter it is
    for sym, (r, c) in zip(symbols, cells):
        got = letters[r][c]
        if sym is None or got is None:
            continue
        s_tok, g_tok = is_token(sym), is_token(got)
        if s_tok and g_tok:
            continue                     # two unknowns: no letter is implied
        if s_tok:
            if word_means.setdefault(sym, got) != got:
                return False
        elif g_tok:
            if grid_means.setdefault(got, sym) != sym:
                return False
    return True


class Solver:
    def __init__(self, letters: list[list[str | None]]) -> None:
        self.letters = letters
        self.rows = len(letters)
        self.cols = len(letters[0]) if letters else 0

    def _match_at(self, symbols, r: int, c: int,
                  dr: int, dc: int) -> Placement | None:
        cells, wildcards = [], 0
        binding: dict[str, str] = {}
        for i, sym in enumerate(symbols):
            rr, cc = r + dr * i, c + dc * i
            if not (0 <= rr < self.rows and 0 <= cc < self.cols):
                return None
            ok, wild = _compatible(self.letters[rr][cc], sym, binding)
            if not ok:
                return None
            wildcards += wild
            cells.append((rr, cc))
        if not _binding_consistent(symbols, cells, self.letters):
            return None
        return Placement(symbols=tuple(symbols), start=(r, c),
                         direction=(dr, dc), cells=tuple(cells),
                         wildcards=wildcards)

    def find_all(self, symbols) -> list[Placement]:
        """Every placement, cleanest (fewest wildcards) first."""
        symbols = _as_symbols(symbols)
        if not symbols or self.rows == 0:
            return []
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                head = self.letters[r][c]
                if (head is not None and symbols[0] is not None
                        and not is_token(head) and not is_token(symbols[0])
                        and head != symbols[0]):
                    continue
                for dr, dc in DIRECTIONS:
                    p = self._match_at(symbols, r, c, dr, dc)
                    if p is not None:
                        out.append(p)
        out.sort(key=lambda p: p.wildcards)
        return out

    def find(self, symbols) -> Placement | None:
        hits = self.find_all(symbols)
        return hits[0] if hits else None

    def solve(self, words) -> tuple[list[Placement], list[str]]:
        """Place as many words as possible. Returns (placements, unplaced).

        Words with exactly one placement are committed first; that resolves
        wildcard cells, which can in turn disambiguate the rest.
        """
        pending = [_as_symbols(w) for w in words]
        placements: list[Placement] = []
        letters = [row[:] for row in self.letters]

        progress = True
        while pending and progress:
            progress = False
            solver = Solver(letters)
            scored = [(syms, solver.find_all(syms)) for syms in pending]
            scored.sort(key=lambda t: (len(t[1]) != 1,
                                       t[1][0].wildcards if t[1] else 99))
            for syms, hits in scored:
                if not hits:
                    continue
                best = hits[0]
                if len(hits) > 1 and best.wildcards and hits[1].wildcards == 0:
                    continue
                for (r, c), sym in zip(best.cells, syms):
                    if sym is not None:
                        letters[r][c] = sym     # a placed word reveals its cells
                placements.append(best)
                pending.remove(syms)
                progress = True
                break

        return placements, [symbols_to_text(s) for s in pending]


def _as_symbols(word) -> list[str | None]:
    """Accept a plain string or an already-split symbol list."""
    if isinstance(word, str):
        return [ch.upper() for ch in word]
    return list(word)


def repair_unplaced(symbols, letters, lexicon=None, max_subs: int = 2):
    """Find a reading of a word that does place, when the given one does not.

    A word that cannot be found anywhere in the grid is almost always a
    misread pill rather than a missing word — the game does not list words it
    has not hidden. VINEGAR once read as VGNEGAR, which appears nowhere, so the
    level could never be finished.

    The grid is a far stronger constraint than the pill image, so it is used to
    correct the reading: blank out one position, and if exactly one placement
    survives, the grid itself says what that letter must be. Of the 175
    single-letter variants of VGNEGAR, exactly one places — VINEGAR.

    Two positions are tried only if one fails, and then the result must be
    unambiguous or a dictionary word, since blanking more of a short word makes
    coincidences easy (KGWG admits both KIWI and RGWS).

    Returns (repaired_symbols, placement) or None.
    """
    from itertools import combinations

    solver = Solver(letters)
    n = len(symbols)
    original = [s for s in symbols]

    for k in range(1, max(1, min(max_subs, n - 1)) + 1):
        found: dict[str, tuple[list, Placement]] = {}
        for combo in combinations(range(n), k):
            trial = list(symbols)
            for i in combo:
                trial[i] = None
            hits = solver.find_all(trial)
            if len(hits) != 1:
                continue
            p = hits[0]
            fixed = list(symbols)
            for i in combo:
                r, c = p.cells[i]
                fixed[i] = letters[r][c]
            if any(f is None or is_token(f) for f in fixed):
                continue
            if fixed == original:
                continue
            # Hand back a placement carrying the corrected letters, not the
            # wildcards used to find it, so the caller can verify and learn
            # from it like any other word.
            corrected = Placement(symbols=tuple(fixed), start=p.start,
                                  direction=p.direction, cells=p.cells,
                                  wildcards=0)
            found["".join(fixed)] = (fixed, corrected)

        if not found:
            continue
        if len(found) == 1:
            return next(iter(found.values()))
        if lexicon is not None and getattr(lexicon, "words", None):
            in_lex = {w: v for w, v in found.items() if w in lexicon.words}
            if len(in_lex) == 1:
                return next(iter(in_lex.values()))
        return None            # several equally good readings; do not guess
    return None


def solve_board(board) -> tuple[list[Placement], list[str]]:
    """Solve every word the board still needs.

    Unlike a plain-text approach, a word containing unidentified glyphs is
    still solvable here, because its tokens match the grid's tokens.
    """
    targets = [w.symbols for w in board.words if not w.found]
    return Solver(board.letters).solve(targets)
