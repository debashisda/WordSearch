"""Resolve unidentified glyphs to real letters, automatically.

When the library has not seen a letter, board.py gives it a token (`?1`) and
the solver plays through it. That is enough to finish the level, but the board
still reads `?1HOMAS`, and nothing is learned.

This module closes the loop with no help from the user. Two constraints turn
an ambiguous pattern into a unique answer:

  1. **Dictionary.** `?1HOMAS` matches exactly one word of that shape.

  2. **A token cannot be a letter the library already knows.** Tokens exist
     precisely because matching *refused* them, and matching only refuses a
     glyph that looks like nothing in the library. So a token's letter must
     come from the library's missing set.

Constraint 2 is what makes this work in practice. `?2ACO?5` has five
dictionary matches — BACON, JACOB, MACON, RACON, VACOA — but with the library
holding `ACDEGHILMNOPRSVWY`, only JACOB assigns both tokens to letters it does
not have. Likewise `MAR?3` narrows from twelve candidates to MARK alone.

Tokens are then tied together across the whole board: the same token must mean
the same letter in every word it appears in, so a word that resolves uniquely
also constrains the others. The result feeds straight into the glyph library,
so the letters are learned permanently and the next level reads them normally.

Two deliberate non-assumptions:

  * Distinct tokens are *not* required to be distinct letters. Clustering
    splits on sub-pixel rendering, so `?2` and `?4` were both J on a real
    board. Requiring distinctness would have rejected the correct answer.
  * A resolution is accepted only if the resolved word still places in the
    grid. The dictionary proposes; the board disposes.
"""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

from .glyphs import ALPHABET, is_token

DEFAULT_WORDS = Path(__file__).with_name("words.txt.gz")


class Lexicon:
    """A word list indexed by length, for fast pattern lookup."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.by_len: dict[int, list[str]] = defaultdict(list)
        self.words: set[str] = set()
        p = Path(path) if path else DEFAULT_WORDS
        if not p.exists():
            return
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                w = line.strip().upper()
                if w.isalpha():
                    self.words.add(w)
                    self.by_len[len(w)].append(w)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def available(self) -> bool:
        return bool(self.words)

    def candidates(self, symbols, allowed: set[str]) -> list[str]:
        """Words matching this symbol pattern, with tokens limited to `allowed`.

        `allowed` is the set of letters the glyph library does *not* hold.
        """
        if not self.words:
            return []
        syms = list(symbols)
        if any(s is None for s in syms):
            return []                      # a blank position tells us nothing
        if not any(is_token(s) for s in syms):
            return ["".join(syms)]

        cls = "[" + "".join(sorted(allowed)) + "]" if allowed else "."
        pattern = "".join(cls if is_token(s) else re.escape(s) for s in syms)
        rx = re.compile(f"^{pattern}$")
        return [w for w in self.by_len.get(len(syms), ()) if rx.match(w)]


def _assign(symbols, word) -> dict[str, str] | None:
    """Token -> letter mapping implied by reading `symbols` as `word`."""
    out: dict[str, str] = {}
    for s, ch in zip(symbols, word):
        if is_token(s):
            if out.setdefault(s, ch) != ch:
                return None
        elif s != ch:
            return None
    return out


def resolve(board, lib, lexicon: Lexicon,
            verify_placement: bool = True) -> dict[str, str]:
    """Work out what each token means. Returns token -> letter.

    Only assignments that are unique *and* consistent across the whole board
    are returned; anything still ambiguous is left as a token.
    """
    if not lexicon.available:
        return {}

    allowed = set(ALPHABET) - set(lib.letters)
    if not allowed:
        return {}

    targets = [w.symbols for w in board.words if any(is_token(s) for s in w.symbols)]
    if not targets:
        return {}

    options: list[list[dict[str, str]]] = []
    for syms in targets:
        maps = []
        for cand in lexicon.candidates(syms, allowed):
            m = _assign(syms, cand)
            if m is not None:
                maps.append(m)
        if maps:
            options.append(maps)

    if not options:
        return {}

    # Fixed point: commit every word that has exactly one surviving reading,
    # then re-filter the rest with what that told us.
    bound: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for i, maps in enumerate(options):
            live = [m for m in maps
                    if all(bound.get(t, v) == v for t, v in m.items())]
            options[i] = live
            if len(live) == 1:
                for t, v in live[0].items():
                    if bound.get(t) != v:
                        bound[t] = v
                        changed = True

    # Second pass, relaxed. "A token is never a letter the library holds" is a
    # strong heuristic, not a law: a borderline render can be refused even for
    # a letter that *is* present (a word-pill N once scored 0.879 against a
    # library containing N). So any word still unresolved is retried against
    # the whole alphabet, with candidates filtered by whether they actually
    # place in the grid. The grid is a far tighter filter than the dictionary —
    # `WI?` has many dictionary matches but only WIN exists on the board.
    leftovers = [w.symbols for w in board.words
                 if any(is_token(s) and s not in bound for s in w.symbols)]
    for syms in leftovers:
        live = []
        for cand in lexicon.candidates(syms, set(ALPHABET)):
            m = _assign(syms, cand)
            if m is None:
                continue
            if any(bound.get(t, v) != v for t, v in m.items()):
                continue
            if _places(board, syms, cand, bound):
                live.append(m)
        if len(live) == 1:
            bound.update(live[0])

    if bound and verify_placement:
        bound = _keep_placeable(board, bound)
    return bound


def _places(board, symbols, word, bound: dict[str, str]) -> bool:
    """Could this word, read this way, actually sit somewhere in the grid?"""
    from .solver import Solver

    letters = [[bound.get(ch, ch) if is_token(ch) else ch for ch in row]
               for row in board.letters]
    return bool(Solver(letters).find_all(list(word)))


def _keep_placeable(board, bound: dict[str, str]) -> dict[str, str]:
    """Drop the mapping if the resolved words no longer fit the grid.

    A dictionary match that cannot be found in the grid is a wrong guess, not
    a discovery, so the whole mapping is discarded rather than half-applied.
    """
    from .solver import Solver

    letters = [[bound.get(ch, ch) if is_token(ch) else ch for ch in row]
               for row in board.letters]
    words = [[bound.get(s, s) if is_token(s) else s for s in w.symbols]
             for w in board.words if not w.found]
    if not words:
        return bound
    _placed, unplaced = Solver(letters).solve(words)
    return {} if unplaced else bound


def auto_resolve(img, board, lib, lexicon: Lexicon, rounds: int = 4,
                 learn: bool = True, verbose: bool = False):
    """Resolve tokens, learn the letters, re-read, repeat.

    Iterating matters. On a real level-8 board the first pass resolved T, J, B
    and U but left `MAR?3` ambiguous between MARK and MART, because T was still
    an unknown letter and so still a legal value for a token. Learning T and
    re-reading removes it from the candidate set, and MARK resolves on the next
    pass. If the token really had been another T, re-reading would simply match
    it as T instead — either way the ambiguity disappears.

    Returns (board, letters_resolved, learned_sample_count). Token names are
    not returned across rounds because re-reading renumbers them; what carries
    over is the set of letters established.
    """
    from .board import learn_pill, read_board, unique_pill

    total: set[str] = set()
    learned = 0

    for _ in range(max(1, rounds)):
        if not board.tokens:
            break
        bound = resolve(board, lib, lexicon)
        if not bound:
            break

        apply_resolution(board, bound)
        total.update(bound.values())
        if verbose:
            print("    resolved " + ", ".join(
                f"?{t[1:]}={v}" for t, v in sorted(bound.items())))

        if not learn:
            continue

        from .solver import Solver
        solver = Solver(board.letters)
        placements, _ = Solver(board.letters).solve(
            [w.symbols for w in board.words if not w.found])
        for p in placements:
            if any(is_token(s) for s in p.symbols):
                continue
            # Only learn from a word that sits in exactly one place; an
            # ambiguous placement would label the wrong cells.
            if len(solver.find_all(list(p.symbols))) != 1:
                continue
            learned += lib.learn_from_word(img, board.grid, p.cells, p.symbols)
            w = unique_pill(board, p.symbols)
            if w is not None:
                learned += learn_pill(img, w.box, p.symbols, lib)

        board = read_board(img, lib, grid=board.grid)

    return board, total, learned


def apply_resolution(board, bound: dict[str, str]) -> None:
    """Rewrite a board's tokens in place with their resolved letters."""
    if not bound:
        return
    for r in range(board.rows):
        for c in range(board.cols):
            s = board.letters[r][c]
            if is_token(s) and s in bound:
                board.letters[r][c] = bound[s]
    for w in board.words:
        w.symbols = [bound.get(s, s) if is_token(s) else s for s in w.symbols]
    board.tokens = [t for t in board.tokens if t not in bound]
