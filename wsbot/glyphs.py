"""Stages 2 and 3 — normalise a cell into a glyph, then identify it.

Stage 2, `normalise()`:
    inset the cell, Otsu-threshold it, take the minority class as ink,
    keep the largest component, tight-crop to its bounding box, scale to a
    fixed box preserving aspect ratio, centre it on a square canvas.

    Ink is the *minority* class rather than "the dark one". The panel carries a
    radial gradient and a found word flips the cell to white-on-colour, so
    absolute brightness is not a reliable guide to polarity. Ink coverage is
    about 20% of a cell either way, so the smaller class is always the letter.

Stage 3, `GlyphLibrary.match()` — correlation behind a *structural gate*.

    Normalised cross-correlation alone is not safe when the library is
    incomplete, and that is the common case: a level only shows the letters it
    happens to use. Measured by holding each letter out of the library in turn,
    an out-of-vocabulary glyph matched the wrong letter at a median score of
    0.783 and a maximum of 0.897 — comfortably past any threshold that still
    accepts true matches, which fall as low as 0.894 once the grid shrinks and
    glyphs render smaller. The two distributions overlap, so no single
    threshold separates them. In practice that meant T read as I, B as D, U as
    O and K as R.

    The fix is to gate candidates on two scale-invariant structural features
    before their correlation is considered at all:

      holes   topological hole count, area-filtered so a thin stroke closing up
              at small scale cannot invent one. B=2, A D O P Q R=1, rest=0.
              This alone separates B/D, U/O and K/R.

      aspect  width/height of the tight crop. Stable to within 1.11x for a
              given letter across a 2x range of render sizes, while I (0.17)
              and T (~0.79) differ by 4.6x. This separates T/I.

    A candidate must match on holes exactly and on aspect within `aspect_tol`
    before its score counts. If nothing survives the gate, the glyph is
    reported unknown rather than guessed — which is the behaviour that matters,
    because a wrong letter corrupts the board silently while an unknown one is
    recoverable (see UnknownPool below).

`UnknownPool` — why an incomplete alphabet is no longer fatal.

    An unreadable glyph does not have to be identified to be *useful*. The grid
    and the word list are rendered in the same typeface, so the same unknown
    glyph appearing in both can be recognised as the same symbol without ever
    learning its name. Unknown glyphs are clustered by correlation and given
    stable tokens (#1, #2, ...) shared across the whole board, so a word that
    reads `#1 A C O #2` can still be located in a grid that reads the same way,
    and dragged correctly. JACOB gets played without the bot knowing what a J
    or a B looks like.
"""

from __future__ import annotations

import hashlib
import string
from pathlib import Path

import cv2
import numpy as np

GLYPH_SIZE = 32          # normalised canvas, px
GLYPH_PAD = 4            # keeps strokes off the canvas edge
ALPHABET = string.ascii_uppercase

# Tuned from measurements described in the module docstring.
MIN_SCORE = 0.90         # OOV letters top out at 0.877; true natives sit >=0.961
MATCH_MARGIN = 0.03      # required lead over the runner-up surviving letter
ASPECT_TOL = 1.45        # max width/height ratio mismatch (within-letter: 1.11)
HOLE_MIN_FRAC = 0.012    # a hole must be this fraction of the glyph's bbox
CLUSTER_SCORE = 0.88     # two unknown glyphs are the same symbol above this


# ---------------------------------------------------------------- stage 2

def normalise(cell_bgr: np.ndarray,
              out: int = GLYPH_SIZE,
              inset: float = 0.12,
              min_ink: int = 60) -> np.ndarray | None:
    """Crop -> binarize -> tight-crop -> resize. Returns a uint8 {0,255} square.

    `inset` trims the cell border so a neighbouring colour stripe or the panel
    edge cannot leak in. Returns None when the cell holds no usable glyph.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return None

    h, w = cell_bgr.shape[:2]
    m = int(min(h, w) * inset)
    sub = cell_bgr[m:h - m, m:w - m]
    if sub.size == 0:
        return None

    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ink is whichever class is smaller — polarity-agnostic by construction.
    if (th > 0).mean() > 0.5:
        th = 255 - th

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_ink:
        return None

    x, y, bw, bh, _ = stats[idx]
    crop = (lab[y:y + bh, x:x + bw] == idx).astype(np.uint8) * 255

    span = out - 2 * GLYPH_PAD
    scale = min(span / bw, span / bh)
    rw, rh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out, out), np.uint8)
    ox, oy = (out - rw) // 2, (out - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = crop
    return canvas


def cell_glyph(img: np.ndarray, grid, r: int, c: int) -> np.ndarray | None:
    """Normalised glyph for cell (r, c) of a full-screen frame."""
    x, y, w, h = grid.cell_box(r, c)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return normalise(img[y0:y1, x0:x1])


# -------------------------------------------------- structural features

def glyph_features(glyph: np.ndarray) -> tuple[float, int] | None:
    """(aspect, holes) for a normalised glyph. Both scale-invariant.

    `aspect` is width/height of the tight ink bounding box. normalise()
    preserves aspect when fitting the canvas, so this survives the round trip.

    `holes` counts background regions fully enclosed by ink, ignoring any
    smaller than HOLE_MIN_FRAC of the bounding box. That filter matters: at
    small render sizes the legs of an M can close up and fake a hole, which
    without it made M's hole count unstable across scales.
    """
    if glyph is None:
        return None
    ys, xs = np.nonzero(glyph)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    aspect = bw / bh

    crop = glyph[y0:y1 + 1, x0:x1 + 1]
    pad = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8)
    pad[1:-1, 1:-1] = crop
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (pad == 0).astype(np.uint8), 4)

    outside = lab[0, 0]
    area = crop.shape[0] * crop.shape[1]
    holes = sum(1 for i in range(1, n)
                if i != outside and stats[i, cv2.CC_STAT_AREA] >= HOLE_MIN_FRAC * area)
    return aspect, holes


def _aspect_ok(a: float, b: float, tol: float = ASPECT_TOL) -> bool:
    lo, hi = min(a, b), max(a, b)
    return lo > 0 and (hi / lo) <= tol


# ---------------------------------------------------------------- stage 3

class GlyphLibrary:
    """Reference glyphs on disk, matched by correlation behind a structural gate."""

    def __init__(self, root: str | Path = "glyphs",
                 min_score: float = MIN_SCORE,
                 margin: float = MATCH_MARGIN,
                 aspect_tol: float = ASPECT_TOL,
                 use_structure: bool = True,
                 readonly: bool = False) -> None:
        self.root = Path(root)
        self.min_score = min_score
        self.margin = margin
        self.aspect_tol = aspect_tol
        self.use_structure = use_structure
        self.readonly = readonly      # learn in memory, never touch the disk
        self.samples: dict[str, list[np.ndarray]] = {}
        self._feats: dict[str, list[tuple[float, int]]] = {}
        # Vectorised index, rebuilt whenever samples change. See _reindex().
        self._vecs: np.ndarray | None = None
        self._vec_letter: np.ndarray | None = None
        self._vec_aspect: np.ndarray | None = None
        self._vec_holes: np.ndarray | None = None
        self._letters_seen: list[str] = []
        self.load()

    # ------------------------------------------------------------- storage

    def load(self) -> None:
        self.samples, self._feats = {}, {}
        if not self.root.exists():
            return
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.upper() not in ALPHABET:
                continue
            imgs = []
            for p in sorted(d.glob("*.png")):
                a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if a is None:
                    continue
                if a.shape != (GLYPH_SIZE, GLYPH_SIZE):
                    a = cv2.resize(a, (GLYPH_SIZE, GLYPH_SIZE),
                                   interpolation=cv2.INTER_AREA)
                imgs.append(a)
            if imgs:
                letter = d.name.upper()
                self.samples[letter] = imgs
                self._feats[letter] = [f for f in map(glyph_features, imgs)
                                       if f is not None]
        self._reindex()

    def add(self, letter: str, glyph: np.ndarray) -> bool:
        """Store a labelled sample. Near-duplicates are skipped."""
        letter = letter.upper()
        if letter not in ALPHABET or glyph is None:
            return False

        # Do not file a glyph under a letter it looks nothing like.
        #
        # The library teaches itself, so one bad sample is permanent and
        # poisons the structural gate for that letter from then on: a single
        # narrow glyph stored under G would let G pass the aspect check against
        # any narrow glyph, and an I could then be read as G. Existing samples
        # of a letter agree closely on holes and aspect, so a candidate that
        # matches none of them is rejected. A letter with no samples yet is
        # always accepted — that is how a new letter gets learned at all.
        refs = self._feats.get(letter) or []
        feat = glyph_features(glyph)
        if refs and feat is not None:
            aspect, holes = feat
            if not any(h == holes and _aspect_ok(a, aspect, self.aspect_tol)
                       for a, h in refs):
                return False

        for existing in self.samples.get(letter, []):
            if float(cv2.matchTemplate(existing, glyph,
                                       cv2.TM_CCOEFF_NORMED)[0][0]) > 0.985:
                return False

        if not self.readonly:
            d = self.root / letter
            d.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha1(glyph.tobytes()).hexdigest()[:12] + ".png"
            cv2.imwrite(str(d / name), glyph)
        self.samples.setdefault(letter, []).append(glyph)
        f = glyph_features(glyph)
        if f is not None:
            self._feats.setdefault(letter, []).append(f)
        self._reindex()
        return True

    # --------------------------------------------------------- fast index

    @staticmethod
    def _normalise_vec(a: np.ndarray) -> np.ndarray:
        v = a.astype(np.float32).ravel()
        v -= v.mean()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else v

    def _reindex(self) -> None:
        """Stack every sample as a unit vector, for scoring in one product.

        Matching used to call cv2.matchTemplate once per stored sample per
        cell, so a board read grew linearly with the library: at 53 samples a
        10x8 board took 0.22 s, at 1060 it took 1.95 s, and the three-frame
        consensus read at the start of each level took almost six seconds.

        For templates and images of equal size, TM_CCOEFF_NORMED at the single
        valid position *is* the Pearson correlation of the two as vectors. So
        each sample is stored zero-meaned and scaled to unit length once, and
        scoring a glyph against the whole library becomes one matrix-vector
        product — the same numbers to within 6e-7, a couple of hundred times
        faster, and no longer sensitive to how much the library has learned.
        """
        vecs, letters, aspects, holes = [], [], [], []
        for letter in sorted(self.samples):
            for g in self.samples[letter]:
                f = glyph_features(g)
                vecs.append(self._normalise_vec(g))
                letters.append(letter)
                aspects.append(f[0] if f else 0.0)
                holes.append(f[1] if f else -1)
        if not vecs:
            self._vecs = None
            self._letters_seen = []
            return
        self._vecs = np.stack(vecs)
        self._letters_seen = sorted(set(letters))
        index = {L: i for i, L in enumerate(self._letters_seen)}
        self._vec_letter = np.array([index[L] for L in letters], np.int32)
        self._vec_aspect = np.array(aspects, np.float32)
        self._vec_holes = np.array(holes, np.int32)

    # ------------------------------------------------------------ matching

    @property
    def letters(self) -> list[str]:
        return sorted(self.samples)

    @property
    def missing(self) -> list[str]:
        return [c for c in ALPHABET if c not in self.samples]

    def __len__(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def _structurally_allows(self, letter: str, feat) -> bool:
        """Could `letter` plausibly be this glyph, on structure alone?"""
        if not self.use_structure or feat is None:
            return True
        refs = self._feats.get(letter) or []
        if not refs:
            return True
        aspect, holes = feat
        return any(h == holes and _aspect_ok(a, aspect, self.aspect_tol)
                   for a, h in refs)

    def score_all(self, glyph: np.ndarray, gated: bool = True
                  ) -> list[tuple[float, str]]:
        """Best score per letter, descending. `gated` applies the structural gate."""
        if glyph is None or self._vecs is None:
            return []

        scores = self._vecs @ self._normalise_vec(glyph)

        if gated:
            feat = glyph_features(glyph)
            if feat is not None:
                aspect, holes = feat
                lo = np.minimum(self._vec_aspect, aspect)
                hi = np.maximum(self._vec_aspect, aspect)
                allowed = ((self._vec_holes == holes)
                           & (lo > 0) & (hi <= lo * self.aspect_tol))
                scores = np.where(allowed, scores, -2.0)

        # Best score per letter, in one pass.
        best = np.full(len(self._letters_seen), -2.0, np.float32)
        np.maximum.at(best, self._vec_letter, scores)

        out = [(float(b), L) for b, L in zip(best, self._letters_seen)
               if b > -1.5]
        out.sort(reverse=True)
        return out

    def match(self, glyph: np.ndarray) -> tuple[str | None, float]:
        """Identify a glyph. Returns (letter, score); letter is None if unsure.

        None means "this is not a letter I hold", which is recoverable.
        A wrong letter is not, so the thresholds here are deliberately strict.
        """
        ranked = self.score_all(glyph)
        if not ranked:
            return None, 0.0
        score, letter = ranked[0]
        if score < self.min_score:
            return None, score
        if len(ranked) > 1 and score - ranked[1][0] < self.margin:
            return None, score          # too close to call
        return letter, score

    def explain(self, glyph: np.ndarray) -> str:
        """Human-readable reason for a match or refusal. Used by debug tools."""
        feat = glyph_features(glyph)
        if feat is None:
            return "no ink in cell"
        aspect, holes = feat
        gated = self.score_all(glyph, gated=True)
        raw = self.score_all(glyph, gated=False)
        head = f"aspect={aspect:.2f} holes={holes}"
        if not gated:
            top = ", ".join(f"{l}:{s:.2f}" for s, l in raw[:3])
            return (f"{head} -> no letter passes the structural gate "
                    f"(ungated best: {top}) -> unknown")
        letter, score = self.match(glyph)
        top = ", ".join(f"{l}:{s:.2f}" for s, l in gated[:3])
        return f"{head} -> {letter or 'unknown'} ({score:.2f}); candidates {top}"

    # ------------------------------------------------------------ learning

    def learn_from_word(self, img: np.ndarray, grid, cells, word) -> int:
        """Label the cells a confirmed word covers, and store them.

        `word` may be a string or a list of symbols; tokens like '#1' are
        skipped, since we do not know what letter they are.
        """
        added = 0
        for (r, c), ch in zip(cells, list(word)):
            if not (isinstance(ch, str) and len(ch) == 1 and ch.isalpha()):
                continue
            g = cell_glyph(img, grid, r, c)
            if g is not None and self.add(ch.upper(), g):
                added += 1
        return added


# ----------------------------------------------------- unknown-glyph tokens

class UnknownPool:
    """Gives unreadable glyphs stable tokens, shared across one board read.

    Two glyphs the library cannot name are treated as the same symbol when they
    correlate above CLUSTER_SCORE and agree structurally. That is enough to
    match a word against the grid without ever identifying the letter.
    """

    def __init__(self, threshold: float = CLUSTER_SCORE,
                 aspect_tol: float = ASPECT_TOL) -> None:
        self.threshold = threshold
        self.aspect_tol = aspect_tol
        self.protos: list[np.ndarray] = []
        self.feats: list[tuple[float, int] | None] = []

    def token(self, glyph: np.ndarray) -> str | None:
        """Stable '#n' token for this glyph, or None if it has no ink."""
        if glyph is None:
            return None
        feat = glyph_features(glyph)
        if feat is None:
            return None

        best_i, best_s = -1, -1.0
        for i, proto in enumerate(self.protos):
            pf = self.feats[i]
            if pf is not None and feat is not None:
                if pf[1] != feat[1] or not _aspect_ok(pf[0], feat[0], self.aspect_tol):
                    continue
            s = float(cv2.matchTemplate(proto, glyph, cv2.TM_CCOEFF_NORMED)[0][0])
            if s > best_s:
                best_i, best_s = i, s

        if best_i >= 0 and best_s >= self.threshold:
            return f"#{best_i + 1}"

        self.protos.append(glyph)
        self.feats.append(feat)
        return f"#{len(self.protos)}"

    def __len__(self) -> int:
        return len(self.protos)

    @property
    def tokens(self) -> list[str]:
        return [f"#{i + 1}" for i in range(len(self.protos))]


def is_token(sym) -> bool:
    return isinstance(sym, str) and sym.startswith("#")


def symbols_to_text(symbols) -> str:
    """Render a symbol list for display: letters as themselves, tokens as ?1."""
    out = []
    for s in symbols:
        if s is None:
            out.append("?")
        elif is_token(s):
            out.append("?" + s[1:])
        else:
            out.append(s)
    return "".join(out)
