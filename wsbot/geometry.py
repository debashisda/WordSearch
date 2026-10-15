"""Stage 1 — locate the grid and fit the cell lattice.

Nothing here is hardcoded to a grid size. Every call re-derives rows, columns
and pitch from the frame, so a level that grows from 6x7 to 12x14 needs no
config change.

How it works:

  1. The grid sits on a near-white rounded panel. Mask near-white pixels, fill
     the holes punched by found-word colour stripes, take the largest solid
     blob. This is the single most reliable landmark on the screen.

  2. Inside the panel, find glyph blobs using *local contrast* rather than
     absolute brightness. A cell is either a dark letter on white or — once its
     word has been found — a white letter on saturated colour. Comparing each
     pixel against a blurred copy of its own neighbourhood catches both
     polarities with one rule and no colour reasoning.

  3. Choose the integer (rows, cols) pair whose lattice best explains where
     those blobs sit. Cells in this game are square and the panel wraps them
     tightly, so a candidate fixes the pitch as panel_w/cols and panel_h/rows;
     candidates whose two pitches disagree are rejected outright. The winner is
     the one minimising mean squared centroid-to-cell-centre distance.

Step 3 beats clustering gaps or projection profiles because it degrades
gracefully: a board where most words are already found still has enough
well-centred glyphs to pick the right lattice, and stray blobs from panel
edges or stripe ends are outvoted rather than allowed to shift a boundary.

Measured on a 1080x2340 device at level 7: panel (24, 584) 1032x1204, 7 rows x
6 cols, pitch 172.0. On that board the correct lattice scores ~17x better than
the runner-up, on both an untouched and a six-words-solved frame.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# Search bounds for grid size. Generous — the scoring picks the winner.
MIN_COLS, MAX_COLS = 2, 16
MIN_ROWS, MAX_ROWS = 2, 20
MIN_PITCH_PX = 40          # below this a cell is too small to hold a glyph
MAX_PITCH_SKEW = 0.05      # cells are square; reject candidates that disagree


@dataclass
class Grid:
    """A fitted cell lattice."""
    panel: tuple[int, int, int, int]   # x, y, w, h of the white panel
    rows: int
    cols: int
    x0: float                          # screen x of the centre of column 0
    y0: float                          # screen y of the centre of row 0
    pitch_x: float
    pitch_y: float
    confidence: float                  # 0..1; margin over the runner-up lattice

    def centre(self, r: int, c: int) -> tuple[int, int]:
        """Screen coordinates of the centre of cell (row, col)."""
        return (int(round(self.x0 + c * self.pitch_x)),
                int(round(self.y0 + r * self.pitch_y)))

    def cell_box(self, r: int, c: int) -> tuple[int, int, int, int]:
        """x, y, w, h of cell (row, col), sized to the pitch."""
        cx, cy = self.centre(r, c)
        w, h = int(round(self.pitch_x)), int(round(self.pitch_y))
        return (cx - w // 2, cy - h // 2, w, h)

    def cells(self):
        for r in range(self.rows):
            for c in range(self.cols):
                yield r, c

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Grid":
        d = dict(d)
        d["panel"] = tuple(d["panel"])
        return Grid(**d)

    def __str__(self) -> str:
        return (f"{self.rows}x{self.cols} grid, pitch {self.pitch_x:.1f}, "
                f"panel {self.panel}, confidence {self.confidence:.2f}")


class GeometryError(RuntimeError):
    pass


# -------------------------------------------------------------------- panel

def find_panel(img: np.ndarray,
               min_frac: float = 0.08) -> tuple[int, int, int, int]:
    """Bounding box of the near-white grid panel.

    Colour stripes over found words punch holes in the white mask; those are
    filled by flood-filling inward from the border, so a heavily-solved board
    still yields the full panel rectangle.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    white = ((V > 185) & (S < 45)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    ff = white.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    filled = cv2.bitwise_or(white, (ff == 0).astype(np.uint8))

    n, _lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w * 0.35 or bh < h * 0.08:
            continue
        if area < min_frac * w * h:
            continue
        if area < 0.80 * bw * bh:        # the panel fills its own bbox
            continue
        if area > best_area:
            best, best_area = (int(x), int(y), int(bw), int(bh)), int(area)

    if best is None:
        raise GeometryError(
            "Grid panel not found. Is the game on a level screen? "
            "Run scripts/debug_view.py to see what the detector sees."
        )
    return best


# ------------------------------------------------------------------- glyphs

def glyph_blobs(panel_img: np.ndarray,
                sigma: float = 21.0,
                contrast: int = 28) -> list[tuple[float, float, float]]:
    """Letter-like blobs inside the panel as (x, y, area), panel-local.

    Polarity-agnostic: each pixel is compared against a blurred copy of its own
    surroundings, so dark-on-white and white-on-colour are both picked up.
    """
    g = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigma))
    m = (diff > contrast).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    H, W = panel_img.shape[:2]
    lim = max(H, W) * 0.22          # a glyph is small next to the whole panel

    out = []
    for i in range(1, n):
        _x, _y, bw, bh, area = stats[i]
        if area < 150 or bh < 10 or bw < 5:
            continue
        if bw > lim or bh > lim:
            continue
        if bw > 5 * bh or bh > 5 * bw:   # reject stripe edges and hairlines
            continue
        out.append((float(cent[i][0]), float(cent[i][1]), float(area)))
    return out


# ------------------------------------------------------------------ lattice

def _lattice_error(blobs, pitch_x: float, pitch_y: float) -> float:
    """Area-weighted mean squared distance from blob centroids to cell centres,
    in units of a cell. Lower is a better-fitting lattice."""
    err = tot = 0.0
    for bx, by, area in blobs:
        cx = (round(bx / pitch_x - 0.5) + 0.5) * pitch_x
        cy = (round(by / pitch_y - 0.5) + 0.5) * pitch_y
        dx = (bx - cx) / pitch_x
        dy = (by - cy) / pitch_y
        err += area * (dx * dx + dy * dy)
        tot += area
    return err / max(1e-9, tot)


def fit_lattice(blobs, pw: int, ph: int):
    """Rank every plausible (rows, cols). Returns the sorted candidate list."""
    results = []
    for cols in range(MIN_COLS, MAX_COLS + 1):
        for rows in range(MIN_ROWS, MAX_ROWS + 1):
            px_, py_ = pw / cols, ph / rows
            if px_ < MIN_PITCH_PX or py_ < MIN_PITCH_PX:
                continue
            if abs(px_ - py_) / max(px_, py_) > MAX_PITCH_SKEW:
                continue
            results.append((_lattice_error(blobs, px_, py_), rows, cols, px_, py_))
    results.sort()
    return results


def _is_harmonic(rows: int, cols: int, r: int, c: int) -> bool:
    """Is (r, c) the same lattice as (rows, cols), subdivided or combined?

    True when one is an exact integer multiple of the other in *both*
    dimensions — 6x4 against 12x8 or 18x12, or against 3x2.
    """
    if (rows, cols) == (r, c):
        return True
    for (a1, b1), (a2, b2) in (((rows, cols), (r, c)), ((r, c), (rows, cols))):
        if a2 % a1 == 0 and b2 % b1 == 0 and a2 // a1 == b2 // b1 >= 2:
            return True
    return False


def detect_grid(img: np.ndarray,
                panel: tuple[int, int, int, int] | None = None,
                min_blobs: int = 6) -> Grid:
    """Detect the panel and fit the cell lattice. Size is derived, not assumed."""
    px, py, pw, ph = panel if panel is not None else find_panel(img)
    sub = img[py:py + ph, px:px + pw]

    blobs = glyph_blobs(sub)
    if len(blobs) < min_blobs:
        raise GeometryError(
            f"only {len(blobs)} glyph blobs inside the panel — too few to fit a "
            f"lattice. Run scripts/debug_view.py to inspect."
        )

    ranked = fit_lattice(blobs, pw, ph)
    if not ranked:
        raise GeometryError("no plausible grid size for this panel")

    err, rows, cols, pitch_x, pitch_y = ranked[0]

    # Confidence is the margin over the best *genuinely different* lattice.
    #
    # A lattice at an integer multiple of the winner's pitch is not a rival
    # reading — it is the same lattice subdivided, and it centres the same
    # blobs almost as well. On a 6x4 board, 18x12 scored 0.0052 against the
    # winner's 0.0040, which dragged confidence down to 0.24 and made the board
    # look like no board at all. Harmonics are therefore skipped when choosing
    # what to compare against.
    runner_up = None
    for e, r, c, _px, _py in ranked[1:]:
        if _is_harmonic(rows, cols, r, c):
            continue
        runner_up = e
        break
    if runner_up is None:
        runner_up = err * 10

    confidence = float(np.clip(1.0 - err / max(1e-9, runner_up), 0.0, 1.0))

    return Grid(
        panel=(px, py, pw, ph),
        rows=rows, cols=cols,
        x0=px + pitch_x / 2.0,
        y0=py + pitch_y / 2.0,
        pitch_x=pitch_x, pitch_y=pitch_y,
        confidence=confidence,
    )
