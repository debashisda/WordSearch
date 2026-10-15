"""Buttons on the screens that are not a level board.

Three such screens have been seen, and they share no colour:

  * COMPLETED, with a green button carrying the next level's number;
  * a photo card with a purple COLLECT button, shown first sometimes;
  * the main menu, with a red PLAY, a yellow THEMES and a green INFINITE.

So buttons are found by shape and colour *uniformity*, never by a specific
colour. Three things were needed to make that work on this game, each learned
from a case where the simpler version picked the wrong thing.

**Hue bands must be circular and overlapping.** Red sits at both ends of the
hue scale, so fixed bands cut a red button in half and each half was too small
to pass — which is why the launcher once pressed the yellow THEMES, the
topmost button it could actually see. A boundary can cut any button, too:
measured on the real menu, PLAY is tight (hue 2.9, spread +/-0.9) while THEMES
is a gradient spanning +/-10.7, so no tiling of narrow bands holds both.
Centres every 10 degrees with a +/-14 tolerance give each button at least one
band containing all of it.

**Saturation must be banded as well.** The menu's sunset background is
orange-red, the same hue family as the PLAY button, so on hue alone the two
merge into a blob spanning the full screen width. They differ in saturation —
PLAY is a flat 255, the sky 119-187 — so splitting that range separates them.

**Candidates are ranked by fill, not size.** Overlapping bands find the same
button several times, and a loose band can swallow two buttons and the gap
between them. That merged blob is the largest thing on screen but the least
solid: 0.79 against 0.93 for the correct PLAY box. Keeping the most solid
version of each region, and discarding whatever substantially overlaps it,
leaves exactly the real buttons.
"""

from __future__ import annotations

import cv2
import numpy as np

HUE_STEP = 10               # band centres, over OpenCV's 0-179 hue scale
HUE_TOL = 14                # half-width; bands overlap so none splits a button
SAT_BANDS = ((100, 215), (180, 256), (235, 256))
MIN_VAL = 100
MIN_AREA_FRAC = 0.012       # of the whole screen
MIN_WIDTH_FRAC = 0.22
MAX_WIDTH_FRAC = 0.90       # a button never spans the whole screen
MAX_AREA_FRAC = 0.18        # nor most of it
MIN_FILL = 0.82             # a button is a solid block, not a ragged region
ASPECT_RANGE = (1.2, 6.5)
BOARD_CONFIDENCE = 0.50     # real boards measure >= 0.86; a photo card 0.11


def button_candidates(img: np.ndarray, scale: int = 4):
    """Every button-shaped block on screen, largest first.

    Searched on a quarter-size copy: a button is hundreds of pixels wide, so
    the detail is not needed and the scan costs a fraction as much.
    """
    if img is None or img.size == 0:
        return []
    full_h, full_w = img.shape[:2]
    if scale > 1:
        img = cv2.resize(img, (full_w // scale, full_h // scale),
                         interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_i = hue.astype(np.int16)
    bright = val > MIN_VAL
    kernel = np.ones((3, 3), np.uint8)

    found: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for centre in range(0, 180, HUE_STEP):
        dist = np.abs(hue_i - centre)
        dist = np.minimum(dist, 180 - dist)
        in_hue = bright & (dist <= HUE_TOL)
        if not in_hue.any():
            continue
        for lo, hi in SAT_BANDS:
            band = (in_hue & (sat >= lo) & (sat < hi)).astype(np.uint8)
            if not band.any():
                continue
            band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(band, 8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < MIN_AREA_FRAC * W * H or w < MIN_WIDTH_FRAC * W:
                    continue
                if w > MAX_WIDTH_FRAC * W or area > MAX_AREA_FRAC * W * H:
                    continue
                if h < 0.02 * H:
                    continue
                fill = area / float(w * h)
                if fill < MIN_FILL:
                    continue
                if not (ASPECT_RANGE[0] <= w / max(1, h) <= ASPECT_RANGE[1]):
                    continue
                found.append((fill, int(area) * scale * scale,
                              (int(x) * scale, int(y) * scale,
                               int(w) * scale, int(h) * scale)))

    found.sort(key=lambda t: (-t[0], t[1]))
    kept = _dedupe(found)
    kept.sort(key=lambda b: b[2] * b[3], reverse=True)
    return kept


def _overlap(a, b) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y = max(0, min(ay + ah, by + bh) - max(ay, by))
    return x * y


def _dedupe(found):
    """Keep the most solid version of each region.

    Overlap, not centre-containment: a band that swallowed two buttons and the
    gap between them has its centre in that gap, inside neither of them, so a
    centre test would keep it alongside the real ones — and being the largest,
    it would then be the one pressed.
    """
    kept: list[tuple[int, int, int, int]] = []
    for _fill, _area, box in found:
        area = box[2] * box[3]
        if any(_overlap(box, k) >= 0.5 * min(area, k[2] * k[3]) for k in kept):
            continue
        kept.append(box)
    return kept


def find_button(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest button-shaped block, whatever its colour."""
    boxes = button_candidates(img)
    return boxes[0] if boxes else None


find_green_button = find_button          # old name, kept for callers


def button_centre(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def looks_like_board(img: np.ndarray) -> bool:
    """True when a letter grid is on screen, i.e. we are on a level.

    Fitting the lattice, not just finding a pale panel: the photo card on the
    COLLECT screen has a bright border that passes a panel test, and would
    otherwise be mistaken for a board forever.
    """
    from .geometry import GeometryError, detect_grid
    try:
        return detect_grid(img).confidence >= BOARD_CONFIDENCE
    except GeometryError:
        return False
    except Exception:
        return False


def find_continue_button(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """The button to press to move on, but only when not on a level board."""
    if looks_like_board(img):
        return None
    return find_button(img)


def describe(img: np.ndarray) -> str:
    """One-line description of what screen we appear to be on."""
    if looks_like_board(img):
        return "level board"
    boxes = button_candidates(img)
    if boxes:
        where = ", ".join(str(button_centre(b)) for b in boxes[:3])
        return f"between levels, {len(boxes)} button(s) at {where}"
    return "unknown screen (ad, loading, or a dialog)"
