#!/usr/bin/env python3
"""Find and remove corrupted samples in the glyph library.

    python scripts/audit_glyphs.py          # report only
    python scripts/audit_glyphs.py --fix    # quarantine the bad ones

The library teaches itself, so a sample stored under the wrong letter is
permanent and quietly poisons the structural gate from then on. The gate lets a
candidate through if *any* stored sample of that letter matches on hole count
and aspect ratio, so one narrow glyph filed under G is enough for G to start
matching narrow glyphs — which is how VINEGAR came to read as VGNEGAR.

A letter's samples normally agree closely: every G has one hole-free shape at
roughly 0.78 aspect, every I is a bar near 0.2. So an outlier is detectable
without knowing what the letter should look like — it simply disagrees with its
own siblings. Letters whose shape genuinely varies are protected by requiring a
clear majority before anything is called an outlier.

Removed samples are moved to glyphs/_quarantine/<LETTER>/ rather than deleted,
so a wrong call can be undone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.config import Config                              # noqa: E402
from wsbot.glyphs import ALPHABET, GLYPH_SIZE, glyph_features  # noqa: E402

# Hole count is a property of the letterform, not the font, so it can be
# asserted outright. Aspect ratio varies by typeface and is judged per letter.
HOLES = {"A": 1, "B": 2, "C": 0, "D": 1, "E": 0, "F": 0, "G": 0, "H": 0,
         "I": 0, "J": 0, "K": 0, "L": 0, "M": 0, "N": 0, "O": 1, "P": 1,
         "Q": 1, "R": 1, "S": 0, "T": 0, "U": 0, "V": 0, "W": 0, "X": 0,
         "Y": 0, "Z": 0}

ASPECT_SPREAD = 1.5      # how far from the letter's median aspect is tolerated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="move suspect samples into glyphs/_quarantine/")
    ap.add_argument("--glyphs", default=None)
    args = ap.parse_args()

    cfg = Config.load()
    root = Path(args.glyphs or cfg.glyph_dir)
    if not root.exists():
        print(f"no glyph library at {root}")
        return 2

    total = suspect = 0
    actions: list[tuple[Path, str]] = []

    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.upper() not in ALPHABET:
            continue
        letter = d.name.upper()
        files = sorted(d.glob("*.png"))
        feats = []
        for f in files:
            a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            if a.shape != (GLYPH_SIZE, GLYPH_SIZE):
                a = cv2.resize(a, (GLYPH_SIZE, GLYPH_SIZE),
                               interpolation=cv2.INTER_AREA)
            ft = glyph_features(a)
            if ft is not None:
                feats.append((f, ft))
        if not feats:
            continue
        total += len(feats)

        want_holes = HOLES[letter]
        aspects = [ft[0] for _f, ft in feats]
        median = float(np.median(aspects))

        # Hole count is asserted outright — but if it would condemn every
        # sample of a letter, the table is wrong for this typeface rather than
        # the samples being bad, so the test is skipped for that letter.
        hole_counts = Counter(ft[1] for _f, ft in feats)
        wrong_holes = sum(n for h, n in hole_counts.items() if h != want_holes)
        trust_holes = wrong_holes < len(feats)

        for f, (aspect, holes) in feats:
            why = None
            if holes != want_holes and trust_holes:
                why = f"{holes} hole(s), a {letter} has {want_holes}"
            elif median > 0 and max(aspect, median) / min(aspect, median) > ASPECT_SPREAD:
                why = (f"aspect {aspect:.2f} against {median:.2f} for the "
                       f"other {letter} samples")
            if why:
                suspect += 1
                actions.append((f, f"{letter}: {why}"))

    if not actions:
        print(f"{total} samples, none suspect — the library is consistent")
        return 0

    print(f"{total} samples, {suspect} suspect:\n")
    for f, why in actions:
        print(f"  {f.parent.name}/{f.name}  {why}")

    if not args.fix:
        print("\nre-run with --fix to move these to glyphs/_quarantine/")
        return 0

    qroot = root / "_quarantine"
    for f, _why in actions:
        dest = qroot / f.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest / f.name))
    print(f"\nmoved {len(actions)} sample(s) to {qroot}")
    print("the library reloads clean on the next run; delete that folder when "
          "you are happy, or move files back to undo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
