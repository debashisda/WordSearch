"""Optional config.json — overrides and tuning, all of it with a default.

The bot runs with no config at all: geometry is detected per level. Calibration
writes a file mainly so you can see what was detected, pin it if auto-detection
ever picks the wrong lattice, and record a tap coordinate for the
level-complete screen if your build needs one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

DEFAULT_PATH = Path("config.json")


@dataclass
class Config:
    serial: str | None = None
    backend: str = "auto"              # auto | u2 | adb
    glyph_dir: str = "glyphs"
    package: str = "com.blackout.word"   # the game, for --launch
    word_list: str | None = None       # defaults to the bundled wsbot/words.txt.gz

    # Pin the lattice only if auto-detection misbehaves. Null means detect.
    panel: list[int] | None = None     # [x, y, w, h]
    rows: int | None = None
    cols: int | None = None

    # Recognition
    min_score: float = 0.90            # below this a glyph is "unknown"
    match_margin: float = 0.03         # required lead over the runner-up letter
    aspect_tol: float = 1.45           # structural gate: max aspect mismatch
    use_structure: bool = True         # gate candidates on holes + aspect
    consensus_frames: int = 3

    # Gestures
    ms_per_point: int = 55
    settle: float = 1.6                # max wait for a word to register, not a sleep
    poll: float = 0.10                 # how often to re-check the pill
    gap: float = 0.12                  # pause between drags in batch mode
    batch: bool = True                 # drag all words, then verify once
    max_attempts: int = 4

    # Level-complete screen: [x, y] to tap, or null to tap the lower middle.
    advance_tap: list[int] | None = None

    extras: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path = DEFAULT_PATH) -> "Config":
        p = Path(path)
        if not p.exists():
            return Config()
        data = json.loads(p.read_text())
        known = {f for f in Config.__dataclass_fields__}
        extras = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known}
        clean["extras"] = extras
        return Config(**clean)

    def save(self, path: str | Path = DEFAULT_PATH) -> None:
        d = asdict(self)
        extras = d.pop("extras", {})
        d.update(extras)
        Path(path).write_text(json.dumps(d, indent=2) + "\n")

    def pinned_grid(self):
        """A Grid built from the config, or None to auto-detect."""
        if not (self.panel and self.rows and self.cols):
            return None
        from .geometry import Grid
        x, y, w, h = self.panel
        px, py = w / self.cols, h / self.rows
        return Grid(panel=(x, y, w, h), rows=self.rows, cols=self.cols,
                    x0=x + px / 2, y0=y + py / 2,
                    pitch_x=px, pitch_y=py, confidence=1.0)
