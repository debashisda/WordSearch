"""wsbot — an Android Word Search solver driven over ADB.

Pipeline, one module per stage:

    adb.py       capture frames from the device, send gestures back
    geometry.py  locate the grid panel and fit the cell lattice  (stage 1)
    glyphs.py    crop / binarize / tight-crop / resize + match    (stages 2 & 3)
    board.py     assemble a Board: letters, word list, found flags
    solver.py    find every target word in the grid, 8 directions
    player.py    turn a placement into a swipe and verify the result

Nothing here hardcodes a grid size. The lattice is re-fitted on every level,
so a 6x7 board and a 12x14 board are handled by the same code path.
"""

__version__ = "1.0.0"

__all__ = ["adb", "geometry", "glyphs", "board", "solver", "player", "config"]
