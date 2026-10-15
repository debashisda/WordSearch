"""Start the game and get as far as a playable board.

Launching is only the first step. Traced from a screen recording, opening the
app goes:

    app icon -> blue splash -> dark loading screen with a progress bar
             -> main menu -> board

The loading screen lasts several seconds and shows nothing clickable, and the
main menu needs a press before any board exists. So this waits and watches
rather than launching and hoping.

The menu is the part that needs care. It carries three large buttons stacked
vertically — a red PLAY with the level number, a yellow THEMES, and a green
INFINITE — and pressing the wrong one starts a different mode or opens a
settings page. Measured on the real menu, PLAY is both the largest (124k px
against INFINITE's 85k) and the topmost, so both signals agree; this requires
them to agree before pressing, and waits rather than guessing when they do not.

The level-complete screens have a single button, so the same rule covers them.
"""

from __future__ import annotations

import time

from .screens import button_centre, find_button, looks_like_board

DEFAULT_PACKAGE = "com.blackout.word"


def _all_buttons(img):
    """Every button on screen, largest first (see screens.button_candidates)."""
    from .screens import button_candidates
    return button_candidates(img)


def find_play_button(img):
    """The button that starts a level, or None if it is not safe to say.

    With several buttons stacked, the largest and the topmost must be the same
    one. On the real menu PLAY satisfies both; if a future layout made them
    disagree, pressing nothing and waiting is better than starting the wrong
    mode.
    """
    boxes = _all_buttons(img)
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    largest = boxes[0]
    topmost = min(boxes, key=lambda b: b[1])
    return largest if largest == topmost else None


def launch(device, package: str = DEFAULT_PACKAGE, force_stop: bool = False,
           timeout: float = 90.0, poll: float = 0.8,
           verbose: bool = True) -> bool:
    """Open the game and press through to a board. True once one is on screen."""
    if not hasattr(device, "shell"):
        if verbose:
            print("this device cannot run shell commands; open the game yourself")
        return looks_like_board(device.screencap())

    if force_stop:
        if verbose:
            print(f"stopping {package}")
        device.shell("am", "force-stop", package)
        time.sleep(0.6)

    if verbose:
        print(f"launching {package}")
    # monkey resolves the launcher activity itself, so the activity name does
    # not have to be known or kept up to date.
    out = device.shell("monkey", "-p", package, "-c",
                       "android.intent.category.LAUNCHER", "1")
    if "No activities found" in out or "Error" in out:
        # Fall back to asking the package manager which activity to start.
        brief = device.shell("cmd", "package", "resolve-activity",
                             "--brief", package).strip().splitlines()
        comp = brief[-1].strip() if brief else ""
        if "/" in comp:
            device.shell("am", "start", "-n", comp)
        else:
            if verbose:
                print(f"could not launch {package} — is it installed?")
            return False

    deadline = time.time() + timeout
    pressed = 0
    last = None
    while time.time() < deadline:
        img = device.screencap()

        if looks_like_board(img):
            if verbose:
                print("  board is up")
            return True

        box = find_play_button(img)
        if box is not None:
            x, y = button_centre(box)
            if verbose and (x, y) != last:
                print(f"  menu; pressing ({x},{y})")
            last = (x, y)
            device.tap(x, y)
            pressed += 1
        elif verbose and pressed == 0:
            print("  waiting for the game to load...")

        time.sleep(poll)

    if verbose:
        print(f"  gave up after {timeout:.0f}s without reaching a board")
    return False
