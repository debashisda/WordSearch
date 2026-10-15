"""Device I/O: pull frames off the phone, push gestures back.

Two gesture backends:

  u2   uiautomator2's swipe_points() — a real multi-point gesture. Every cell
       centre along the word becomes a MOVE sample, so the game cannot miss an
       intermediate letter. This is what pattern-unlock automation uses, and it
       is the backend you want.

  adb  `adb shell input swipe` — only two points. Android interpolates MOVE
       events between them, which is usually enough for a straight line, but
       the sampling rate is not guaranteed. Kept as a zero-dependency fallback.

Capture always goes through `adb exec-out screencap -p`, which streams the PNG
to stdout. Nothing is written to the device's storage.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import time
from typing import Sequence

import cv2
import numpy as np


class DeviceError(RuntimeError):
    pass


class Device:
    def __init__(self, serial: str | None = None, backend: str = "auto",
                 adb_path: str = "adb") -> None:
        self.serial = serial
        self.adb_path = adb_path
        self._u2 = None
        self._capture: str | None = None      # 'raw' or 'png', benchmarked once
        self._capture_timings: dict[str, float] = {}

        if shutil.which(adb_path) is None:
            raise DeviceError(
                f"'{adb_path}' not found on PATH. Install platform-tools and "
                f"make sure `adb devices` lists your phone."
            )

        if backend == "auto":
            backend = "u2" if self._try_u2() else "adb"
        elif backend == "u2":
            if not self._try_u2():
                raise DeviceError(
                    "uiautomator2 backend requested but unavailable. "
                    "pip install uiautomator2, then run `python -m uiautomator2 init`."
                )
        self.backend = backend

    # ---------------------------------------------------------------- setup

    def _try_u2(self) -> bool:
        try:
            import uiautomator2 as u2
        except ImportError:
            return False
        try:
            self._u2 = u2.connect(self.serial) if self.serial else u2.connect()
            self._u2.info  # force a round trip so a dead link fails here
            return True
        except Exception:
            self._u2 = None
            return False

    def _cmd(self, *args: str) -> list[str]:
        base = [self.adb_path]
        if self.serial:
            base += ["-s", self.serial]
        return base + list(args)

    def shell(self, *args: str, timeout: float = 30.0) -> str:
        """Run an adb shell command and return its stdout."""
        proc = subprocess.run(self._cmd("shell", *args),
                              capture_output=True, text=True, timeout=timeout)
        return proc.stdout or ""

    def check(self) -> str:
        """Confirm exactly one usable device and return its serial."""
        out = subprocess.run(
            self._cmd("devices"), capture_output=True, text=True, timeout=20
        ).stdout
        devices = [
            ln.split("\t")[0]
            for ln in out.splitlines()[1:]
            if ln.strip() and ln.strip().endswith("device")
        ]
        if not devices:
            raise DeviceError(
                "No authorised device. Enable USB debugging, reconnect, and "
                "accept the RSA prompt on the phone."
            )
        if len(devices) > 1 and not self.serial:
            raise DeviceError(
                f"Multiple devices attached: {devices}. Pass --serial to choose one."
            )
        return self.serial or devices[0]

    # -------------------------------------------------------------- capture

    def _screencap_png(self) -> np.ndarray | None:
        """`screencap -p`: the phone PNG-encodes the frame before sending it."""
        proc = subprocess.run(self._cmd("exec-out", "screencap", "-p"),
                              capture_output=True, timeout=30)
        if not proc.stdout:
            return None
        img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None and img.size else None

    def _screencap_raw(self) -> np.ndarray | None:
        """`screencap` with no flag: raw RGBA, no encoding step on the phone.

        PNG-encoding a 1080x2340 frame is the single slowest thing in the loop
        on many devices — often over a second. Raw skips it entirely. The
        payload is larger (~10 MB against ~2 MB) but USB transfer is usually
        much cheaper than the encode, so this is normally the faster route.

        Layout: three little-endian uint32 (width, height, format), then
        width*height*4 bytes of RGBA. Newer Android adds a fourth uint32
        (colorspace), so the header is 12 or 16 bytes; the size tells us which.
        Anything that does not add up is rejected, which also catches the old
        Windows adb bug that rewrote newlines inside binary output.
        """
        proc = subprocess.run(self._cmd("exec-out", "screencap"),
                              capture_output=True, timeout=30)
        buf = proc.stdout
        if not buf or len(buf) < 16:
            return None
        w, h, _fmt = struct.unpack_from("<III", buf, 0)
        if not (0 < w <= 8192 and 0 < h <= 8192):
            return None
        need = w * h * 4
        for header in (12, 16):
            if len(buf) == need + header:
                arr = np.frombuffer(buf, np.uint8, count=need, offset=header)
                rgba = arr.reshape(h, w, 4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return None

    def _pick_capture(self) -> None:
        """Time both capture methods once and keep the faster one."""
        timings = {}
        for name, fn in (("raw", self._screencap_raw), ("png", self._screencap_png)):
            try:
                t0 = time.time()
                img = fn()
                dt = time.time() - t0
            except Exception:
                continue
            if img is not None:
                timings[name] = dt
        if not timings:
            self._capture = "png"
            return
        self._capture = min(timings, key=timings.get)
        self._capture_timings = timings

    def screencap(self, retries: int = 2) -> np.ndarray:
        """Grab the current screen as a BGR ndarray."""
        if self._capture is None:
            self._pick_capture()

        order = ([self._screencap_raw, self._screencap_png]
                 if self._capture == "raw"
                 else [self._screencap_png, self._screencap_raw])

        for attempt in range(retries + 1):
            for fn in order:
                try:
                    img = fn()
                except Exception:
                    img = None
                if img is not None:
                    return img
            if attempt < retries:
                time.sleep(0.4)
        raise DeviceError(
            "screencap returned no image. If the game blocks capture you will "
            "need scrcpy or a device that permits it."
        )

    # ------------------------------------------------------------- gestures

    def swipe_path(self, points: Sequence[tuple[int, int]],
                   ms_per_point: int = 55, min_ms: int = 320) -> None:
        """Drag through every point in order, finger down throughout.

        `points` should be the centre of every cell the word covers, including
        both ends. Duration scales with length so long words are not rushed.
        """
        if len(points) < 2:
            raise ValueError("a drag needs at least two points")

        duration_ms = max(min_ms, int(ms_per_point * len(points)))

        if self.backend == "u2" and self._u2 is not None:
            # swipe_points takes seconds *between* samples.
            per_step = max(0.02, (duration_ms / 1000.0) / max(1, len(points) - 1))
            self._u2.swipe_points([(int(x), int(y)) for x, y in points], per_step)
            return

        # Fallback: a two-point swipe. Android interpolates between the ends,
        # which tracks a straight line well enough for this game.
        x1, y1 = points[0]
        x2, y2 = points[-1]
        subprocess.run(
            self._cmd("shell", "input", "swipe",
                      str(int(x1)), str(int(y1)),
                      str(int(x2)), str(int(y2)), str(duration_ms)),
            capture_output=True, timeout=30,
        )

    def swipe_many(self, paths, ms_per_point: int = 55, min_ms: int = 320,
                   gap_ms: int = 120) -> bool:
        """Send several drags in one go. Returns True if it handled them.

        On the adb backend every `adb shell input swipe` is a separate process
        launch on the host *and* a JVM start on the phone, because `input` is a
        Java program. Chaining the commands into one shell invocation removes
        the host-side launches and the USB round trips between them, which is
        worth ~100 ms per word on Windows. The JVM cost stays — only the
        uiautomator2 backend avoids that, since it talks to an agent already
        running on the device.
        """
        if not paths:
            return True

        if self.backend == "u2" and self._u2 is not None:
            for pts in paths:
                duration = max(min_ms, int(ms_per_point * len(pts)))
                per_step = max(0.02, (duration / 1000.0) / max(1, len(pts) - 1))
                self._u2.swipe_points([(int(x), int(y)) for x, y in pts], per_step)
                time.sleep(gap_ms / 1000.0)
            return True

        parts = []
        for pts in paths:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            ms = max(min_ms, int(ms_per_point * len(pts)))
            parts.append(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {ms}")
        script = f"; sleep {gap_ms / 1000.0:.2f}; ".join(parts)

        total_ms = sum(max(min_ms, ms_per_point * len(p)) for p in paths)
        timeout = 30 + (total_ms + gap_ms * len(paths)) / 1000.0
        try:
            subprocess.run(self._cmd("shell", script),
                           capture_output=True, timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def tap(self, x: int, y: int) -> None:
        if self.backend == "u2" and self._u2 is not None:
            self._u2.click(int(x), int(y))
        else:
            subprocess.run(
                self._cmd("shell", "input", "tap", str(int(x)), str(int(y))),
                capture_output=True, timeout=20,
            )


class ReplayDevice:
    """Stand-in that replays PNG/JPG stills instead of talking to a phone.

    Lets every stage downstream of capture be developed and tested offline
    against frames pulled out of a screen recording. Gestures are logged.
    """

    def __init__(self, frames: Sequence[str]) -> None:
        self.frames = list(frames)
        self.index = 0
        self.gestures: list[list[tuple[int, int]]] = []
        self.backend = "replay"
        self.serial = "replay"
        self._capture = "replay"
        self._capture_timings: dict[str, float] = {}

    def check(self) -> str:
        return "replay"

    def screencap(self, retries: int = 2) -> np.ndarray:
        if not self.frames:
            raise DeviceError("ReplayDevice has no frames")
        path = self.frames[min(self.index, len(self.frames) - 1)]
        img = cv2.imread(path)
        if img is None:
            raise DeviceError(f"could not read frame {path}")
        return img

    def swipe_path(self, points, ms_per_point: int = 55, min_ms: int = 320) -> None:
        self.gestures.append([(int(x), int(y)) for x, y in points])
        self.index = min(self.index + 1, len(self.frames) - 1)

    def swipe_many(self, paths, **kw) -> bool:
        for pts in paths:
            self.swipe_path(pts)
        return True

    def tap(self, x: int, y: int) -> None:
        self.gestures.append([(int(x), int(y))])
