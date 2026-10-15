#!/usr/bin/env python3
"""Measure where the time per word actually goes on your setup.

    python scripts/benchmark.py

Answers three questions:

  * is the cable (or port) the bottleneck?
  * is PNG or raw capture faster on this phone?
  * how much is `adb shell input swipe` costing?

That last one is usually the surprise. `input` is a Java program, so Android
starts a JVM for every single call. On many devices that is most of a second
before the drag even begins, which no cable can fix — the uiautomator2 backend
is what removes it, because it talks to an agent already running on the phone.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsbot.adb import Device, DeviceError      # noqa: E402
from wsbot.config import Config                # noqa: E402

N = 5


def timed(fn, n=N):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        out.append((time.perf_counter() - t0) * 1000)
    return out


def show(label: str, ms, extra: str = "") -> float | None:
    if not ms:
        print(f"  {label:<34} failed")
        return None
    med = statistics.median(ms)
    print(f"  {label:<34} {med:7.0f} ms   (min {min(ms):.0f}, max {max(ms):.0f}){extra}")
    return med


def main() -> int:
    cfg = Config.load()
    try:
        dev = Device(serial=cfg.serial, backend=cfg.backend)
        serial = dev.check()
    except DeviceError as e:
        print(f"device error: {e}")
        return 2

    print(f"device {serial}, backend {dev.backend}\n")

    print("adb round trip (process spawn + USB latency)")
    base = show("adb shell echo", timed(
        lambda: subprocess.run(dev._cmd("shell", "echo", "x"),
                               capture_output=True, timeout=20)))

    print("\nscreen capture")
    png = show("screencap -p  (phone encodes PNG)", timed(dev._screencap_png, 3))
    raw = show("screencap     (raw RGBA)", timed(dev._screencap_raw, 3))

    img = dev.screencap()
    h, w = img.shape[:2]
    raw_mb = w * h * 4 / 1e6
    print(f"\n  screen {w}x{h}: raw payload {raw_mb:.1f} MB")
    if raw and base:
        transfer = max(1.0, raw - base)
        tput = raw_mb / (transfer / 1000)
        print(f"  implied USB throughput        {tput:6.1f} MB/s")
        if tput < 8:
            print("     -> very slow. A USB 2 cable/port tops out near 30-40 MB/s;")
            print("        under 8 suggests a poor cable, an extension, a hub, or")
            print("        a port that negotiated down. Try a different cable and")
            print("        a port directly on the machine.")
        elif tput < 25:
            print("     -> USB 2 territory. A USB 3 port and cable would roughly")
            print("        halve the raw-capture time.")
        else:
            print("     -> healthy; the cable is not your bottleneck.")

    print("\ngesture")
    sw = show("adb shell input swipe", timed(
        lambda: subprocess.run(
            dev._cmd("shell", "input", "swipe", str(w // 2), str(h // 2),
                     str(w // 2 + 10), str(h // 2), "60"),
            capture_output=True, timeout=20), 3))
    if sw and base:
        jvm = sw - base - 60
        print(f"  of which JVM startup for `input` ~{max(0, jvm):.0f} ms")
        if jvm > 300:
            print("     -> this is pure overhead and the single biggest win available.")
            print("        pip install uiautomator2 && python -m uiautomator2 init")
            print("        then run with --backend u2 (or let auto pick it).")

    try:
        import uiautomator2            # noqa: F401
        print("\n  uiautomator2 is installed"
              + ("" if dev.backend == "u2" else " but not in use — pass --backend u2"))
    except ImportError:
        print("\n  uiautomator2 is NOT installed — installing it is likely the "
              "largest single speedup")

    best_cap = min([x for x in (png, raw) if x], default=0)
    per_word = (sw or 0) + best_cap
    print(f"\nestimated floor per word with the current setup: {per_word / 1000:.1f}s")
    print("  (one gesture + one capture; the bot now does exactly this)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
