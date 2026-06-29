#!/usr/bin/env python3
"""Benchmark Sharpa fingertip tactile sampling.

Standalone: talks to the Sharpa SDK directly (like read_tactile.py), no
retargeting_teleop import. Measures, per channel:

  - poll_hz : how many fetch_tactile_frame() calls/sec we can do (our ceiling)
  - frame_hz: distinct device frames/sec (the sensor's true stream rate, from
              counting unique frame "ts" timestamps) -> THIS is the real limit
  - latency : mean age of frames (now - ts)

Use the smallest frame_hz across the channels you care about as the safe max for
the teleop tactile/force sampling rate (force_hz). Polling faster than frame_hz
just re-reads the same frame.

Usage (from anywhere):
    python3 sharpa_teleop/sharpa_controller/tools/tactile_benchmark.py
    python3 .../tactile_benchmark.py --duration 5 --fingers index,thumb --calibrate
"""

import argparse
import sys
import time
from pathlib import Path

# Bootstrap the SDK import path (adds sharpa_controller/ so `import sharpa_hand`
# wires up /opt/sharpa-wave-sdk), exactly like read_tactile.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sharpa_hand  # noqa: F401  # bootstrap SDK paths

from sharpa import DeviceType, HandSide, SharpaWaveManager  # noqa: E402

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb"]


def connect_hand(serial, discovery_timeout_s):
    manager = SharpaWaveManager.get_instance()
    deadline = time.time() + discovery_timeout_s
    while time.time() < deadline:
        hands = [
            info
            for info in manager.get_all_devices()
            if info.device_type == DeviceType.HAND
        ]
        if hands:
            sns = [info.sn for info in hands]
            target = serial if serial in sns else sns[0]
            if serial and serial not in sns:
                raise RuntimeError(f"Device {serial!r} not found. Available: {sns}")
            print(f"Connecting to {target}")
            hand = manager.connect(target)
            if hand is None:
                raise RuntimeError(f"Failed to connect to {target}")
            return manager, hand, hand.get_device_info()
        time.sleep(0.5)
    raise RuntimeError(f"No Sharpa hand found within {discovery_timeout_s:.0f}s")


def channel_for_finger(hand_side, finger):
    base = FINGER_NAMES.index(finger)
    return base + 5 if hand_side == HandSide.LEFT else base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Device serial (default: first found)")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to sample")
    parser.add_argument(
        "--fingers",
        default="index,thumb",
        help="Comma list of fingers to benchmark (default: index,thumb)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.005,
        help="Per-fetch timeout in seconds (default: 0.005)",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate tactile before benchmark"
    )
    parser.add_argument("--discovery-timeout", type=float, default=10.0)
    args = parser.parse_args()

    fingers = [f.strip() for f in args.fingers.split(",") if f.strip()]
    unknown = [f for f in fingers if f not in FINGER_NAMES]
    if unknown:
        raise SystemExit(f"Unknown finger(s) {unknown}; choose from {FINGER_NAMES}")

    manager, hand, info = connect_hand(args.serial, args.discovery_timeout)
    if not hand.start():
        manager.disconnect_all()
        raise RuntimeError("Failed to start hand (check tactile port / host binding)")

    try:
        if args.calibrate:
            print("Calibrating tactile sensors...")
            if not hand.calib_tactile():
                raise RuntimeError("Tactile calibration failed")
            print("Calibration complete")

        channels = {f: channel_for_finger(info.hand_side, f) for f in fingers}
        print(
            f"Benchmarking {args.duration:.1f}s on {info.sn} "
            f"({'left' if info.hand_side == HandSide.LEFT else 'right'} hand): "
            f"{channels}"
        )

        polls = {f: 0 for f in fingers}
        seen_ts = {f: set() for f in fingers}
        latency_sum = {f: 0.0 for f in fingers}
        latency_n = {f: 0 for f in fingers}

        end = time.time() + args.duration
        while time.time() < end:
            for f, ch in channels.items():
                frame = hand.fetch_tactile_frame(ch, args.timeout)
                polls[f] += 1
                if not frame or "content" not in frame:
                    continue
                ts = frame.get("ts")
                if ts is not None:
                    seen_ts[f].add(ts)
                    latency_sum[f] += time.time() - ts
                    latency_n[f] += 1

        dur = args.duration
        print("\nResults:")
        print(f"  {'finger':<8} {'poll_hz':>9} {'frame_hz':>9} {'latency_ms':>11}")
        for f in fingers:
            poll_hz = polls[f] / dur
            frame_hz = len(seen_ts[f]) / dur
            lat = (latency_sum[f] / latency_n[f] * 1000) if latency_n[f] else float("nan")
            print(f"  {f:<8} {poll_hz:>9.1f} {frame_hz:>9.1f} {lat:>11.1f}")

        worst = min((len(seen_ts[f]) / dur) for f in fingers)
        print(
            f"\nSafe max tactile/force sampling rate (force_hz) <= {worst:.0f} Hz "
            f"(slowest channel's true frame rate)."
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        hand.stop()
        manager.disconnect_all()


if __name__ == "__main__":
    main()
