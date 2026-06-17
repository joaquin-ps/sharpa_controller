#!/usr/bin/env python3
"""Read and print tactile sensor data from a connected Sharpa Wave hand."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sharpa_hand  # noqa: F401  # bootstrap SDK

from sharpa import DeviceType, HandSide, SharpaWaveManager  # noqa: E402

# Channels 0-4 (right) and 5-9 (left): little finger to thumb
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb"]
F6_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def device_has_tactile(device_info) -> bool:
    checker = getattr(device_info, "has_fingertip_tactile", None)
    if callable(checker):
        return bool(checker())
    return bool(checker)


def connect_hand(serial: str | None, discovery_timeout_s: float):
    manager = SharpaWaveManager.get_instance()
    deadline = time.time() + discovery_timeout_s

    while time.time() < deadline:
        hand_devices = [
            info
            for info in manager.get_all_devices()
            if info.device_type == DeviceType.HAND
        ]
        if hand_devices:
            sns = [info.sn for info in hand_devices]
            target = serial if serial in sns else sns[0]
            if serial and serial not in sns:
                raise RuntimeError(f"Device {serial!r} not found. Available: {sns}")

            print(f"Connecting to {target}")
            hand = manager.connect(target)
            if hand is None:
                raise RuntimeError(f"Failed to connect to {target}")

            info = hand.get_device_info()
            if not device_has_tactile(info):
                manager.disconnect_all()
                raise RuntimeError(
                    f"Device {target} does not support fingertip tactile sensors"
                )
            return manager, hand, info
        time.sleep(0.5)

    raise RuntimeError(
        f"No Sharpa Wave hand found within {discovery_timeout_s:.0f}s"
    )


def channels_for_hand(hand_side: HandSide) -> range:
    if hand_side == HandSide.LEFT:
        return range(5, 10)
    return range(0, 5)


def finger_name(hand_side: HandSide, channel: int) -> str:
    idx = channel - 5 if hand_side == HandSide.LEFT else channel
    return FINGER_NAMES[idx]


def hand_label(hand_side: HandSide) -> str:
    if hand_side == HandSide.LEFT:
        return "left"
    if hand_side == HandSide.RIGHT:
        return "right"
    return "unknown"


def as_list(values):
    if values is None:
        return None
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def format_f6(values) -> str:
    values = as_list(values) or []
    parts = []
    for label, value in zip(F6_LABELS, values):
        parts.append(f"{label}={float(value):.4f}")
    return ", ".join(parts)


def format_contact_points(points) -> str:
    points = as_list(points)
    if not points:
        return "none"

    if isinstance(points[0], (list, tuple)):
        formatted = []
        for point in points:
            if len(point) >= 3:
                formatted.append(
                    f"({float(point[0]):.1f}, {float(point[1]):.1f}, conf={float(point[2]):.3f})"
                )
        return "; ".join(formatted) if formatted else "none"

    if len(points) >= 3:
        return (
            f"({float(points[0]):.1f}, {float(points[1]):.1f}, "
            f"conf={float(points[2]):.3f})"
        )
    return str(points)


def describe_array(name: str, data) -> str:
    if data is None:
        return f"{name}: none"
    shape = getattr(data, "shape", None)
    if shape is not None:
        return f"{name}: shape={tuple(shape)}"
    return f"{name}: {type(data).__name__}"


def save_tactile_images(save_dir: Path, hand_side: HandSide, channel: int, content: dict):
    import cv2
    import numpy as np

    finger = finger_name(hand_side, channel)
    prefix = f"{hand_label(hand_side)}_{finger}_ch{channel}"

    raw = content.get("RAW")
    if raw is not None:
        img = np.asarray(raw).squeeze()
        if img.size == 76800:
            img = img.reshape(240, 320)
        elif img.size == 57600:
            img = img.reshape(240, 240)
        cv2.imwrite(str(save_dir / f"{prefix}_raw.png"), img.astype(np.uint8))

    deform = content.get("DEFORM")
    if deform is not None:
        img = np.asarray(deform).squeeze().reshape(240, 240)
        cv2.imwrite(str(save_dir / f"{prefix}_deform.png"), img.astype(np.uint8))


def read_tactile_frame(hand, channel: int, timeout_s: float):
    frame = hand.fetch_tactile_frame(channel, timeout_s)
    if frame is None:
        return None

    if not isinstance(frame, dict) or "content" not in frame:
        raise RuntimeError(f"Invalid tactile frame on channel {channel}")

    return frame


def print_frame(hand_side: HandSide, channel: int, frame: dict, verbose: bool):
    ts = frame.get("ts")
    latency = time.time() - ts if ts is not None else None
    content = frame["content"]
    finger = finger_name(hand_side, channel)

    print(f"  [{channel}] {finger}")
    if ts is not None:
        print(f"      timestamp: {ts:.6f}")
    if latency is not None:
        print(f"      latency:   {latency * 1000:.1f} ms")
    print(f"      F6:        {format_f6(content.get('F6'))}")
    print(f"      contact:   {format_contact_points(content.get('CONTACT_POINT'))}")

    if verbose:
        for key in ("RAW", "DEFORM", "F6", "CONTACT_POINT"):
            print(f"      {describe_array(key, content.get(key))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        help="Device serial number (default: first discovered hand)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Read rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Read once and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print array shapes for each tactile stream",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run tactile calibration before reading",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Save RAW/DEFORM images to this directory",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="Per-channel fetch timeout in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for device discovery (default: 10)",
    )
    args = parser.parse_args()

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    manager, hand, info = connect_hand(args.serial, args.discovery_timeout)
    hand_side = info.hand_side
    channels = list(channels_for_hand(hand_side))

    if not hand.start():
        manager.disconnect_all()
        raise RuntimeError("Failed to start hand (check tactile port / host binding)")

    if args.calibrate:
        print("Calibrating tactile sensors...")
        if not hand.calib_tactile():
            hand.stop()
            manager.disconnect_all()
            raise RuntimeError("Tactile calibration failed")
        print("Calibration complete")

    period = 1.0 / args.rate if args.rate > 0 else 0.0

    try:
        while True:
            print(f"\nTactile data ({hand_label(hand_side)} hand, sn={info.sn}):")
            for channel in channels:
                frame = read_tactile_frame(hand, channel, args.timeout)
                if frame is None:
                    print(f"  [{channel}] {finger_name(hand_side, channel)}: no frame")
                    continue

                print_frame(hand_side, channel, frame, args.verbose)
                if args.save_dir is not None:
                    save_tactile_images(args.save_dir, hand_side, channel, frame["content"])

            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        hand.stop()
        manager.disconnect_all()


if __name__ == "__main__":
    main()
