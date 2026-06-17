#!/usr/bin/env python3
"""Read and print per-joint torques from a connected Sharpa Wave hand."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sharpa_hand import SharpaHand  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        help="Device serial number (default: first discovered device)",
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
        "--velocities",
        action="store_true",
        help="Also print joint velocities from the same state packet",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for device discovery (default: 10)",
    )
    args = parser.parse_args()

    hand = SharpaHand(
        serial=args.serial,
        discovery_timeout_s=args.discovery_timeout,
    )
    hand.connect()
    hand.configure()
    hand.start()

    period = 1.0 / args.rate if args.rate > 0 else 0.0

    try:
        while True:
            state = hand.read_state()

            print("\nJoint torques (Nm):")
            print(SharpaHand.format_joints(state.torques, "Nm"))

            if args.velocities:
                print("\nJoint velocities (rad/s):")
                print(SharpaHand.format_joints(state.velocities, "rad/s"))

            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
