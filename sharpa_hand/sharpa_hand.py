"""High-level wrapper around the Sharpa Wave SDK."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

_SDK_PYTHON = "/opt/sharpa-wave-sdk/python"
_SDK_LIB = "/opt/sharpa-wave-sdk/lib"
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)
os.environ["LD_LIBRARY_PATH"] = f"{_SDK_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from sharpa import ControlMode, ControlSource, SharpaWaveManager  # noqa: E402

from .constants import (  # noqa: E402
    JOINT_NAME_TO_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    clamp_angle_rad,
)


@dataclass
class SharpaJointState:
    """Full-hand joint state (length NUM_JOINTS)."""

    angles: list[float]
    velocities: list[float]
    torques: list[float]
    timestamp: float = 0.0


class SharpaHand:
    """Connect, configure, and control a Sharpa Wave hand."""

    def __init__(
        self,
        serial: str | None = None,
        enabled_joints: list[str] | None = None,
        speed_coeff: float = 0.3,
        current_coeff: float = 0.6,
        discovery_timeout_s: float = 10.0,
        io_frequency_hz: float = 400.0,
        verbose: bool = True,
    ):
        self.serial = serial
        self.speed_coeff = speed_coeff
        self.current_coeff = current_coeff
        self.discovery_timeout_s = discovery_timeout_s
        self.io_frequency_hz = io_frequency_hz
        self.verbose = verbose

        self._manager: SharpaWaveManager | None = None
        self._hand = None
        self._running = False
        self._loop_thread: threading.Thread | None = None
        self._loop_stop = threading.Event()

        self._io_thread: threading.Thread | None = None
        self._io_stop = threading.Event()
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._latest_state: SharpaJointState | None = None
        self._command_vector = [0.0] * NUM_JOINTS

        self._enabled_mask = [False] * NUM_JOINTS
        self._hold_positions_rad = [0.0] * NUM_JOINTS

        if enabled_joints:
            self.set_enabled_joints(enabled_joints)

    @property
    def hand(self):
        if self._hand is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._hand

    def connect(self) -> None:
        """Discover and connect to the hand."""
        if self._hand is not None:
            return

        self._manager = SharpaWaveManager.get_instance()
        deadline = time.time() + self.discovery_timeout_s

        while time.time() < deadline:
            devices = self._manager.get_all_device_sn()
            if devices:
                if self.serial and self.serial not in devices:
                    raise RuntimeError(
                        f"Device {self.serial!r} not found. Available: {devices}"
                    )
                target = self.serial if self.serial in devices else devices[0]
                if self.verbose:
                    print(f"Connecting to Sharpa hand {target}")
                self._hand = self._manager.connect(target)
                if self._hand is None:
                    raise RuntimeError(f"Failed to connect to {target}")
                return
            time.sleep(0.5)

        raise RuntimeError(
            f"No Sharpa Wave device found within {self.discovery_timeout_s:.0f}s"
        )

    def configure(self, control_mode=ControlMode.POSITION) -> None:
        """Apply SDK control settings (call before start())."""
        hand = self.hand

        steps = [
            ("control mode", hand.set_control_mode(control_mode)),
            ("speed coeff", hand.set_speed_coeff(self.speed_coeff)),
            ("current coeff", hand.set_current_coeff(self.current_coeff)),
            ("control source", hand.set_control_source(ControlSource.SDK)),
        ]
        for label, error in steps:
            if error.code != 0:
                raise RuntimeError(f"Failed to set {label}: {error.message}")

    def start(self) -> None:
        """Enter streaming / running mode and start the IO thread."""
        if self._running:
            return
        self.hand.start()
        self._running = True
        self._capture_hold_positions()
        with self._command_lock:
            self._command_vector = list(self._hold_positions_rad)
        self._start_io_loop()

    def stop(self) -> None:
        """Stop streaming and the IO thread."""
        if not self._running:
            return
        self.stop_loop()
        self._stop_io_loop()
        self.hand.stop()
        self._running = False

    def disconnect(self) -> None:
        """Stop and disconnect from the device."""
        self.stop()
        if self._manager is not None:
            self._manager.disconnect_all()
        self._manager = None
        self._hand = None

    def set_enabled_joints(self, names: list[str]) -> None:
        """Enable position control only on the named joints."""
        unknown = [n for n in names if n not in JOINT_NAME_TO_INDEX]
        if unknown:
            raise ValueError(f"Unknown joint names: {unknown}")

        self._enabled_mask = [False] * NUM_JOINTS
        for name in names:
            self._enabled_mask[JOINT_NAME_TO_INDEX[name]] = True

        if self._hand is not None and self._running:
            self._capture_hold_positions()
            with self._command_lock:
                self._command_vector = list(self._hold_positions_rad)

    def _capture_hold_positions(self) -> None:
        """Snapshot current angles for disabled joints."""
        state = self._read_state_from_sdk()
        for i in range(NUM_JOINTS):
            if not self._enabled_mask[i]:
                self._hold_positions_rad[i] = state.angles[i]

    def _read_state_from_sdk(self) -> SharpaJointState:
        """Read angles, velocities, and torques directly from the SDK."""
        state = self.hand.get_states()
        angles = list(state.angles) if state.angles else [0.0] * NUM_JOINTS
        velocities = list(state.velocities) if state.velocities else [0.0] * NUM_JOINTS
        torques = list(state.torques) if state.torques else [0.0] * NUM_JOINTS

        if len(angles) < NUM_JOINTS:
            angles.extend([0.0] * (NUM_JOINTS - len(angles)))
        if len(velocities) < NUM_JOINTS:
            velocities.extend([0.0] * (NUM_JOINTS - len(velocities)))
        if len(torques) < NUM_JOINTS:
            torques.extend([0.0] * (NUM_JOINTS - len(torques)))

        return SharpaJointState(
            angles=angles[:NUM_JOINTS],
            velocities=velocities[:NUM_JOINTS],
            torques=torques[:NUM_JOINTS],
            timestamp=float(getattr(state, "timestamp", 0.0)),
        )

    def read_state(self) -> SharpaJointState:
        """Return the latest state from the IO thread (non-blocking)."""
        with self._state_lock:
            if self._latest_state is not None:
                s = self._latest_state
                return SharpaJointState(
                    angles=list(s.angles),
                    velocities=list(s.velocities),
                    torques=list(s.torques),
                    timestamp=s.timestamp,
                )
        if self._hand is not None and self._running:
            return self._read_state_from_sdk()
        raise RuntimeError("Not connected. Call connect() and start() first.")

    def _send_command_vector_to_sdk(
        self,
        command_vector: list[float],
        *,
        interpolate: bool = True,
    ) -> None:
        error = self.hand.set_joint_position(command_vector, interpolate)
        if error.code != 0:
            raise RuntimeError(f"set_joint_position failed: {error.message}")

    def send_positions(
        self,
        targets: dict[str, float],
        *,
        interpolate: bool = True,
    ) -> None:
        """Update position targets; the IO thread sends them at io_frequency_hz."""
        del interpolate  # IO loop always uses interpolation for smooth streaming.
        with self._command_lock:
            full = list(self._command_vector)
            for name, angle_rad in targets.items():
                idx = JOINT_NAME_TO_INDEX[name]
                if not self._enabled_mask[idx]:
                    raise ValueError(f"Joint {name!r} is not enabled")
                full[idx] = clamp_angle_rad(idx, angle_rad)
            self._command_vector = full

    def _io_loop(self) -> None:
        period = 1.0 / self.io_frequency_hz
        while not self._io_stop.is_set():
            cycle_start = time.perf_counter()

            state = self._read_state_from_sdk()
            with self._state_lock:
                self._latest_state = state

            with self._command_lock:
                command = list(self._command_vector)
            self._send_command_vector_to_sdk(command, interpolate=True)

            elapsed = time.perf_counter() - cycle_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _start_io_loop(self) -> None:
        if self._io_thread is not None and self._io_thread.is_alive():
            return
        self._io_stop.clear()
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self._io_thread.start()
        if self.verbose:
            print(f"Sharpa IO loop started at {self.io_frequency_hz} Hz")

    def _stop_io_loop(self) -> None:
        self._io_stop.set()
        if self._io_thread is not None:
            self._io_thread.join(timeout=2.0)
            self._io_thread = None

    def run_loop(
        self,
        callback: Callable[[], None],
        rate_hz: float = 100.0,
        *,
        blocking: bool = False,
    ) -> None:
        """Run callback at fixed rate in a background thread (or blocking)."""
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")

        period = 1.0 / rate_hz

        def _loop() -> None:
            while not self._loop_stop.is_set():
                cycle_start = time.perf_counter()
                callback()
                elapsed = time.perf_counter() - cycle_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self._loop_stop.clear()
        if blocking:
            _loop()
            return

        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("Control loop already running")

        self._loop_thread = threading.Thread(target=_loop, daemon=True)
        self._loop_thread.start()

    def stop_loop(self) -> None:
        """Stop a background control loop started with run_loop()."""
        self._loop_stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None

    @staticmethod
    def format_joints(values: list[float], units: str) -> str:
        lines = []
        for i, value in enumerate(values):
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}"
            lines.append(f"  [{i:2d}] {name:<35} {value:8.3f} {units}")
        return "\n".join(lines)

    @classmethod
    def from_config(cls, sharpa_cfg, verbose: bool = True) -> SharpaHand:
        """Construct from a Hydra/OmegaConf sharpa config node."""
        serial = sharpa_cfg.get("serial")
        if serial in (None, "null"):
            serial = None
        return cls(
            serial=serial,
            enabled_joints=list(sharpa_cfg.get("enabled_joints", [])),
            speed_coeff=float(sharpa_cfg.get("speed_coeff", 0.3)),
            current_coeff=float(sharpa_cfg.get("current_coeff", 0.6)),
            discovery_timeout_s=float(sharpa_cfg.get("discovery_timeout_s", 10.0)),
            io_frequency_hz=float(sharpa_cfg.get("io_frequency_hz", 400.0)),
            verbose=verbose,
        )
