"""Joint names and angle limits for the Sharpa Wave hand (22 DOF)."""

import math

JOINT_NAMES = [
    "Thumb CMC Flexion/Extension",
    "Thumb CMC Abduction/Adduction",
    "Thumb MCP Flexion/Extension",
    "Thumb MCP Abduction/Adduction",
    "Thumb DIP Flexion/Extension",
    
    "Index MCP Flexion/Extension",
    "Index MCP Abduction/Adduction",
    "Index PIP Flexion/Extension",
    "Index DIP Flexion/Extension",
    
    "Middle MCP Flexion/Extension",
    "Middle MCP Abduction/Adduction",
    "Middle PIP Flexion/Extension",
    "Middle DIP Flexion/Extension",
    
    "Ring MCP Flexion/Extension",
    "Ring MCP Abduction/Adduction",
    "Ring PIP Flexion/Extension",
    "Ring DIP Flexion/Extension",
    
    "Pinky CMC Flexion/Extension",
    "Pinky MCP Flexion/Extension",
    "Pinky MCP Abduction/Adduction",
    "Pinky PIP Flexion/Extension",
    "Pinky DIP Flexion/Extension",
]

# Angle ranges in degrees (from Sharpa SDK example).
ANGLE_RANGES_DEG = [
    (0, 50),    # Thumb CMC Flexion/Extension
    (0, 10),    # Thumb CMC Abduction/Adduction
    (0, 30),    # Thumb MCP Flexion/Extension
    (0, 10),    # Thumb MCP Abduction/Adduction
    (0, 40),    # Thumb DIP Flexion/Extension

    (0, 40),    # (0, 20) # Index MCP Flexion/Extension 
    (-20, 20),  # Index MCP Abduction/Adduction
    (0, 60),    # (0, 20) # Index PIP Flexion/Extension
    (0, 60),    # (0, 20) # Index DIP Flexion/Extension

    (0, 20),    # Middle MCP Flexion/Extension
    (-20, 20),  # Middle MCP Abduction/Adduction
    (0, 20),    # Middle PIP Flexion/Extension
    (0, 20),    # Middle DIP Flexion/Extension

    (0, 20),    # Ring MCP Flexion/Extension
    (-20, 20),  # Ring MCP Abduction/Adduction
    (0, 20),    # Ring PIP Flexion/Extension
    (0, 20),    # Ring DIP Flexion/Extension

    (0, 10),    # Pinky CMC Flexion/Extension
    (0, 20),    # Pinky MCP Flexion/Extension
    (-20, 20),  # Pinky MCP Abduction/Adduction
    (0, 20),    # Pinky PIP Flexion/Extension
    (0, 20),    # Pinky DIP Flexion/Extension
]

NUM_JOINTS = len(JOINT_NAMES)

JOINT_NAME_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def angle_limits_rad(joint_index: int) -> tuple[float, float]:
    """Return (min, max) joint angle in radians for a joint index."""
    lo_deg, hi_deg = ANGLE_RANGES_DEG[joint_index]
    return math.radians(lo_deg), math.radians(hi_deg)


def clamp_angle_rad(joint_index: int, angle_rad: float) -> float:
    """Clamp angle to the SDK-reported joint limits."""
    lo, hi = angle_limits_rad(joint_index)
    return max(lo, min(hi, angle_rad))
