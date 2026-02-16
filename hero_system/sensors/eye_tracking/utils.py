"""
Eye Tracking Utility Functions
Rotation matrices, normalization, and coordinate transformations
Directly from original implementation
"""

import math
import numpy as np


def rot_x(a: float) -> np.ndarray:
    """
    Rotation matrix around X-axis

    Args:
        a: Angle in radians

    Returns:
        3x3 rotation matrix
    """
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ], dtype=float)


def rot_y(a: float) -> np.ndarray:
    """
    Rotation matrix around Y-axis

    Args:
        a: Angle in radians

    Returns:
        3x3 rotation matrix
    """
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector

    Args:
        v: Vector to normalize

    Returns:
        Normalized vector (or original if norm too small)
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_scale(points_3d: np.ndarray) -> float:
    """
    Compute average pairwise distance between 3D points
    Used for scale-invariant head tracking

    Args:
        points_3d: Nx3 array of 3D points

    Returns:
        Average pairwise distance
    """
    n = len(points_3d)
    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1

    return total / count if count > 0 else 1.0


def draw_wireframe_cube(
        frame: np.ndarray,
        center: np.ndarray,
        R: np.ndarray,
        size: int = 80,
        color: tuple = (255, 128, 0),
        thickness: int = 2
):
    """
    Draw a wireframe cube aligned with rotation matrix
    Used for head pose visualization

    Args:
        frame: Image to draw on
        center: 3D center position
        R: 3x3 rotation matrix
        size: Cube size
        color: BGR color tuple
        thickness: Line thickness
    """
    import cv2

    # Coordinate frame from rotation matrix
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (
                center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward
        )

    # Generate 8 corners
    corners = [
        corner(x, y, z)
        for x in [-1, 1]
        for y in [1, -1]
        for z in [-1, 1]
    ]

    # Project to 2D
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Define cube edges
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Front face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
    ]

    # Draw edges
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], color, thickness)


def draw_coordinate_axes(
        frame: np.ndarray,
        center: np.ndarray,
        R: np.ndarray,
        size: int = 80
):
    """
    Draw RGB coordinate axes (X=red, Y=green, Z=blue)

    Args:
        frame: Image to draw on
        center: 3D center position
        R: 3x3 rotation matrix
        size: Axis length
    """
    import cv2

    axis_length = size * 1.2

    # Axis directions from rotation matrix
    axis_dirs = [
        R[:, 0],  # X-axis (right)
        -R[:, 1],  # Y-axis (up)
        -R[:, 2]  # Z-axis (forward)
    ]

    # RGB colors for XYZ
    axis_colors = [
        (0, 0, 255),  # Red for X
        (0, 255, 0),  # Green for Y
        (255, 0, 0)  # Blue for Z
    ]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(
            frame,
            (int(center[0]), int(center[1])),
            (int(end_pt[0]), int(end_pt[1])),
            axis_colors[i],
            2
        )
