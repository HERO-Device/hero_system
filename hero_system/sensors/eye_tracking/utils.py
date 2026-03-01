"""
Eye Tracking Shared Utilities
Used by calibrator.py and processor.py
"""

import cv2
import numpy as np
from .config import EyeTrackingConfig


def correct_frame(frame: np.ndarray, flip_code: int) -> np.ndarray:
    """
    Flip a camera frame to correct for physical mounting orientation.

    Args:
        frame:     BGR numpy array from the DepthAI queue.
        flip_code: cv2.flip code — 0=vertical, 1=horizontal, -1=both.

    Returns:
        Flipped BGR numpy array.
    """
    return cv2.flip(frame, flip_code)

def extract_features(landmarks, img_w: int, img_h: int, config: EyeTrackingConfig) -> np.ndarray:
    """
    Compute a 4-element normalised iris feature vector from FaceMesh landmarks.

    Args:
        landmarks: MediaPipe face landmark list.
        img_w:     Frame width in pixels.
        img_h:     Frame height in pixels.
        config:    EyeTrackingConfig providing iris and eye corner landmark indices.

    Returns:
        Numpy array [left_x, left_y, right_x, right_y] of normalised iris positions.
    """
    def norm_iris(iris_idx, inner_idx, outer_idx):
        iris  = landmarks[iris_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        ix, iy = iris.x * img_w,  iris.y * img_h
        inx    = inner.x * img_w
        outx   = outer.x * img_w
        eye_w  = abs(inx - outx) + 1e-6
        eye_cy = (inner.y + outer.y) / 2 * img_h
        return (ix - outx) / eye_w, ((iy - eye_cy) * 5) / eye_w

    lx, ly = norm_iris(config.left_iris_idx,  config.left_eye_inner,  config.left_eye_outer)
    rx, ry = norm_iris(config.right_iris_idx, config.right_eye_inner, config.right_eye_outer)
    return np.array([lx, ly, rx, ry], dtype=np.float32)

# Legacy stubs — retained so any stale imports do not raise AttributeError.
# These functions are not used in the current implementation.
def rot_x(a): pass
def rot_y(a): pass
def normalize(v): return v
def compute_scale(pts): return 1.0
def draw_wireframe_cube(*a, **k): pass
def draw_coordinate_axes(*a, **k): pass
