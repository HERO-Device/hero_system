"""
Eye Tracking Configuration
9-point polynomial regression approach
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EyeTrackingConfig:
    """Eye tracking configuration - polynomial regression implementation"""

    # Operating mode
    mode: str = 'session'  # 'calibration' or 'session'

    # Calibration grid
    samples_per_point: int = 60
    dot_radius: int = 18
    cursor_radius: int = 20
    
    # Validation
    screen_distance_mm = 530.0  # Eye-to-screen distance (mm)
    screen_width_mm = 527.0     # Physical screen width (mm)
    validation_samples_per_point = 40

    # Gaze smoothing (EMA)
    smoothing_alpha: float = 0.1

    # Camera orientation: 0=flip vertical, 1=horizontal, -1=both
    camera_flip_code: int = 0

    # Polynomial regression
    poly_degree: int = 2
    ridge_alpha: float = 0.01

    # Iris landmark indices (MediaPipe FaceMesh, refine_landmarks=True)
    left_iris_idx: int = 468
    right_iris_idx: int = 473

    # Eye corner landmarks for iris normalisation
    left_eye_inner: int = 133
    left_eye_outer: int = 33
    right_eye_inner: int = 362
    right_eye_outer: int = 263

    # MediaPipe FaceMesh settings
    mp_max_num_faces: int = 1
    mp_refine_landmarks: bool = True
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5

    # Camera settings for DepthAI
    preview_width: int = 640
    preview_height: int = 480

    # GPIO button pin for calibration capture
    gpio_button_pin: int = 23

    # Database batch commit size (session mode)
    batch_commit_size: int = 30

    # Validation
    validation_samples: int = 40
    screen_distance_mm: float = 430.0
    screen_width_mm: float = 222.0

    # Calibration point order: centre first, then corners, then edges
    calib_order: Tuple[int, ...] = (4, 0, 8, 2, 6, 1, 7, 3, 5)

    @classmethod
    def for_calibration(cls) -> 'EyeTrackingConfig':
        return cls(mode='calibration')

    @classmethod
    def for_session(cls) -> 'EyeTrackingConfig':
        return cls(mode='session')
        
