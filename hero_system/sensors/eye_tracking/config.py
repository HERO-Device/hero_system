"""
Eye Tracking Sensor Configuration
Mimics original setup with database integration
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EyeTrackingConfig:
    """Eye tracking configuration - based on original implementation"""

    # Operating mode
    mode: str = 'session'  # 'calibration' or 'session'

    # Smoothing and visualization
    filter_length: int = 10        # Frames for gaze smoothing
    gaze_length: int = 350         # Gaze ray visualization length

    # Virtual screen dimensions
    virtual_width: int = 1920
    virtual_height: int = 1080

    # Camera physical setup
    camera_tilt_angle: float = 25.0      # degrees upward tilt
    camera_below_screen: int = 60        # pixels below screen bottom edge
    screen_height_pixels: int = 1080     # Full screen height

    # Field of view parameters (from original convert_gaze_to_screen_coordinates)
    yaw_degrees: float = 15.0      # ±15° horizontal range (5 * 3)
    pitch_degrees: float = 5.0     # ±5° vertical range (2.0 * 2.5)

    # Eye sphere tracking
    base_radius: int = 20          # Base radius for eye sphere

    # MediaPipe FaceMesh settings
    mp_static_image_mode: bool = False
    mp_max_num_faces: int = 1
    mp_refine_landmarks: bool = True
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5

    # Landmark indices
    left_iris_idx: int = 468
    right_iris_idx: int = 473

    # Nose landmarks for head pose (from original)
    nose_indices: Tuple[int, ...] = (
        4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
        461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
        3, 248
    )

    # Camera settings for DepthAI
    preview_width: int = 640
    preview_height: int = 480

    # Database settings
    batch_commit_size: int = 30  # Commit every N samples

    @property
    def center_x(self) -> int:
        return self.virtual_width // 2

    @property
    def center_y(self) -> int:
        return self.virtual_height // 2

    @property
    def camera_vertical_offset_pixels(self) -> int:
        """Total vertical offset from camera to screen center"""
        return (self.screen_height_pixels // 2) + self.camera_below_screen

    @classmethod
    def for_calibration(cls) -> 'EyeTrackingConfig':
        """Configuration for calibration phase"""
        return cls(mode='calibration')

    @classmethod
    def for_session(cls) -> 'EyeTrackingConfig':
        """Configuration for active session phase"""
        return cls(mode='session')
    