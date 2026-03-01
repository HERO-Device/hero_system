"""
Eye Tracking Sensor Module for HERO System
Real-time gaze tracking using ArduCam Pinsight AI and MediaPipe FaceMesh

Architecture:
- EyeTrackingCalibrator: Interactive 9-point calibration, saves coefficients to DB
- GazeSystem:            Pygame-based calibration and collection for CalibrationScreen
- EyeTrackingProcessor:  Loads calibration from DB, runs live gaze collection
- EyeTrackingConfig:     Configuration parameters
- utils:                 Shared frame correction and feature extraction utilities

Usage:
    # Calibration (via orchestrator.py)
    calibrator = EyeTrackingCalibrator(session_id, db_session)
    calibrator.start()
    calibrator.run_calibration()
    calibrator.save_to_database()
    calibrator.stop()

    # Session tracking (via pipeline.py)
    calibration_data = EyeTrackingCalibrator.load_from_database(db_session, session_id)
    processor = EyeTrackingProcessor(session_id, db_session, coordinator, config)
    processor.load_calibration(calibration_data)
    processor.start()
    gaze_x, gaze_y = processor.get_current_gaze()
    processor.stop()
"""

from .config import EyeTrackingConfig
from .processor import EyeTrackingProcessor
from .calibrator import EyeTrackingCalibrator
from .utils import (
    rot_x,
    rot_y,
    normalize,
    compute_scale,
    draw_wireframe_cube,
    draw_coordinate_axes
)

__all__ = [
    'EyeTrackingProcessor',
    'EyeTrackingCalibrator',
    'EyeTrackingConfig',
]

__version__ = '1.0.0'
