"""
Eye Tracking Sensor Module for HERO System
Real-time gaze tracking using ArduCam Pinsight AI and MediaPipe FaceMesh

Architecture:
- processor.py: Camera initialization, gaze computation, database storage
- calibrator.py: Interactive calibration workflow
- config.py: Configuration parameters
- utils.py: Mathematical utilities

Usage:
    # Calibration (in Calibration UI)
    calibrator = EyeTrackingCalibrator(session_id, db_session)
    calibrator.start()

    while calibrating:
        frame = calibrator.get_frame()
        # Display frame, handle user input
        if user_presses_c:
            calibrator.calibrate_eye_spheres()
        if user_presses_s:
            calibrator.calibrate_screen_center()

    calibration_data = calibrator.get_calibration_data()
    calibrator.stop()

    # Session tracking (in Main App)
    processor = EyeTrackingProcessor(session_id, db_session, coordinator, config)
    processor.load_calibration(calibration_data)
    processor.start()

    # Games can access gaze
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
    # Main classes
    'EyeTrackingProcessor',
    'EyeTrackingCalibrator',
    'EyeTrackingConfig',

    # Utility functions
    'rot_x',
    'rot_y',
    'normalize',
    'compute_scale',
    'draw_wireframe_cube',
    'draw_coordinate_axes',
]

__version__ = '1.0.0'