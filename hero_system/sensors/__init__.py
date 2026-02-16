"""
HERO System Sensors
Multi-modal biosensor data collection
"""

from .eeg import EEGCollector, EEGProcessor, EEGConfig
from .eye_tracking import EyeTrackingProcessor, EyeTrackingCalibrator, EyeTrackingConfig

# TODO: Add other sensors as they're implemented
# from .heart_rate import HeartRateSensor
# from .pulse_oximeter import PulseOximeterSensor
# from .accelerometer import AccelerometerSensor
# from .gyroscope import GyroscopeSensor

__all__ = [
    # EEG
    'EEGCollector',
    'EEGProcessor',
    'EEGConfig',

    # Eye Tracking
    'EyeTrackingProcessor',
    'EyeTrackingCalibrator',
    'EyeTrackingConfig',
]
