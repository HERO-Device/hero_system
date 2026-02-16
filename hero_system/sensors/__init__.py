"""
HERO System Sensors
Multi-modal biosensor data collection

Available Sensors:
- EEG: 4-channel OpenBCI Ganglion (200 Hz)
- Eye Tracking: ArduCam Pinsight AI with MediaPipe (~30 Hz)
- MAX30102: Heart rate and SpO2 pulse oximeter (~100 Hz)
- MPU6050: 3-axis accelerometer and gyroscope (100 Hz)

All sensors support:
- Dual-mode operation (calibration with real-time processing, session with raw collection)
- Coordinator-based timestamp synchronization
- Post-session processing and metric computation
- Quality assessment and validation
"""

from .eeg import EEGCollector, EEGProcessor, EEGConfig
from .eye_tracking import EyeTrackingProcessor, EyeTrackingCalibrator, EyeTrackingConfig
from .max30102 import MAX30102Collector, MAX30102Processor, MAX30102Config
from .mpu6050 import MPU6050Collector, MPU6050Processor, MPU6050Config

__all__ = [
    # EEG (4 channels)
    'EEGCollector',
    'EEGProcessor',
    'EEGConfig',

    # Eye Tracking
    'EyeTrackingProcessor',
    'EyeTrackingCalibrator',
    'EyeTrackingConfig',

    # MAX30102 (Heart Rate + SpO2)
    'MAX30102Collector',
    'MAX30102Processor',
    'MAX30102Config',

    # MPU6050 (Accelerometer + Gyroscope)
    'MPU6050Collector',
    'MPU6050Processor',
    'MPU6050Config',
]

__version__ = '1.0.0'
