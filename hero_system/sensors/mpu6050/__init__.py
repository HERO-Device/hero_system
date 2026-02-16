"""
MPU6050 Sensor Module for HERO System
3-axis Accelerometer and Gyroscope with tremor detection

Architecture:
- Collector: Raw X,Y,Z data collection for both accelerometer and gyroscope
- Processor: Tremor detection using bandpass filtering and PSD analysis

Dual-mode operation:
- Calibration mode: Real-time magnitude display for signal verification
- Session mode: Raw X,Y,Z collection only (process after session)

Post-processing capabilities:
- Tremor detection (4-6 Hz bandpass filtering)
- Power Spectral Density analysis using Welch's method
- Tremor classification (rotational vs linear)
- Peak frequency identification
- Tremor power ratio calculation

Usage:
    # Session collection
    collector = MPU6050Collector(session_id, db_session, coordinator)
    collector.start()
    # ... collect during session ...
    collector.stop()

    # Post-session processing
    processor = MPU6050Processor()
    metrics_count = processor.process_session_data(session_id, db_session)
"""

from .collector import MPU6050Collector
from .processor import MPU6050Processor
from .config import MPU6050Config

__all__ = [
    'MPU6050Collector',
    'MPU6050Processor',
    'MPU6050Config',
]

__version__ = '1.0.0'
