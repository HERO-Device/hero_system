"""
EEG Sensor Module for HERO System
4-channel EEG data collection using BrainFlow (OpenBCI Ganglion)

Supports dual-mode operation:
- Calibration mode: Real-time processing for UI feedback
- Session mode: Raw collection only (process later)
"""

from .collector import EEGCollector
from .processor import EEGProcessor
from .config import EEGConfig, BandPowerRanges

__all__ = [
    'EEGCollector',
    'EEGProcessor',
    'EEGConfig',
    'BandPowerRanges',
]

__version__ = '1.0.0'