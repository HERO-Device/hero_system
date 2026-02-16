"""
MAX30102 Sensor Module for HERO System
Heart Rate and SpO2 measurement using MAX30102 pulse oximeter

Architecture:
- Collector: Raw IR/Red signal collection (all modes)
- Processor: HR/SpO2 computation (real-time for calibration, post-session for analysis)

Dual-mode operation:
- Calibration mode: Real-time HR/SpO2 calculation for signal verification
- Session mode: Raw IR/Red signal collection only (process after session)

Post-processing capabilities:
- Heart rate (BPM)
- Blood oxygen saturation (SpO2%)
- Heart rate variability (HRV - SDNN, RMSSD)
"""

from .collector import MAX30102Collector
from .processor import MAX30102Processor
from .config import MAX30102Config

__all__ = [
    'MAX30102Collector',
    'MAX30102Processor',
    'MAX30102Config',
]

__version__ = '1.0.0'
