"""
HERO System Sensor Coordinator
Manages multi-sensor data collection with synchronized timestamps
"""

from .clock import CentralClock
from .coordinator import SensorCoordinator

__all__ = [
    'CentralClock',
    'SensorCoordinator',
]

__version__ = '1.0.0'
