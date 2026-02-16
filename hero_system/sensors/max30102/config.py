"""
MAX30102 Sensor Configuration
Heart rate and SpO2 measurement parameters
"""

from dataclasses import dataclass


@dataclass
class MAX30102Config:
    """MAX30102 sensor configuration parameters"""

    # Operating mode
    mode: str = 'session'  # 'calibration' or 'session'

    # Hardware settings
    i2c_bus: int = 1
    i2c_address: int = 0x57

    # Sampling settings
    buffer_size: int = 100  # Samples needed for HR/SpO2 calculation
    collection_interval: float = 0.01  # 100 Hz polling rate

    # Processing settings (for calibration mode)
    realtime_processing: bool = False  # Set based on mode
    hr_smoothing_window: int = 5  # Number of HR measurements to average
    display_update_interval: float = 3.0  # Seconds between updates in calibration

    # Quality thresholds
    min_signal_threshold: int = 10000  # Minimum IR signal for valid reading
    max_signal_threshold: int = 200000  # Maximum IR signal (saturation check)

    # Database settings
    batch_commit_size: int = 100  # Commit every N samples

    @classmethod
    def for_calibration(cls) -> 'MAX30102Config':
        """Configuration for calibration phase - real-time processing enabled"""
        config = cls(
            mode='calibration',
            realtime_processing=True,
        )
        return config

    @classmethod
    def for_session(cls) -> 'MAX30102Config':
        """Configuration for active session - raw collection only"""
        config = cls(
            mode='session',
            realtime_processing=False,
        )
        return config
    