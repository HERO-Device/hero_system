"""
MAX30102 Sensor Configuration
Heart rate and SpO2 measurement parameters
"""

from dataclasses import dataclass


@dataclass
class MAX30102Config:
    """
    Configuration parameters for the MAX30102 heart rate and pulse oximeter sensor.

    Controls I2C hardware settings, sampling rate, signal quality thresholds,
    and processing mode for both calibration and session operation.
    """

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
        """
        Create a configuration for the calibration phase.

        Enables real-time HR/SpO2 processing for live signal verification.

        Returns:
            MAX30102Config with mode='calibration' and realtime_processing=True.
        """
        config = cls(
            mode='calibration',
            realtime_processing=True,
        )
        return config

    @classmethod
    def for_session(cls) -> 'MAX30102Config':
        """
        Create a configuration for an active session.

        Disables real-time processing — raw IR and Red signals are collected
        and processed post-session.

        Returns:
            MAX30102Config with mode='session' and realtime_processing=False.
        """
        config = cls(
            mode='session',
            realtime_processing=False,
        )
        return config
    