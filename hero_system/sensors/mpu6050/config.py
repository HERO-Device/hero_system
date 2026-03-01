"""
MPU6050 Sensor Configuration
Accelerometer and gyroscope measurement parameters
"""

from dataclasses import dataclass


@dataclass
class MPU6050Config:
    """MPU6050 sensor configuration parameters"""

    # Operating mode
    mode: str = 'session'  # 'calibration' or 'session'

    # Hardware settings
    i2c_bus: int = 1
    i2c_address: int = 0x68

    # Sampling settings
    sample_rate: int = 100  # Hz (from your teammate's code)
    collection_interval: float = 0.01  # 1/100 = 10ms between samples

    # Accelerometer settings
    accel_range: int = 0x00  # ±2g range (register value)
    accel_sensitivity: float = 16384.0  # LSB/g for ±2g range
    gravity: float = 9.81  # m/s² conversion factor

    # Gyroscope settings
    gyro_range: int = 0x00  # ±250°/s range (register value)
    gyro_sensitivity: float = 131.0  # LSB/(°/s) for ±250°/s range

    # Processing settings (for calibration mode)
    realtime_processing: bool = False  # Set based on mode

    # Tremor detection parameters (for post-processing)
    tremor_window_seconds: int = 4  # Window duration for analysis
    tremor_freq_low: float = 4.0  # Hz - lower bound of tremor band
    tremor_freq_high: float = 6.0  # Hz - upper bound of tremor band
    tremor_filter_order: int = 4  # Butterworth filter order
    tremor_threshold: float = 0.3  # Tremor power ratio threshold for detection

    # Database settings
    batch_commit_size: int = 100  # Commit every N samples

    @property
    def window_size(self) -> int:
        """
        Calculate the tremor analysis window size in samples.

        Returns:
            Number of samples in one tremor analysis window.
        """
        return self.sample_rate * self.tremor_window_seconds

    @classmethod
    def for_calibration(cls) -> 'MPU6050Config':
        """
        Create a configuration for the calibration phase.

        Enables real-time processing for live signal verification.

        Returns:
            MPU6050Config with mode='calibration' and realtime_processing=True.
        """
        config = cls(
            mode='calibration',
            realtime_processing=True,
        )
        return config

    @classmethod
    def for_session(cls) -> 'MPU6050Config':
        """
        Create a configuration for an active session.

        Disables real-time processing — raw data is collected and
        processed post-session.

        Returns:
            MPU6050Config with mode='session' and realtime_processing=False.
        """
        config = cls(
            mode='session',
            realtime_processing=False,
        )
        return config
    