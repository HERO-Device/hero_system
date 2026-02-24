"""
EEG Sensor Configuration
Defines frequency bands, filter parameters, and hardware settings
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from brainflow.board_shim import BoardIds


@dataclass
class BandPowerRanges:
    """EEG frequency band definitions (Hz)"""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 14.0)
    beta: Tuple[float, float] = (14.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 50.0)

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """Convert to dictionary for easy iteration"""
        return {
            'delta': self.delta,
            'theta': self.theta,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
        }


@dataclass
class EEGConfig:
    """EEG sensor configuration parameters"""

    # Operating mode
    mode: str = 'session'  # 'calibration' or 'session'

    # Hardware settings - OpenBCI Ganglion
    board_id: int = BoardIds.GANGLION_BOARD.value
    serial_port: str = ''
    mac_address: str = ''
    num_channels: int = 4

    # Sampling settings
    sampling_rate: int = 200  # Ganglion native sampling rate
    buffer_size: int = 512  # Samples to collect before processing
    collection_interval: float = 0.005  # 200 Hz = 5ms between samples
    eeg_poll_interval: float = 0.1  # Poll brainflow buffer every 100ms

    # Signal processing
    bandpass_low: float = 3.0
    bandpass_high: float = 45.0
    bandpass_order: int = 2

    # Notch filters (power line interference)
    notch_50hz: Tuple[float, float] = (48.0, 52.0)  # 50 Hz (Europe)
    notch_60hz: Tuple[float, float] = (58.0, 62.0)  # 60 Hz (North America)
    notch_order: int = 2

    # Band power ranges
    bands: BandPowerRanges = None

    def __post_init__(self):
        """Initialize bands if not provided"""
        if self.bands is None:
            self.bands = BandPowerRanges()

    # Quality assessment
    quality_check_enabled: bool = True
    quality_threshold: float = 0.7  # 0-1 scale

    # Database settings
    batch_commit_size: int = 50  # Commit every N samples

    # Processing settings (mode-dependent)
    realtime_processing: bool = False  # Set based on mode
    processing_interval: int = 512  # Process every N samples

    @classmethod
    def for_calibration(cls, serial_port: str = '', mac_address: str = '') -> 'EEGConfig':
        """Configuration for calibration phase - real-time processing enabled"""
        config = cls(
            mode='calibration',
            board_id=BoardIds.GANGLION_BOARD.value,
            serial_port=serial_port,
            realtime_processing=True,
            processing_interval=256,  # Process more frequently during calibration
        )
        return config

    @classmethod
    def for_session(cls, serial_port: str = '', mac_address: str = '') -> 'EEGConfig':
        """Configuration for active session - collection only"""
        config = cls(
            mode='session',
            board_id=BoardIds.GANGLION_BOARD.value,
            serial_port=serial_port,
            realtime_processing=False,
        )
        return config

    @classmethod
    def for_synthetic(cls, mode: str = 'session') -> 'EEGConfig':
        """Configuration for synthetic/testing board"""
        config = cls(
            mode=mode,
            board_id=BoardIds.SYNTHETIC_BOARD.value,
            realtime_processing=(mode == 'calibration'),
        )
        return config
