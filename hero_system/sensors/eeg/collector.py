"""
EEG Data Collector
Handles raw EEG data collection from BrainFlow with optional real-time processing
Integrates with SensorCoordinator for synchronized timestamps
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from uuid import UUID
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
from sqlalchemy.orm import Session

from .config import EEGConfig
from .processor import EEGProcessor

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from hero_system.coordinator import SensorCoordinator

logger = logging.getLogger(__name__)


class EEGCollector:
    """
    EEG data collector with dual-mode operation and coordinator integration

    Modes:
    - Calibration mode: Real-time processing for UI feedback
    - Session mode: Raw collection only (post-process later)

    Uses coordinator's central clock for timestamp synchronization across all sensors
    """

    def __init__(
        self,
        session_id: UUID,
        db_session: Session,
        coordinator: 'SensorCoordinator',
        config: Optional[EEGConfig] = None
    ):
        """
        Initialize EEG collector

        Args:
            session_id: UUID of the current test session
            db_session: SQLAlchemy database session
            coordinator: SensorCoordinator instance for timestamp sync
            config: EEG configuration (uses session defaults if None)
        """
        self.session_id = session_id
        self.db_session = db_session
        self.coordinator = coordinator
        self.config = config if config else EEGConfig.for_session()

        # Initialize BrainFlow parameters
        self.params = BrainFlowInputParams()
        if self.config.serial_port:
            self.params.serial_port = self.config.serial_port

        # Board components
        self.board = None
        self.board_id = self.config.board_id
        self.sampling_rate = None
        self.eeg_channels = None

        # Signal processor (initialized after board)
        self.processor = None

        # State management
        self.is_running = False
        self.collection_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Sample tracking
        self.sample_count = 0
        self.sample_buffer = []  # For real-time processing

        # Import database models
        try:
            from hero_core.database.models.sensors import SensorEEG, MetricsProcessed
            self.SensorEEG = SensorEEG
            self.MetricsProcessed = MetricsProcessed
        except ImportError:
            logger.error("Could not import database models. Install hero_core package.")
            self.SensorEEG = None
            self.MetricsProcessed = None

        logger.info(
            f"EEG Collector initialized for session {session_id} "
            f"(mode: {self.config.mode}, realtime: {self.config.realtime_processing})"
        )

    def start(self):
        """Start EEG data collection"""
        if self.is_running:
            logger.warning("EEG collector already running")
            return

        try:
            # Initialize BrainFlow board
            BoardShim.enable_dev_board_logger()
            self.board = BoardShim(self.board_id, self.params)

            # Get board info
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

            logger.info(f"Board initialized: {self.board_id}")
            logger.info(f"Sampling rate: {self.sampling_rate} Hz")
            logger.info(f"EEG channels: {self.eeg_channels}")

            # Initialize processor
            self.processor = EEGProcessor(self.config, self.sampling_rate)

            # Prepare and start BrainFlow session
            self.board.prepare_session()
            self.board.start_stream()
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'EEG stream started')

            # Wait for buffer to stabilize
            logger.info("Waiting for EEG buffer to stabilize...")
            time.sleep(2)

            # Clear any accumulated data
            self.board.get_board_data()

            # Start collection
            self.is_running = True
            self.stop_event.clear()
            self.sample_count = 0
            self.sample_buffer = []

            # Start collection thread
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="EEG-Collection-Thread",
                daemon=True
            )
            self.collection_thread.start()

            # Start processing thread if in calibration mode
            if self.config.realtime_processing:
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    name="EEG-Processing-Thread",
                    daemon=True
                )
                self.processing_thread.start()
                logger.info("✓ Real-time processing enabled (calibration mode)")

            logger.info("✓ EEG data collection started successfully")

        except Exception as e:
            logger.error(f"✗ Failed to start EEG collector: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """Stop EEG data collection"""
        if not self.is_running:
            logger.warning("EEG collector not running")
            return

        try:
            logger.info("Stopping EEG data collection...")

            # Signal threads to stop
            self.stop_event.set()

            # Wait for collection thread
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5)
                if self.collection_thread.is_alive():
                    logger.warning("Collection thread did not stop gracefully")

            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
                if self.processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")

            # Stop BrainFlow stream
            if self.board:
                self.board.stop_stream()
                self.board.release_session()

            # Final commit of any pending data
            try:
                self.db_session.commit()
                logger.info(f"Final commit: {self.sample_count} total samples collected")
            except Exception as e:
                logger.error(f"Error in final commit: {e}")
                self.db_session.rollback()

            self.is_running = False
            logger.info("✓ EEG data collection stopped successfully")

        except Exception as e:
            logger.error(f"✗ Error stopping EEG collector: {e}", exc_info=True)
            raise

    def _collection_loop(self):
        """Main data collection loop - runs in separate thread"""
        logger.info("EEG collection loop started")

        while not self.stop_event.is_set():
            try:
                # Get current board data
                data = self.board.get_current_board_data(self.config.buffer_size)

                if data.shape[1] > 0:
                    # Store raw samples
                    self._store_raw_samples(data)

                    # Add to buffer for real-time processing if enabled
                    if self.config.realtime_processing:
                        self.sample_buffer.append(data)

                # Sleep to maintain collection rate
                time.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("EEG collection loop stopped")

    def _store_raw_samples(self, data: np.ndarray):
        """
        Store raw EEG samples to database with synchronized timestamp

        Args:
            data: Board data array (channels x samples)
        """
        if self.SensorEEG is None:
            logger.warning("SensorEEG model not available")
            return

        try:
            # Use coordinator's central timestamp for synchronization
            timestamp = self.coordinator.get_central_timestamp()

            # Validate channels
            if len(self.eeg_channels) < self.config.num_channels:
                logger.warning(
                    f"Expected {self.config.num_channels} channels, "
                    f"got {len(self.eeg_channels)}"
                )
                return

            # Extract channel data
            ch1_data = data[self.eeg_channels[0]]
            ch2_data = data[self.eeg_channels[1]]
            ch3_data = data[self.eeg_channels[2]]
            ch4_data = data[self.eeg_channels[3]]

            # Store each sample
            for i in range(data.shape[1]):
                # Quality assessment (simple version)
                quality_flag = 100
                is_valid = True

                if self.config.quality_check_enabled and i % 10 == 0:
                    # Check quality periodically
                    window = ch1_data[max(0, i-10):i+1]
                    quality_score, is_valid = self.processor.assess_signal_quality(window)
                    quality_flag = int(quality_score * 100)

                # Create database record with synchronized timestamp
                eeg_sample = self.SensorEEG(
                    time=timestamp,
                    session_id=self.session_id,
                    channel_1=float(ch1_data[i]),
                    channel_2=float(ch2_data[i]),
                    channel_3=float(ch3_data[i]),
                    channel_4=float(ch4_data[i]),
                    quality_flag=quality_flag,
                    is_valid=is_valid
                )
                self.db_session.add(eeg_sample)
                self.sample_count += 1

            # Batch commit
            if self.sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()
                logger.debug(f"Committed batch: {self.sample_count} total samples")

        except Exception as e:
            logger.error(f"Error storing raw samples: {e}", exc_info=True)
            self.db_session.rollback()

    def _processing_loop(self):
        """Real-time processing loop - only runs in calibration mode"""
        logger.info("EEG processing loop started (calibration mode)")

        accumulated_samples = []

        while not self.stop_event.is_set():
            try:
                # Check if we have enough samples to process
                if len(self.sample_buffer) > 0:
                    # Get buffered data
                    data_chunks = self.sample_buffer.copy()
                    self.sample_buffer = []

                    # Concatenate chunks
                    for chunk in data_chunks:
                        accumulated_samples.append(chunk)

                    # Check if we have enough samples
                    total_samples = sum(chunk.shape[1] for chunk in accumulated_samples)

                    if total_samples >= self.config.processing_interval:
                        # Concatenate all accumulated data
                        combined_data = np.hstack(accumulated_samples)

                        # Process using the processor (proper separation of concerns!)
                        self._process_and_store_metrics(combined_data)

                        # Clear accumulated samples
                        accumulated_samples = []

                # Sleep briefly
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("EEG processing loop stopped")

    def _process_and_store_metrics(self, data: np.ndarray):
        """
        Process EEG data and store metrics (calibration mode only)
        Delegates actual processing to EEGProcessor

        Args:
            data: Combined board data array
        """
        try:
            # Get synchronized timestamp from coordinator
            timestamp = self.coordinator.get_central_timestamp()

            # Let the processor do the processing!
            metrics, band_powers = self.processor.compute_and_create_metrics(
                data=data,
                eeg_channels=self.eeg_channels,
                session_id=self.session_id,
                timestamp=timestamp,
                computation_method='welch_psd_realtime'
            )

            # Collector just stores the results
            for metric in metrics:
                self.db_session.add(metric)

            self.db_session.commit()

            logger.debug(
                f"Stored band powers: δ={band_powers['delta']:.2f}, "
                f"θ={band_powers['theta']:.2f}, α={band_powers['alpha']:.2f}, "
                f"β={band_powers['beta']:.2f}, γ={band_powers['gamma']:.2f}"
            )

        except Exception as e:
            logger.error(f"Error processing and storing metrics: {e}", exc_info=True)
            self.db_session.rollback()

    def get_status(self) -> dict:
        """Get current collector status"""
        return {
            'sensor_type': 'EEG',
            'mode': self.config.mode,
            'is_running': self.is_running,
            'realtime_processing': self.config.realtime_processing,
            'session_id': str(self.session_id),
            'board_id': self.board_id,
            'sampling_rate': self.sampling_rate,
            'channels': len(self.eeg_channels) if self.eeg_channels else 0,
            'samples_collected': self.sample_count,
        }

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        return f"<EEGCollector(mode={self.config.mode}, status={status})>"
