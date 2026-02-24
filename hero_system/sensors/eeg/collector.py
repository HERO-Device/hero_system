"""
EEG Data Collector
Handles raw EEG data collection from BrainFlow with optional real-time processing
Integrates with SensorCoordinator for synchronized timestamps
"""

import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING
from uuid import UUID
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
from sqlalchemy.orm import Session

from .config import EEGConfig
from .processor import EEGProcessor

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
        self.session_id = session_id
        self.db_session = db_session
        self.coordinator = coordinator
        self.config = config if config else EEGConfig.for_session()

        self.params = BrainFlowInputParams()
        if self.config.serial_port:
            self.params.serial_port = self.config.serial_port
        if self.config.mac_address:
            self.params.mac_address = self.config.mac_address

        self.board = None
        self.board_id = self.config.board_id
        self.sampling_rate = None
        self.eeg_channels = None
        self.processor = None

        self.is_running = False
        self.collection_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()

        self.sample_count = 0
        self.sample_buffer = []

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
            BoardShim.enable_dev_board_logger()
            self.board = BoardShim(self.board_id, self.params)

            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

            logger.info(f"Board initialized: {self.board_id}")
            logger.info(f"Sampling rate: {self.sampling_rate} Hz")
            logger.info(f"EEG channels: {self.eeg_channels}")

            self.processor = EEGProcessor(self.config, self.sampling_rate)

            self.board.prepare_session()
            self.board.start_stream()
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'EEG stream started')

            logger.info("Waiting for EEG buffer to stabilize...")
            time.sleep(2)

            # Clear buffer before starting collection
            self.board.get_board_data()

            self.is_running = True
            self.stop_event.clear()
            self.sample_count = 0
            self.sample_buffer = []

            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="EEG-Collection-Thread",
                daemon=True
            )
            self.collection_thread.start()

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
            self.stop_event.set()

            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5)
                if self.collection_thread.is_alive():
                    logger.warning("Collection thread did not stop gracefully")

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
                if self.processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")

            if self.board:
                self.board.stop_stream()
                self.board.release_session()

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
        """Main data collection loop - polls brainflow and stores only new samples"""
        logger.info("EEG collection loop started")
        poll_interval = 0.1  # Poll every 100ms = ~20 new samples at 200Hz
        last_count = 0
        sample_interval_us = int(1_000_000 / self.config.sampling_rate)

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                current_count = self.board.get_board_data_count()
                new_samples = current_count - last_count
                logger.debug(f"EEG poll: count={current_count}, new={new_samples}")
                if new_samples > 0:
                    # get_current_board_data fetches the most recent N samples
                    # Since last_count tracks our position, new_samples are truly new
                    data = self.board.get_current_board_data(new_samples)
                    if data.shape[1] > 0:
                        self._store_raw_samples(data, sample_interval_us)
                        last_count = current_count
                        if self.config.realtime_processing:
                            self.sample_buffer.append(data)
                elapsed = time.perf_counter() - loop_start
                remaining = poll_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                time.sleep(0.1)
        logger.info("EEG collection loop stopped")

    def _store_raw_samples(self, data: np.ndarray, sample_interval_us: int):
        """
        Store raw EEG samples to database with per-sample timestamps

        Args:
            data: Board data array (channels x samples)
            sample_interval_us: Microseconds between samples (1e6 / sampling_rate)
        """
        if self.SensorEEG is None:
            logger.warning("SensorEEG model not available")
            return

        try:
            # Base timestamp for this batch — each sample gets a unique offset
            batch_start = self.coordinator.get_central_timestamp()

            if len(self.eeg_channels) < self.config.num_channels:
                logger.warning(
                    f"Expected {self.config.num_channels} channels, "
                    f"got {len(self.eeg_channels)}"
                )
                return

            ch1_data = data[self.eeg_channels[0]]
            ch2_data = data[self.eeg_channels[1]]
            ch3_data = data[self.eeg_channels[2]]
            ch4_data = data[self.eeg_channels[3]]

            for i in range(data.shape[1]):
                # Each sample gets a unique timestamp spaced by 1/sampling_rate
                timestamp = batch_start + timedelta(microseconds=i * sample_interval_us)

                quality_flag = 100
                is_valid = True

                if self.config.quality_check_enabled and i % 10 == 0:
                    window = ch1_data[max(0, i-10):i+1]
                    quality_score, is_valid = self.processor.assess_signal_quality(window)
                    quality_flag = int(quality_score * 100)

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
                if len(self.sample_buffer) > 0:
                    data_chunks = self.sample_buffer.copy()
                    self.sample_buffer = []
                    for chunk in data_chunks:
                        accumulated_samples.append(chunk)
                    total_samples = sum(chunk.shape[1] for chunk in accumulated_samples)
                    if total_samples >= self.config.processing_interval:
                        combined_data = np.hstack(accumulated_samples)
                        self._process_and_store_metrics(combined_data)
                        accumulated_samples = []
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("EEG processing loop stopped")

    def _process_and_store_metrics(self, data: np.ndarray):
        """Process EEG data and store metrics (calibration mode only)"""
        try:
            timestamp = self.coordinator.get_central_timestamp()
            metrics, band_powers = self.processor.compute_and_create_metrics(
                data=data,
                eeg_channels=self.eeg_channels,
                session_id=self.session_id,
                timestamp=timestamp,
                computation_method='welch_psd_realtime'
            )
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
        
