"""
MAX30102 Data Collector
Raw IR and Red signal collection only - all processing delegated to processor
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from uuid import UUID

from max30102 import MAX30102

from .config import MAX30102Config

if TYPE_CHECKING:
    from hero_system.coordinator import SensorCoordinator

logger = logging.getLogger(__name__)


class MAX30102Collector:
    """
    MAX30102 collector - raw data collection only

    Collects raw IR and Red signals from MAX30102 sensor and stores to database.
    All HR/SpO2 processing is delegated to MAX30102Processor.

    Uses coordinator's central clock for timestamp synchronization.
    """

    def __init__(
        self,
        session_id: UUID,
        db_session,
        coordinator: 'SensorCoordinator',
        config: Optional[MAX30102Config] = None
    ):
        """
        Initialize MAX30102 collector

        Args:
            session_id: UUID of current session
            db_session: Database session
            coordinator: Sensor coordinator for timestamps
            config: MAX30102 configuration
        """
        self.session_id = session_id
        self.db_session = db_session
        self.coordinator = coordinator
        self.config = config if config else MAX30102Config.for_session()

        # Hardware
        self.sensor = None

        # State management
        self.is_running = False
        self.collection_thread = None
        self.stop_event = threading.Event()

        # Sample tracking
        self.sample_count = 0

        # Import database models
        try:
            from hero_core.database.models.sensors import SensorHeartRate, SensorOximeter
            self.SensorHeartRate = SensorHeartRate
            self.SensorOximeter = SensorOximeter
        except ImportError:
            logger.error("Could not import database models")
            self.SensorHeartRate = None
            self.SensorOximeter = None

        logger.info(f"MAX30102 Collector initialized for session {session_id}")

    def start(self):
        """Start MAX30102 data collection"""
        if self.is_running:
            logger.warning("MAX30102 collector already running")
            return

        try:
            # Initialize sensor
            logger.info("Initializing MAX30102 sensor...")
            self.sensor = MAX30102()

            logger.info(f"✓ MAX30102 ready at address 0x{self.config.i2c_address:02X}")

            # Start collection
            self.is_running = True
            self.stop_event.clear()
            self.sample_count = 0

            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="MAX30102-Collection-Thread",
                daemon=True
            )
            self.collection_thread.start()

            logger.info("✓ MAX30102 data collection started successfully")

        except Exception as e:
            logger.error(f"✗ Failed to start MAX30102: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """Stop MAX30102 data collection"""
        if not self.is_running:
            logger.warning("MAX30102 collector not running")
            return

        try:
            logger.info("Stopping MAX30102 data collection...")

            # Signal thread to stop
            self.stop_event.set()

            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5)

            # Final commit
            try:
                self.db_session.commit()
                logger.info(f"Final commit: {self.sample_count} total samples collected")
            except Exception as e:
                logger.error(f"Error in final commit: {e}")
                self.db_session.rollback()

            self.is_running = False
            logger.info("✓ MAX30102 data collection stopped successfully")

        except Exception as e:
            logger.error(f"✗ Error stopping MAX30102: {e}", exc_info=True)
            raise

    def _collection_loop(self):
        """Main data collection loop - runs in separate thread"""
        logger.info("MAX30102 collection loop started")

        while not self.stop_event.is_set():
            try:
                # Check if data is available in FIFO
                num_bytes = self.sensor.get_data_present()

                if num_bytes > 0:
                    # Read all available data from FIFO
                    while num_bytes > 0:
                        red, ir = self.sensor.read_fifo()
                        num_bytes -= 1

                        # Store raw data to database
                        self._store_raw_sample(ir, red)

                # Sleep to maintain polling rate
                time.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("MAX30102 collection loop stopped")

    def _store_raw_sample(self, ir: int, red: int):
        """
        Store raw IR and Red signals to database

        Args:
            ir: Infrared signal value
            red: Red signal value
        """
        if self.SensorHeartRate is None or self.SensorOximeter is None:
            return

        try:
            # Get synchronized timestamp from coordinator
            timestamp = self.coordinator.get_central_timestamp()

            # Check signal quality
            is_valid = self._check_signal_quality(ir, red)
            quality = self._calculate_quality_score(ir, red)

            # Store heart rate sensor data (IR signal)
            hr_sample = self.SensorHeartRate(
                time=timestamp,
                session_id=self.session_id,
                raw_signal=float(ir),
                quality=quality,
                is_valid=is_valid
            )
            self.db_session.add(hr_sample)

            # Store oximeter data (Red and IR signals)
            ox_sample = self.SensorOximeter(
                time=timestamp,
                session_id=self.session_id,
                red_signal=float(red),
                infrared_signal=float(ir),
                is_valid=is_valid
            )
            self.db_session.add(ox_sample)

            self.sample_count += 1

            # Batch commit
            if self.sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()
                logger.debug(f"Committed batch: {self.sample_count} total samples")

        except Exception as e:
            logger.error(f"Error storing raw sample: {e}", exc_info=True)
            self.db_session.rollback()

    def _check_signal_quality(self, ir: int, red: int) -> bool:
        """
        Check if signal quality is acceptable

        Args:
            ir: Infrared signal value
            red: Red signal value

        Returns:
            True if signal quality is good
        """
        # Check if signals are within valid range
        if ir < self.config.min_signal_threshold or ir > self.config.max_signal_threshold:
            return False
        if red < self.config.min_signal_threshold or red > self.config.max_signal_threshold:
            return False

        return True

    def _calculate_quality_score(self, ir: int, red: int) -> int:
        """
        Calculate signal quality score (0=poor, 1=fair, 2=good)

        Args:
            ir: Infrared signal value
            red: Red signal value

        Returns:
            Quality score (0-2)
        """
        if not self._check_signal_quality(ir, red):
            return 0  # Poor

        # Check if signal is strong
        if ir > 50000 and red > 50000:
            return 2  # Good
        else:
            return 1  # Fair

    def get_status(self) -> dict:
        """Get current collector status"""
        return {
            'sensor_type': 'MAX30102',
            'mode': self.config.mode,
            'is_running': self.is_running,
            'session_id': str(self.session_id),
            'samples_collected': self.sample_count,
        }

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        return f"<MAX30102Collector(status={status})>"
    