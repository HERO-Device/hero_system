"""
MPU6050 Data Collector
Raw accelerometer and gyroscope data collection
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from uuid import UUID
import numpy as np

import smbus2

from .config import MPU6050Config

if TYPE_CHECKING:
    from hero_system.coordinator import SensorCoordinator

logger = logging.getLogger(__name__)


class MPU6050Collector:
    """
    MPU6050 collector - raw accelerometer and gyroscope data collection

    Collects raw 3-axis acceleration and rotation data from MPU6050 sensor.
    All processing (tremor detection, PSD analysis) is delegated to MPU6050Processor.

    Uses coordinator's central clock for timestamp synchronization.
    """

    def __init__(
            self,
            session_id: UUID,
            db_session,
            coordinator: 'SensorCoordinator',
            config: Optional[MPU6050Config] = None
    ):
        """
        Initialize MPU6050 collector

        Args:
            session_id: UUID of current session
            db_session: Database session
            coordinator: Sensor coordinator for timestamps
            config: MPU6050 configuration
        """
        self.session_id = session_id
        self.db_session = db_session
        self.coordinator = coordinator
        self.config = config if config else MPU6050Config.for_session()

        # I2C bus
        self.bus = None

        # State management
        self.is_running = False
        self.collection_thread = None
        self.stop_event = threading.Event()

        # Sample tracking
        self.accel_sample_count = 0
        self.gyro_sample_count = 0

        # Import database models
        try:
            from hero_core.database.models.sensors import SensorAccelerometer, SensorGyroscope
            self.SensorAccelerometer = SensorAccelerometer
            self.SensorGyroscope = SensorGyroscope
        except ImportError:
            logger.error("Could not import database models")
            self.SensorAccelerometer = None
            self.SensorGyroscope = None

        logger.info(f"MPU6050 Collector initialized for session {session_id}")

    def start(self):
        """
        Initialise the I2C bus, configure the MPU6050, and start the collection thread.

        Raises:
            Exception if the I2C bus cannot be opened or the sensor fails to configure.

        Returns:
            None.
        """
        if self.is_running:
            logger.warning("MPU6050 collector already running")
            return

        try:
            # Initialize I2C bus
            logger.info("Initializing MPU6050 sensor...")
            self.bus = smbus2.SMBus(self.config.i2c_bus)

            # Wake up MPU6050 (disable sleep mode)
            self.bus.write_byte_data(self.config.i2c_address, 0x6B, 0x00)
            time.sleep(0.1)

            # Configure accelerometer range
            self.bus.write_byte_data(self.config.i2c_address, 0x1C, self.config.accel_range)

            # Configure gyroscope range
            self.bus.write_byte_data(self.config.i2c_address, 0x1B, self.config.gyro_range)
            time.sleep(0.1)

            logger.info(f"✓ MPU6050 ready at address 0x{self.config.i2c_address:02X}")
            logger.info(f"  Accelerometer: ±2g range")
            logger.info(f"  Gyroscope: ±250°/s range")

            # Start collection
            self.is_running = True
            self.stop_event.clear()
            self.accel_sample_count = 0
            self.gyro_sample_count = 0

            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="MPU6050-Collection-Thread",
                daemon=True
            )
            self.collection_thread.start()

            logger.info("✓ MPU6050 data collection started successfully")

        except Exception as e:
            logger.error(f"✗ Failed to start MPU6050: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """
        Signal the collection thread to stop, close the I2C bus, and flush remaining samples.

        Returns:
            None.
        """
        if not self.is_running:
            logger.warning("MPU6050 collector not running")
            return

        try:
            logger.info("Stopping MPU6050 data collection...")

            # Signal thread to stop
            self.stop_event.set()

            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5)

            # Close I2C bus
            if self.bus:
                self.bus.close()

            # Final commit
            try:
                self.db_session.commit()
                logger.info(f"Final commit: {self.accel_sample_count} accel + {self.gyro_sample_count} gyro samples")
            except Exception as e:
                logger.error(f"Error in final commit: {e}")
                self.db_session.rollback()

            self.is_running = False
            logger.info("✓ MPU6050 data collection stopped successfully")

        except Exception as e:
            logger.error(f"✗ Error stopping MPU6050: {e}", exc_info=True)
            raise

    def _collection_loop(self):
        """
        Main data collection loop — runs in a background thread.

        Reads raw accelerometer and gyroscope registers at the configured
        sample rate, converts to physical units, and stores to the database.

        Returns:
            None.
        """
        logger.info("MPU6050 collection loop started")

        while not self.stop_event.is_set():
            try:
                # Read accelerometer data
                accel_x_raw = self._read_word_2c(0x3B)
                accel_y_raw = self._read_word_2c(0x3D)
                accel_z_raw = self._read_word_2c(0x3F)

                # Read gyroscope data
                gyro_x_raw = self._read_word_2c(0x43)
                gyro_y_raw = self._read_word_2c(0x45)
                gyro_z_raw = self._read_word_2c(0x47)

                # Convert to physical units
                # Accelerometer: ±2g range, sensitivity = 16384 LSB/g
                accel_x = (accel_x_raw / self.config.accel_sensitivity) * self.config.gravity  # m/s²
                accel_y = (accel_y_raw / self.config.accel_sensitivity) * self.config.gravity
                accel_z = (accel_z_raw / self.config.accel_sensitivity) * self.config.gravity

                # Gyroscope: ±250°/s range, sensitivity = 131 LSB/(°/s)
                gyro_x = gyro_x_raw / self.config.gyro_sensitivity  # °/s
                gyro_y = gyro_y_raw / self.config.gyro_sensitivity
                gyro_z = gyro_z_raw / self.config.gyro_sensitivity

                self._store_accel_sample(accel_x, accel_y, accel_z)
                self._store_gyro_sample(gyro_x, gyro_y, gyro_z)

                # Sleep to maintain sample rate
                time.sleep(self.config.collection_interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("MPU6050 collection loop stopped")

    def _read_word_2c(self, reg: int) -> int:
        """
        Read signed 16-bit value from I2C register

        Args:
            reg: Register address

        Returns:
            Signed 16-bit integer value
        """
        high = self.bus.read_byte_data(self.config.i2c_address, reg)
        low = self.bus.read_byte_data(self.config.i2c_address, reg + 1)
        val = (high << 8) + low

        # Convert to signed value
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val

    def _store_accel_sample(self, x: float, y: float, z: float):
        """
        Store accelerometer sample to database

        Args:
            x: X-axis acceleration (m/s²)
            y: Y-axis acceleration (m/s²)
            z: Z-axis acceleration (m/s²)
        """
        if self.SensorAccelerometer is None:
            return

        try:
            # Get synchronized timestamp from coordinator
            timestamp = self.coordinator.get_central_timestamp()

            # Calculate magnitude for quality assessment
            magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            quality_score = self._assess_accel_quality(magnitude)

            # Store accelerometer data
            accel_sample = self.SensorAccelerometer(
                time=timestamp,
                session_id=self.session_id,
                x=float(x),
                y=float(y),
                z=float(z),
                quality_score=quality_score,
                is_valid=True
            )
            self.db_session.add(accel_sample)

            self.accel_sample_count += 1

            # Batch commit
            if self.accel_sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()
                logger.debug(f"Committed batch: {self.accel_sample_count} accel samples")

        except Exception as e:
            logger.error(f"Error storing accel sample: {e}", exc_info=True)
            self.db_session.rollback()

    def _store_gyro_sample(self, x: float, y: float, z: float):
        """
        Store gyroscope sample to database

        Args:
            x: X-axis rotation (°/s)
            y: Y-axis rotation (°/s)
            z: Z-axis rotation (°/s)
        """
        if self.SensorGyroscope is None:
            return

        try:
            # Get synchronized timestamp from coordinator
            timestamp = self.coordinator.get_central_timestamp()

            # Calculate magnitude for quality assessment
            magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            quality_score = self._assess_gyro_quality(magnitude)

            # Store gyroscope data
            gyro_sample = self.SensorGyroscope(
                time=timestamp,
                session_id=self.session_id,
                x=float(x),
                y=float(y),
                z=float(z),
                quality_score=quality_score,
                is_valid=True
            )
            self.db_session.add(gyro_sample)

            self.gyro_sample_count += 1

            # Batch commit
            if self.gyro_sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()
                logger.debug(f"Committed batch: {self.gyro_sample_count} gyro samples")

        except Exception as e:
            logger.error(f"Error storing gyro sample: {e}", exc_info=True)
            self.db_session.rollback()

    def _assess_accel_quality(self, magnitude: float) -> float:
        """
        Assess accelerometer signal quality based on magnitude

        Args:
            magnitude: Acceleration magnitude (m/s²)

        Returns:
            Quality score (0.0 - 1.0)
        """
        # At rest, magnitude should be ~9.81 m/s² (gravity)
        # Quality degrades as we deviate from this
        deviation = abs(magnitude - self.config.gravity)

        if deviation < 1.0:
            return 1.0  # Excellent
        elif deviation < 3.0:
            return 0.8  # Good
        elif deviation < 5.0:
            return 0.6  # Fair
        else:
            return 0.4  # Poor (excessive movement or sensor issue)

    def _assess_gyro_quality(self, magnitude: float) -> float:
        """
        Assess gyroscope signal quality based on magnitude

        Args:
            magnitude: Rotation magnitude (°/s)

        Returns:
            Quality score (0.0 - 1.0)
        """
        # At rest, magnitude should be ~0 °/s
        # Quality degrades with excessive rotation

        if magnitude < 5.0:
            return 1.0  # Excellent (stable)
        elif magnitude < 20.0:
            return 0.8  # Good (minor movement)
        elif magnitude < 50.0:
            return 0.6  # Fair (moderate movement)
        else:
            return 0.4  # Poor (excessive rotation)

    def get_status(self) -> dict:
        """
        Return the current collector state.

        Returns:
            Dict containing sensor type, mode, running state, session ID,
            and sample counts for accelerometer and gyroscope.
        """
        return {
            'sensor_type': 'MPU6050',
            'mode': self.config.mode,
            'is_running': self.is_running,
            'session_id': str(self.session_id),
            'accel_samples_collected': self.accel_sample_count,
            'gyro_samples_collected': self.gyro_sample_count,
        }

    def __repr__(self):
        """String representation showing running state."""
        status = "running" if self.is_running else "stopped"
        return f"<MPU6050Collector(status={status})>"
    