"""
HERO System - Sensor Pipeline
==============================
Central module that owns the full lifecycle of all sensors.

Usage in run.py:
    pipeline = SensorPipeline(session_id, db.session, clock)
    pipeline.start()
    # ... cognitive games run ...
    pipeline.stop()

Sensors managed:
    - MPU6050      : 3-axis accelerometer + gyroscope (I2C)
    - MAX30102     : Heart rate (PPG) + pulse oximeter (I2C)
    - EEG          : 4-channel OpenBCI Ganglion via BrainFlow (Bluetooth/BLED112)
    - Eye Tracking : ArduCam Pinsight AI + MediaPipe gaze (USB/CSI)

Failure policy:
    If a sensor fails to initialise, it is skipped and its failure is recorded
    in the sensor_calibration table so it's traceable per session.
    The session continues with whichever sensors are available.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from hero_core.coordinator.clock import CentralClock
from hero_core.coordinator.coordinator import SensorCoordinator
from hero_core.database.models.connection import get_db_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensor type labels — match what goes into sensor_calibration.sensor_type
# ---------------------------------------------------------------------------
SENSOR_MPU6050       = 'mpu6050'
SENSOR_MAX30102      = 'max30102'
SENSOR_EEG           = 'eeg'
SENSOR_EYE_TRACKING  = 'eye_tracking'


class SensorPipeline:
    """
    Owns the lifecycle of all HERO sensors for a single session.

    Responsibilities:
      - Import and initialise each sensor's Collector/Processor
      - Register all sensors with the SensorCoordinator
      - Log each sensor's init outcome to sensor_calibration
      - Provide a clean start() / stop() interface for run.py
      - Report which sensors are active via get_status()
    """

    def __init__(
        self,
        session_id: UUID,
        db_session: Session,
        clock: CentralClock,
    ):
        """
        Args:
            session_id : UUID of the current test session
            db_session : SQLAlchemy session (already connected)
            clock      : Shared CentralClock from run.py — keeps all
                         sensor timestamps in sync with the game layer
        """
        self.session_id = session_id
        self.db_session = db_session
        self.clock = clock

        # Build a coordinator that uses the shared clock rather than
        # creating its own internal one.
        self.coordinator = SensorCoordinator(
            session_id=session_id,
            db_session=db_session,
        )
        # Inject the shared clock so all sensors and games share one
        # time reference.
        self.coordinator.clock = clock

        # Tracks which sensors successfully initialised
        self._active_sensors: list[str] = []
        self._failed_sensors: list[str] = []

        # Collector references (set during _init_* methods)
        self._mpu6050_collector   = None
        self._max30102_collector  = None
        self._eeg_collector       = None
        self._eye_tracking_processor = None

        logger.info(f"SensorPipeline created for session {session_id}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self):
        """
        Initialise and start all sensors.
        Failed sensors are logged to the DB and skipped.
        """
        logger.info("=" * 55)
        logger.info("  HERO Sensor Pipeline — starting")
        logger.info("=" * 55)

        self._init_mpu6050()
        self._init_max30102()
        self._init_eeg()
#        self._init_eye_tracking()  # disabled — pending calibration

        logger.info(
            f"Pipeline ready — active: {self._active_sensors or 'none'} | "
            f"failed: {self._failed_sensors or 'none'}"
        )

    def stop(self):
        """
        Gracefully stop all active sensors and flush any buffered data.
        """
        logger.info("Stopping sensor pipeline...")
        self.coordinator.stop_all_sensors()
        logger.info("✓ Sensor pipeline stopped")

    def _new_session(self):
        """Create a fresh independent DB session for a sensor."""
        _, session = get_db_connection(
            host='localhost', port=5432, user='postgres',
            password='pgdbadmin', dbname='hero_db'
        )
        return session

    def get_status(self) -> dict:
        """
        Return a summary of sensor states for logging / UI display.
        """
        return {
            'session_id'     : str(self.session_id),
            'active_sensors' : self._active_sensors,
            'failed_sensors' : self._failed_sensors,
            'coordinator'    : self.coordinator.get_coordinator_status(),
        }

    # -----------------------------------------------------------------------
    # Private — sensor initialisation helpers
    # -----------------------------------------------------------------------

    def _init_mpu6050(self):
        """Initialise MPU6050 accelerometer/gyroscope."""
        sensor_name = SENSOR_MPU6050
        try:
            from hero_system.sensors.mpu6050.collector import MPU6050Collector
            from hero_system.sensors.mpu6050.config import MPU6050Config

            config = MPU6050Config.for_session()
            collector = MPU6050Collector(
                session_id=self.session_id,
                db_session=self._new_session(),
                coordinator=self.coordinator,
                config=config,
            )

            self.coordinator.register_sensor(sensor_name, collector, config)
            self.coordinator.start_sensor(sensor_name)

            self._mpu6050_collector = collector
            self._active_sensors.append(sensor_name)

            self._log_sensor_status(
                sensor_type=sensor_name,
                status='active',
                sampling_rate_hz=config.sample_rate,
                params={
                    'i2c_address': hex(config.i2c_address),
                    'i2c_bus'    : config.i2c_bus,
                    'accel_range': config.accel_range,
                    'gyro_range' : config.gyro_range,
                    'mode'       : config.mode,
                },
            )
            logger.info(f"✓ MPU6050 initialised ({config.sample_rate} Hz)")

        except Exception as e:
            self._handle_sensor_failure(sensor_name, e)

    def _init_max30102(self):
        """Initialise MAX30102 heart rate / SpO2 sensor."""
        sensor_name = SENSOR_MAX30102
        try:
            from hero_system.sensors.max30102.collector import MAX30102Collector
            from hero_system.sensors.max30102.config import MAX30102Config

            config = MAX30102Config.for_session()
            collector = MAX30102Collector(
                session_id=self.session_id,
                db_session=self._new_session(),
                coordinator=self.coordinator,
                config=config,
            )

            self.coordinator.register_sensor(sensor_name, collector, config)
            self.coordinator.start_sensor(sensor_name)

            self._max30102_collector = collector
            self._active_sensors.append(sensor_name)

            # Derive Hz from the polling interval
            hz = int(1.0 / config.collection_interval) if config.collection_interval else 0
            self._log_sensor_status(
                sensor_type=sensor_name,
                status='active',
                sampling_rate_hz=hz,
                params={
                    'i2c_address'  : hex(config.i2c_address),
                    'i2c_bus'      : config.i2c_bus,
                    'buffer_size'  : config.buffer_size,
                    'mode'         : config.mode,
                    'realtime'     : config.realtime_processing,
                },
            )
            logger.info(f"✓ MAX30102 initialised (~{hz} Hz)")

        except Exception as e:
            self._handle_sensor_failure(sensor_name, e)

    def _init_eeg(self):
        """Initialise OpenBCI Ganglion EEG via BrainFlow."""
        sensor_name = SENSOR_EEG
        try:
            from hero_system.sensors.eeg.collector import EEGCollector
            from hero_system.sensors.eeg.config import EEGConfig

            config = EEGConfig.for_session(serial_port='/dev/ttyACM0')
            config.mac_address = 'cb:1c:86:2e:73:2c'
            collector = EEGCollector(
                session_id=self.session_id,
                db_session=self._new_session(),
                coordinator=self.coordinator,
                config=config,
            )

            self.coordinator.register_sensor(sensor_name, collector, config)
            self.coordinator.start_sensor(sensor_name)

            self._eeg_collector = collector
            self._active_sensors.append(sensor_name)

            self._log_sensor_status(
                sensor_type=sensor_name,
                status='active',
                sampling_rate_hz=200,  # Ganglion fixed at 200 Hz
                params={
                    'board_id'   : config.board_id,
                    'channels'   : config.num_channels,
                    'mode'       : config.mode,
                    'realtime'   : config.realtime_processing,
                },
            )
            logger.info("✓ EEG (OpenBCI Ganglion) initialised (200 Hz)")

        except Exception as e:
            self._handle_sensor_failure(sensor_name, e)

    def _init_eye_tracking(self):
        """Initialise ArduCam Pinsight AI + MediaPipe eye tracking."""
        sensor_name = SENSOR_EYE_TRACKING
        try:
            from hero_system.sensors.eye_tracking.processor import EyeTrackingProcessor
            from hero_system.sensors.eye_tracking.config import EyeTrackingConfig

            config = EyeTrackingConfig.for_session()
            processor = EyeTrackingProcessor(
                session_id=self.session_id,
                db_session=self._new_session(),
                coordinator=self.coordinator,
                config=config,
            )

            # Eye tracking uses a processor (not a separate collector) since
            # all processing is inline — register under the same interface.
            self.coordinator.register_sensor(sensor_name, processor, config)
            self.coordinator.start_sensor(sensor_name)

            self._eye_tracking_processor = processor
            self._active_sensors.append(sensor_name)

            fps = getattr(config, 'fps', 30)
            self._log_sensor_status(
                sensor_type=sensor_name,
                status='active',
                sampling_rate_hz=fps,
                params={
                    'camera'  : 'ArduCam Pinsight AI (DepthAI)',
                    'mode'    : config.mode,
                    'fps'     : fps,
                },
            )
            logger.info(f"✓ Eye tracking initialised ({fps} fps)")

        except Exception as e:
            self._handle_sensor_failure(sensor_name, e)

    # -----------------------------------------------------------------------
    # Private — failure handling + DB logging
    # -----------------------------------------------------------------------

    def _handle_sensor_failure(self, sensor_name: str, exc: Exception):
        """
        Log a failed sensor to the DB and mark it as skipped.
        The session continues — we just won't have data for this sensor.
        """
        self._failed_sensors.append(sensor_name)

        logger.warning(
            f"⚠ {sensor_name} failed to initialise — skipping. "
            f"Error: {exc}"
        )

        self._log_sensor_status(
            sensor_type=sensor_name,
            status='failed',
            sampling_rate_hz=None,
            params={},
            notes=f"{type(exc).__name__}: {exc}",
        )

    def _log_sensor_status(
        self,
        sensor_type: str,
        status: str,
        sampling_rate_hz: Optional[int],
        params: dict,
        notes: str = None,
    ):
        """
        Write a row to sensor_calibration so every session has a
        full record of which sensors were active and what their config was.
        """
        try:
            from hero_core.database.models.calibration import SensorCalibration

            record = SensorCalibration(
                calibration_id=uuid.uuid4(),
                session_id=self.session_id,
                sensor_type=sensor_type,
                sampling_rate_hz=sampling_rate_hz,
                calibration_timestamp=datetime.now(timezone.utc),
                calibration_params=params or {},
                sensor_status=status,
                notes=notes,
            )
            self.db_session.add(record)
            self.db_session.commit()

        except Exception as e:
            # Don't let a logging failure cascade — just warn
            logger.warning(f"Could not write sensor_calibration record for {sensor_type}: {e}")
            try:
                self.db_session.rollback()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __repr__(self):
        return (
            f"<SensorPipeline("
            f"active={self._active_sensors}, "
            f"failed={self._failed_sensors})>"
        )
