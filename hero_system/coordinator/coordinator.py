"""
Sensor Coordinator
Manages lifecycle and synchronization of all HERO system sensors
"""

import logging
from typing import Dict, Optional, Any
from uuid import UUID
from sqlalchemy.orm import Session

from .clock import CentralClock

logger = logging.getLogger(__name__)


class SensorCoordinator:
    """
    Coordinates multiple sensors with synchronized timestamps

    Responsibilities:
    - Manage sensor lifecycle (start/stop)
    - Provide central timestamp synchronization
    - Track sensor status
    - Handle graceful shutdown
    """

    def __init__(self, session_id: UUID, db_session: Session):
        """
        Initialize sensor coordinator

        Args:
            session_id: UUID of the current test session
            db_session: SQLAlchemy database session
        """
        self.session_id = session_id
        self.db_session = db_session

        # Central clock for timestamp synchronization
        self.clock = CentralClock()

        # Sensor registry
        self.sensors: Dict[str, Any] = {}
        self.sensor_configs: Dict[str, Any] = {}

        logger.info(f"Sensor Coordinator initialized for session {session_id}")

    def get_central_timestamp(self):
        """
        Get synchronized timestamp for all sensors
        Thread-safe central time reference

        Returns:
            datetime: Current synchronized timestamp
        """
        return self.clock.now()

    def register_sensor(self, sensor_name: str, sensor_instance: Any, config: Optional[Any] = None):
        """
        Register a sensor with the coordinator

        Args:
            sensor_name: Unique identifier for sensor (e.g., 'eeg', 'eye_tracking')
            sensor_instance: Sensor object instance
            config: Optional sensor configuration
        """
        if sensor_name in self.sensors:
            logger.warning(f"Sensor '{sensor_name}' already registered, replacing")

        self.sensors[sensor_name] = sensor_instance
        if config:
            self.sensor_configs[sensor_name] = config

        logger.info(f"✓ Registered sensor: {sensor_name}")

    def start_sensor(self, sensor_name: str):
        """
        Start a registered sensor

        Args:
            sensor_name: Name of sensor to start
        """
        if sensor_name not in self.sensors:
            logger.error(f"Sensor '{sensor_name}' not registered")
            raise ValueError(f"Unknown sensor: {sensor_name}")

        try:
            sensor = self.sensors[sensor_name]
            sensor.start()
            logger.info(f"✓ Started sensor: {sensor_name}")
        except Exception as e:
            logger.error(f"✗ Failed to start sensor '{sensor_name}': {e}", exc_info=True)
            raise

    def stop_sensor(self, sensor_name: str):
        """
        Stop a registered sensor

        Args:
            sensor_name: Name of sensor to stop
        """
        if sensor_name not in self.sensors:
            logger.warning(f"Sensor '{sensor_name}' not registered")
            return

        try:
            sensor = self.sensors[sensor_name]
            sensor.stop()
            logger.info(f"✓ Stopped sensor: {sensor_name}")
        except Exception as e:
            logger.error(f"✗ Error stopping sensor '{sensor_name}': {e}", exc_info=True)

    def start_all_sensors(self):
        """Start all registered sensors"""
        logger.info(f"Starting {len(self.sensors)} sensors...")

        for sensor_name in self.sensors:
            try:
                self.start_sensor(sensor_name)
            except Exception as e:
                logger.error(f"Failed to start {sensor_name}, continuing with others")

        logger.info("✓ All sensors started")

    def stop_all_sensors(self):
        """Stop all registered sensors"""
        logger.info(f"Stopping {len(self.sensors)} sensors...")

        for sensor_name in self.sensors:
            try:
                self.stop_sensor(sensor_name)
            except Exception as e:
                logger.error(f"Error stopping {sensor_name}, continuing with others")

        logger.info("✓ All sensors stopped")

    def get_sensor_status(self, sensor_name: str) -> Optional[dict]:
        """
        Get status of a specific sensor

        Args:
            sensor_name: Name of sensor

        Returns:
            dict: Sensor status or None if not found
        """
        if sensor_name not in self.sensors:
            return None

        sensor = self.sensors[sensor_name]

        # Try to get status if sensor implements get_status()
        if hasattr(sensor, 'get_status'):
            return sensor.get_status()

        return {'sensor_name': sensor_name, 'registered': True}

    def get_all_status(self) -> Dict[str, dict]:
        """
        Get status of all sensors

        Returns:
            dict: Mapping of sensor names to their status
        """
        status = {}
        for sensor_name in self.sensors:
            status[sensor_name] = self.get_sensor_status(sensor_name)

        return status

    def get_coordinator_status(self) -> dict:
        """
        Get overall coordinator status

        Returns:
            dict: Coordinator status information
        """
        return {
            'session_id': str(self.session_id),
            'registered_sensors': list(self.sensors.keys()),
            'sensor_count': len(self.sensors),
            'clock_stats': self.clock.get_stats(),
            'sensors': self.get_all_status(),
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.stop_all_sensors()

    def __repr__(self):
        return f"<SensorCoordinator(session={self.session_id}, sensors={len(self.sensors)})>"