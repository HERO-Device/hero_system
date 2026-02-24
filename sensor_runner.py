"""
HERO Sensor Runner - Standalone sensor collection process
Spawned by run.py after session is created.
Usage: python sensor_runner.py <session_id>
"""
import os
import sys
import signal
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hero_app', 'session'))

from hero_core.coordinator.clock import CentralClock
from hero_core.database.models.connection import get_db_connection
from hero_system.pipeline import SensorPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('sensor_runner')

def main():
    if len(sys.argv) < 2:
        print("Usage: sensor_runner.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    logger.info(f"Sensor runner starting for session {session_id}")
    fh = logging.FileHandler(f'/tmp/hero_sensor_{session_id[:8]}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(fh)

    _, db_session = get_db_connection(
        host='localhost', port=5432, user='postgres',
        password='pgdbadmin', dbname='hero_db'
    )

    clock = CentralClock()
    pipeline = SensorPipeline(session_id=session_id, clock=clock, db_session=db_session)

    # Handle SIGTERM gracefully
    def shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping pipeline...")
        pipeline.stop()
        db_session.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    pipeline.start()
    logger.info(f"✓ Pipeline active: {pipeline.get_status()['active_sensors']}")

    # Signal ready to run.py
    ready_file = f"/tmp/hero_sensors_ready_{session_id}"
    open(ready_file, 'w').close()

    # Keep alive until signal
    signal.pause()

if __name__ == '__main__':
    main()
