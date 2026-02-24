"""
HERO System - Main Entry Point
Orchestrates the full session flow:
  1. Connect to DB
  2. Start central clock
  3. Login patient
  4. Start test session in DB
  5. Start sensors (stubbed for now)
  6. Run cognitive games (shared clock passed in)
  7. Stop sensors
  8. End session in DB
"""

import os
import sys
import logging
import pygame as pg

# Ensure hero_system root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hero_app', 'session'))

from db.db_access import HeroDB
from hero_core.coordinator.clock import CentralClock
from hero_app.session.hero.consultation.orchestrator import Consultation
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('hero')

# ------------------------------------------------------------------
# Config — change these for Pi vs laptop
# ------------------------------------------------------------------

PI_MODE       = False   # Set True on Raspberry Pi
SCALE         = 0.9     # Scale factor for laptop testing
ENABLE_SPEECH = False   # Set True on Pi with audio configured


# ------------------------------------------------------------------
# Sensor stubs — replace with real SensorCoordinator when ready
# ------------------------------------------------------------------

def start_sensors(session_id, clock=None, db_session=None):
    """Spawn sensor collection as an independent process."""
    logger.info(f"Spawning sensor process for session {session_id}...")
    try:
        venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '.venv', 'bin', 'python3')
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensor_runner.py')
        proc = subprocess.Popen(
            [venv_python, runner, str(session_id)],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        logger.info(f"✓ Sensor process started (PID {proc.pid})")
        return proc
    except Exception as e:
        logger.error(f"✗ Sensor process failed: {e}")
        return None


def stop_sensors(proc):
    """Stop the sensor process."""
    if proc and proc.poll() is None:
        logger.info(f"Stopping sensor process (PID {proc.pid})...")
        proc.terminate()
        proc.wait(timeout=30)
        logger.info("✓ Sensor process stopped")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print()
    print("=" * 50)
    print("  HERO System")
    print("=" * 50)
    print()

    # 1. Connect to DB
    logger.info("Connecting to database...")
    try:
        db = HeroDB()
    except Exception as e:
        logger.error(f"Cannot connect to database: {e}")
        print("\n✗ Could not connect to the database.")
        print("  Make sure PostgreSQL is running.")
        sys.exit(1)

    # 2. Start central clock — shared by sensors and games
    clock = CentralClock()
    logger.info(f"✓ Central clock started: {clock}")

    try:
        # 3. Init pygame
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pg.init()
        pg.font.init()
        pg.event.pump()

        # 4. Start sensors (stubbed — session_id set after login)
        pipeline = None

        def on_session_start(session_id):
            nonlocal pipeline
            import time
            pipeline = start_sensors(session_id=session_id)
            # Wait for sensors to be ready (max 15s)
            ready_file = f"/tmp/hero_sensors_ready_{session_id}"
            for _ in range(30):
                if os.path.exists(ready_file):
                    os.remove(ready_file)
                    logger.info("✓ Sensors ready")
                    break
                time.sleep(0.5)
            else:
                logger.warning("⚠ Sensors did not signal ready in time")

        coordinator = None  # Will be set via callback

        # 5. Run consultation — pass in db and shared clock
        logger.info("Starting consultation...")
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hero_app', 'session'))
        consult = Consultation(
            pi=PI_MODE,
            authenticate=True,
            seamless=True,
            local=True,          # Also save locally as JSON backup
            enable_speech=ENABLE_SPEECH,
            scale=SCALE,
            db=db,
            clock=clock,         # Shared clock for synchronized timestamps
            on_session_start=on_session_start,
        )

        consult.loop()

        # 6. Stop sensors
        stop_sensors(pipeline)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    finally:
        db.close()
        pg.quit()
        print("\n✓ HERO session complete.")


if __name__ == '__main__':
    main()
    
