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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('hero')

# ------------------------------------------------------------------
# Config — change these for Pi vs laptop
# ------------------------------------------------------------------

PI_MODE       = False   # Set True on Raspberry Pi
SCALE         = 0.8     # Scale factor for laptop testing
ENABLE_SPEECH = False   # Set True on Pi with audio configured


# ------------------------------------------------------------------
# Sensor stubs — replace with real SensorCoordinator when ready
# ------------------------------------------------------------------

def start_sensors(session_id, clock):
    """Stub — replace with real sensor startup when hardware ready."""
    logger.info("⚠ Sensors stubbed — skipping hardware init")
    # When ready:
    # from hero_system.sensors.imu import IMUSensor
    # from hero_core.coordinator.coordinator import SensorCoordinator
    # coordinator = SensorCoordinator(session_id=session_id, db_session=db.session)
    # coordinator.register_sensor('imu', IMUSensor())
    # coordinator.start_all_sensors()
    # return coordinator
    return None


def stop_sensors(coordinator):
    """Stub — replace with real sensor shutdown when hardware ready."""
    logger.info("⚠ Sensors stubbed — skipping hardware stop")
    # When ready:
    # if coordinator:
    #     coordinator.stop_all_sensors()


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
        coordinator = start_sensors(session_id=None, clock=clock)

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
        )

        consult.loop()

        # 6. Stop sensors
        stop_sensors(coordinator)

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
    
