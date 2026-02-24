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
import signal
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

def _force_exit(sig, frame):
    """Ctrl+C / kill signal — exit cleanly."""
    logger.info("Force exit requested")
    pg.quit()
    sys.exit(0)

signal.signal(signal.SIGINT,  _force_exit)
signal.signal(signal.SIGTERM, _force_exit)

def _start_exit_monitor():
    """Background thread: sends SIGINT to self if Ctrl+Z pressed on keyboard."""
    import threading

    def monitor():
        try:
            import evdev, selectors
            devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
            keyboards = [d for d in devices if evdev.ecodes.EV_KEY in d.capabilities()]
            if not keyboards:
                return
            sel = selectors.DefaultSelector()
            for dev in keyboards:
                sel.register(dev, selectors.EVENT_READ)
            ctrl_held = False
            while True:
                for key, _ in sel.select(timeout=0.1):
                    dev = key.fileobj
                    for event in dev.read():
                        if event.type == evdev.ecodes.EV_KEY:
                            if event.code in (evdev.ecodes.KEY_LEFTCTRL, evdev.ecodes.KEY_RIGHTCTRL):
                                ctrl_held = event.value in (1, 2)
                            if event.code == evdev.ecodes.KEY_Z and event.value == 1 and ctrl_held:
                                logger.info("Ctrl+Z detected — forcing exit")
                                os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            pass

    threading.Thread(target=monitor, daemon=True).start()

# ------------------------------------------------------------------
# Config — change these for Pi vs laptop
# ------------------------------------------------------------------

PI_MODE       = True    # Set True on Raspberry Pi
SCALE         = 1.0     # Scale factor for laptop testing
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
    _start_exit_monitor()
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
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1029,600"  # Top physical screen (HDMI-A-2)
        pg.init()
        pg.font.init()
        pg.event.pump()

        # 4. Start sensors (stubbed — session_id set after login)
        pipeline = None

        def on_session_start(session_id):
            """Called after login — runs calibration screen then spawns sensors."""
            nonlocal pipeline

            # Run calibration screen (sensor checks + eye tracking calibration)
            from hero_app.calibration.calibration_screen import CalibrationScreen

            def after_calibration():
                nonlocal pipeline
                pipeline = start_sensors(session_id=session_id)
                import time
                ready_file = f"/tmp/hero_sensors_ready_{session_id}"
                for _ in range(30):
                    if os.path.exists(ready_file):
                        os.remove(ready_file)
                        logger.info("✓ Sensors ready")
                        break
                    time.sleep(0.5)
                else:
                    logger.warning("⚠ Sensors did not signal ready in time")

                # Start gaze collection in background (camera already open from calibration)
                if calib_screen.gaze_system:
                    try:
                        calib_screen.gaze_system.begin_collection()
                        logger.info("✓ Gaze collection started")
                    except Exception as e:
                        logger.warning(f"Could not start gaze collection: {e}")

            # Get the pygame window from the consultation — reuse it
            # We draw on the top half (display_size height)
            display_size = pg.Vector2(1024, 600) * SCALE
            window = pg.display.get_surface()

            # Get db session for calibration
            _, calib_db_session = db.engine.connect(), None
            try:
                from hero_core.database.models.connection import get_db_connection
                _, calib_db_session = get_db_connection(
                    host='localhost', port=5432, user='postgres',
                    password='pgdbadmin', dbname='hero_db'
                )
            except Exception as e:
                logger.warning(f"Could not open calibration DB session: {e}")

            calib_screen = CalibrationScreen(
                session_id=session_id,
                db_session=calib_db_session,
                display_size=display_size,
                window=window,
                on_complete=after_calibration,
                pi=PI_MODE,
            )
            calib_screen.run()

            if calib_db_session:
                calib_db_session.close()

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
        # Stop gaze collection if running
        try:
            if 'calib_screen' in dir() and calib_screen.gaze_system:
                calib_screen.gaze_system.stop()
        except Exception:
            pass
        pg.quit()
        import subprocess
        subprocess.run(['sudo', 'pkill', '-f', 'depthai'], capture_output=True)
        print("\n✓ HERO session complete.")


if __name__ == '__main__':
    main()
    
