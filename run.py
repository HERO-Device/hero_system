"""
HERO System - Main Entry Point
Orchestrates the full session flow:
  1. Connect to DB
  2. Start central clock
  3. Login patient
  4. Start test session in DB
  5. Start sensors
  6. Run cognitive games (shared clock passed in)
  7. Stop sensors
  8. End session in DB
  9. Print session summary
 10. Wait for Vol Down button or Enter to close
"""

import os
import sys
import signal
import logging
import pygame as pg

# Ensure hero_system root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hero_system"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hero_app', 'session'))

from db.db_access import HeroDB
from hero_core.coordinator.clock import CentralClock
from hero_app.session.consultation.orchestrator import Consultation
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('hero')


def _force_exit(sig, frame):
    logger.info("Force exit requested")
    pg.quit()
    sys.exit(0)

signal.signal(signal.SIGINT,  _force_exit)
signal.signal(signal.SIGTERM, _force_exit)


def _start_exit_monitor():
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


PI_MODE       = True
SCALE         = 1
ENABLE_SPEECH = False
VOL_DOWN_PIN  = 27


def start_sensors(session_id, clock=None, db_session=None):
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
    if proc and proc.poll() is None:
        logger.info(f"Stopping sensor process (PID {proc.pid})...")
        proc.terminate()
        proc.wait(timeout=30)
        logger.info("✓ Sensor process stopped")


def print_session_summary(db, session_id):
    from hero_core.database.models.session import TestSession
    from hero_core.database.models.game_results import GameResult
    from hero_core.database.models.events import Event
    from hero_core.database.models.user import User
    try:
        W = 60
        print()
        print("=" * W)
        print("  SESSION SUMMARY")
        print("=" * W)
        session = db.session.query(TestSession).filter_by(session_id=session_id).first()
        if session:
            user = db.session.query(User).filter_by(user_id=session.user_id).first()
            username = user.username if user else str(session.user_id)
            duration = (session.ended_at - session.started_at).total_seconds() \
                       if session.ended_at else None
            print(f"  Patient   : {username}")
            print(f"  Session ID: {session_id}")
            print(f"  Started   : {session.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if duration:
                print(f"  Duration  : {int(duration)}s ({int(duration)//60}m {int(duration)%60}s)")
        print("-" * W)
        results = db.session.query(GameResult).filter_by(
            session_id=session_id
        ).order_by(GameResult.game_number).all()
        if results:
            print(f"  {'Game':<12} {'Score':>6} {'Max':>6} {'Acc%':>6} {'Dur(s)':>8} {'Status':<12}")
            print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*12}")
            for r in results:
                score  = str(r.final_score)          if r.final_score        is not None else '-'
                max_s  = str(r.max_score)            if r.max_score          is not None else '-'
                acc    = f"{r.accuracy_percent:.1f}" if r.accuracy_percent   is not None else '-'
                dur    = f"{r.duration_seconds:.1f}" if r.duration_seconds   is not None else '-'
                status = r.completion_status or '-'
                print(f"  {r.game_name:<12} {score:>6} {max_s:>6} {acc:>6} {dur:>8} {status:<12}")
        else:
            print("  No game results recorded.")
        print("-" * W)
        try:
            from sqlalchemy import text
            sensor_tables = ['eeg_data', 'accelerometer_data', 'heart_rate_data', 'gaze_data']
            print("  Sensor data rows:")
            for table in sensor_tables:
                try:
                    count = db.session.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE session_id = :sid"),
                        {'sid': str(session_id)}
                    ).scalar()
                    print(f"    {table:<25}: {count:>6} rows")
                except Exception:
                    print(f"    {table:<25}: (table not found)")
        except Exception as e:
            print(f"  Could not query sensor tables: {e}")
        event_count = db.session.query(Event).filter_by(session_id=session_id).count()
        print(f"  Events logged : {event_count}")
        print("=" * W)
    except Exception as e:
        print(f"  Could not generate summary: {e}")


def wait_for_close():
    print("\n  Press VOL DOWN button or Enter to close...")
    btn_request = None
    if PI_MODE:
        try:
            import gpiod
            from gpiod.line import Direction, Bias
            btn_request = gpiod.request_lines(
                '/dev/gpiochip0',
                consumer='HeroClose',
                config={VOL_DOWN_PIN: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    bias=Bias.PULL_UP,
                )}
            )
        except Exception as e:
            logger.warning(f"Could not initialise Vol Down button: {e}")
            btn_request = None

    import threading, time
    done = threading.Event()

    def wait_enter():
        try:
            input()
        except Exception:
            pass
        done.set()

    threading.Thread(target=wait_enter, daemon=True).start()

    if btn_request:
        from gpiod.line import Value
        prev = btn_request.get_value(VOL_DOWN_PIN)
        while not done.is_set():
            val = btn_request.get_value(VOL_DOWN_PIN)
            if val == Value.INACTIVE and prev == Value.ACTIVE:
                done.set()
                break
            prev = val
            time.sleep(0.02)
        try:
            btn_request.release()
        except Exception:
            pass
    else:
        done.wait()


def main():
    print()
    print("=" * 50)
    print("  HERO System")
    print("=" * 50)
    print()

    subprocess.run(['sudo', 'pkill', '-f', 'depthai'], capture_output=True)

    logger.info("Connecting to database...")
    _start_exit_monitor()
    try:
        db = HeroDB()
    except Exception as e:
        logger.error(f"Cannot connect to database: {e}")
        print("\n✗ Could not connect to the database.")
        print("  Make sure PostgreSQL is running.")
        sys.exit(1)

    clock = CentralClock()
    logger.info(f"✓ Central clock started: {clock}")

    session_id_holder = [None]

    try:
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1029,600"
        pg.init()
        pg.font.init()
        pg.event.pump()

        pipeline = None

        def on_session_start(session_id):
            nonlocal pipeline
            session_id_holder[0] = session_id

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

                if calib_screen.gaze_system:
                    try:
                        calib_screen.gaze_system.begin_collection()
                        logger.info("✓ Gaze collection started")
                    except Exception as e:
                        logger.warning(f"Could not start gaze collection: {e}")

            display_size = pg.Vector2(1024, 600) * SCALE
            window = pg.display.get_surface()

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

        logger.info("Starting consultation...")
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hero_app', 'session'))
        consult = Consultation(
            pi=PI_MODE,
            authenticate=True,
            seamless=True,
            local=True,
            enable_speech=ENABLE_SPEECH,
            scale=SCALE,
            db=db,
            clock=clock,
            on_session_start=on_session_start,
        )

        consult.loop()
        stop_sensors(pipeline)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    finally:
        if session_id_holder[0]:
            print_session_summary(db, session_id_holder[0])

        db.close()

        try:
            if 'calib_screen' in dir() and calib_screen.gaze_system:
                calib_screen.gaze_system.stop()
        except Exception:
            pass

        pg.quit()
        subprocess.run(['sudo', 'pkill', '-f', 'depthai'], capture_output=True)

        print("\n✓ HERO session complete.")
        wait_for_close()


if __name__ == '__main__':
    main()
