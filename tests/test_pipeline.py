#!/usr/bin/env python3
"""
HERO System - Standalone Pipeline Test
=======================================
Tests SensorPipeline in isolation — no pygame, no consultation UI.

Usage:
    cd /home/hero/hero_system
    python tests/test_pipeline.py              # Run all sensors for 15s
    python tests/test_pipeline.py --duration 30
    python tests/test_pipeline.py --sensors mpu6050 max30102   # subset only

The script:
  1. Creates a real DB connection + test user/session
  2. Starts the SensorPipeline
  3. Prints live sample counts every 2 seconds
  4. Stops after --duration seconds
  5. Queries the DB to verify rows actually landed
  6. Cleans up the test session

Sensor failures are expected during development (e.g. EEG Bluetooth not
connected) — the pipeline will skip them and continue. The test passes as
long as at least one sensor commits data successfully.
"""

import sys
import os
import time
import uuid
import argparse
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup — run from hero_system root
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERO_DEVICE = os.path.dirname(ROOT)  # /home/hero/HERO-Device

sys.path.insert(0, ROOT)                                      # hero_system/
sys.path.insert(0, os.path.join(HERO_DEVICE, 'hero_core'))   # hero_core/

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from hero_core.coordinator.clock import CentralClock
from hero_system.pipeline import SensorPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('pipeline_test')

# ---------------------------------------------------------------------------
# DB config — matches db_access.py
# ---------------------------------------------------------------------------
DB_URL = 'postgresql://postgres:pgdbadmin@localhost:5432/hero_db'

# ---------------------------------------------------------------------------
# DB table → sample count queries
# ---------------------------------------------------------------------------
SENSOR_TABLES = {
    'mpu6050'     : ['sensor_accelerometer', 'sensor_gyroscope'],
    'max30102'    : ['sensor_heart_rate', 'sensor_oximeter'],
    'eeg'         : ['sensor_eeg'],
    'eye_tracking': ['sensor_eye_tracking'],
}


def create_test_session(db_session, username: str) -> tuple:
    """Insert a throwaway user + session, return (user_id, session_id)."""
    user_id    = uuid.uuid4()
    session_id = uuid.uuid4()

    db_session.execute(
        text("INSERT INTO users (user_id, username) VALUES (:uid, :uname)"),
        {'uid': user_id, 'uname': username}
    )
    db_session.execute(
        text("INSERT INTO test_sessions (session_id, user_id) VALUES (:sid, :uid)"),
        {'sid': session_id, 'uid': user_id}
    )
    db_session.commit()
    logger.info(f"✓ Test session created: {session_id}")
    return user_id, session_id


def cleanup_test_session(db_session, user_id, session_id):
    """Remove the test rows — CASCADE deletes sensor data too."""
    try:
        db_session.execute(
            text("DELETE FROM test_sessions WHERE session_id = :sid"),
            {'sid': session_id}
        )
        db_session.execute(
            text("DELETE FROM users WHERE user_id = :uid"),
            {'uid': user_id}
        )
        db_session.commit()
        logger.info("✓ Test session cleaned up")
    except Exception as e:
        logger.warning(f"Cleanup failed (non-fatal): {e}")
        db_session.rollback()


def count_rows(db_session, table: str, session_id) -> int:
    """Count rows in a sensor table for this session."""
    try:
        result = db_session.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE session_id = :sid"),
            {'sid': session_id}
        )
        return result.scalar() or 0
    except Exception as e:
        logger.warning(f"Could not count {table}: {e}")
        return -1


def print_live_status(pipeline: SensorPipeline, db_session, session_id):
    """Print a snapshot of collector sample counts."""
    lines = []

    if pipeline._mpu6050_collector:
        c = pipeline._mpu6050_collector
        lines.append(
            f"  MPU6050    accel={c.accel_sample_count:>6}  gyro={c.gyro_sample_count:>6}"
        )

    if pipeline._max30102_collector:
        c = pipeline._max30102_collector
        lines.append(
            f"  MAX30102   samples={c.sample_count:>6}"
        )

    if pipeline._eeg_collector:
        c = pipeline._eeg_collector
        lines.append(
            f"  EEG        samples={c.sample_count:>6}"
        )

    if pipeline._eye_tracking_processor:
        p = pipeline._eye_tracking_processor
        # EyeTrackingProcessor may expose sample_count differently
        count = getattr(p, 'sample_count', getattr(p, 'frame_count', '?'))
        lines.append(
            f"  EyeTrack   frames={count}"
        )

    if not lines:
        lines.append("  (no active sensors)")

    print("\n".join(lines))


def verify_db_counts(db_session, session_id, active_sensors: list) -> dict:
    """Query DB for row counts per sensor table. Returns {table: count}."""
    results = {}
    for sensor in active_sensors:
        for table in SENSOR_TABLES.get(sensor, []):
            results[table] = count_rows(db_session, table, session_id)
    return results


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(duration: int):
    print()
    print("=" * 60)
    print("  HERO Sensor Pipeline — Standalone Test")
    print("=" * 60)
    print()

    # 1. Connect to DB
    print("Connecting to database...")
    try:
        engine     = create_engine(DB_URL)
        DbSession  = sessionmaker(bind=engine)
        db_session = DbSession()
        db_session.execute(text("SELECT 1"))
        print("✓ Connected to hero_db\n")
    except Exception as e:
        print(f"✗ DB connection failed: {e}")
        sys.exit(1)

    # 2. Create throwaway session
    username = f"pipeline_test_{int(time.time())}"
    user_id, session_id = create_test_session(db_session, username)

    # 3. Start pipeline
    clock    = CentralClock()
    pipeline = SensorPipeline(
        session_id=session_id,
        db_session=db_session,
        clock=clock,
    )

    print("\nInitialising sensors...")
    print("-" * 60)
    pipeline.start()
    print("-" * 60)

    status = pipeline.get_status()
    print(f"\n  Active  : {status['active_sensors'] or 'none'}")
    print(f"  Failed  : {status['failed_sensors'] or 'none'}")
    print()

    if not status['active_sensors']:
        print("⚠ No sensors came up — check hardware connections.")
        cleanup_test_session(db_session, user_id, session_id)
        db_session.close()
        sys.exit(1)

    # 4. Collect for duration seconds, printing status every 2s
    print(f"Collecting for {duration} seconds...\n")
    start      = time.time()
    last_print = 0

    try:
        while time.time() - start < duration:
            elapsed = time.time() - start

            if time.time() - last_print >= 2.0:
                print(f"[{elapsed:5.1f}s]")
                print_live_status(pipeline, db_session, session_id)
                print()
                last_print = time.time()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted early")

    # 5. Stop pipeline
    print("\nStopping pipeline...")
    pipeline.stop()
    print("✓ Pipeline stopped\n")

    # 6. Verify DB
    print("Verifying database rows...")
    print("-" * 60)
    db_counts  = verify_db_counts(db_session, session_id, status['active_sensors'])
    all_passed = True

    for table, count in db_counts.items():
        if count > 0:
            print(f"  ✓ {table:<30} {count:>6} rows")
        elif count == 0:
            print(f"  ✗ {table:<30}      0 rows  ← nothing committed!")
            all_passed = False
        else:
            print(f"  ? {table:<30}   error querying")

    # Also check sensor_calibration
    cal_count = count_rows(db_session, 'sensor_calibration', session_id)
    print(f"  ✓ {'sensor_calibration':<30} {cal_count:>6} rows  (init records)")

    print("-" * 60)
    print()

    # 7. Cleanup
    cleanup_test_session(db_session, user_id, session_id)
    db_session.close()

    # 8. Result
    if all_passed:
        print("✓ PIPELINE TEST PASSED — all active sensors committed data\n")
        sys.exit(0)
    else:
        print("✗ PIPELINE TEST FAILED — some sensors had 0 DB rows\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HERO Sensor Pipeline Test')
    parser.add_argument(
        '--duration', type=int, default=15,
        help='How long to collect data in seconds (default: 15)'
    )
    args = parser.parse_args()

    run_test(duration=args.duration)
    
