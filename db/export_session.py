"""
HERO System - Session Data Exporter
Exports anonymised session data to CSV files organised by age range.

Only anonymised sessions (those that have been processed by anonymise_sessions.py)
can be exported. If a session has not yet been anonymised, the script will
reject it and prompt you to run anonymise_sessions.py first.

Usage:
    # Export a single session:
    python db/export_session.py --session-id <uuid>

    # Export multiple sessions:
    python db/export_session.py --session-id <uuid1> --session-id <uuid2>

    # Non-standard port:
    python db/export_session.py --session-id <uuid> --db-port 5433

    # Custom output directory:
    python db/export_session.py --session-id <uuid> --out-dir /data/exports

Output directory structure:
    exports/
    └── <age_range>/
        └── <session_id>/
            ├── export_manifest.txt
            ├── session_info.csv
            ├── game_results.csv
            ├── sensor_accelerometer.csv
            ├── sensor_gyroscope.csv
            ├── sensor_eeg.csv
            ├── sensor_eye_tracking.csv
            ├── sensor_heart_rate.csv
            ├── sensor_oximeter.csv
            ├── calibration.csv
            ├── calibration_eye_tracking.csv
            └── events.csv
"""

import argparse
import csv
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hero_core.database.models.connection import get_db_connection
from hero_core.database.models.session import TestSession
from hero_core.database.models.anon_demographics import AnonDemographics
from hero_core.database.models.game_results import GameResult
from hero_core.database.models.sensors import (
    SensorAccelerometer, SensorGyroscope, SensorEEG,
    SensorEyeTracking, SensorHeartRate, SensorOximeter,
    CalibrationEyeTracking,
)
from hero_core.database.models.calibration import SensorCalibration
from hero_core.database.models.events import Event

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "user":     "postgres",
    "password": "pgdbadmin",
    "dbname":   "hero_db",
}

SENSOR_TABLES = {
    "sensor_accelerometer": SensorAccelerometer,
    "sensor_gyroscope":     SensorGyroscope,
    "sensor_eeg":           SensorEEG,
    "sensor_eye_tracking":  SensorEyeTracking,
    "sensor_heart_rate":    SensorHeartRate,
    "sensor_oximeter":      SensorOximeter,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_dict(row) -> dict:
    """Convert a SQLAlchemy model instance to a plain dict."""
    return {col.name: getattr(row, col.name) for col in row.__table__.columns}


def _write_csv(filepath: Path, rows: list[dict]) -> int:
    """Write a list of dicts to a CSV file. Returns row count written."""
    if not rows:
        filepath.write_text("# no data\n")
        return 0
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _resolve_uuid(raw: str) -> uuid.UUID:
    """Parse and validate a UUID string, exit cleanly on failure."""
    try:
        return uuid.UUID(raw)
    except ValueError:
        logger.error(f"Invalid UUID: '{raw}'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_session(db_session, session: TestSession, export_root: Path) -> Path:
    """
    Export all data for a single anonymised session to a structured subdirectory.

    Directory structure: <export_root>/<age_range>/<session_id>/

    Args:
        db_session:  SQLAlchemy session.
        session:     TestSession ORM object (must be anonymised).
        export_root: Root exports directory.

    Returns:
        Path to the created export directory.
    """
    demo = db_session.query(AnonDemographics).filter_by(anon_id=session.anon_id).first()
    age_range = demo.age_range if demo else "unknown"

    export_dir = export_root / age_range / str(session.session_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting to: {export_dir}")

    file_counts = {}

    # ---- Session info ----
    session_row = _model_to_dict(session)
    session_row["age_range"] = age_range
    n = _write_csv(export_dir / "session_info.csv", [session_row])
    file_counts["session_info.csv"] = n

    # ---- Game results ----
    game_rows = db_session.query(GameResult).filter_by(session_id=session.session_id).all()
    n = _write_csv(export_dir / "game_results.csv", [_model_to_dict(r) for r in game_rows])
    file_counts["game_results.csv"] = n

    # ---- Sensor tables ----
    for label, model_cls in SENSOR_TABLES.items():
        rows = db_session.query(model_cls).filter_by(session_id=session.session_id).all()
        n = _write_csv(export_dir / f"{label}.csv", [_model_to_dict(r) for r in rows])
        file_counts[f"{label}.csv"] = n

    # ---- Calibration ----
    cal_rows = db_session.query(SensorCalibration).filter_by(session_id=session.session_id).all()
    n = _write_csv(export_dir / "calibration.csv", [_model_to_dict(r) for r in cal_rows])
    file_counts["calibration.csv"] = n

    # ---- Eye tracking calibration ----
    eye_cal = db_session.query(CalibrationEyeTracking).filter_by(session_id=session.session_id).first()
    if eye_cal:
        n = _write_csv(export_dir / "calibration_eye_tracking.csv", [_model_to_dict(eye_cal)])
    else:
        n = _write_csv(export_dir / "calibration_eye_tracking.csv", [])
    file_counts["calibration_eye_tracking.csv"] = n

    # ---- Events ----
    event_rows = db_session.query(Event).filter_by(session_id=session.session_id).order_by(Event.time).all()
    n = _write_csv(export_dir / "events.csv", [_model_to_dict(r) for r in event_rows])
    file_counts["events.csv"] = n

    # ---- Manifest ----
    manifest = [
        "HERO Session Export",
        "===================",
        f"Exported at  : {datetime.now(timezone.utc).isoformat()}",
        f"Session ID   : {session.session_id}",
        f"Anon ID      : {session.anon_id}",
        f"Age range    : {age_range}",
        f"Anonymised at: {session.anonymised_at}",
        f"Session start: {session.started_at}",
        f"Session end  : {session.ended_at}",
        f"Notes        : {session.notes or ''}",
        "",
        "Files",
        "-----",
    ]
    for fname, count in file_counts.items():
        manifest.append(f"  {fname:<36} {count} row(s)")

    (export_dir / "export_manifest.txt").write_text("\n".join(manifest) + "\n")

    logger.info(f"✓ Export complete — {age_range}/{session.session_id}")
    return export_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    CLI entry point for the session exporter.

    Parses --session-id arguments, connects to hero_db, validates that each
    session exists and has been anonymised, then calls export_session() for
    each one. Prints a summary of exported and skipped sessions on completion.

    Exits with code 1 if no sessions were successfully exported.
    """
    parser = argparse.ArgumentParser(
        description="Export anonymised HERO session data to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-id", action="append", dest="session_ids", metavar="UUID",
        required=True,
        help="Session UUID to export. Can be specified multiple times.",
    )
    parser.add_argument(
        "--out-dir", default="exports",
        help="Root output directory (default: ./exports).",
    )
    parser.add_argument(
        "--db-port", type=int, default=5432,
        help="PostgreSQL port (default: 5432).",
    )
    args = parser.parse_args()

    try:
        _, db_session = get_db_connection(**{**DB_CONFIG, "port": args.db_port})
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        sys.exit(1)

    try:
        export_root = Path(args.out_dir)
        export_dirs = []
        skipped     = []

        for raw_id in args.session_ids:
            session_uuid = _resolve_uuid(raw_id)
            session = db_session.query(TestSession).filter_by(session_id=session_uuid).first()

            if session is None:
                logger.warning(f"Session '{raw_id}' not found — skipping.")
                skipped.append((raw_id, "not found"))
                continue

            if not session.is_anonymised:
                logger.warning(
                    f"Session '{raw_id}' has not been anonymised — skipping. "
                    f"Run anonymise_sessions.py first."
                )
                skipped.append((raw_id, "not anonymised"))
                continue

            export_dir = export_session(db_session, session, export_root)
            export_dirs.append(export_dir)

        # Summary
        print(f"\nExported {len(export_dirs)} session(s):")
        for d in export_dirs:
            print(f"  → {d}")

        if skipped:
            print(f"\nSkipped {len(skipped)} session(s):")
            for sid, reason in skipped:
                print(f"  ✗ {sid}  ({reason})")
            if any(r == "not anonymised" for _, r in skipped):
                print("\n  Tip: run anonymise_sessions.py to anonymise sessions before exporting.")

        if not export_dirs:
            sys.exit(1)

    except SQLAlchemyError as e:
        logger.error(f"Database error during export: {e}")
        sys.exit(1)
    finally:
        db_session.close()


if __name__ == "__main__":
    main()