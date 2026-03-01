"""
HERO System - Session Data Exporter
Pulls all data for one or more sessions into a local subdirectory.

Works with both identified sessions (pre-anonymisation, still linked to a user)
and anonymised sessions (post-anonymisation, linked to anon_demographics).

Usage:
    # Export a single session by ID:
    python db/export_session.py --session-id <uuid>

    # Export multiple sessions:
    python db/export_session.py --session-id <uuid1> --session-id <uuid2>

    # Non-standard port:
    python db/export_session.py --session-id <uuid> --db-port 5433

    # Custom output directory:
    python db/export_session.py --session-id <uuid> --out-dir /data/exports

Output directory structure:
    exports/
    └── <participant_label>_<session_id_short>_<timestamp>/
        ├── export_manifest.txt
        ├── session_info.csv
        ├── game_results.csv
        ├── sensor_accelerometer.csv
        ├── sensor_gyroscope.csv
        ├── sensor_eeg.csv
        ├── sensor_eye_tracking.csv
        ├── sensor_heart_rate.csv
        ├── sensor_oximeter.csv
        └── calibration.csv

    For identified sessions, participant_label = username.
    For anonymised sessions, participant_label = anon_id (e.g. "testing_41-50_1").
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
from hero_core.database.models.user import User
from hero_core.database.models.session import TestSession
from hero_core.database.models.anon_demographics import AnonDemographics
from hero_core.database.models.game_results import GameResult
from hero_core.database.models.sensors import (
    SensorAccelerometer, SensorGyroscope, SensorEEG,
    SensorEyeTracking, SensorHeartRate, SensorOximeter,
)
from hero_core.database.models.calibration import SensorCalibration

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


def _participant_label(session: TestSession, db_session) -> tuple[str, dict]:
    """
    Return a short human-readable label for the session participant
    and a dict of participant metadata to include in session_info.csv.

    For identified sessions: uses username.
    For anonymised sessions: uses anon_id + age_range.
    """
    if session.is_anonymised:
        demo = db_session.query(AnonDemographics).filter_by(anon_id=session.anon_id).first()
        label = session.anon_id
        meta  = {
            "participant_type": "anonymised",
            "anon_id":          session.anon_id,
            "age_range":        demo.age_range if demo else "unknown",
            "anonymised_at":    str(session.anonymised_at),
        }
    else:
        user  = db_session.query(User).filter_by(user_id=session.user_id).first()
        label = user.username if user else str(session.user_id)[:8]
        meta  = {
            "participant_type": "identified",
            "username":         user.username  if user else None,
            "full_name":        user.full_name if user else None,
            "user_id":          str(session.user_id),
        }

    return label, meta


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_session(db_session, session: TestSession, export_root: Path) -> Path:
    """
    Export all data for a single TestSession to a timestamped subdirectory.

    Handles both identified and anonymised sessions transparently —
    participant metadata is resolved from whichever table is appropriate.

    Args:
        db_session:  SQLAlchemy session.
        session:     TestSession ORM object.
        export_root: Parent directory where the export folder will be created.

    Returns:
        Path to the created export directory.
    """
    participant_label, participant_meta = _participant_label(session, db_session)

    ts_tag    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sid_short = str(session.session_id)[:8]
    export_dir = export_root / f"{participant_label}_{sid_short}_{ts_tag}"
    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting to: {export_dir}")

    file_counts = {}

    # ---- Session info ----
    session_row = _model_to_dict(session)
    session_row.update(participant_meta)   # Merge participant fields in
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

    # ---- Manifest ----
    manifest = [
        "HERO Session Export",
        "===================",
        f"Exported at  : {datetime.now(timezone.utc).isoformat()}",
        f"Session ID   : {session.session_id}",
        f"Session start: {session.started_at}",
        f"Session end  : {session.ended_at}",
        f"Notes        : {session.notes or ''}",
        f"Anonymised   : {'yes — ' + str(session.anonymised_at) if session.is_anonymised else 'no'}",
        "",
        "Participant",
        "-----------",
    ]
    for k, v in participant_meta.items():
        manifest.append(f"  {k:<20}: {v}")
    manifest += [
        "",
        "Files",
        "-----",
    ]
    for fname, count in file_counts.items():
        manifest.append(f"  {fname:<32} {count} row(s)")

    (export_dir / "export_manifest.txt").write_text("\n".join(manifest) + "\n")

    logger.info(f"✓ Export complete — {export_dir.name}")
    return export_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export HERO session data to CSV files.",
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

        for raw_id in args.session_ids:
            session_uuid = _resolve_uuid(raw_id)
            session = db_session.query(TestSession).filter_by(session_id=session_uuid).first()

            if session is None:
                logger.warning(f"Session '{raw_id}' not found — skipping.")
                continue

            export_dir = export_session(db_session, session, export_root)
            export_dirs.append(export_dir)

        if not export_dirs:
            logger.error("No sessions were exported.")
            sys.exit(1)

        print(f"\nExported {len(export_dirs)} session(s):")
        for d in export_dirs:
            print(f"  → {d}")

    except SQLAlchemyError as e:
        logger.error(f"Database error during export: {e}")
        sys.exit(1)
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
