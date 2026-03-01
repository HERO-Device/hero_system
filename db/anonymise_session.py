"""
HERO System - Ethics Anonymisation Runner
Anonymises sessions that have passed the retention threshold.

Anonymisation means:
  1. A row is created in anon_demographics with a cohort label and age range
     (or an existing matching row is reused).
  2. The session's user_id FK is set to NULL and anon_id is set to the
     new demographics row.
  3. The user's PII is scrubbed from the users table once ALL of that
     user's sessions have been anonymised.
  4. Every action is recorded in data_lifecycle_log for audit purposes.

The central anonymisation timestamp is session.started_at — the clinically
meaningful moment that matches all sensor hypertable timestamps.

Age ranges (neurodegenerative disease research population):
    18-24, 25-30, 31-40, 41-50, 51-60, 61-70

Usage:
    # Always dry-run first:
    python db/anonymise_sessions.py --dry-run

    # Live run:
    python db/anonymise_sessions.py

    # Override retention threshold (days):
    python db/anonymise_sessions.py --days 14

    # Non-standard port:
    python db/anonymise_sessions.py --db-port 5433

    # Print the SQL migration to run once before first use:
    python db/anonymise_sessions.py --print-migration
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hero_core.database.models.connection import get_db_connection
from hero_core.database.models.user import User
from hero_core.database.models.session import TestSession
from hero_core.database.models.anon_demographics import AnonDemographics, AGE_RANGES
from hero_core.database.models.ethics import DataLifecycleLog
from hero_core.config.config import ETHICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "user":     "postgres",
    "password": "pgdbadmin",
    "dbname":   "hero_db",
}

MIGRATION_NOTE = """
-- Run 08_anonymisation.sql from hero_core/database/schema/ before using this script:
--
--   PGPASSWORD=pgdbadmin psql -U postgres -h localhost -d hero_db \\
--       -f hero_core/database/schema/08_anonymisation.sql
--
-- That migration:
--   - Creates the anon_demographics table
--   - Drops NOT NULL on test_sessions.user_id
--   - Adds test_sessions.anon_id, test_sessions.anonymised_at
--   - Adds the participant_link_exclusive CHECK constraint
--   - Patches users table with is_anonymized / anonymized_at columns
"""


# ---------------------------------------------------------------------------
# Age range calculation
# ---------------------------------------------------------------------------

# Half-open intervals [low, high) mapping to label
# Covers 18–70, which is the full HERO research population.
_AGE_BANDS = [
    (18, 25, "18-24"),
    (25, 31, "25-30"),
    (31, 41, "31-40"),
    (41, 51, "41-50"),
    (51, 61, "51-60"),
    (61, 71, "61-70"),
]


def compute_age_range(date_of_birth: datetime, reference_date: datetime) -> str:
    """
    Return the age-range band for a patient at the time of their session.

    Using session.started_at rather than today ensures the demographic stub
    reflects the patient's age at assessment, which is what matters for
    longitudinal ML analysis.

    Args:
        date_of_birth:  Patient DOB.
        reference_date: Session started_at timestamp.

    Returns:
        Age range string e.g. "41-50", or "unknown" if DOB is None.
    """
    if date_of_birth is None:
        return "unknown"

    dob = date_of_birth.replace(tzinfo=None) if date_of_birth.tzinfo else date_of_birth
    ref = reference_date.replace(tzinfo=None) if reference_date.tzinfo else reference_date
    age = (ref - dob).days // 365

    for low, high, label in _AGE_BANDS:
        if low <= age < high:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Cohort label generation
# ---------------------------------------------------------------------------

def make_cohort_label(age_range: str, existing_labels: set[str]) -> str:
    """
    Generate the next available cohort label for a given age range.

    Labels follow the pattern: testing_<age_range>_<n>
    e.g. "testing_41-50_1", "testing_41-50_2", ...

    Args:
        age_range:       Age band string e.g. "41-50".
        existing_labels: Set of anon_id values already in the DB.

    Returns:
        Next unused label string.
    """
    n = 1
    while True:
        candidate = f"testing_{age_range}_{n}"
        if candidate not in existing_labels:
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Core anonymisation logic
# ---------------------------------------------------------------------------

def get_or_create_anon_demographics(
    db_session,
    age_range: str,
    existing_labels: set[str],
    dry_run: bool,
) -> str:
    """
    Find an existing anon_demographics row for this age range, or create one.

    Each unique age range gets its own cohort label. If one already exists
    for this band, we reuse it — all sessions in the same age band share
    a single demographics record.

    Args:
        db_session:      SQLAlchemy session.
        age_range:       Age band string.
        existing_labels: Current set of anon_id values (updated in-place).
        dry_run:         If True, simulate but don't write.

    Returns:
        The anon_id string to assign to the session.
    """
    # Try to reuse an existing row for this age range
    existing = db_session.query(AnonDemographics).filter_by(age_range=age_range).first()
    if existing:
        return existing.anon_id

    # Create a new one
    label = make_cohort_label(age_range, existing_labels)
    existing_labels.add(label)

    if not dry_run:
        demo = AnonDemographics(anon_id=label, age_range=age_range)
        db_session.add(demo)
        db_session.flush()  # Ensure the row exists before sessions FK to it

    return label


def anonymise_session(
    db_session,
    session: TestSession,
    user: User,
    anon_id: str,
    dry_run: bool,
) -> dict:
    """
    Anonymise a single session: detach from user, attach to anon_demographics.

    Args:
        db_session: SQLAlchemy session.
        session:    TestSession ORM object.
        user:       User ORM object.
        anon_id:    The cohort label to assign.
        dry_run:    If True, describe changes only.

    Returns:
        Summary dict of the action taken.
    """
    summary = {
        "session_id": str(session.session_id),
        "username":   user.username,
        "started_at": str(session.started_at),
        "age_range":  db_session.query(AnonDemographics).filter_by(anon_id=anon_id).first().age_range
                      if not dry_run
                      else anon_id,
        "anon_id":    anon_id,
        "dry_run":    dry_run,
    }

    if dry_run:
        return summary

    session.user_id       = None
    session.anon_id       = anon_id
    session.anonymised_at = datetime.now(timezone.utc)

    db_session.add(DataLifecycleLog(
        action_type  = "anonymized",
        target_type  = "session",
        target_id    = session.session_id,
        performed_by = "system_automated",
        details      = {
            "anon_id":   anon_id,
            "triggered": "anonymise_sessions.py",
        },
    ))

    return summary


def scrub_user_pii(db_session, user: User, dry_run: bool) -> None:
    """
    Scrub PII from a User row once all their sessions are anonymised.

    Zeroes: full_name, email, date_of_birth, password.
    Replaces username with "anon_<uuid_short>" to preserve FK integrity
    for any remaining consent or audit records.

    Args:
        db_session: SQLAlchemy session.
        user:       User ORM object.
        dry_run:    If True, log only.
    """
    action = "[DRY RUN] Would scrub" if dry_run else "Scrubbing"
    logger.info(f"  {action} PII for user '{user.username}' ({user.user_id})")

    if dry_run:
        return

    short_id = str(user.user_id)[:8]
    user.full_name     = None
    user.email         = None
    user.date_of_birth = None
    user.password      = "[redacted]"
    user.username      = f"anon_{short_id}"
    user.is_anonymized = True
    user.anonymized_at = datetime.now(timezone.utc)

    db_session.add(DataLifecycleLog(
        action_type  = "anonymized",
        target_type  = "user",
        target_id    = user.user_id,
        performed_by = "system_automated",
        details      = {"pii_cleared": ["full_name", "email", "date_of_birth", "password", "username"]},
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Anonymise HERO sessions that have passed the retention threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making any changes.")
    parser.add_argument("--days", type=int, default=None,
                        help=f"Override retention threshold in days "
                             f"(default: {ETHICS['anonymization_days']} from config).")
    parser.add_argument("--db-port", type=int, default=5432,
                        help="PostgreSQL port (default: 5432).")
    parser.add_argument("--print-migration", action="store_true",
                        help="Print migration instructions and exit.")
    args = parser.parse_args()

    if args.print_migration:
        print(MIGRATION_NOTE)
        sys.exit(0)

    threshold_days = args.days if args.days is not None else ETHICS["anonymization_days"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=threshold_days)

    logger.info(f"Retention threshold : {threshold_days} day(s)")
    logger.info(f"Cutoff timestamp    : {cutoff.isoformat()}")
    if args.dry_run:
        logger.info("DRY RUN — no changes will be written.")

    try:
        _, db_session = get_db_connection(**{**DB_CONFIG, "port": args.db_port})
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        sys.exit(1)

    try:
        # Sessions eligible for anonymisation:
        #  - still linked to a real user (user_id IS NOT NULL)
        #  - not already anonymised (anonymised_at IS NULL)
        #  - started before the cutoff
        candidates = (
            db_session.query(TestSession)
            .filter(
                TestSession.user_id.isnot(None),
                TestSession.anonymised_at.is_(None),
                TestSession.started_at < cutoff,
            )
            .all()
        )

        if not candidates:
            logger.info("No sessions eligible for anonymisation. Nothing to do.")
            return

        logger.info(f"Found {len(candidates)} session(s) eligible for anonymisation.")

        # Pre-load existing anon_id labels to avoid duplicates
        existing_labels = {
            row.anon_id for row in db_session.query(AnonDemographics.anon_id).all()
        }

        # Cache users to avoid repeated queries
        user_cache: dict[uuid.UUID, User] = {}
        results = []

        for session in candidates:
            uid = session.user_id
            if uid not in user_cache:
                user_cache[uid] = db_session.query(User).filter_by(user_id=uid).first()
            user = user_cache[uid]

            age_range = compute_age_range(user.date_of_birth, session.started_at)
            anon_id   = get_or_create_anon_demographics(
                db_session, age_range, existing_labels, args.dry_run
            )

            summary = anonymise_session(db_session, session, user, anon_id, args.dry_run)
            results.append(summary)

            logger.info(
                f"  {'[DRY RUN] ' if args.dry_run else ''}"
                f"Session {summary['session_id'][:8]}…  "
                f"user={summary['username']}  "
                f"age_range={age_range}  anon_id={anon_id}"
            )

        # Scrub PII for any user whose sessions are now all anonymised
        for uid, user in user_cache.items():
            if user is None:
                continue

            remaining_linked = (
                db_session.query(TestSession)
                .filter(TestSession.user_id == uid)
                .count()
            )
            # In dry-run, sessions haven't been NULLed yet, so adjust the count
            if args.dry_run:
                being_anon = sum(1 for r in results if r.get("username") == user.username)
                remaining_linked -= being_anon

            if remaining_linked == 0:
                scrub_user_pii(db_session, user, args.dry_run)

        if not args.dry_run:
            db_session.commit()
            logger.info(f"✓ Committed anonymisation of {len(results)} session(s).")
        else:
            db_session.rollback()
            logger.info(f"[DRY RUN] Would anonymise {len(results)} session(s). No changes written.")

        # Print summary table
        print(f"\n{'=' * 70}")
        print(f"{'DRY RUN — ' if args.dry_run else ''}Anonymisation summary")
        print(f"{'=' * 70}")
        print(f"{'Session ID':<38}  {'Username':<18}  {'Age range':<10}  Cohort")
        print(f"{'-' * 38}  {'-' * 18}  {'-' * 10}  {'-' * 20}")
        for r in results:
            age = r.get("age_range", "unknown")
            print(f"{r['session_id']:<38}  {r['username']:<18}  {age:<10}  {r['anon_id']}")
        print(f"{'=' * 70}")
        print(f"Total: {len(results)} session(s)\n")

    except SQLAlchemyError as e:
        db_session.rollback()
        logger.error(f"Database error — rolled back: {e}")
        sys.exit(1)
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
    