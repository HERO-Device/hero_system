"""
HERO System - Database Access Layer
Wraps hero_core models to provide a simple interface for:
- Connecting to the database
- Managing users and sessions
- Writing game results and events
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from hero_core.database.models.connection import get_db_connection
from hero_core.database.models.user import User
from hero_core.database.models.session import TestSession
from hero_core.database.models.game_results import GameResult
from hero_core.database.models.events import Event

logger = logging.getLogger(__name__)

# DB credentials
DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'user':     'postgres',
    'password': 'pgdbadmin',
    'dbname':   'hero_db',
}


class HeroDB:
    """
    Database access layer for the HERO system.

    Wraps hero_core ORM models to provide a simple interface for
    the session app. Handles user authentication, session lifecycle,
    game result storage, and event logging. All operations use a
    single SQLAlchemy session created at init.

    Usage:
        db = HeroDB()
        user = db.verify_login(username, password)
        session_id = db.start_session(user.user_id)
        ...
        db.end_session(session_id)
        db.close()
    """
    def __init__(self, config: dict = None):
        cfg = config or DB_CONFIG
        try:
            self.engine, self.session = get_db_connection(**cfg)
            logger.info("✓ Connected to hero_db")
        except Exception as e:
            logger.error(f"✗ Failed to connect to hero_db: {e}")
            raise

    # ------------------------------------------------------------------
    # User
    # ------------------------------------------------------------------

    def get_user(self, username: str) -> Optional[User]:
        """
        Fetch a user by username.
        Args:
            username: The patient's login handle.
        Returns:
            User object if found, None otherwise.
        """
        try:
            return self.session.query(User).filter_by(username=username).first()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching user '{username}': {e}")
            return None

    def verify_login(self, username: str, password: str) -> Optional[User]:
        """
        Verify login credentials against the database.
        Args:
            username: The patient's login handle.
            password: Plaintext password — to be hashed in a future security pass.
        Returns:
            User object if credentials are valid, None otherwise.
        """
        user = self.get_user(username)
        if user and user.password == password:
            return user
        return None

    def create_user(self, username: str, password: str,
                    full_name: str = None,
                    email: str = None,
                    date_of_birth: datetime = None) -> Optional[User]:
        """
        Create and persist a new patient account.
        Args:
            username:      Unique login handle.
            password:      Plaintext password.
            full_name:     Patient's display name.
            email:         Optional contact email.
            date_of_birth: Optional date of birth for demographic analysis.
        Returns:
            Newly created User object, or None if creation failed.
        """
        try:
            user = User(
                user_id=uuid.uuid4(),
                username=username,
                password=password,
                full_name=full_name,
                email=email,
                date_of_birth=date_of_birth,
            )
            self.session.add(user)
            self.session.commit()
            logger.info(f"✓ Created user '{username}' ({user.user_id})")
            return user
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"✗ Failed to create user '{username}': {e}")
            return None

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def start_session(self, user_id: uuid.UUID, notes: str = None) -> Optional[uuid.UUID]:
        """
        Create a new test session in the database.
        Args:
            user_id: UUID of the authenticated patient.
            notes:   Optional free-text (e.g. consultation reference).
        Returns:
            session_id UUID if successful, None otherwise.
        """
        try:
            session_id = uuid.uuid4()
            test_session = TestSession(
                session_id=session_id,
                user_id=user_id,
                started_at=datetime.now(timezone.utc),
                notes=notes,
            )
            self.session.add(test_session)
            self.session.commit()
            logger.info(f"✓ Started session {session_id}")
            return session_id
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"✗ Failed to start session: {e}")
            return None

    def end_session(self, session_id: uuid.UUID):
        """
        Mark a session as complete by writing ended_at.
        Args:
            session_id: UUID of the session to close.
        Returns:
            None.
        """
        try:
            test_session = self.session.query(TestSession).filter_by(
                session_id=session_id
            ).first()
            if test_session:
                test_session.ended_at = datetime.now(timezone.utc)
                self.session.commit()
                logger.info(f"✓ Ended session {session_id}")
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"✗ Failed to end session {session_id}: {e}")

    # ------------------------------------------------------------------
    # Game Results
    # ------------------------------------------------------------------

    def save_game_result(self,
                         session_id: uuid.UUID,
                         game_name: str,
                         game_number: int,
                         started_at: datetime,
                         completed_at: datetime,
                         final_score: int = None,
                         max_score: int = None,
                         accuracy_percent: float = None,
                         correct_answers: int = None,
                         incorrect_answers: int = None,
                         average_reaction_time_ms: float = None,
                         game_data: dict = None,
                         completion_status: str = 'completed') -> Optional[uuid.UUID]:
        """
        Persist the result of a completed cognitive test.
        Args:
            session_id:               UUID of the current session.
            game_name:                Game identifier e.g. 'Spiral', 'Trail', 'Shapes', 'Memory'.
            game_number:              Ordinal position of the game in the session.
            started_at:               UTC timestamp when the game loop began.
            completed_at:             UTC timestamp when the game loop ended.
            final_score:              Raw score achieved.
            max_score:                Maximum possible score.
            accuracy_percent:         Score as a percentage of max_score.
            correct_answers:          Count of correct responses.
            incorrect_answers:        Count of incorrect responses.
            average_reaction_time_ms: Mean response time across all trials in ms.
            game_data:                Dict of full game-specific results for JSONB storage.
            completion_status:        Outcome string, defaults to 'completed'.
        Returns:
            result_id UUID if successful, None otherwise.
        """
        try:
            duration = (completed_at - started_at).total_seconds()
            result_id = uuid.uuid4()

            result = GameResult(
                result_id=result_id,
                session_id=session_id,
                game_name=game_name,
                game_number=game_number,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                final_score=final_score,
                max_score=max_score,
                accuracy_percent=accuracy_percent,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
                average_reaction_time_ms=average_reaction_time_ms,
                game_data=game_data,
                completion_status=completion_status,
            )
            self.session.add(result)
            self.session.commit()
            logger.info(f"✓ Saved game result for '{game_name}' ({result_id})")
            return result_id
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"✗ Failed to save game result for '{game_name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(self,
                  session_id: uuid.UUID,
                  event_type: str,
                  event_category: str,
                  game_name: str = None,
                  game_number: int = None,
                  screen_x: float = None,
                  screen_y: float = None,
                  event_data: dict = None):
        """
        Log a discrete system or game event.
        Args:
            session_id:     UUID of the current session.
            event_type:     Event identifier e.g. 'game_start', 'game_end'.
            event_category: Broad grouping e.g. 'game', 'system'.
            game_name:      Game this event relates to. None for system events.
            game_number:    Ordinal position of the game. None for system events.
            screen_x:       Optional X coordinate of a screen interaction in pixels.
            screen_y:       Optional Y coordinate of a screen interaction in pixels.
            event_data:     Optional dict of additional event metadata.
        Returns:
            None.
        """
        try:
            event = Event(
                event_id=uuid.uuid4(),
                time=datetime.now(timezone.utc),
                session_id=session_id,
                event_type=event_type,
                event_category=event_category,
                game_name=game_name,
                game_number=game_number,
                screen_x=screen_x,
                screen_y=screen_y,
                event_data=event_data,
            )
            self.session.add(event)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"✗ Failed to log event '{event_type}': {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """
        Close the SQLAlchemy session.
        Returns:
            None.
        """
        try:
            self.session.close()
            logger.info("✓ DB session closed")
        except Exception as e:
            logger.error(f"Error closing DB session: {e}")

    def __enter__(self):
        """
        Allow use as a context manager.
        Returns:
            Self.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close DB session on context manager exit.
        Args:
            exc_type: Exception type if raised, None otherwise.
            exc_val:  Exception value if raised, None otherwise.
            exc_tb:   Traceback if raised, None otherwise.
        Returns:
            None.
        """
        self.close()
