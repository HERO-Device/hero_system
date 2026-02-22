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
    Main database access class for the HERO system.
    Handles all DB operations for the session app.
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
        """Fetch a user by username."""
        try:
            return self.session.query(User).filter_by(username=username).first()
        except SQLAlchemyError as e:
            logger.error(f"Error fetching user '{username}': {e}")
            return None

    def verify_login(self, username: str, password: str) -> Optional[User]:
        """
        Verify login credentials.
        Returns the User object if valid, None otherwise.
        NOTE: passwords stored as plaintext for now — hash in production.
        """
        user = self.get_user(username)
        if user and user.password == password:
            return user
        return None

    def create_user(self, username: str, password: str,
                    full_name: str = None,
                    email: str = None,
                    date_of_birth: datetime = None) -> Optional[User]:
        """Create a new patient user."""
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
        Create a new test session in the DB.
        Returns the session_id UUID.
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
        """Mark a session as ended."""
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
        Save a game result to the DB.
        Returns the result_id UUID.
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
        """Log a single event to the DB."""
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
        """Close the DB session."""
        try:
            self.session.close()
            logger.info("✓ DB session closed")
        except Exception as e:
            logger.error(f"Error closing DB session: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
