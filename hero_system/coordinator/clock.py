"""
Central Clock System
Provides synchronized timestamps for all sensors in the HERO system
"""

import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class CentralClock:
    """
    Thread-safe central clock for multi-sensor timestamp synchronization

    Ensures all sensors use the same time reference with:
    - Thread-safe access (multiple sensors can call simultaneously)
    - Monotonic timestamps (always increasing, no duplicates)
    - Microsecond precision (suitable for high-frequency sampling)
    """

    def __init__(self):
        """Initialize central clock"""
        self._lock = threading.Lock()
        self._last_timestamp: Optional[datetime] = None
        self._call_count = 0

        logger.info("Central clock initialized")

    def now(self) -> datetime:
        """
        Get current synchronized timestamp

        Returns:
            datetime: Current UTC timestamp with microsecond precision
        """
        with self._lock:
            current_time = datetime.now(timezone.utc)

            # Ensure monotonic increasing timestamps
            if self._last_timestamp and current_time <= self._last_timestamp:
                # If system clock hasn't advanced, increment by 1 microsecond
                current_time = self._last_timestamp + timedelta(microseconds=1)
                logger.debug("Adjusted timestamp to maintain monotonic sequence")

            self._last_timestamp = current_time
            self._call_count += 1

            return current_time

    def get_timestamp(self) -> datetime:
        """
        Alias for now() - more explicit naming

        Returns:
            datetime: Current synchronized timestamp
        """
        return self.now()

    def reset(self):
        """Reset clock state (useful for testing)"""
        with self._lock:
            self._last_timestamp = None
            self._call_count = 0
            logger.info("Central clock reset")

    def get_stats(self) -> dict:
        """
        Get clock statistics

        Returns:
            dict: Clock usage statistics
        """
        with self._lock:
            return {
                'total_calls': self._call_count,
                'last_timestamp': self._last_timestamp.isoformat() if self._last_timestamp else None,
            }

    def __repr__(self):
        return f"<CentralClock(calls={self._call_count})>"
