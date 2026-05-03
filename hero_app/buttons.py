"""
HERO Button Watcher
===================
Singleton background thread that polls all GPIO buttons and fires callbacks.

Usage:
    from hero_app.buttons import ButtonWatcher

    watcher = ButtonWatcher.instance()
    watcher.on('Info', lambda: print("Info pressed"))
    watcher.on('Home', lambda: print("Home pressed"))

    # Remove a callback:
    watcher.off('Info')

    # Block until a specific button is pressed:
    watcher.wait_for('Home')

Button names: 'Power', 'Home', 'Vol_Up', 'Vol_Down', 'Info'
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)

BUTTON_PINS = {
    4:  'Power',
    17: 'Home',
    23: 'Info',
    27: 'Vol_Down',
    22: 'Vol_Up',
}

POLL_INTERVAL = 0.02  # seconds


class ButtonWatcher:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls):
        """Return the singleton ButtonWatcher, creating it if needed."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._callbacks = {}   # name -> callable
        self._events = {}      # name -> threading.Event (for wait_for)
        self._request = None
        self._prev_states = {name: 1 for name in BUTTON_PINS.values()}
        self._running = False
        self._thread = None
        self._start()

    def _start(self):
        """Initialise GPIO and start polling thread."""
        try:
            import gpiod
            from gpiod.line import Direction
            self._request = gpiod.request_lines(
                '/dev/gpiochip0',
                consumer='HeroButtons',
                config={
                    tuple(BUTTON_PINS.keys()): gpiod.LineSettings(
                        direction=Direction.INPUT
                    )
                }
            )
            logger.info("✓ ButtonWatcher: GPIO lines acquired")
        except Exception as e:
            logger.warning(f"ButtonWatcher: Could not acquire GPIO lines: {e}")
            self._request = None

        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        """Background polling loop — detects falling edges (active-low buttons)."""
        from gpiod.line import Value
        while self._running:
            if self._request:
                try:
                    for pin, name in BUTTON_PINS.items():
                        val = self._request.get_value(pin)
                        state = 1 if val == Value.ACTIVE else 0
                        if state == 0 and self._prev_states[name] == 1:
                            self._fire(name)
                        self._prev_states[name] = state
                except Exception as e:
                    logger.warning(f"ButtonWatcher poll error: {e}")
            time.sleep(POLL_INTERVAL)

    def _fire(self, name):
        """Fire callback and set event for the named button."""
        logger.debug(f"Button pressed: {name}")
        # Fire registered callback
        cb = self._callbacks.get(name)
        if cb:
            try:
                cb()
            except Exception as e:
                logger.warning(f"ButtonWatcher callback error for {name}: {e}")
        # Set event if anyone is waiting
        ev = self._events.get(name)
        if ev:
            ev.set()

    def on(self, button_name: str, callback):
        """Register a callback for a button press. Replaces any existing callback."""
        self._callbacks[button_name] = callback

    def off(self, button_name: str):
        """Remove the callback for a button."""
        self._callbacks.pop(button_name, None)

    def wait_for(self, button_name: str, timeout: float = None) -> bool:
        """
        Block until the named button is pressed.

        Args:
            button_name: Name of the button to wait for.
            timeout:     Max seconds to wait. None = wait forever.

        Returns:
            True if button was pressed, False if timed out.
        """
        ev = threading.Event()
        self._events[button_name] = ev
        result = ev.wait(timeout=timeout)
        self._events.pop(button_name, None)
        return result

    def stop(self):
        """Stop the polling thread and release GPIO lines."""
        self._running = False
        if self._request:
            try:
                self._request.release()
            except Exception:
                pass
        ButtonWatcher._instance = None
