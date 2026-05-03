"""
HERO Buttons
============
Reads button presses from the button daemon log file.
Each call to wait_for_button() only sees presses that happen AFTER it starts waiting.
"""

import os
import time

LOG_PATH = '/tmp/hero_button_log'


def wait_for_button(name, timeout=None):
    """
    Block until the named button is pressed after this call starts.
    Returns True if pressed, False if timed out.
    """
    # Record current file size — only look at new lines written after this point
    try:
        start_pos = os.path.getsize(LOG_PATH)
    except OSError:
        start_pos = 0

    start_time = time.time()

    while True:
        try:
            with open(LOG_PATH, 'r') as f:
                f.seek(start_pos)
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2 and parts[1] == name:
                        return True
        except OSError:
            pass

        if timeout and (time.time() - start_time) > timeout:
            return False
        time.sleep(0.05)
