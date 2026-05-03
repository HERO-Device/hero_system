"""
HERO Launcher
Waits for the Info button (GPIO pin 23) to be pressed, then launches run.py.
Runs persistently — after a session completes, returns to waiting.
"""

import os
import sys
import time
import subprocess
import gpiod
from gpiod.line import Direction, Bias, Value

INFO_PIN = 23
CHIP     = '/dev/gpiochip0'

HERO_DIR    = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(HERO_DIR, '..', '.venv', 'bin', 'python3')
RUN_SCRIPT  = os.path.join(HERO_DIR, 'run.py')


def wait_for_button(request):
    """Block until Info button is pressed (falling edge)."""
    prev = request.get_value(INFO_PIN)
    while True:
        val = request.get_value(INFO_PIN)
        if val == Value.INACTIVE and prev == Value.ACTIVE:
            return
        prev = val
        time.sleep(0.02)


def main():
    print()
    print("=" * 50)
    print("  HERO Launcher — waiting for Info button")
    print("=" * 50)
    print()

    try:
        request = gpiod.request_lines(
            CHIP,
            consumer='HeroLauncher',
            config={INFO_PIN: gpiod.LineSettings(
                direction=Direction.INPUT,
                bias=Bias.PULL_UP,
            )}
        )
    except Exception as e:
        print(f"✗ Could not initialise Info button (pin {INFO_PIN}): {e}")
        print("  Launching immediately without button wait...")
        os.execv(VENV_PYTHON, [VENV_PYTHON, RUN_SCRIPT])
        return

    while True:
        print("  Press INFO button to start a session...")
        wait_for_button(request)
        print("  INFO pressed — launching HERO System\n")

        proc = subprocess.run([VENV_PYTHON, RUN_SCRIPT])

        print()
        print("  Session ended. Press INFO button to start another.")
        print()
        time.sleep(1)  # debounce


if __name__ == '__main__':
    main()
