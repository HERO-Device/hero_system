"""
HERO Launcher
Starts ButtonWatcher on boot, waits for Info button, launches run.py.
After session ends, returns to waiting.
"""

import os
import sys
import time
import subprocess

HERO_DIR    = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(HERO_DIR, '..', '.venv', 'bin', 'python3')
RUN_SCRIPT  = os.path.join(HERO_DIR, 'run.py')

sys.path.insert(0, HERO_DIR)


def main():
    print()
    print("=" * 50)
    print("  HERO Launcher")
    print("=" * 50)
    print()

    from hero_app.buttons import ButtonWatcher
    watcher = ButtonWatcher.instance()

    while True:
        print("  Press INFO button to start a session...")
        pressed = watcher.wait_for('Info')
        if pressed:
            print("  INFO pressed — launching HERO System\n")
            watcher.off('Info')
            subprocess.run([VENV_PYTHON, RUN_SCRIPT])
            print("\n  Session ended. Press INFO to start another.")
            time.sleep(1)  # debounce


if __name__ == '__main__':
    main()
