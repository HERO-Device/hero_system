"""
HERO Button Daemon
==================
Runs as a systemd service on boot.
Polls GPIO buttons, launches run.py in a terminal on Vol Up,
and appends button presses to a log file that readers can poll.

Log file: /tmp/hero_button_log
"""

import os
import time
import subprocess
import gpiod
from gpiod.line import Direction, Value

LOG_PATH    = '/tmp/hero_button_log'
VENV_PYTHON = '/home/hero/HERO-Device/.venv/bin/python'
RUN_SCRIPT  = '/home/hero/HERO-Device/hero_system/run.py'

BUTTON_DICT = {
    27: "Vol Down",
    17: "Home",
    23: "Info",
    4:  "Vol Up",
    22: "Power",
}

POLL_INTERVAL = 0.02
DISPLAY_ENV = {
    **os.environ,
    'DISPLAY': ':0',
    'WAYLAND_DISPLAY': 'wayland-1',
    'XDG_RUNTIME_DIR': '/run/user/1000',
}


def main():
    # Clear log on start
    open(LOG_PATH, 'w').close()

    print(f"HERO Button Daemon starting — log: {LOG_PATH}")

    request = gpiod.request_lines(
        '/dev/gpiochip0',
        consumer="HeroButtonDaemon",
        config={
            tuple(BUTTON_DICT.keys()): gpiod.LineSettings(direction=Direction.INPUT)
        }
    )
    print("GPIO lines acquired")

    states = {name: 1 for name in BUTTON_DICT.values()}
    hero_proc = None

    while True:
        for pin, name in BUTTON_DICT.items():
            val = request.get_value(pin)
            state = 1 if val == Value.ACTIVE else 0
            if state == 0 and states[name] == 1:
                print(f"Button pressed: {name}")

                # Append to log file with timestamp
                with open(LOG_PATH, 'a') as f:
                    f.write(f"{time.time():.3f} {name}\n")

                # Launch HERO in terminal on Vol Up
                if name == "Power":
                    if hero_proc is None or hero_proc.poll() is not None:
                        print("Launching HERO System in terminal...")
                        hero_proc = subprocess.Popen(
                            ['x-terminal-emulator', '-e',
                             f'{VENV_PYTHON} {RUN_SCRIPT}'],
                            env=DISPLAY_ENV
                        )

            states[name] = state
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
