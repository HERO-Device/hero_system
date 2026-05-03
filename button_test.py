"""
HERO Button Testbench
Tests GPIO buttons on pins 17, 27, 22, 23, 4 via gpiod.
Press each button to verify it registers. Press Ctrl+C to exit.
"""
import time
import gpiod
from gpiod.line import Direction, Value

BUTTONS = {
    4:  "Home",
    17: "Vol_Down",
    23: "Info",
    27: "Power",
    22: "Vol_Up",
}

print("=" * 40)
print("  HERO Button Testbench")
print("=" * 40)
print("Press each button to test. Ctrl+C to exit.\n")

# Request all lines
button_lines = []
for pin_num, name in BUTTONS.items():
    try:
        line_req = gpiod.request_lines(
            '/dev/gpiochip4',
            consumer="HeroButtonTest",
            config={pin_num: gpiod.LineSettings(direction=Direction.INPUT)}
        )
        button_lines.append((pin_num, line_req, name))
        print(f"  ✓ Pin {pin_num:2d} ({name}) — ready")
    except Exception as e:
        print(f"  ✗ Pin {pin_num:2d} ({name}) — FAILED: {e}")

print("\nListening for button presses...\n")

prev_values = {pin: Value.ACTIVE for pin, _, _ in button_lines}

try:
    while True:
        for pin_num, line_req, name in button_lines:
            val = line_req.get_value(pin_num)
            if val == Value.INACTIVE and prev_values[pin_num] == Value.ACTIVE:
                print(f"  PRESSED: {name} (pin {pin_num})")
            prev_values[pin_num] = val
        time.sleep(0.02)
except KeyboardInterrupt:
    print("\nExiting.")
finally:
    for _, line_req, _ in button_lines:
        line_req.release()
