"""
HERO Calibration Screen
Terminal-style sensor init display, then eye tracking calibration on bottom screen.
"""

import logging
import time
import pygame as pg
from uuid import UUID
from typing import Optional, Callable

logger = logging.getLogger(__name__)

BG    = (10,  10,  10)
FG    = (0,   255, 0)
WHITE = (220, 220, 220)
YELLOW= (255, 255, 0)
RED   = (255, 60,  60)
CYAN  = (0,   255, 255)


class CalibrationScreen:

    SENSORS = [
        ('mpu6050',  'MPU6050',  'Accelerometer / Gyroscope'),
        ('max30102', 'MAX30102', 'Heart Rate / SpO2'),
        ('eeg',      'EEG',      'Impedance Check (4 channels)'),
    ]

    def __init__(
        self,
        session_id: UUID,
        db_session,
        display_size: pg.Vector2,
        window: pg.Surface,
        on_complete: Optional[Callable] = None,
        pi: bool = True,
    ):
        self.session_id   = session_id
        self.db_session   = db_session
        self.display_size = display_size
        self.window       = window
        self.on_complete  = on_complete
        self.pi           = pi
        self.gaze_system  = None
        self.lines        = []

        self.bottom_screen = window.subsurface(
            (0, int(display_size.y)),
            (int(display_size.x), int(display_size.y))
        )

        pg.font.init()
        self.font = pg.font.SysFont('couriernew', 18, bold=True)

    def run(self):
        self._add_line("HERO Calibration", CYAN)
        self._add_line("", FG)
        self._render()

        for sensor_key, sensor_name, _ in self.SENSORS:
            self.lines.append([f"  {sensor_name:<14} initialising...", YELLOW, sensor_key])
            self._render()

            if sensor_key == 'eeg':
                self.lines[-1][0] = f"  {sensor_name:<14} checking impedance..."
                self._render()
                self._check_eeg_impedance()
            else:
                success, msg = self._test_sensor(sensor_key)
                self.lines[-1][0] = f"  {sensor_name:<14} {msg}"
                self.lines[-1][1] = FG if success else RED

            self._render()
            time.sleep(0.1)
            self._add_line("", FG)

        self._add_line("  EYE TRACKING   initialising...", YELLOW)
        self._render()
        try:
            from hero_system.sensors.eye_tracking.gaze import GazeSystem
            from hero_system.sensors.eye_tracking.config import EyeTrackingConfig
            self.gaze_system = GazeSystem(
                session_id    = self.session_id,
                db_session    = self.db_session,
                window        = self.window,
                bottom_screen = self.bottom_screen,
                config        = EyeTrackingConfig.for_session(),
            )
            self.gaze_system.start()
            self.lines[-1][0] = "  EYE TRACKING   Calibrated"
            self.lines[-1][1] = FG
        except Exception as e:
            logger.warning(f"Eye tracking unavailable: {e}")
            self.lines[-1][0] = f"  EYE TRACKING   FAILED — {str(e)[:40]}"
            self.lines[-1][1] = RED
            self.gaze_system = None

        self._render()
        self._add_line("", FG)
        self._add_line("  Click anywhere to start", WHITE)
        self._render()

        self._wait_for_input()

        if self.on_complete:
            self.on_complete()

    def _wait_for_input(self):
        pg.event.clear()
        while True:
            for event in pg.event.get():
                if event.type in (pg.MOUSEBUTTONDOWN, pg.FINGERDOWN, pg.KEYDOWN):
                    return
            time.sleep(0.01)

    def _test_sensor(self, sensor_key: str):
        try:
            if sensor_key == 'mpu6050':
                import smbus2
                bus = smbus2.SMBus(1)
                who_am_i = bus.read_byte_data(0x68, 0x75)
                bus.close()
                return (True, "Detected at 0x68") if who_am_i == 0x68 \
                       else (False, f"Bad ID: {hex(who_am_i)}")
            elif sensor_key == 'max30102':
                import smbus2
                bus = smbus2.SMBus(1)
                part_id = bus.read_byte_data(0x57, 0xFF)
                bus.close()
                return (True, "Detected at 0x57") if part_id == 0x15 \
                       else (False, f"Bad ID: {hex(part_id)}")
        except Exception as e:
            return False, str(e)[:50]
        return False, "Unknown sensor"

    def _find_serial_port(self):
        import glob
        ports = sorted(glob.glob('/dev/ttyACM*'))
        return ports[0] if ports else None

    def _check_eeg_impedance(self):
        THRESHOLD = 50000
        try:
            import numpy as np
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            import time as _t

            port = self._find_serial_port()
            if not port:
                self._add_line("  EEG            No serial port found", RED)
                self._render()
                return

            params = BrainFlowInputParams()
            params.serial_port = port
            params.mac_address = 'cb:1c:86:2e:73:2c'
            BoardShim.disable_board_logger()
            board = BoardShim(BoardIds.GANGLION_BOARD, params)
            board.prepare_session()
            board.config_board('impedance_mode:1')
            board.start_stream()
            _t.sleep(3)
            data = board.get_board_data()
            board.stop_stream()
            board.config_board('impedance_mode:0')
            board.release_session()

            if data.shape[1] == 0:
                self._add_line("  EEG            No data received", RED)
                self._render()
                return

            for i, ch in enumerate(BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value)):
                imp  = float(np.mean(np.abs(data[ch])))
                kohm = imp / 1000
                ok   = imp < THRESHOLD
                self._add_line(
                    f"  EEG C{i+1:<12} {kohm:.0f} kO  -- {'OK' if ok else 'HIGH'}",
                    FG if ok else RED
                )
                self._render()

        except Exception as e:
            self._add_line(f"  EEG            FAILED -- {str(e)[:45]}", RED)
            self._render()

    def _add_line(self, text: str, colour):
        self.lines.append([text, colour, None])

    def _render(self):
        w   = int(self.display_size.x)
        h   = int(self.display_size.y)
        sur = pg.Surface((w, h))
        sur.fill(BG)

        line_h  = 26
        visible = h // line_h
        shown   = self.lines[-visible:]

        y = (h - len(shown) * line_h) // 2
        for item in shown:
            txt = self.font.render(item[0], True, item[1])
            sur.blit(txt, (w // 2 - txt.get_width() // 2, y))
            y += line_h

        self.window.blit(sur, (0, 0))
        pg.display.flip()
        pg.event.pump()
        
