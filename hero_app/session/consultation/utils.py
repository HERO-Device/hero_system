"""
Shared utilities for the HERO consultation UI.

Provides MediaPipe face data extraction, screenshot capture, JSON encoding,
and hardware button input handling.
"""

import cv2
import datetime
import pygame as pg
import numpy as np
import json
from datetime import date, datetime
from enum import Enum
import subprocess

try:
    import gpiod
    from gpiod.line import Direction, Value
except ImportError:
    gpiod = None


def get_pipe_data(detector, image):
    """
    Extract face landmarks, blendshapes, and pose matrix from a MediaPipe detection.

    Args:
        detector: MediaPipe FaceLandmarker detector instance.
        image: MediaPipe Image object to run detection on.

    Returns:
        Tuple of (landArray, blend_scores, pose_matrix) where each is a numpy
        array on success, or (None, None, None) if no face is detected.
    """
    faceDetection = detector.detect(image)

    try:
        landmarks = faceDetection.face_landmarks[0]

        landArray = np.zeros((len(landmarks), 3))
        for idx, coord in enumerate(landmarks):
            landArray[idx, :] = [coord.x, coord.y, coord.z]

        blend = faceDetection.face_blendshapes[0]

        blend_scores = [AU.score for AU in blend]
        pose_matrix = faceDetection.facial_transformation_matrixes[0]

    except Exception:
        landArray, blend_scores, pose_matrix = None, None, None

    return landArray, blend_scores, pose_matrix


def take_screenshot(screen, filename=None):
    """
    Save the current pygame surface as a PNG screenshot.

    Args:
        screen: pygame Surface to capture.
        filename: Output filename (without extension). Defaults to current datetime.
    """
    print("Taking Screenshot")
    img_array = pg.surfarray.array3d(screen)
    img_array = cv2.transpose(img_array)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    if filename is None:
        filename = datetime.datetime.now()
    cv2.imwrite(f"screenshots/{filename}.png", img_array)


class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy and datetime types.

    Extends the default JSONEncoder to serialise numpy integers, floats,
    arrays, and datetime/date objects.
    """

    def default(self, obj):
        """
        Convert numpy or datetime objects to JSON-serialisable types.

        Args:
            obj: Object to serialise.

        Returns:
            JSON-serialisable equivalent of obj.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        return super(NpEncoder, self).default(obj)


class Buttons(Enum):
    """Physical button identifiers for the HERO hardware button module."""

    power = "Power"
    Home = "Home"
    vol_up = "Vol_Up"
    vol_down = "Vol_Down"
    info = "Info"


class ButtonModule:
    """
    Hardware button input handler supporting both Pi GPIO and keyboard emulation.

    On Pi, reads physical GPIO lines via gpiod. In non-Pi mode, maps keyboard
    keys 1–5 to the equivalent button actions for development use.

    Attributes:
        pi: Whether GPIO mode is active.
        button_dict: Mapping of pin numbers (Pi) or key codes (non-Pi) to button names.
        states: Current debounce state for each button.
        buttons: Reference to the Buttons enum.
        volume: Current system volume level (0–100).
    """

    def __init__(self, pi=True):
        """
        Initialise the button module and configure GPIO lines or keyboard mappings.

        Args:
            pi: If True, use GPIO input via gpiod. If False, use keyboard key emulation.
        """
        self.pi = pi
        if self.pi:
            # Define Raspberry Pi button pins
            self.button_dict = {
                4:  "Home",
                17: "Vol_Down",
                23: "Info",
                27: "Power",
                22: "Vol_Up",
            }

            self.button_lines = []
            for pin_num, name in self.button_dict.items():
                line_req = gpiod.request_lines(
                    '/dev/gpiochip4',
                    consumer="HeroButton",
                    config={pin_num: gpiod.LineSettings(direction=Direction.INPUT)}
                )
                self.button_lines.append((pin_num, line_req, name))
        else:

            self.button_dict = {
                pg.K_1: "Power",  # Pin number 7
                pg.K_2: "Home",  # Pin number 11
                pg.K_3: "Vol_Up",  # Pin number 16
                pg.K_4: "Vol_Down",  # Pin number 13
                pg.K_5: "Info",  # Pin number 15
            }

        self.states = {
            "Power": 0,
            "Home": 0,
            "Vol_Up": 0,
            "Vol_Down": 0,
            "Info": 0,
        }

        self.buttons = Buttons

        self.volume = 70

    def check_pressed(self):
        """
        Poll all buttons and return the first newly pressed button.

        Volume buttons are handled internally (amixer call) and return None.

        Returns:
            Buttons enum member for the pressed button, or None if no new press.
        """
        if self.pi:
            for (pin_num, line_req, name) in self.button_lines:
                button_state = line_req.get_value(pin_num) == Value.ACTIVE

                if button_state and not self.states[name]:
                    self.states[name] = button_state

                    if self.buttons(name) == Buttons.vol_up:
                        self.volume = min([100, self.volume + 10])
                        proc = subprocess.Popen(f'/usr/bin/amixer sset Master {self.volume}%', shell=True, stdout=subprocess.PIPE)
                        proc.wait()
                        return None

                    elif self.buttons(name) == Buttons.vol_down:
                        self.volume = max([0, self.volume - 10])
                        proc = subprocess.Popen(f'/usr/bin/amixer sset Master {self.volume}%', shell=True, stdout=subprocess.PIPE)
                        proc.wait()
                        return None

                    return self.buttons(name)

                self.states[name] = button_state

        else:
            pressed = pg.key.get_pressed()
            for val, name in self.button_dict.items():

                if pressed[val] and not self.states[name]:
                    self.states[name] = pressed[val]

                    return self.buttons(name)

                self.states[name] = pressed[val]

        return None
