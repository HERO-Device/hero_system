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



def sigmoid(x):
    return


def get_pipe_data(detector, image):
    faceDetection = detector.detect(image)

    try:
        landmarks = faceDetection.face_landmarks[0]

        landArray = np.zeros((len(landmarks), 3))
        for idx, coord in enumerate(landmarks):
            landArray[idx, :] = [coord.x, coord.y, coord.z]

        blend = faceDetection.face_blendshapes[0]

        blend_scores = [AU.score for AU in blend]
        pose_matrix = faceDetection.facial_transformation_matrixes[0]

    except:
        landArray, blend_scores, pose_matrix = None, None, None

    return landArray, blend_scores, pose_matrix


def take_screenshot(screen, filename=None):
    print("Taking Screenshot")
    img_array = pg.surfarray.array3d(screen)
    img_array = cv2.transpose(img_array)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    if filename is None:
        filename = datetime.datetime.now()
    cv2.imwrite(f"screenshots/{filename}.png", img_array)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
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
    power = "Power"
    Home = "Home"
    vol_up = "Vol_Up"
    vol_down = "Vol_Down"
    info = "Info"


class ButtonModule:
    def __init__(self, pi=True):
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
            "Power": 0,  # to track the current state (on/off)
            "Home": 0,  # to track the current state (on/off)
            "Vol_Up": 0,  # to track the current state (on/off)
            "Vol_Down": 0,  # to track the current state (on/off)
            "Info": 0,  # to track the current state (on/off)
        }

        self.buttons = Buttons

        self.volume = 70

    def check_pressed(self):
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
                    self.states[name] = pressed[val]  # update the internal state

                    return self.buttons(name)

                self.states[name] = pressed[val]

        return None


if __name__ == "__main__":
    pg.init()
    pg.event.pump()

    pi = True

    buttons = ButtonModule(pi=pi)

    while True:
        if not pi:
            pg.event.pump()

            pressed = buttons.check_pressed()
            if pressed:
                print(f"{pressed}!")
        else:
            pressed = buttons.check_pressed()
            if pressed:
                print(f"{pressed}!")
                
