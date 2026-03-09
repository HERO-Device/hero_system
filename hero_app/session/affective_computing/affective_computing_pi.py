"""
Affective computing module for the HERO consultation system.

Captures facial imagery and audio during patient question sessions,
runs facial expression recognition via a Keras model, and stores
per-question affective labels for downstream analysis.
"""

import numpy as np
import pygame as pg
from consultation.touch_screen import TouchScreen, GameButton, GameObjects
from consultation.display_screen import DisplayScreen
from consultation.screen import Colours, BlitLocation
from consultation.utils import take_screenshot, NpEncoder, get_pipe_data

import cv2
import wave
import pyaudio
import os
import keras
from scipy.special import softmax
import random
import shutil
import json
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pygame import Rect, Vector2


def segment_face(landmarks, img_array):
    """
    Crop a face region from an image using MediaPipe landmark bounding box.

    Expands the bounding box by a fixed scale factor to include the full
    head region beyond the landmark extents.

    Args:
        landmarks: Numpy array of shape (N, 3) containing normalised landmark coordinates.
        img_array: numpy array of the source image (H, W, C).

    Returns:
        Cropped numpy image array of the face region.
    """
    px_locations_x = landmarks[:, 0] * img_array.shape[1]
    px_locations_y = landmarks[:, 1] * img_array.shape[0]

    max_x, min_x = max(px_locations_x), min(px_locations_x)
    max_y, min_y = max(px_locations_y), min(px_locations_y)

    # create bounding box of face and scale to adjust for full head region
    scale = Vector2(1.8, 1.6)
    bbox = np.asarray([min_x, min_y, max_x - min_x, max_y - min_y], dtype=np.int16)
    face_rect = Rect(bbox).scale_by(scale.x, scale.y)
    face_rect = face_rect.clip(Rect((0, 0), img_array.shape[:2]))
    cropped_img = img_array[face_rect.top:face_rect.bottom, face_rect.left:face_rect.right]

    return cropped_img


class AffectiveModulePi:
    """
    Consultation module that records facial and audio data during patient responses.

    Captures images via PiCamera2 or OpenCV and audio via PyAudio across a
    configurable number of questions. On completion, runs facial expression
    recognition using a pre-trained Keras model and stores per-question labels.

    Attributes:
        parent: Parent Consultation instance, or None in standalone mode.
        display_size: Vector2 dimensions of the display area.
        display_screen: DisplayScreen instance for the upper screen.
        touch_screen: TouchScreen instance for the lower screen.
        main_button: Begin/stop GameButton displayed during questions.
        listening: True while audio and image capture is active.
        running: Main loop control flag.
        pi: True if running on Raspberry Pi with PiCamera2.
        question_count: Total number of questions to ask.
        question_idx: Index of the current question.
        picam: PiCamera2 instance (Pi only).
        cv2_cam: OpenCV VideoCapture instance (non-Pi only).
        pyaud: PyAudio instance for microphone capture.
        audio_rate: Sample rate of the default audio input device.
        cleanse_files: If True, delete captured data after classification.
        classify: If True, run FER classification in exit_sequence.
        detector: MediaPipe FaceLandmarker for face cropping.
        auto_run: If True, skip capture and use synthetic data.
        label_data: Dict of per-question affective labels populated by exit_sequence.
    """

    def __init__(self, size=(1024, 600), parent=None, pi=True, cleanse_files=True, classify=True, auto_run=False):
        """
        Initialise the affective module, camera, microphone, and face detector.

        Args:
            size: Tuple (width, height) used in standalone mode without a parent.
            parent: Parent Consultation instance. If provided, screens are shared.
            pi: If True, use PiCamera2 for image capture. If False, use OpenCV.
            cleanse_files: If True, delete captured images and audio after classification.
            classify: If True, run the FER model in exit_sequence.
            auto_run: If True, bypass capture and generate synthetic labels.
        """
        self.parent = parent
        if parent is not None:
            self.display_size = parent.display_size
            self.bottom_screen = parent.bottom_screen
            self.top_screen = parent.top_screen
            self.display_screen = DisplayScreen(self.display_size, avatar=parent.avatar)

        else:
            self.display_size = pg.Vector2(size)
            self.window = pg.display.set_mode((self.display_size.x, self.display_size.y * 2), pg.SRCALPHA)

            self.top_screen = self.window.subsurface(((0, 0), self.display_size))
            self.bottom_screen = self.window.subsurface((0, self.display_size.y), self.display_size)
            self.display_screen = DisplayScreen(self.display_size)

        self.touch_screen = TouchScreen(self.display_size)

        self.display_screen.speech_text = "How have you been feeling in the past week?"

        button_size = pg.Vector2(self.display_size.x*0.9, self.display_size.y*0.15)
        self.main_button = GameButton(pg.Vector2(0.5*self.display_size.x, 0.85*self.display_size.y) - button_size / 2,
                                      button_size, id=1, text="Begin", colour=Colours.hero_blue)
        self.touch_screen.sprites = GameObjects([self.main_button])

        self.listening = False
        self.running = False
        self.pi = pi
        self.question_count = 2
        self.question_idx = 0

        if pi:
            picamera = __import__('picamera2')
            self.picam = picamera.Picamera2()

            # specify RGB only
            config = self.picam.create_preview_configuration({'format': 'BGR888'})
            self.picam.configure(config)
            print("config done!!")
            print(self.picam)
            self.cv2_cam = None

        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cv2_cam = None
            self.picam = None
            print("pi not specified, will use cv2 instead")

        if not os.path.isdir("data/affective_images"):
            os.mkdir("data/affective_images")

        if not os.path.isdir("data/nlp_audio"):
            os.mkdir("data/nlp_audio")

        try:
            self.pyaud = pyaudio.PyAudio()
            device_info = self.pyaud.get_default_input_device_info()
            self.audio_rate = int(device_info["defaultSampleRate"])

        except Exception:
            print("Microphone error")
            self.pyaud = None
            self.audio_rate = None

        self.cleanse_files = cleanse_files
        self.classify = classify

        base_options = python.BaseOptions(model_asset_path='models/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.auto_run = auto_run
        self.label_data = None

    def update_display(self):
        """Refresh both screens and blit the current state to the physical displays."""
        self.display_screen.refresh()
        self.touch_screen.refresh()

        if self.listening:
            self.touch_screen.load_image("hero/consultation/resources/graphics/listening.png", size=pg.Vector2(300, 300),
                                         pos=pg.Vector2(self.display_size.x*0.5, self.display_size.y*0.4),
                                         location=BlitLocation.centre)

        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))
        self.bottom_screen.blit(self.touch_screen.get_surface(), (0, 0))
        pg.display.flip()

    def entry_sequence(self):
        """Initialise the module state and show the start prompt."""
        self.running = True
        self.display_screen.instruction = "Press the button to start"
        self.update_display()

    def crop_face(self, frame):
        """
        Detect and crop the face region from a camera frame.

        Args:
            frame: numpy image array (H, W, C) in RGB format.

        Returns:
            Cropped face numpy array, or None if no face is detected.
        """
        img_mp = mp.Image(data=frame, image_format=mp.ImageFormat.SRGB)
        face_landmarks, _, _ = get_pipe_data(self.detector, img_mp)
        if face_landmarks is not None:
            img_face = segment_face(face_landmarks, frame)
            return img_face
        else:
            print("no face found")
            return None

    def question_loop(self, max_time=10):
        """
        Run a single question capture session, recording audio and facial images.

        Captures audio at the device sample rate and samples camera frames
        every 16 audio chunks. Images are cropped to the face region where
        possible. All data is saved under data/affective_images/question_{idx}/
        and data/nlp_audio/.

        Args:
            max_time: Maximum recording duration in seconds.
        """
        self.display_screen.instruction = "Press the button to stop"
        self.main_button.text = "I'm finished"
        self.update_display()

        if self.auto_run:
            time.sleep(0.5)
            return

        question_path = f"data/affective_images/question_{self.question_idx}"
        if not os.path.isdir(question_path):
            os.mkdir(question_path)

        image_directory = os.path.join(question_path, "images")
        if not os.path.isdir(image_directory):
            os.mkdir(image_directory)

        self.listening = True
        self.update_display()

        if self.picam:
            self.picam.start()
            print("starting picam")
        else:
            self.cv2_cam = cv2.VideoCapture(0)

        chunk_size = 1024
        stream = self.pyaud.open(format=pyaudio.paInt16, channels=1, rate=int(self.audio_rate), input=True, frames_per_buffer=chunk_size)
        frames = []

        iter_per_second = int((self.audio_rate / chunk_size))
        for i in range(0, iter_per_second * max_time):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)

            if (i % 16) == 0:
                if self.picam:
                    frame = cv2.flip(self.picam.capture_array(), 0)
                elif self.cv2_cam:
                    ret, frame = self.cv2_cam.read()

                if frame is not None:
                    img_face = self.crop_face(frame)
                    if img_face is not None:
                        cv2.imwrite(os.path.join(image_directory, f"im_{i}.png"), cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(os.path.join(image_directory, f"im_{i}.png"),
                                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        stream.stop_stream()
        stream.close()

        wf = wave.open(f"data/nlp_audio/question_{self.question_idx}.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.pyaud.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.audio_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        if self.picam:
            self.picam.stop()

        if self.cv2_cam:
            self.cv2_cam.release()

        print("question complete")

        self.display_screen.instruction = "Press the button to start"
        self.main_button.text = "begin"
        self.listening = False
        self.update_display()

    def exit_sequence(self):
        """
        Run FER classification on captured data and store per-question labels.

        In auto_run mode, generates synthetic random predictions. Otherwise,
        loads the Keras FER model and runs inference on the saved image datasets.
        Results are stored in self.label_data and written to
        data/affective_predictions.json. Captured files are removed if
        cleanse_files is True.
        """
        if self.classify:
            fer_class_labels = ["Negative", "Neutral", "Positive"]
            if self.auto_run:
                label_data = {}
                mood = random.choices([0, 1, 2], weights=(2, 1, 2), k=1)

                for q_idx in range(self.question_count):
                    predictions = np.random.normal(1, 5, (100, 3))
                    predictions[:, mood] *= 10

                    fer_labels = np.argmax(predictions, axis=1)
                    unique, counts = np.unique(fer_labels, return_counts=True)
                    label = "Test label"

                    label_data[f"question_{q_idx}"] = {
                        "affective_data": dict(zip(fer_class_labels, counts.tolist())),
                        "nlp_label": label
                    }

                self.label_data = label_data
                return

            label_data = {}

            try:
                affective_model = keras.models.load_model("models/AffectInceptionResNetV3.keras")
            except Exception:
                print("models failed to load")

            image_shape = (224, 224, 3)
            for q_idx in range(self.question_idx):
                affective_data = {}
                image_directory = f"data/affective_images/question_{q_idx}"

                image_ds = keras.utils.image_dataset_from_directory(
                    directory=image_directory,
                    batch_size=16,
                    image_size=image_shape[0:2],
                    shuffle=False)

                print("found the images")

                predictions = affective_model.predict(image_ds)
                fer_labels = np.argmax(predictions, axis=1)

                unique, counts = np.unique(fer_labels, return_counts=True)
                affective_data["label_counts"] = dict(zip(unique, counts))

                label_data[f"question_{q_idx}"] = {"affective_data": affective_data}

            self.label_data = label_data

            with open("data/affective_predictions.json", "w") as write_file:
                json.dump(label_data, write_file, cls=NpEncoder, indent=4)

        if self.cleanse_files:
            print("cleansing files")
            shutil.rmtree("data/affective_images")
            shutil.rmtree("data/nlp_audio")
            print("successfully cleansed files")

    def loop(self):
        """
        Main event loop — called by the orchestrator.

        Waits for the patient to press the Begin button, runs question_loop()
        for each question, then hands control back on completion.
        """
        self.entry_sequence()
        while self.running:
            if self.auto_run:
                self.running = False

            else:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_s:
                            if self.parent:
                                take_screenshot(self.parent.window)
                            else:
                                take_screenshot(self.window, "affective_module")

                        elif event.key == pg.K_ESCAPE:
                            self.running = False

                    elif event.type == pg.MOUSEBUTTONDOWN:
                        if self.parent:
                            pos = self.parent.get_relative_mose_pos()
                        else:
                            pos = pg.Vector2(pg.mouse.get_pos()) - pg.Vector2(0, self.display_size.y)

                        button_id = self.touch_screen.click_test(pos)
                        if button_id is not None:
                            if button_id:
                                self.question_loop()

                                self.question_idx += 1
                                if self.question_idx == self.question_count:
                                    self.running = False
                                    print("quitting")

                    elif event.type == pg.QUIT:
                        self.running = False

        self.exit_sequence()
