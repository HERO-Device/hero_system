"""
Eye Tracking Calibration Module
================================
Fits into the Consultation module_order as "EyeCalib".

Flow:
  1. Opens camera + MediaPipe
  2. 9-point calibration (button press per point)
  3. 9-point validation
  4. Shows validation result to assisting person (GOOD / ACCEPTABLE / POOR)
     with mean error in degrees
  5. RETRY button → repeats from step 2 with same camera
     ACCEPT button → saves model to DB, starts collection thread
  6. On ACCEPT: calls parent.begin_gaze_collection()

Deliberately does NOT use OpenCV windows — all rendering is via pygame
on bottom_screen so it matches the rest of the UI.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pygame as pg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

# ── Colours ────────────────────────────────────────────────────────────────
BLACK  = (0,   0,   0)
GREEN  = (0,   220, 80)
WHITE  = (240, 240, 240)
CYAN   = (0,   210, 210)
YELLOW = (220, 200, 0)
DIM    = (50,  50,  50)
RED    = (220, 60,  60)
ORANGE = (220, 140, 0)
GREY   = (100, 100, 100)

# ── Calibration grid (9 points, normalised 0-1) ────────────────────────────
_CALIB_GRID = np.array([
    [0.05, 0.05], [0.50, 0.05], [0.95, 0.05],
    [0.05, 0.50], [0.50, 0.50], [0.95, 0.50],
    [0.05, 0.95], [0.50, 0.95], [0.95, 0.95],
], dtype=np.float32)

# Validation points chosen NOT to overlap with calibration grid
_VALID_GRID = np.array([
    [0.25, 0.25], [0.75, 0.25],
    [0.50, 0.50],
    [0.25, 0.75], [0.75, 0.75],
    [0.15, 0.50], [0.85, 0.50],
    [0.50, 0.10], [0.50, 0.90],
], dtype=np.float32)

# Calibration order: centre first, then corners, then edges
_CALIB_ORDER = (4, 0, 8, 2, 6, 1, 7, 3, 5)


class EyeCalibModule:
    """
    Plugs into Consultation.modules and Consultation.module_order.

    parent must expose:
      parent.bottom_screen  : pg.Surface  (bottom physical display)
      parent.window         : pg.Surface  (full combined window)
      parent.session_id     : UUID
      parent.db_session     : SQLAlchemy session  (or None)
      parent.clock          : CentralClock        (or None)
      parent.gaze_system    : set to self after ACCEPT so run.py
                              can call begin_collection() later
    """

    # Thresholds for rating
    GOOD_DEG       = 1.0   # < 1.0° → GOOD
    ACCEPTABLE_DEG = 2.0   # < 2.0° → ACCEPTABLE, else POOR

    # Samples collected per calibration / validation point
    SAMPLES_PER_CALIB = 60
    SAMPLES_PER_VALID = 40

    # Physical screen geometry for degree calculation
    SCREEN_WIDTH_MM    = 222.0
    SCREEN_DISTANCE_MM = 430.0

    # Camera / MediaPipe
    FLIP_CODE = 0          # cv2.flip code (0 = vertical)

    # MediaPipe iris landmark indices (refine_landmarks=True required)
    LEFT_IRIS  = 468;  LEFT_INNER  = 133;  LEFT_OUTER  = 33
    RIGHT_IRIS = 473;  RIGHT_INNER = 362;  RIGHT_OUTER = 263

    # Regression
    POLY_DEGREE  = 2
    RIDGE_ALPHA  = 0.01

    # DB batch size for collection
    BATCH_SIZE = 30

    # EMA smoothing for live gaze
    SMOOTHING_ALPHA = 0.15

    def __init__(self, parent, auto_run=False):
        self.parent    = parent
        self.auto_run  = auto_run
        self.running   = False
        self.results   = {}

        # Camera state (opened once, reused across retries)
        self._device   = None
        self._queue    = None
        self._face_mesh = None

        # Fitted models
        self.poly      = None
        self.model_x   = None
        self.model_y   = None
        self.is_calibrated = False

        # Collection thread
        self._stop_evt  = threading.Event()
        self._thread    = None
        self._gaze_lock = threading.Lock()
        self._gaze_x    = 0
        self._gaze_y    = 0
        self.sample_count = 0

        # Screen geometry (set in loop())
        self._w = 0
        self._h = 0

        pg.font.init()
        self._font_large = pg.font.SysFont('couriernew', 28, bold=True)
        self._font_med   = pg.font.SysFont('couriernew', 20)
        self._font_small = pg.font.SysFont('couriernew', 16)

    # ── Public API (Consultation interface) ────────────────────────────────

    def loop(self):
        """Called by Consultation.loop() — blocks until calibration accepted."""
        self.running = True
        surf = self.parent.bottom_screen
        self._w = surf.get_width()
        self._h = surf.get_height()

        try:
            self._open_camera()
            self._open_face_mesh()

            while self.running:
                # ── Calibration phase ──────────────────────────────────────
                self._show_message("Eye Tracking Calibration",
                                   "Press the button to begin each point",
                                   colour=CYAN)
                self._wait_for_button()

                feats, tgts = self._run_calibration_phase()

                if feats is None:
                    # User aborted (q key)
                    self.running = False
                    break

                # ── Fit models ─────────────────────────────────────────────
                self._fit_models(feats, tgts)

                # ── Validation phase ───────────────────────────────────────
                self._show_message("Validation",
                                   "Press the button for each point",
                                   colour=YELLOW)
                self._wait_for_button()

                mean_deg, std_deg, rating = self._run_validation_phase()

                # ── Show result to assisting person ────────────────────────
                action = self._show_validation_result(mean_deg, std_deg, rating)

                if action == 'accept':
                    self._save_calibration(feats, tgts)
                    self._save_validation_record(mean_deg, std_deg, rating)
                    self.results = {
                        'mean_error_deg': round(mean_deg, 3),
                        'std_deg':        round(std_deg,  3),
                        'rating':         rating,
                        'calibrated':     True,
                    }
                    self.is_calibrated = True
                    # Expose self so run.py can call begin_collection()
                    self.parent.gaze_system = self
                    self.running = False

                # else: retry — loop back to calibration

        except Exception as e:
            logger.error(f"EyeCalibModule failed: {e}", exc_info=True)
            self._show_message("Eye Tracking FAILED",
                               str(e)[:60],
                               colour=RED)
            time.sleep(3)
            self.results = {'calibrated': False, 'error': str(e)}
            self.running = False

    def begin_collection(self):
        """Start background gaze collection thread. Called after ACCEPT."""
        if not self.is_calibrated:
            logger.warning("begin_collection called before calibration — ignoring")
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._collection_loop, daemon=True,
            name="EyeGaze-Collection"
        )
        self._thread.start()
        logger.info("✓ Gaze collection started")

    def stop(self):
        """Stop collection thread and release camera."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._close_camera()
        logger.info(f"EyeCalibModule stopped — {self.sample_count} gaze samples")

    def get_current_gaze(self):
        """Thread-safe read of latest smoothed gaze coords."""
        with self._gaze_lock:
            return self._gaze_x, self._gaze_y

    # ── Camera ─────────────────────────────────────────────────────────────

    def _open_camera(self):
        import depthai as dai
        pipeline = dai.Pipeline()
        cam  = pipeline.create(dai.node.ColorCamera)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.preview.link(xout.input)
        self._device = dai.Device(pipeline)
        self._queue  = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        logger.info("✓ DepthAI camera opened")

    def _open_face_mesh(self):
        import mediapipe as mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _close_camera(self):
        try:
            if self._face_mesh:
                self._face_mesh.close()
        except Exception:
            pass
        try:
            if self._device:
                self._device.close()
        except Exception:
            pass
        self._device = self._queue = self._face_mesh = None
        try:
            import subprocess
            subprocess.run(['sudo', 'pkill', '-f', 'depthai'], capture_output=True)
        except Exception:
            pass

    def _get_frame(self):
        f = self._queue.get()
        return cv2.flip(f.getCvFrame(), self.FLIP_CODE)

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract_features(self, landmarks, img_w, img_h) -> np.ndarray:
        def norm_iris(iris_idx, inner_idx, outer_idx):
            iris  = landmarks[iris_idx]
            inner = landmarks[inner_idx]
            outer = landmarks[outer_idx]
            ix, iy = iris.x * img_w, iris.y * img_h
            inx  = inner.x * img_w
            outx = outer.x * img_w
            eye_w  = abs(inx - outx) + 1e-6
            eye_cy = (inner.y + outer.y) / 2 * img_h
            return (ix - outx) / eye_w, ((iy - eye_cy) * 5) / eye_w

        l
