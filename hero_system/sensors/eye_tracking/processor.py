"""
Eye Tracking Processor
Loads polynomial regression calibration from DB and runs live gaze tracking,
storing gaze coordinates to sensor_eye_tracking alongside the cognitive games.
"""

import logging
import time
import threading
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from uuid import UUID
from datetime import datetime, timezone
from collections import deque

import cv2
import depthai as dai
import mediapipe as mp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from .config import EyeTrackingConfig
from .utils import correct_frame as _correct_frame, extract_features as _extract_features

if TYPE_CHECKING:
    from hero_core.coordinator.coordinator import SensorCoordinator

logger = logging.getLogger(__name__)


class EyeTrackingProcessor:
    """
    Live gaze processor for HERO system.
    Loads fitted polynomial models from DB (saved by EyeTrackingCalibrator),
    runs gaze computation in a background thread, stores to sensor_eye_tracking.
    Games can call get_current_gaze() at any time.
    """

    def __init__(
        self,
        session_id: UUID,
        db_session,
        coordinator: 'SensorCoordinator',
        config: Optional[EyeTrackingConfig] = None,
    ):
        """
        Initialise the EyeTrackingProcessor.

        Args:
            session_id:  UUID of the current session.
            db_session:  SQLAlchemy session for writing gaze samples.
            coordinator: SensorCoordinator providing the shared central clock.
            config:      EyeTrackingConfig instance. Defaults to EyeTrackingConfig.for_session().

        Returns:
            None.
        """
        self.session_id  = session_id
        self.db_session  = db_session
        self.coordinator = coordinator
        self.config      = config or EyeTrackingConfig.for_session()

        # DepthAI
        self.pipeline = None
        self.device   = None
        self.queue    = None

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = None

        # Polynomial models (loaded from DB)
        self.poly    = None
        self.model_x = None
        self.model_y = None

        # Threading
        self.is_running       = False
        self.processing_thread = None
        self.stop_event       = threading.Event()

        # EMA-smoothed gaze (shared with games)
        self._gaze_x    = None
        self._gaze_y    = None
        self._gaze_lock = threading.Lock()

        self.sample_count = 0

        try:
            from hero_core.database.models.sensors import SensorEyeTracking
            self.SensorEyeTracking = SensorEyeTracking
        except ImportError:
            logger.error("Could not import SensorEyeTracking model")
            self.SensorEyeTracking = None

        logger.info(f"EyeTrackingProcessor initialised for session {session_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_calibration(self, calibration_data: dict):
        """
        Reconstruct polynomial regression models from coefficients stored in the database.

        Must be called before start().

        Args:
            calibration_data: Dict with keys coeff_x, coeff_y, intercept_x, intercept_y,
                              poly_degree — as returned by EyeTrackingCalibrator.load_from_database().

        Returns:
            None.
        """
        degree = calibration_data.get('poly_degree', 2)
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)
        # Fit on a proper dummy so sklearn internal state is fully initialised
        self.poly.fit(np.zeros((1, 4), dtype=np.float32))

        self.model_x = Ridge()
        self.model_y = Ridge()

        self.model_x.coef_      = np.array(calibration_data['coeff_x'],  dtype=np.float64)
        self.model_x.intercept_ = np.float64(calibration_data['intercept_x'])
        self.model_y.coef_      = np.array(calibration_data['coeff_y'],  dtype=np.float64)
        self.model_y.intercept_ = np.float64(calibration_data['intercept_y'])

        # Minimal sklearn attributes needed for predict()
        self.model_x.n_features_in_ = self.poly.n_output_features_
        self.model_y.n_features_in_ = self.poly.n_output_features_

        logger.info(f"✓ Calibration loaded ({self.poly.n_output_features_} polynomial terms, degree {degree})")

    def start(self):
        """
        Open the DepthAI camera, initialise MediaPipe FaceMesh, and start the processing thread.

        Raises:
            RuntimeError if called before load_calibration().
            Exception if the camera or FaceMesh cannot be initialised.

        Returns:
            None.
        """
        if not self._is_calibrated():
            raise RuntimeError("Load calibration before calling start()")

        try:
            self.pipeline = dai.Pipeline()
            cam  = self.pipeline.create(dai.node.ColorCamera)
            xout = self.pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.setInterleaved(False)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.preview.link(xout.input)

            self.device = dai.Device(self.pipeline)
            self.queue  = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.config.mp_max_num_faces,
                refine_landmarks=self.config.mp_refine_landmarks,
                min_detection_confidence=self.config.mp_min_detection_confidence,
                min_tracking_confidence=self.config.mp_min_tracking_confidence,
            )

            self.is_running = True
            self.stop_event.clear()
            self.sample_count = 0

            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="EyeTracking-Thread",
                daemon=True,
            )
            self.processing_thread.start()
            logger.info("✓ EyeTrackingProcessor started")

        except Exception as e:
            logger.error(f"✗ Failed to start EyeTrackingProcessor: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """
        Stop the processing thread, release camera and MediaPipe resources, and flush remaining samples.

        Returns:
            None.
        """
        if not self.is_running:
            return

        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        if self.device:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        self.pipeline = None
        self.queue = None

        try:
            self.db_session.commit()
            logger.info(f"✓ EyeTrackingProcessor stopped — {self.sample_count} samples saved")
        except Exception as e:
            logger.error(f"Error in final commit: {e}")
            self.db_session.rollback()

        self.is_running = False

    def get_current_gaze(self) -> Optional[Tuple[int, int]]:
        """
        Thread-safe read of the latest EMA-smoothed gaze position.

        Returns:
            Tuple of (gaze_x, gaze_y) in pixels, or None if no gaze has been computed yet.
        """
        with self._gaze_lock:
            if self._gaze_x is not None and self._gaze_y is not None:
                return (self._gaze_x, self._gaze_y)
            return None

    def get_status(self) -> dict:
        """
        Return the current processor state.

        Returns:
            Dict containing sensor type, running state, calibration state,
            sample count, and current gaze coordinates.
        """
        return {
            'sensor_type':      'eye_tracking',
            'is_running':       self.is_running,
            'is_calibrated':    self._is_calibrated(),
            'samples_collected': self.sample_count,
            'current_gaze':     self.get_current_gaze(),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_calibrated(self) -> bool:
        """
        Check whether polynomial models have been loaded.

        Returns:
            True if poly, model_x and model_y are all set.
        """
        return self.poly is not None and self.model_x is not None and self.model_y is not None

    def _processing_loop(self):
        """
        Background thread — grabs frames, computes EMA-smoothed gaze, and stores to the database.

        Returns:
            None.
        """
        logger.info("Eye tracking processing loop started")

        # Initialise EMA cursor at screen centre
        import subprocess, re
        try:
            out = subprocess.check_output(["xrandr"]).decode()
            match = re.search(r"current (\d+) x (\d+)", out)
            screen_w = int(match.group(1)) if match else 1024
            screen_h = int(match.group(2)) if match else 600
        except Exception:
            screen_w, screen_h = 1024, 600

        gaze_x = screen_w // 2
        gaze_y = screen_h // 2

        while not self.stop_event.is_set():
            try:
                f     = self.queue.get()
                frame = _correct_frame(f.getCvFrame(), self.config.camera_flip_code)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm   = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]

                    feat      = _extract_features(lm, w, h, self.config)
                    feat_poly = self.poly.transform(feat.reshape(1, -1))

                    raw_x = float(self.model_x.predict(feat_poly)[0])
                    raw_y = float(self.model_y.predict(feat_poly)[0])

                    raw_x = np.clip(raw_x, 0, screen_w - 1)
                    raw_y = np.clip(raw_y, 0, screen_h - 1)

                    # EMA smoothing
                    a     = self.config.smoothing_alpha
                    gaze_x = int(a * raw_x + (1 - a) * gaze_x)
                    gaze_y = int(a * raw_y + (1 - a) * gaze_y)

                    # Update shared state for games
                    with self._gaze_lock:
                        self._gaze_x = gaze_x
                        self._gaze_y = gaze_y

                    # Store to DB
                    self._store_sample(gaze_x, gaze_y)

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in gaze processing loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Eye tracking processing loop stopped")

    def _store_sample(self, gaze_x: int, gaze_y: int):
        """
        Write one gaze sample to sensor_eye_tracking.

        Args:
            gaze_x: Smoothed X gaze coordinate in pixels.
            gaze_y: Smoothed Y gaze coordinate in pixels.

        Returns:
            None.
        """
        if self.SensorEyeTracking is None:
            return

        try:
            timestamp = self.coordinator.get_central_timestamp()
            sample = self.SensorEyeTracking(
                time=timestamp,
                session_id=self.session_id,
                gaze_x=float(gaze_x),
                gaze_y=float(gaze_y),
                is_valid=True,
            )
            self.db_session.add(sample)
            self.sample_count += 1

            if self.sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()

        except Exception as e:
            logger.error(f"Error storing gaze sample: {e}", exc_info=True)
            self.db_session.rollback()

    def __repr__(self):
        """String representation showing running state and sample count."""
        status = "running" if self.is_running else "stopped"
        return f"<EyeTrackingProcessor(status={status}, samples={self.sample_count})>"
