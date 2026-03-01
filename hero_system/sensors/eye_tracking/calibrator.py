"""
Eye Tracking Calibrator
9-point polynomial regression calibration.
GPIO23 external button replaces keyboard SPACE for point capture.
"""

import logging
import numpy as np
import cv2
import depthai as dai
import mediapipe as mp
import subprocess
import re
from typing import Optional, Tuple
from uuid import UUID

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from .config import EyeTrackingConfig
from .utils import correct_frame as _correct_frame, extract_features as _extract_features

logger = logging.getLogger(__name__)

# 9-point calibration grid (normalised 0-1 screen coords)
_CALIB_POINTS_BASE = np.array([
    [0.05, 0.05], [0.5, 0.05], [0.95, 0.05],
    [0.05, 0.5],  [0.5, 0.5],  [0.95, 0.5],
    [0.05, 0.95], [0.5, 0.95], [0.95, 0.95],
], dtype=np.float32)


def _get_pi_resolution() -> tuple:
    """
    Get the Pi display resolution.

    Returns:
        Tuple of (width, height) in pixels. Always (1024, 600) for the official Pi display.
    """
    try:
        import subprocess, re
        out = subprocess.check_output(["xrandr"]).decode()
        match = re.search(r"current (\d+) x (\d+)", out)
        if match:
            return 1024, 600
            # return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return 1024, 600


def _setup_gpio_button(gpio_pin: int):
    """
    Configure a GPIO pin as input using gpiod v2.

    Args:
        gpio_pin: BCM pin number to configure.

    Returns:
        gpiod request object if successful, None if gpiod is unavailable.
    """
    try:
        import gpiod
        from gpiod.line import Direction
        request = gpiod.request_lines(
            '/dev/gpiochip0',
            consumer="HeroCalibButton",
            config={
                (gpio_pin,): gpiod.LineSettings(direction=Direction.INPUT)
            }
        )
        return request
    except Exception as e:
        logger.warning(f"gpiod unavailable — falling back to SPACE key: {e}")
        return None


def _wait_for_gpio_button(request, gpio_pin: int) -> bool:
    """
    Block until the GPIO button is pressed or the user presses 'q'.

    Falls back to SPACE key if no gpiod request is provided.

    Args:
        request:  gpiod request object from _setup_gpio_button, or None for keyboard fallback.
        gpio_pin: BCM pin number to poll.

    Returns:
        True on button press, False if 'q' is pressed.
    """
    if request is None:
        # Keyboard fallback
        while True:
            key = cv2.waitKey(10)
            if key == ord(' '):
                return True
            if key == ord('q'):
                return False
    else:
        from gpiod.line import Value
        prev = Value.ACTIVE  # unpressed = ACTIVE (active-low)
        while True:
            val = request.get_value(gpio_pin)
            # Pressed = INACTIVE (active-low), detect falling edge
            if val == Value.INACTIVE and prev == Value.ACTIVE:
                return True
            prev = val
            if cv2.waitKey(10) == ord('q'):
                return False


class EyeTrackingCalibrator:
    """
    Interactive 9-point gaze calibration using polynomial regression.

    Runs a fullscreen OpenCV calibration UI on the bottom Pi display.
    The patient presses the GPIO23 button to capture each calibration point.
    Fits 2nd-order Ridge regression models for X and Y screen coordinates
    and saves the coefficients to CalibrationEyeTracking in the database.
    """
    def __init__(
        self,
        session_id: UUID,
        db_session,
        config: Optional[EyeTrackingConfig] = None
    ):
        """
        Initialise the calibrator.

        Args:
            session_id: UUID of the current session.
            db_session: SQLAlchemy session for saving calibration data.
            config:     EyeTrackingConfig instance. Defaults to EyeTrackingConfig.for_calibration().

        Returns:
            None.
        """
        self.session_id = session_id
        self.db_session = db_session
        self.config = config or EyeTrackingConfig.for_calibration()

        # DepthAI
        self.pipeline = None
        self.device   = None
        self.queue    = None

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = None

        # sklearn models (set after calibration)
        self.poly    = None
        self.model_x = None
        self.model_y = None

        # Raw calibration data (stored for DB)
        self._calib_features: list = []
        self._calib_targets:  list = []

        self.is_calibrated = False
        self.is_running    = False

        logger.info(f"EyeTrackingCalibrator initialised for session {session_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """
        Open the DepthAI camera pipeline and initialise MediaPipe FaceMesh.

        Raises:
            Exception if the camera or FaceMesh cannot be initialised.

        Returns:
            None.
        """
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
            logger.info("✓ EyeTrackingCalibrator started")
        except Exception as e:
            logger.error(f"✗ Failed to start calibrator: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """
        Release GPIO, close MediaPipe and DepthAI resources, and destroy OpenCV windows.

        Returns:
            None.
        """
        if hasattr(self, '_gpio_request') and self._gpio_request:
            try:
                self._gpio_request.release()
            except Exception:
                pass
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
        self.is_running = False
        cv2.destroyAllWindows()
        import time as _t
        _t.sleep(1.0)
        logger.info("✓ EyeTrackingCalibrator stopped")

    def run_calibration(self) -> bool:
        """
        Run the full 9-point calibration loop.

        Displays each calibration point on the bottom Pi screen. The patient
        presses the GPIO23 button to capture each point. Fits polynomial
        regression models on completion.

        Raises:
            RuntimeError if called before start().

        Returns:
            True if calibration completed successfully, False if aborted.
        """
        if not self.is_running:
            raise RuntimeError("Call start() before run_calibration()")

        screen_w, screen_h = _get_pi_resolution()
        calib_points = _CALIB_POINTS_BASE[list(self.config.calib_order)]

        # Setup GPIO once before the loop
        self._gpio_request = _setup_gpio_button(self.config.gpio_button_pin)
        if self._gpio_request is None:
            logger.warning("gpiod unavailable — using SPACE key fallback")

        # Bottom physical screen position (HDMI-A-1: 1024x600 at +1025+1200)
        BOTTOM_SCREEN_X = 1025
        BOTTOM_SCREEN_Y = 1200

        cv2.namedWindow("gaze", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("gaze", screen_w, screen_h)
        blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.imshow("gaze", blank)
        cv2.waitKey(1)
        cv2.moveWindow("gaze", BOTTOM_SCREEN_X, BOTTOM_SCREEN_Y)
        cv2.setWindowProperty("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self._calib_features = []
        self._calib_targets  = []

        for pt_idx, (tx_n, ty_n) in enumerate(calib_points):
            tx = int(tx_n * screen_w)
            ty = int(ty_n * screen_h)

            # --- Wait for button press ---
            while True:
                f     = self.queue.get()
                frame = _correct_frame(f.getCvFrame(), self.config.camera_flip_code)

                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                for fp in calib_points[pt_idx:]:
                    cv2.circle(canvas, (int(fp[0]*screen_w), int(fp[1]*screen_h)),
                               self.config.dot_radius, (60, 60, 60), -1)
                cv2.circle(canvas, (tx, ty), self.config.dot_radius, (0, 255, 0), -1)
                cv2.putText(canvas,
                            f"Point {pt_idx+1}/9 — look at dot and press button",
                            (screen_w//2 - 320, screen_h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow("gaze", canvas)

                pressed = _wait_for_gpio_button(self._gpio_request, self.config.gpio_button_pin)
                if not pressed:
                    logger.warning("Calibration aborted by user")
                    return False
                break

            # --- Collect samples ---
            samples = []
            while len(samples) < self.config.samples_per_point:
                f     = self.queue.get()
                frame = _correct_frame(f.getCvFrame(), self.config.camera_flip_code)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(canvas, (tx, ty), self.config.dot_radius, (0, 200, 255), -1)
                progress = int((len(samples) / self.config.samples_per_point) * 200)
                cv2.rectangle(canvas, (tx-100, ty+30), (tx-100+progress, ty+45), (0, 200, 255), -1)
                cv2.imshow("gaze", canvas)
                cv2.waitKey(1)

                if results.multi_face_landmarks:
                    lm   = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    feat = _extract_features(lm, w, h, self.config)
                    samples.append(feat)

            self._calib_features.append(np.mean(samples, axis=0))
            self._calib_targets.append([tx, ty])
            logger.info(f"  ✓ Point {pt_idx+1}/9 captured")

        # --- Fit models ---
        self.poly    = PolynomialFeatures(degree=self.config.poly_degree, include_bias=True)
        X = self.poly.fit_transform(np.array(self._calib_features))
        T = np.array(self._calib_targets, dtype=np.float32)

        self.model_x = Ridge(alpha=self.config.ridge_alpha)
        self.model_y = Ridge(alpha=self.config.ridge_alpha)
        self.model_x.fit(X, T[:, 0])
        self.model_y.fit(X, T[:, 1])

        self.is_calibrated = True
        logger.info("✓ Polynomial models fitted successfully")
        return True

    def save_to_database(self) -> bool:
        """
        Persist fitted model coefficients and intercepts to CalibrationEyeTracking.

        Replaces any existing calibration record for this session.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.is_calibrated:
            logger.error("Cannot save — calibration not complete")
            return False

        try:
            from datetime import datetime, timezone
            from hero_core.database.models.sensors import CalibrationEyeTracking

            existing = self.db_session.query(CalibrationEyeTracking).filter(
                CalibrationEyeTracking.session_id == self.session_id
            ).first()
            if existing:
                self.db_session.delete(existing)

            record = CalibrationEyeTracking(
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc),
                coeff_x=self.model_x.coef_.tolist(),
                coeff_y=self.model_y.coef_.tolist(),
                intercept_x=float(self.model_x.intercept_),
                intercept_y=float(self.model_y.intercept_),
                poly_degree=self.config.poly_degree,
                calib_features=np.array(self._calib_features).tolist(),
                calib_targets=self._calib_targets,
            )
            self.db_session.add(record)
            self.db_session.commit()
            logger.info(f"✓ Calibration saved for session {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving calibration: {e}", exc_info=True)
            self.db_session.rollback()
            return False

    @staticmethod
    def load_from_database(db_session, session_id: UUID) -> Optional[dict]:
        """
        Load calibration coefficients for a session from the database.

        Args:
            db_session: SQLAlchemy session.
            session_id: UUID of the session to load calibration for.

        Returns:
            Dict with keys coeff_x, coeff_y, intercept_x, intercept_y, poly_degree,
            or None if no calibration record exists.
        """
        try:
            from hero_core.database.models.sensors import CalibrationEyeTracking
            record = db_session.query(CalibrationEyeTracking).filter(
                CalibrationEyeTracking.session_id == session_id
            ).first()
            if not record:
                return None
            return {
                'coeff_x':     record.coeff_x,
                'coeff_y':     record.coeff_y,
                'intercept_x': record.intercept_x,
                'intercept_y': record.intercept_y,
                'poly_degree': record.poly_degree,
            }
        except Exception as e:
            logger.error(f"Error loading calibration: {e}", exc_info=True)
            return None

    def __repr__(self):
        """String representation showing calibration state."""
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"<EyeTrackingCalibrator({status})>"
