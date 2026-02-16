"""
Eye Tracking Calibrator
Interactive calibration workflow for eye sphere locking and screen center calibration
"""

import logging
import numpy as np
from typing import Optional, Tuple
from uuid import UUID

import cv2
import depthai as dai
import mediapipe as mp
from scipy.spatial.transform import Rotation as Rscipy

from .config import EyeTrackingConfig
from .utils import (
    rot_x, normalize, compute_scale,
    draw_wireframe_cube, draw_coordinate_axes
)

logger = logging.getLogger(__name__)


class EyeTrackingCalibrator:
    """
    Interactive calibration for eye tracking
    Handles eye sphere locking and screen center calibration
    """

    def __init__(
            self,
            session_id: UUID,
            db_session,
            config: Optional[EyeTrackingConfig] = None
    ):
        """
        Initialize calibrator

        Args:
            session_id: UUID of current session
            db_session: Database session
            config: Eye tracking configuration
        """
        self.session_id = session_id
        self.db_session = db_session
        self.config = config if config else EyeTrackingConfig.for_calibration()

        # DepthAI pipeline and device
        self.pipeline = None
        self.device = None
        self.queue = None

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None

        # Calibration state
        self.is_running = False

        # Head pose tracking
        self.R_ref_nose = None

        # Calibration data
        self.left_sphere_offset = None
        self.left_nose_scale = None
        self.right_sphere_offset = None
        self.right_nose_scale = None
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0

        # Calibration status flags
        self.eye_spheres_calibrated = False
        self.screen_center_calibrated = False

        # Latest frame data (for calibration computation)
        self.latest_iris_3d_left = None
        self.latest_iris_3d_right = None
        self.latest_head_center = None
        self.latest_R_final = None
        self.latest_nose_points_3d = None

        logger.info(f"Eye Tracking Calibrator initialized for session {session_id}")

    def start(self):
        """Start calibration session"""
        if self.is_running:
            logger.warning("Calibrator already running")
            return

        try:
            # Initialize DepthAI pipeline
            logger.info("Setting up DepthAI pipeline for calibration...")
            self.pipeline = dai.Pipeline()

            camRgb = self.pipeline.create(dai.node.ColorCamera)
            xoutRgb = self.pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")

            camRgb.setInterleaved(False)
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            camRgb.setPreviewSize(self.config.preview_width, self.config.preview_height)

            camRgb.preview.link(xoutRgb.input)

            # Connect to device
            logger.info("Connecting to ArduCam...")
            self.device = dai.Device(self.pipeline)
            logger.info(f"✓ Connected: {self.device.getDeviceName()}")

            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Initialize MediaPipe
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.config.mp_static_image_mode,
                max_num_faces=self.config.mp_max_num_faces,
                refine_landmarks=self.config.mp_refine_landmarks,
                min_detection_confidence=self.config.mp_min_detection_confidence,
                min_tracking_confidence=self.config.mp_min_tracking_confidence
            )

            self.is_running = True

            logger.info("✓ Calibration session started")
            logger.info("  Press 'c' to calibrate eye spheres")
            logger.info("  Press 's' to calibrate screen center (after 'c')")

        except Exception as e:
            logger.error(f"✗ Failed to start calibration: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """Stop calibration session"""
        if not self.is_running:
            logger.warning("Calibrator not running")
            return

        try:
            logger.info("Stopping calibration session...")

            # Cleanup
            if self.face_mesh:
                self.face_mesh.close()

            if self.device:
                self.device.close()

            self.is_running = False
            logger.info("✓ Calibration session stopped")

        except Exception as e:
            logger.error(f"✗ Error stopping calibration: {e}", exc_info=True)
            raise

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get and process current frame
        Updates internal state for calibration

        Returns:
            Processed frame with overlays, or None if no frame available
        """
        if not self.is_running:
            return None

        try:
            # Get frame from camera
            inRgb = self.queue.get()
            frame = inRgb.getCvFrame()

            h, w = frame.shape[:2]

            # Process with MediaPipe
            frame_rgb = frame
            results = self.face_mesh.process(frame_rgb)

            # Convert to BGR for display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Compute head pose
                head_center, R_final, nose_points_3d = self._compute_head_pose(
                    face_landmarks, w, h
                )

                # Draw head coordinate frame
                draw_wireframe_cube(frame_bgr, head_center, R_final, size=80)
                draw_coordinate_axes(frame_bgr, head_center, R_final, size=80)

                # Get iris positions
                left_iris = face_landmarks[self.config.left_iris_idx]
                right_iris = face_landmarks[self.config.right_iris_idx]

                iris_3d_left = np.array([
                    left_iris.x * w,
                    left_iris.y * h,
                    left_iris.z * w
                ])

                iris_3d_right = np.array([
                    right_iris.x * w,
                    right_iris.y * h,
                    right_iris.z * w
                ])

                # Store latest data for calibration
                self.latest_iris_3d_left = iris_3d_left
                self.latest_iris_3d_right = iris_3d_right
                self.latest_head_center = head_center
                self.latest_R_final = R_final
                self.latest_nose_points_3d = nose_points_3d

                # Draw iris points or eye spheres
                if not self.eye_spheres_calibrated:
                    # Draw iris centers (before calibration)
                    x_iris_l = int(left_iris.x * w)
                    y_iris_l = int(left_iris.y * h)
                    cv2.circle(frame_bgr, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)

                    x_iris_r = int(right_iris.x * w)
                    y_iris_r = int(right_iris.y * h)
                    cv2.circle(frame_bgr, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
                else:
                    # Draw eye spheres (after calibration)
                    current_nose_scale = compute_scale(nose_points_3d)

                    # Left eye sphere
                    scale_ratio_left = current_nose_scale / self.left_nose_scale
                    scaled_offset_left = self.left_sphere_offset * scale_ratio_left
                    sphere_world_left = head_center + R_final @ scaled_offset_left
                    x_sphere_l = int(sphere_world_left[0])
                    y_sphere_l = int(sphere_world_left[1])
                    scaled_radius_left = int(self.config.base_radius * scale_ratio_left)
                    cv2.circle(frame_bgr, (x_sphere_l, y_sphere_l), scaled_radius_left, (255, 255, 25), 2)

                    # Right eye sphere
                    scale_ratio_right = current_nose_scale / self.right_nose_scale
                    scaled_offset_right = self.right_sphere_offset * scale_ratio_right
                    sphere_world_right = head_center + R_final @ scaled_offset_right
                    x_sphere_r = int(sphere_world_right[0])
                    y_sphere_r = int(sphere_world_right[1])
                    scaled_radius_right = int(self.config.base_radius * scale_ratio_right)
                    cv2.circle(frame_bgr, (x_sphere_r, y_sphere_r), scaled_radius_right, (25, 255, 255), 2)

                    # Draw gaze rays if fully calibrated
                    if self.screen_center_calibrated:
                        left_gaze_dir = normalize(iris_3d_left - sphere_world_left)
                        right_gaze_dir = normalize(iris_3d_right - sphere_world_right)
                        combined_gaze = normalize((left_gaze_dir + right_gaze_dir) / 2)

                        combined_origin = (sphere_world_left + sphere_world_right) / 2
                        combined_target = combined_origin + combined_gaze * self.config.gaze_length

                        cv2.line(
                            frame_bgr,
                            (int(combined_origin[0]), int(combined_origin[1])),
                            (int(combined_target[0]), int(combined_target[1])),
                            (255, 255, 10), 3
                        )

            # Draw status text
            self._draw_status_overlay(frame_bgr)

            return frame_bgr

        except Exception as e:
            logger.error(f"Error getting frame: {e}", exc_info=True)
            return None

    def _compute_head_pose(
            self,
            face_landmarks,
            w: int,
            h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute head pose from nose landmarks using PCA"""
        points_3d = np.array([
            [
                face_landmarks[i].x * w,
                face_landmarks[i].y * h,
                face_landmarks[i].z * w
            ]
            for i in self.config.nose_indices
        ])

        center = np.mean(points_3d, axis=0)
        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        r = Rscipy.from_matrix(eigvecs)
        roll, pitch, yaw = r.as_euler('zyx', degrees=False)
        R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

        # Fix coordinate flipping
        if self.R_ref_nose is None:
            self.R_ref_nose = R_final.copy()
        else:
            for i in range(3):
                if np.dot(R_final[:, i], self.R_ref_nose[:, i]) < 0:
                    R_final[:, i] *= -1

        return center, R_final, points_3d

    def _draw_status_overlay(self, frame: np.ndarray):
        """Draw calibration status on frame"""
        y_offset = 30

        # Eye spheres status
        if not self.eye_spheres_calibrated:
            text = "Press 'c' to calibrate eye spheres"
            color = (0, 165, 255)  # Orange
        elif not self.screen_center_calibrated:
            text = "Eye spheres: OK | Look at screen center, press 's'"
            color = (0, 165, 255)  # Orange
        else:
            text = "Calibration complete! Press 'q' to finish"
            color = (0, 255, 0)  # Green

        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    def calibrate_eye_spheres(self) -> bool:
        """
        Calibrate eye sphere positions
        Should be called when user presses 'c'

        Returns:
            True if successful, False otherwise
        """
        if self.latest_nose_points_3d is None:
            logger.warning("No face detected - cannot calibrate")
            return False

        try:
            current_nose_scale = compute_scale(self.latest_nose_points_3d)

            # Compute left eye sphere offset
            self.left_sphere_offset = self.latest_R_final.T @ (
                    self.latest_iris_3d_left - self.latest_head_center
            )
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = self.latest_R_final.T @ camera_dir_world
            self.left_sphere_offset += self.config.base_radius * camera_dir_local
            self.left_nose_scale = current_nose_scale

            # Compute right eye sphere offset
            self.right_sphere_offset = self.latest_R_final.T @ (
                    self.latest_iris_3d_right - self.latest_head_center
            )
            self.right_sphere_offset += self.config.base_radius * camera_dir_local
            self.right_nose_scale = current_nose_scale

            self.eye_spheres_calibrated = True

            logger.info("✓ Eye spheres calibrated!")
            logger.info("  Now look at screen center and press 's'")

            return True

        except Exception as e:
            logger.error(f"Error calibrating eye spheres: {e}", exc_info=True)
            return False

    def calibrate_screen_center(self) -> bool:
        """
        Calibrate screen center offset
        Should be called when user presses 's' while looking at screen center

        Returns:
            True if successful, False otherwise
        """
        if not self.eye_spheres_calibrated:
            logger.warning("Must calibrate eye spheres first")
            return False

        if self.latest_iris_3d_left is None:
            logger.warning("No face detected - cannot calibrate")
            return False

        try:
            # Compute eye sphere world positions
            current_nose_scale = compute_scale(self.latest_nose_points_3d)

            scale_ratio_left = current_nose_scale / self.left_nose_scale
            scaled_offset_left = self.left_sphere_offset * scale_ratio_left
            sphere_world_left = self.latest_head_center + self.latest_R_final @ scaled_offset_left

            scale_ratio_right = current_nose_scale / self.right_nose_scale
            scaled_offset_right = self.right_sphere_offset * scale_ratio_right
            sphere_world_right = self.latest_head_center + self.latest_R_final @ scaled_offset_right

            # Compute gaze directions
            left_gaze_dir = normalize(self.latest_iris_3d_left - sphere_world_left)
            right_gaze_dir = normalize(self.latest_iris_3d_right - sphere_world_right)

            current_combined_direction = normalize((left_gaze_dir + right_gaze_dir) / 2)

            # Compute raw angles
            _, _, raw_yaw, raw_pitch = self._convert_gaze_to_screen_coordinates(
                current_combined_direction, 0, 0
            )

            # Compute offsets to make current gaze point to center
            self.calibration_offset_yaw = 0 - raw_yaw
            self.calibration_offset_pitch = 0 - raw_pitch

            self.screen_center_calibrated = True

            logger.info("✓ Screen center calibrated!")
            logger.info(f"  Offset - Yaw: {self.calibration_offset_yaw:.2f}°, "
                        f"Pitch: {self.calibration_offset_pitch:.2f}°")

            return True

        except Exception as e:
            logger.error(f"Error calibrating screen center: {e}", exc_info=True)
            return False

    def _convert_gaze_to_screen_coordinates(
            self,
            combined_gaze_direction: np.ndarray,
            offset_yaw: float,
            offset_pitch: float
    ) -> Tuple[int, int, float, float]:
        """Convert 3D gaze direction to 2D screen coordinates"""
        import math

        reference_forward = np.array([0, 0, -1])
        avg_direction = normalize(combined_gaze_direction)

        # Apply camera tilt correction
        tilt_rad = math.radians(self.config.camera_tilt_angle)
        rotation_matrix = rot_x(-tilt_rad)
        corrected_direction = rotation_matrix @ avg_direction

        # Calculate yaw
        xz_proj = np.array([corrected_direction[0], 0, corrected_direction[2]])
        xz_proj = normalize(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if corrected_direction[0] < 0:
            yaw_rad = -yaw_rad

        # Calculate pitch
        yz_proj = np.array([0, corrected_direction[1], corrected_direction[2]])
        yz_proj = normalize(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if corrected_direction[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        # Flip yaw
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg

        # Apply calibration offsets
        yaw_deg += offset_yaw
        pitch_deg += offset_pitch

        # Convert to screen coordinates
        yaw_fov = self.config.yaw_degrees
        pitch_fov = self.config.pitch_degrees

        screen_x = int(((yaw_deg + yaw_fov) / (2 * yaw_fov)) * self.config.virtual_width)
        screen_y = int(((pitch_fov - pitch_deg) / (2 * pitch_fov)) * self.config.virtual_height)

        screen_y = screen_y - int(self.config.camera_vertical_offset_pixels * 0.5)

        screen_x = max(10, min(screen_x, self.config.virtual_width - 10))
        screen_y = max(10, min(screen_y, self.config.virtual_height - 10))

        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def is_fully_calibrated(self) -> bool:
        """Check if all calibration steps are complete"""
        return self.eye_spheres_calibrated and self.screen_center_calibrated

    def get_calibration_data(self) -> dict:
        """
        Get calibration data for storage

        Returns:
            Dictionary with all calibration parameters
        """
        if not self.is_fully_calibrated():
            logger.warning("Calibration incomplete")
            return {}

        return {
            'left_sphere_offset_x': float(self.left_sphere_offset[0]),
            'left_sphere_offset_y': float(self.left_sphere_offset[1]),
            'left_sphere_offset_z': float(self.left_sphere_offset[2]),
            'left_nose_scale': float(self.left_nose_scale),
            'right_sphere_offset_x': float(self.right_sphere_offset[0]),
            'right_sphere_offset_y': float(self.right_sphere_offset[1]),
            'right_sphere_offset_z': float(self.right_sphere_offset[2]),
            'right_nose_scale': float(self.right_nose_scale),
            'offset_yaw': float(self.calibration_offset_yaw),
            'offset_pitch': float(self.calibration_offset_pitch),
        }

    def save_to_database(self):
        """
        Save calibration data to database
        Uses the session_id from initialization
        """
        if not self.is_fully_calibrated():
            logger.error("Cannot save - calibration incomplete")
            return False

        try:
            from datetime import datetime, timezone
            from hero_core.database.models.sensors import CalibrationEyeTracking

            # Check if calibration already exists for this session
            existing = self.db_session.query(CalibrationEyeTracking).filter(
                CalibrationEyeTracking.session_id == self.session_id
            ).first()

            if existing:
                logger.warning(f"Calibration already exists for session {self.session_id}, overwriting")
                self.db_session.delete(existing)

            # Create new calibration record
            calibration = CalibrationEyeTracking(
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc),
                left_sphere_offset_x=float(self.left_sphere_offset[0]),
                left_sphere_offset_y=float(self.left_sphere_offset[1]),
                left_sphere_offset_z=float(self.left_sphere_offset[2]),
                left_nose_scale=float(self.left_nose_scale),
                right_sphere_offset_x=float(self.right_sphere_offset[0]),
                right_sphere_offset_y=float(self.right_sphere_offset[1]),
                right_sphere_offset_z=float(self.right_sphere_offset[2]),
                right_nose_scale=float(self.right_nose_scale),
                offset_yaw=float(self.calibration_offset_yaw),
                offset_pitch=float(self.calibration_offset_pitch)
            )

            self.db_session.add(calibration)
            self.db_session.commit()

            logger.info(f"✓ Calibration saved for session {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving calibration: {e}", exc_info=True)
            self.db_session.rollback()
            return False

    @staticmethod
    def load_calibration_from_database(db_session, session_id: UUID) -> Optional[dict]:
        """
        Load calibration for a specific session

        Args:
            db_session: Database session
            session_id: UUID of the session

        Returns:
            Calibration data dict or None if not found
        """
        try:
            from hero_core.database.models.sensors import CalibrationEyeTracking

            calibration = db_session.query(CalibrationEyeTracking).filter(
                CalibrationEyeTracking.session_id == session_id
            ).first()

            if not calibration:
                logger.warning(f"No calibration found for session {session_id}")
                return None

            return {
                'left_sphere_offset_x': calibration.left_sphere_offset_x,
                'left_sphere_offset_y': calibration.left_sphere_offset_y,
                'left_sphere_offset_z': calibration.left_sphere_offset_z,
                'left_nose_scale': calibration.left_nose_scale,
                'right_sphere_offset_x': calibration.right_sphere_offset_x,
                'right_sphere_offset_y': calibration.right_sphere_offset_y,
                'right_sphere_offset_z': calibration.right_sphere_offset_z,
                'right_nose_scale': calibration.right_nose_scale,
                'offset_yaw': calibration.offset_yaw,
                'offset_pitch': calibration.offset_pitch,
            }

        except Exception as e:
            logger.error(f"Error loading calibration: {e}", exc_info=True)
            return None

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        calibrated = "fully calibrated" if self.is_fully_calibrated() else "not calibrated"
        return f"<EyeTrackingCalibrator(status={status}, {calibrated})>"
