"""
Eye Tracking Processor
Camera initialization, gaze computation, and database storage
Combines collection and processing since we only store processed gaze coordinates
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
from scipy.spatial.transform import Rotation as Rscipy

from .config import EyeTrackingConfig
from .utils import rot_x, normalize, compute_scale

if TYPE_CHECKING:
    from hero_system.coordinator import SensorCoordinator

logger = logging.getLogger(__name__)


class EyeTrackingProcessor:
    """
    Eye tracking processor for HERO system
    Handles camera, gaze computation, and database storage
    """

    def __init__(
        self,
        session_id: UUID,
        db_session,
        coordinator: 'SensorCoordinator',
        config: Optional[EyeTrackingConfig] = None
    ):
        """
        Initialize eye tracking processor

        Args:
            session_id: UUID of current session
            db_session: Database session
            coordinator: Sensor coordinator for timestamps
            config: Eye tracking configuration
        """
        self.session_id = session_id
        self.db_session = db_session
        self.coordinator = coordinator
        self.config = config if config else EyeTrackingConfig.for_session()

        # DepthAI pipeline and device
        self.pipeline = None
        self.device = None
        self.queue = None

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None

        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Head pose tracking
        self.R_ref_nose = None

        # Gaze smoothing buffer
        self.combined_gaze_directions = deque(maxlen=config.filter_length)

        # Calibration data (loaded or set during calibration)
        self.left_sphere_offset = None
        self.left_nose_scale = None
        self.right_sphere_offset = None
        self.right_nose_scale = None
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0

        # Current gaze (for games to access)
        self.current_gaze_x = None
        self.current_gaze_y = None
        self._gaze_lock = threading.Lock()

        # Sample tracking
        self.sample_count = 0

        # Import database models
        try:
            from hero_core.database.models.sensors import SensorEyeTracking
            self.SensorEyeTracking = SensorEyeTracking
        except ImportError:
            logger.error("Could not import SensorEyeTracking model")
            self.SensorEyeTracking = None

        logger.info(f"Eye Tracking Processor initialized for session {session_id}")

    def load_calibration(self, calibration_data: dict):
        """
        Load calibration data from database

        Args:
            calibration_data: Dictionary with calibration parameters
        """
        self.left_sphere_offset = np.array([
            calibration_data['left_sphere_offset_x'],
            calibration_data['left_sphere_offset_y'],
            calibration_data['left_sphere_offset_z']
        ])
        self.left_nose_scale = calibration_data['left_nose_scale']

        self.right_sphere_offset = np.array([
            calibration_data['right_sphere_offset_x'],
            calibration_data['right_sphere_offset_y'],
            calibration_data['right_sphere_offset_z']
        ])
        self.right_nose_scale = calibration_data['right_nose_scale']

        self.calibration_offset_yaw = calibration_data['offset_yaw']
        self.calibration_offset_pitch = calibration_data['offset_pitch']

        logger.info("✓ Calibration data loaded")

    def start(self):
        """Start eye tracking"""
        if self.is_running:
            logger.warning("Eye tracking already running")
            return

        try:
            # Initialize DepthAI pipeline
            logger.info("Setting up DepthAI pipeline...")
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
            logger.info(f"Connected: {self.device.getDeviceName()}")

            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Initialize MediaPipe
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.config.mp_static_image_mode,
                max_num_faces=self.config.mp_max_num_faces,
                refine_landmarks=self.config.mp_refine_landmarks,
                min_detection_confidence=self.config.mp_min_detection_confidence,
                min_tracking_confidence=self.config.mp_min_tracking_confidence
            )

            # Start processing
            self.is_running = True
            self.stop_event.clear()
            self.sample_count = 0

            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="EyeTracking-Processing-Thread",
                daemon=True
            )
            self.processing_thread.start()

            logger.info("✓ Eye tracking started successfully")

        except Exception as e:
            logger.error(f"✗ Failed to start eye tracking: {e}", exc_info=True)
            self.is_running = False
            raise

    def stop(self):
        """Stop eye tracking"""
        if not self.is_running:
            logger.warning("Eye tracking not running")
            return

        try:
            logger.info("Stopping eye tracking...")

            # Signal thread to stop
            self.stop_event.set()

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            # Cleanup
            if self.face_mesh:
                self.face_mesh.close()

            if self.device:
                self.device.close()

            # Final commit
            try:
                self.db_session.commit()
                logger.info(f"Final commit: {self.sample_count} total samples")
            except Exception as e:
                logger.error(f"Error in final commit: {e}")
                self.db_session.rollback()

            self.is_running = False
            logger.info("✓ Eye tracking stopped successfully")

        except Exception as e:
            logger.error(f"✗ Error stopping eye tracking: {e}", exc_info=True)
            raise

    def _processing_loop(self):
        """Main processing loop - runs in separate thread"""
        logger.info("Eye tracking processing loop started")

        while not self.stop_event.is_set():
            try:
                # Get frame from camera
                inRgb = self.queue.get()
                frame = inRgb.getCvFrame()

                h, w = frame.shape[:2]

                # Process with MediaPipe
                frame_rgb = frame
                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    # Compute head pose
                    head_center, R_final, nose_points_3d = self._compute_head_pose(
                        face_landmarks, w, h
                    )

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

                    # Only track gaze if calibrated
                    if self._is_calibrated():
                        self._process_and_store_gaze(
                            iris_3d_left,
                            iris_3d_right,
                            head_center,
                            R_final,
                            nose_points_3d
                        )

                # Small sleep to avoid CPU overload
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Eye tracking processing loop stopped")

    def _compute_head_pose(
        self,
        face_landmarks,
        w: int,
        h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute head pose from nose landmarks using PCA"""
        # Extract 3D points for nose landmarks
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

    def _is_calibrated(self) -> bool:
        """Check if calibration data is loaded"""
        return (
            self.left_sphere_offset is not None and
            self.right_sphere_offset is not None and
            self.left_nose_scale is not None and
            self.right_nose_scale is not None
        )

    def _process_and_store_gaze(
        self,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        head_center: np.ndarray,
        R_final: np.ndarray,
        nose_points_3d: np.ndarray
    ):
        """Process gaze and store to database"""
        if self.SensorEyeTracking is None:
            return

        try:
            # Compute current nose scale
            current_nose_scale = compute_scale(nose_points_3d)

            # Compute eye sphere positions with scaling
            scale_ratio_left = current_nose_scale / self.left_nose_scale
            scaled_offset_left = self.left_sphere_offset * scale_ratio_left
            sphere_world_left = head_center + R_final @ scaled_offset_left

            scale_ratio_right = current_nose_scale / self.right_nose_scale
            scaled_offset_right = self.right_sphere_offset * scale_ratio_right
            sphere_world_right = head_center + R_final @ scaled_offset_right

            # Compute gaze directions
            left_gaze_dir = normalize(iris_3d_left - sphere_world_left)
            right_gaze_dir = normalize(iris_3d_right - sphere_world_right)

            # Combine and smooth
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction = normalize(raw_combined_direction)

            self.combined_gaze_directions.append(raw_combined_direction)
            avg_combined_direction = np.mean(self.combined_gaze_directions, axis=0)
            avg_combined_direction = normalize(avg_combined_direction)

            # Convert to screen coordinates
            screen_x, screen_y, raw_yaw, raw_pitch = self._convert_gaze_to_screen_coordinates(
                avg_combined_direction
            )

            # Update current gaze (for games)
            with self._gaze_lock:
                self.current_gaze_x = screen_x
                self.current_gaze_y = screen_y

            # Store to database with coordinator timestamp
            timestamp = self.coordinator.get_central_timestamp()

            gaze_sample = self.SensorEyeTracking(
                time=timestamp,
                session_id=self.session_id,
                gaze_x=float(screen_x),
                gaze_y=float(screen_y),
                raw_yaw=float(raw_yaw),
                raw_pitch=float(raw_pitch),
                confidence=None,  # Could add quality metric
                is_valid=True
            )
            self.db_session.add(gaze_sample)
            self.sample_count += 1

            # Batch commit
            if self.sample_count % self.config.batch_commit_size == 0:
                self.db_session.commit()
                logger.debug(f"Committed batch: {self.sample_count} total samples")

        except Exception as e:
            logger.error(f"Error processing gaze: {e}", exc_info=True)
            self.db_session.rollback()

    def _convert_gaze_to_screen_coordinates(
        self,
        combined_gaze_direction: np.ndarray
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

        # Flip yaw for screen coordinate system
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg

        # Apply calibration offsets
        yaw_deg += self.calibration_offset_yaw
        pitch_deg += self.calibration_offset_pitch

        # Convert to screen coordinates
        yaw_fov = self.config.yaw_degrees
        pitch_fov = self.config.pitch_degrees

        screen_x = int(((yaw_deg + yaw_fov) / (2 * yaw_fov)) * self.config.virtual_width)
        screen_y = int(((pitch_fov - pitch_deg) / (2 * pitch_fov)) * self.config.virtual_height)

        # Apply vertical offset correction
        screen_y = screen_y - int(self.config.camera_vertical_offset_pixels * 0.5)

        # Clamp to screen bounds
        screen_x = max(10, min(screen_x, self.config.virtual_width - 10))
        screen_y = max(10, min(screen_y, self.config.virtual_height - 10))

        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def get_current_gaze(self) -> Optional[Tuple[int, int]]:
        """
        Get current gaze position (for games to access)
        Thread-safe access to latest gaze coordinates

        Returns:
            Tuple of (x, y) or None if no gaze available
        """
        with self._gaze_lock:
            if self.current_gaze_x is not None and self.current_gaze_y is not None:
                return (self.current_gaze_x, self.current_gaze_y)
            return None

    def get_status(self) -> dict:
        """Get current processor status"""
        return {
            'sensor_type': 'EyeTracking',
            'mode': self.config.mode,
            'is_running': self.is_running,
            'is_calibrated': self._is_calibrated(),
            'session_id': str(self.session_id),
            'samples_collected': self.sample_count,
            'current_gaze': self.get_current_gaze(),
        }

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        return f"<EyeTrackingProcessor(mode={self.config.mode}, status={status})>"
    