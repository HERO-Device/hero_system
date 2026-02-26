"""
HERO GazeSystem
Pure pygame eye tracking: calibration + validation on bottom_screen,
then background collection to DB.
No OpenCV windows.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
import pygame as pg

logger = logging.getLogger(__name__)

BLACK  = (0,   0,   0)
GREEN  = (0,   255, 0)
WHITE  = (255, 255, 255)
CYAN   = (0,   220, 220)
YELLOW = (200, 200, 0)
DIM    = (50,  50,  50)
GREY   = (120, 120, 120)


class GazeSystem:

    _CALIB_GRID = np.array([
        [0.05, 0.05], [0.50, 0.05], [0.95, 0.05],
        [0.05, 0.50], [0.50, 0.50], [0.95, 0.50],
        [0.05, 0.95], [0.50, 0.95], [0.95, 0.95],
    ], dtype=np.float32)

    _VALID_GRID = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.50, 0.50],
        [0.25, 0.75], [0.75, 0.75],
        [0.15, 0.50], [0.85, 0.50],
        [0.50, 0.05], [0.50, 0.95],
    ], dtype=np.float32)

    def __init__(self, session_id, db_session, window, bottom_screen, config):
        self.session_id = session_id
        self.db_session = db_session
        self.window     = window
        self.bottom     = bottom_screen
        self.config     = config
        self.w          = bottom_screen.get_width()
        self.h          = bottom_screen.get_height()

        self.poly    = None
        self.model_x = None
        self.model_y = None
        self.device  = None
        self.queue   = None
        self.face_mesh = None
        self._calib_record = None

        self._stop_event  = threading.Event()
        self._thread      = None
        self._gaze_lock   = threading.Lock()
        self._gaze_x      = self.w // 2
        self._gaze_y      = self.h // 2
        self.sample_count = 0

        pg.font.init()
        self._font = pg.font.SysFont('couriernew', 18)

    # ── Public ────────────────────────────────────────────────────────────

    def start(self):
        """Open camera, calibrate, validate. Repeats if POOR, max 3 attempts.
        After 3 POOR attempts, keeps the best result and continues."""
        self._open_camera()
        self._open_face_mesh()

        MAX_ATTEMPTS = 3
        best_mean    = float("inf")
        best_attempt = 1

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._run_calibration()
            rating = self._run_validation()

            current_mean = self._calib_record.validation_mean_deg if self._calib_record else float("inf")
            if current_mean < best_mean:
                best_mean    = current_mean
                best_attempt = attempt

            if rating != "POOR":
                logger.info(f"Calibration accepted on attempt {attempt} ({rating})")
                break

            if attempt < MAX_ATTEMPTS:
                logger.warning(f"Validation POOR (attempt {attempt}/{MAX_ATTEMPTS}) — retrying")
                font = pg.font.SysFont("couriernew", 22, bold=True)
                self._draw(lambda s, f=font, a=attempt: (
                    s.blit(f.render("Calibration POOR — retrying...", True, (220, 60, 60)),
                           (self.w // 2 - 200, self.h // 2 - 20)),
                    s.blit(f.render(f"Attempt {a} of {MAX_ATTEMPTS}", True, (180, 180, 180)),
                           (self.w // 2 - 100, self.h // 2 + 20)),
                ))
                time.sleep(2.0)
            else:
                logger.warning(f"All {MAX_ATTEMPTS} attempts POOR — using best result (attempt {best_attempt}, {best_mean:.2f}deg)")
                font = pg.font.SysFont("couriernew", 22, bold=True)
                self._draw(lambda s, f=font, m=best_mean: (
                    s.blit(f.render("Using best available calibration", True, (220, 140, 0)),
                           (self.w // 2 - 210, self.h // 2 - 20)),
                    s.blit(f.render(f"Mean error: {m:.2f} deg", True, (180, 180, 180)),
                           (self.w // 2 - 110, self.h // 2 + 20)),
                ))
                time.sleep(2.0)

        logger.info("GazeSystem ready")

    def begin_collection(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._close()
        logger.info(f"GazeSystem stopped — {self.sample_count} samples")

    # ── Camera ────────────────────────────────────────────────────────────

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
        self.device = dai.Device(pipeline)
        self.queue  = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)

    def _open_face_mesh(self):
        import mediapipe as mp
        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _close(self):
        try:
            if self.face_mesh: self.face_mesh.close()
        except Exception: pass
        try:
            if self.device: self.device.close()
        except Exception: pass
        self.device = self.queue = self.face_mesh = None
        try:
            import subprocess
            subprocess.run(['sudo', 'pkill', '-f', 'depthai'], capture_output=True)
        except Exception: pass

    # ── Frame helpers ────────────────────────────────────────────────────

    def _get_frame(self):
        import cv2
        f = self.queue.get()
        return cv2.flip(f.getCvFrame(), self.config.camera_flip_code)

    def _extract_features(self, landmarks, img_w, img_h):
        def norm_iris(iris_idx, inner_idx, outer_idx):
            iris  = landmarks[iris_idx]
            inner = landmarks[inner_idx]
            outer = landmarks[outer_idx]
            ix, iy = iris.x * img_w, iris.y * img_h
            inx    = inner.x * img_w
            outx   = outer.x * img_w
            eye_w  = abs(inx - outx) + 1e-6
            eye_cy = (inner.y + outer.y) / 2 * img_h
            return (ix - outx) / eye_w, ((iy - eye_cy) * 5) / eye_w
        lx, ly = norm_iris(self.config.left_iris_idx,  self.config.left_eye_inner,  self.config.left_eye_outer)
        rx, ry = norm_iris(self.config.right_iris_idx, self.config.right_eye_inner, self.config.right_eye_outer)
        return np.array([lx, ly, rx, ry], dtype=np.float32)

    def _predict(self, feat):
        fp = self.poly.transform(feat.reshape(1, -1))
        x  = int(np.clip(float(self.model_x.predict(fp)[0]), 0, self.w - 1))
        y  = int(np.clip(float(self.model_y.predict(fp)[0]), 0, self.h - 1))
        return x, y

    # ── GPIO button ──────────────────────────────────────────────────────

    def _wait_for_button(self):
        try:
            import gpiod
            from gpiod.line import Direction, Value
            req  = gpiod.request_lines(
                '/dev/gpiochip4',
                consumer="HeroGaze",
                config={(23,): gpiod.LineSettings(direction=Direction.INPUT)}
            )
            prev = Value.ACTIVE
            while True:
                val = req.get_value(23)
                if val == Value.INACTIVE and prev == Value.ACTIVE:
                    req.release()
                    return
                prev = val
                for ev in pg.event.get():
                    if ev.type == pg.KEYDOWN and ev.key == pg.K_SPACE:
                        req.release()
                        return
                time.sleep(0.01)
        except Exception:
            while True:
                for ev in pg.event.get():
                    if ev.type == pg.KEYDOWN and ev.key == pg.K_SPACE:
                        return
                time.sleep(0.01)

    # ── Rendering ────────────────────────────────────────────────────────

    def _draw(self, fn):
        self.bottom.fill(BLACK)
        fn(self.bottom)
        pg.display.flip()
        pg.event.pump()

    def _dot(self, surf, tx, ty, colour, future_pts=None):
        if future_pts is not None:
            for fp in future_pts:
                pg.draw.circle(surf, DIM, (int(fp[0]*self.w), int(fp[1]*self.h)), self.config.dot_radius)
        pg.draw.circle(surf, colour, (tx, ty), self.config.dot_radius)

    def _progress(self, surf, tx, ty, colour, n, total):
        pg.draw.circle(surf, colour, (tx, ty), self.config.dot_radius)
        bar_w = int((n / total) * 200)
        pg.draw.rect(surf, colour, (tx - 100, ty + 30, bar_w, 14))

    # ── Calibration ──────────────────────────────────────────────────────

    def _run_calibration(self):
        import cv2
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge

        calib_pts   = self._CALIB_GRID[list(self.config.calib_order)]
        feats, tgts = [], []

        self._draw(lambda s: None)   # blank — no intro text
        self._wait_for_button()

        for pt_idx, (tx_n, ty_n) in enumerate(calib_pts):
            tx, ty   = int(tx_n * self.w), int(ty_n * self.h)
            future   = calib_pts[pt_idx:]

            self._draw(lambda s, tx=tx, ty=ty, f=future: self._dot(s, tx, ty, GREEN, f))
            self._wait_for_button()

            samples = []
            while len(samples) < self.config.samples_per_point:
                frame   = self._get_frame()
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                n       = len(samples)
                self._draw(lambda s, tx=tx, ty=ty, n=n:
                    self._progress(s, tx, ty, CYAN, n, self.config.samples_per_point))
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    samples.append(self._extract_features(lm, w, h))

            feats.append(np.mean(samples, axis=0))
            tgts.append([tx, ty])

        self.poly    = PolynomialFeatures(degree=self.config.poly_degree, include_bias=True)
        X            = self.poly.fit_transform(np.array(feats))
        T            = np.array(tgts, dtype=np.float32)
        self.model_x = Ridge(alpha=self.config.ridge_alpha)
        self.model_y = Ridge(alpha=self.config.ridge_alpha)
        self.model_x.fit(X, T[:, 0])
        self.model_y.fit(X, T[:, 1])

        self._save_calibration(feats, tgts)

    def _save_calibration(self, feats, tgts):
        try:
            from hero_core.database.models.sensors import CalibrationEyeTracking
            # Store the record now; validation columns will be updated after _run_validation
            existing = self.db_session.query(CalibrationEyeTracking).filter(
                CalibrationEyeTracking.session_id == self.session_id
            ).first()
            if existing:
                self.db_session.delete(existing)

            rec = CalibrationEyeTracking(
                session_id     = self.session_id,
                timestamp      = datetime.now(timezone.utc),
                coeff_x        = self.model_x.coef_.tolist(),
                coeff_y        = self.model_y.coef_.tolist(),
                intercept_x    = float(self.model_x.intercept_),
                intercept_y    = float(self.model_y.intercept_),
                poly_degree    = self.config.poly_degree,
                calib_features = [f.tolist() for f in feats],
                calib_targets  = tgts,
            )
            self.db_session.add(rec)
            self.db_session.commit()
            self._calib_record = rec  # keep reference so validation can update it
        except Exception as e:
            logger.warning(f"Could not save calibration: {e}")
            self._calib_record = None

    # ── Validation ───────────────────────────────────────────────────────

    def _run_validation(self):
        import cv2

        # ── Instruction screen ───────────────────────────────────────────
        font_large = pg.font.SysFont("couriernew", 24, bold=True)
        font_med   = pg.font.SysFont("couriernew", 18)

        def _draw_instructions(s):
            s.blit(font_large.render("Validation", True, CYAN),
                   (self.w // 2 - 80, self.h // 2 - 60))
            s.blit(font_med.render("Look at each white dot and press the button", True, WHITE),
                   (self.w // 2 - 230, self.h // 2 - 20))
            s.blit(font_med.render("Press button to begin", True, YELLOW),
                   (self.w // 2 - 115, self.h // 2 + 20))

        self._draw(_draw_instructions)
        self._wait_for_button()

        tgts, preds = [], []

        for tx_n, ty_n in self._VALID_GRID:
            tx, ty = int(tx_n * self.w), int(ty_n * self.h)

            self._draw(lambda s, tx=tx, ty=ty: self._dot(s, tx, ty, WHITE))
            self._wait_for_button()

            raw = []
            while len(raw) < self.config.validation_samples:
                frame   = self._get_frame()
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                n       = len(raw)
                self._draw(lambda s, tx=tx, ty=ty, n=n:
                    self._progress(s, tx, ty, YELLOW, n, self.config.validation_samples))
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    gx, gy = self._predict(self._extract_features(lm, w, h))
                    raw.append([gx, gy])

            tgts.append([tx, ty])
            preds.append(np.mean(raw, axis=0).tolist())

        # Compute metrics
        tgts  = np.array(tgts,  dtype=np.float32)
        preds = np.array(preds, dtype=np.float32)
        err   = np.linalg.norm(preds - tgts, axis=1)
        mm_px = self.config.screen_width_mm / self.w
        deg   = np.degrees(np.arctan((err * mm_px) / self.config.screen_distance_mm))
        mean  = float(np.mean(deg))
        std   = float(np.std(deg))
        rating = "GOOD" if std < 1.0 else "ACCEPTABLE" if std < 2.0 else "POOR"
        logger.info(f"Validation: {std:.2f}deg std ({rating})")

        self._save_validation(mean, std, rating)
        self._draw_validation_results(tgts, preds, deg, mean, std, rating)
        self._wait_for_button()

        return rating

    def _draw_validation_results(self, tgts, preds, deg, mean, std, rating):
        """Show a full-screen scatter plot of target vs predicted gaze points."""
        GOOD_COL       = (0,   210, 80)
        ACCEPTABLE_COL = (220, 180, 0)
        POOR_COL       = (220, 60,  60)
        rating_col = GOOD_COL if rating == "GOOD" else ACCEPTABLE_COL if rating == "ACCEPTABLE" else POOR_COL

        font_large = pg.font.SysFont("couriernew", 24, bold=True)
        font_med   = pg.font.SysFont("couriernew", 17)
        font_small = pg.font.SysFont("couriernew", 14)

        def draw(s):
            # ── Per-point: error line + target + predicted dot ───────────
            for i, (tgt, pred) in enumerate(zip(tgts, preds)):
                tx, ty = int(tgt[0]),  int(tgt[1])
                px, py = int(pred[0]), int(pred[1])
                point_deg = float(deg[i])

                # Colour each point by its individual error
                col = GOOD_COL if point_deg < 1.0 else ACCEPTABLE_COL if point_deg < 2.0 else POOR_COL

                # Line from target to prediction
                pg.draw.line(s, col, (tx, ty), (px, py), 1)
                # Target: white hollow circle
                pg.draw.circle(s, WHITE, (tx, ty), 8, 2)
                # Predicted: filled coloured dot
                pg.draw.circle(s, col, (px, py), 5)
                # Per-point error label
                lbl = font_small.render(f"{point_deg:.1f}d", True, col)
                s.blit(lbl, (px + 7, py - 7))

            # ── Legend ───────────────────────────────────────────────────
            s.blit(font_small.render("O  target",    True, WHITE),  (8, self.h - 52))
            s.blit(font_small.render("●  predicted", True, CYAN),   (8, self.h - 36))
            s.blit(font_small.render("—  error",     True, GREY),   (8, self.h - 20))

            # ── Stats panel (top-left) ────────────────────────────────────
            s.blit(font_large.render(f"Validation: {rating}", True, rating_col), (10, 10))
            s.blit(font_med.render(f"Mean error : {mean:.2f} deg", True, WHITE),  (10, 44))
            s.blit(font_med.render(f"Std dev    : {std:.2f} deg",  True, WHITE),  (10, 66))
            s.blit(font_med.render(f"Points     : {len(tgts)}",    True, WHITE),  (10, 88))

            # ── Thresholds reminder ───────────────────────────────────────
            s.blit(font_small.render("GOOD < 1.0d   ACCEPTABLE < 2.0d   POOR >= 2.0d",
                                     True, GREY), (10, 114))

            # ── Press to continue ─────────────────────────────────────────
            cont = font_med.render("Press button to continue", True, YELLOW)
            s.blit(cont, (self.w // 2 - cont.get_width() // 2, self.h - 72))

        self._draw(draw)

    def _save_validation(self, mean_deg, std_deg, rating):
        try:
            if self._calib_record is not None:
                self._calib_record.validation_mean_deg = mean_deg
                self._calib_record.validation_std_deg  = std_deg
                self._calib_record.validation_rating   = rating
                self.db_session.commit()
                logger.info(f"Validation saved to calibration record: {mean_deg:.2f}deg ({rating})")
        except Exception as e:
            logger.warning(f"Could not save validation: {e}")

    # ── Collection ───────────────────────────────────────────────────────

    def _collection_loop(self):
        import cv2
        from hero_core.database.models.sensors import SensorEyeTracking

        batch  = []
        alpha  = self.config.smoothing_alpha
        gx, gy = self.w // 2, self.h // 2

        while not self._stop_event.is_set():
            try:
                frame   = self._get_frame()
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    rx, ry = self._predict(self._extract_features(lm, w, h))
                    gx = int(alpha * rx + (1 - alpha) * gx)
                    gy = int(alpha * ry + (1 - alpha) * gy)
                    with self._gaze_lock:
                        self._gaze_x, self._gaze_y = gx, gy
                    batch.append(SensorEyeTracking(
                        time       = datetime.now(timezone.utc),
                        session_id = self.session_id,
                        gaze_x     = float(gx),
                        gaze_y     = float(gy),
                        is_valid   = True,
                    ))
                    self.sample_count += 1

                if len(batch) >= self.config.batch_commit_size:
                    self.db_session.bulk_save_objects(batch)
                    self.db_session.commit()
                    batch.clear()

            except Exception as e:
                logger.warning(f"Gaze collection: {e}")
                time.sleep(0.1)

        if batch:
            try:
                self.db_session.bulk_save_objects(batch)
                self.db_session.commit()
            except Exception:
                pass
