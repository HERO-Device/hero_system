#!/usr/bin/env python3 
""" 
Eye gaze estimation using 9-point calibration + 2nd-order polynomial regression. 
Camera: OAK-D via DepthAI. Landmark detection: MediaPipe FaceMesh (with iris refinement). 
After calibration, a fullscreen OpenCV window shows a gaze cursor in real time. 
Press SPACE to capture each calibration sample, Q to quit at any time. 
""" 
 
import cv2 
import numpy as np 
import depthai as dai 
import mediapipe as mp 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge 
import subprocess, re 
import csv 
from datetime import datetime 
 
# ────────────────────────────────────────────── 
# Iris landmark indices (MediaPipe FaceMesh, refine_landmarks=True) 
# 468-472 = left iris  (468 = centre) 
# 473-477 = right iris (473 = centre) 
LEFT_IRIS  = 468 
RIGHT_IRIS = 473 
 
# Eye corner landmarks for normalising iris position within the eye 
# Left eye: outer=33, inner=133 | Right eye: inner=362, outer=263 
LEFT_EYE_INNER  = 133 
LEFT_EYE_OUTER  = 33 
RIGHT_EYE_INNER = 362 
RIGHT_EYE_OUTER = 263 
 
# ────────────────────────────────────────────── 
# Build a 9-point calibration grid (normalised 0-1 screen coords) 
CALIB_POINTS = np.array([ 
    [0.05, 0.05], [0.5, 0.05], [0.95, 0.05], 
    [0.05, 0.5], [0.5, 0.5], [0.95, 0.5], 
    [0.05, 0.95], [0.5, 0.95], [0.95, 0.95], 
], dtype=np.float32) 
CALIB_ORDER = [4, 0, 8, 2, 6, 1, 7, 3, 5]  # centre, corners, edges 
CALIB_POINTS = CALIB_POINTS[CALIB_ORDER] 
 
SAMPLES_PER_POINT = 60   # frames averaged per calibration dot 
CURSOR_RADIUS     = 20 
DOT_RADIUS        = 18 
SMOOTHING_ALPHA   = 0.1  # EMA smoothing factor for gaze output 
# Camera orientation correction: 0 = flip vertical, 1 = horizontal, -1 = both 
CAMERA_FLIP_CODE = 0 
 
def correct_frame(frame): 
    """Apply camera orientation correction to every raw frame.""" 
    return cv2.flip(frame, CAMERA_FLIP_CODE) 
 
 
def run_validation(q, face_mesh, poly, model_x, model_y, screen_w, screen_h): 
    """ 
    9-point validation pass using points interleaved between calibration targets. 
    Collects 40 frames per point, computes Euclidean + angular error, saves to CSV. 
    Press SPACE to capture each point, Q to abort. 
    """ 
    VALIDATION_POINTS = np.array([ 
    [0.25, 0.25], [0.75, 0.25], 
    [0.50, 0.50], 
    [0.25, 0.75], [0.75, 0.75], 
    [0.15, 0.50], [0.85, 0.50], 
    [0.50, 0.05],  # extreme top centre 
    [0.50, 0.95],  # extreme bottom centre 
    ], dtype=np.float32) 
 
    SCREEN_DISTANCE_MM = 430.0  # adjust to your eye-to-screen distance 
    SCREEN_WIDTH_MM    = 222.0  # adjust to your monitor's physical width 
 
    targets_px     = [] 
    predictions_px = [] 
 
    for pt_idx, (tx_n, ty_n) in enumerate(VALIDATION_POINTS): 
        tx = int(tx_n * screen_w) 
        ty = int(ty_n * screen_h) 
 
        # Wait for fixation + SPACE 
        while True: 
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
            for fp in VALIDATION_POINTS[pt_idx:]: 
                cv2.circle(canvas, (int(fp[0]*screen_w), int(fp[1]*screen_h)), DOT_RADIUS, (50, 50, 50), -1) 
            cv2.circle(canvas, (tx, ty), DOT_RADIUS, (255, 255, 255), -1) 
            cv2.putText(canvas, f"Validation {pt_idx+1}/9 — fixate, press SPACE", 
                        (screen_w//2 - 300, screen_h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1, cv2.LINE_AA) 
            cv2.imshow("gaze", canvas) 
            key = cv2.waitKey(1) 
            if key == ord('q'): 
                return 
            if key == ord(' '): 
                break 
 
        # Collect 40 predictions and average 
        raw_preds = [] 
        while len(raw_preds) < 40: 
            f       = q.get() 
            frame   = correct_frame(f.getCvFrame()) 
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            results = face_mesh.process(rgb) 
 
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
            cv2.circle(canvas, (tx, ty), DOT_RADIUS, (200, 200, 0), -1) 
            progress = int((len(raw_preds) / 40) * 200) 
            cv2.rectangle(canvas, (tx - 100, ty + 30), (tx - 100 + progress, ty + 45), (200, 200, 0), -1) 
            cv2.imshow("gaze", canvas) 
            cv2.waitKey(1) 
 
            if results.multi_face_landmarks: 
                lm   = results.multi_face_landmarks[0].landmark 
                h, w = frame.shape[:2] 
                fp   = poly.transform(extract_features(lm, w, h).reshape(1, -1)) 
                raw_preds.append([float(model_x.predict(fp)[0]), float(model_y.predict(fp)[0])]) 
 
        targets_px.append([tx, ty]) 
        predictions_px.append(np.mean(raw_preds, axis=0)) 
 
    targets_px     = np.array(targets_px,     dtype=np.float32) 
    predictions_px = np.array(predictions_px, dtype=np.float32) 
 
    # Euclidean pixel error per point 
    errors_px  = np.linalg.norm(predictions_px - targets_px, axis=1) 
 
    # Convert to visual degrees using screen geometry 
    mm_per_px  = SCREEN_WIDTH_MM / screen_w 
    errors_deg = np.degrees(np.arctan((errors_px * mm_per_px) / SCREEN_DISTANCE_MM)) 
 
    # ── Results overlay ─────────────────────────────────────────────────────── 
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
    for i, (tgt, pred, err_deg) in enumerate(zip(targets_px, predictions_px, errors_deg)): 
        tx, ty = int(tgt[0]),  int(tgt[1]) 
        px, py = int(pred[0]), int(pred[1]) 
        t      = min(err_deg / 5.0, 1.0) 
        col    = (0, int(255*(1-t)), int(255*t))  # green → red 
        cv2.line(canvas, (tx, ty), (px, py), col, 1, cv2.LINE_AA) 
        cv2.circle(canvas, (tx, ty), DOT_RADIUS, (255, 255, 255), 2, cv2.LINE_AA) 
        cv2.circle(canvas, (px, py), 8, col, -1, cv2.LINE_AA) 
        cv2.putText(canvas, f"{err_deg:.1f}°", (tx + 12, ty - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA) 
 
    mean_deg = float(np.mean(errors_deg)) 
    rating   = "GOOD" if mean_deg < 1.0 else "ACCEPTABLE" if mean_deg < 2.0 else "POOR - recalibrate" 
    cv2.putText(canvas, f"Mean: {mean_deg:.2f} deg | Std: {np.std(errors_deg):.2f} | {rating}", 
                (40, screen_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA) 
    cv2.putText(canvas, "Q to exit", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1) 
 
    while True: 
        cv2.imshow("gaze", canvas) 
        if cv2.waitKey(30) == ord('q'): 
            break 
 
    # ── Write CSV ───────────────────────────────────────────────────────────── 
    # filename = f"gaze_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
    # with open(filename, "w", newline="") as f: 
        # writer = csv.writer(f) 
        # writer.writerow(["point", "target_x", "target_y", "pred_x", "pred_y", 
                         # "error_px", "error_deg"]) 
        # for i, (tgt, pred, epx, edeg) in enumerate( 
                # zip(targets_px, predictions_px, errors_px, errors_deg)): 
            # writer.writerow([i+1, round(tgt[0],1), round(tgt[1],1), 
                             # round(pred[0],1), round(pred[1],1), 
                             # round(epx,2), round(edeg,3)]) 
        # writer.writerow([]) 
        # writer.writerow(["mean_error_deg", round(mean_deg, 3)]) 
        # writer.writerow(["std_error_deg",  round(float(np.std(errors_deg)), 3)]) 
        # writer.writerow(["max_error_deg",  round(float(np.max(errors_deg)), 3)]) 
        # writer.writerow(["rating",         rating]) 
    # print(f"Validation results saved to {filename}") 
 
 
def extract_features(landmarks, img_w, img_h): 
    """ 
    Returns a 4-element feature vector: 
      [norm_iris_x_left, norm_iris_y_left, norm_iris_x_right, norm_iris_y_right] 
    Iris position is normalised relative to the eye-corner width so it is 
    invariant to face scale and minor head translations. 
    """ 
    def norm_iris(iris_idx, inner_idx, outer_idx): 
        iris  = landmarks[iris_idx] 
        inner = landmarks[inner_idx] 
        outer = landmarks[outer_idx] 
        # pixel coords 
        ix, iy = iris.x * img_w,  iris.y * img_h 
        inx    = inner.x * img_w 
        outx   = outer.x * img_w 
        eye_w  = abs(inx - outx) + 1e-6 
        eye_cy = (inner.y + outer.y) / 2 * img_h 
        return (ix - outx) / eye_w, ((iy - eye_cy)*5) / eye_w ## adding a weight to boost the y values 
 
    lx, ly = norm_iris(LEFT_IRIS,  LEFT_EYE_INNER,  LEFT_EYE_OUTER) 
    rx, ry = norm_iris(RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER) 
    return np.array([lx, ly, rx, ry], dtype=np.float32) 
 
 
def build_dai_pipeline(): 
    """DepthAI pipeline: 1080p RGB colour camera streamed to host.""" 
    pipeline = dai.Pipeline() 
    cam = pipeline.create(dai.node.ColorCamera) 
    xout = pipeline.create(dai.node.XLinkOut) 
    xout.setStreamName("rgb") 
    cam.setInterleaved(False) 
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) 
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) 
    cam.preview.link(xout.input) 
    return pipeline 
 
def get_display_resolution(): 
    """Query actual display resolution via xrandr (Linux) or fallback.""" 
    try: 
        out = subprocess.check_output(["xrandr"]).decode() 
        match = re.search(r"current (\d+) x (\d+)", out) 
        if match == (1024,600): 
            return int(match.group(1)), int(match.group(2)) 
    except Exception: 
        pass 
    # Fallback: use a known resolution or prompt the user 
    return 1024, 600 
 
def run(): 
    # ── MediaPipe setup ────────────────────────────────────────────────────── 
    mp_fm = mp.solutions.face_mesh 
    face_mesh = mp_fm.FaceMesh( 
        max_num_faces=1, 
        refine_landmarks=True,   # enables iris landmarks 468-477 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
    ) 
 
    poly    = PolynomialFeatures(degree=2, include_bias=True) 
    model_x = Ridge(alpha=0.01) 
    model_y = Ridge(alpha=0.01) 
 
    # ── DepthAI pipeline ───────────────────────────────────────────────────── 
    pipeline = build_dai_pipeline() 
    with dai.Device(pipeline) as device: 
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) 
 
        screen_w, screen_h = get_display_resolution()
        print(f"Using resolution from getdisplayresolution(): {screen_w} x {screen_h}")
 
 
        # Create window, show a black frame first, THEN set fullscreen 
        cv2.namedWindow("gaze", cv2.WINDOW_NORMAL) 
        blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
        cv2.imshow("gaze", blank) 
        cv2.waitKey(1)  # allow the window to render before going fullscreen 
        cv2.setWindowProperty("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
 
        # ── Calibration loop ────────────────────────────────────────────────── 
        calib_features = []  # list of (4,) feature vectors 
        calib_targets  = []  # list of (2,) pixel-space targets 
 
        cv2.namedWindow("gaze", cv2.WND_PROP_FULLSCREEN) 
        cv2.setWindowProperty("gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
 
        for pt_idx, (tx_n, ty_n) in enumerate(CALIB_POINTS): 
            tx = int(tx_n * screen_w) 
            ty = int(ty_n * screen_h) 
 
            # -- Wait for user to fixate and press SPACE -- 
            while True: 
                f = q.get() 
                frame = correct_frame(f.getCvFrame()) 
 
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
                # Draw remaining dots dimly so user knows what is coming 
                for future_pt in CALIB_POINTS[pt_idx:]: 
                    fx, fy = int(future_pt[0]*screen_w), int(future_pt[1]*screen_h) 
                    cv2.circle(canvas, (fx, fy), DOT_RADIUS, (60, 60, 60), -1) 
                # Highlight active dot 
                cv2.circle(canvas, (tx, ty), DOT_RADIUS, (0, 255, 0), -1) 
                label = f"Point {pt_idx+1}/9 — look at dot and press SPACE" 
                cv2.putText(canvas, label, (screen_w//2 - 300, screen_h - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2) 
                cv2.imshow("gaze", canvas) 
 
                key = cv2.waitKey(1) 
                if key == ord('q'): 
                    face_mesh.close() 
                    return 
                if key == ord(' '): 
 
                    break 
 
            # -- Collect SAMPLES_PER_POINT frames and average features -- 
            samples = [] 
            while len(samples) < SAMPLES_PER_POINT: 
                f = q.get() 
                frame = correct_frame(f.getCvFrame()) 
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                results = face_mesh.process(rgb) 
 
                # Show a simple fill-bar so the user knows capture is running 
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
                cv2.circle(canvas, (tx, ty), DOT_RADIUS, (0, 200, 255), -1) 
                progress = int((len(samples) / SAMPLES_PER_POINT) * 200) 
                cv2.rectangle(canvas, 
                              (tx - 100, ty + 30), 
                              (tx - 100 + progress, ty + 45), 
                              (0, 200, 255), -1) 
                cv2.imshow("gaze", canvas) 
                cv2.waitKey(1) 
 
                if results.multi_face_landmarks: 
                    lm   = results.multi_face_landmarks[0].landmark 
                    h, w = frame.shape[:2] 
                    feat = extract_features(lm, w, h) 
                    samples.append(feat) 
 
            calib_features.append(np.mean(samples, axis=0)) 
            calib_targets.append([tx, ty]) 
 
        # ── Fit 2nd-order polynomial regression (separate X and Y models) ──── 
        # PolynomialFeatures on 4 inputs → 15 terms at degree 2 
        X = poly.fit_transform(np.array(calib_features)) 
        T = np.array(calib_targets, dtype=np.float32) 
        model_x.fit(X, T[:, 0]) 
        model_y.fit(X, T[:, 1]) 
 
        run_validation(q, face_mesh, poly, model_x, model_y, screen_w, screen_h) 
 
        # ── Live gaze display loop ──────────────────────────────────────────── 
        gaze_x, gaze_y = screen_w // 2, screen_h // 2  # EMA-smoothed cursor 
 
        while True: 
            f       = q.get() 
            frame   = correct_frame(f.getCvFrame()) 
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            results = face_mesh.process(rgb) 
 
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) 
 
            if results.multi_face_landmarks: 
                lm        = results.multi_face_landmarks[0].landmark 
                h, w      = frame.shape[:2] 
                feat_poly = poly.transform(extract_features(lm, w, h).reshape(1, -1)) 
 
                raw_x = float(model_x.predict(feat_poly)[0]) 
                raw_y = float(model_y.predict(feat_poly)[0]) 
 
                # Clamp to valid screen area 
                raw_x = np.clip(raw_x, 0, screen_w - 1) 
                raw_y = np.clip(raw_y, 0, screen_h - 1) 
 
                # Exponential moving average to reduce jitter 
                gaze_x = int(SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * gaze_x) 
                gaze_y = int(SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * gaze_y) 
 
            # Crosshair + dot gaze cursor 
            cv2.circle(canvas, (gaze_x, gaze_y), CURSOR_RADIUS, (0, 255, 0), 2) 
            cv2.line(canvas, 
                     (gaze_x - CURSOR_RADIUS, gaze_y), 
                     (gaze_x + CURSOR_RADIUS, gaze_y), (0, 255, 0), 1) 
            cv2.line(canvas, 
                     (gaze_x, gaze_y - CURSOR_RADIUS), 
                     (gaze_x, gaze_y + CURSOR_RADIUS), (0, 255, 0), 1) 
            cv2.circle(canvas, (gaze_x, gaze_y), 4, (0, 255, 0), -1) 
            cv2.putText(canvas, "Q to quit", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1) 
 
            cv2.imshow("gaze", canvas) 
            if cv2.waitKey(1) == ord('q'): 
                break 
 
        face_mesh.close() 
        cv2.destroyAllWindows() 
 

if __name__ == "__main__": 
    run() 
