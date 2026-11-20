"""
Air Canvas PRO - Universal Enterprise Edition (v5.0)
Cross-platform architecture engineered for Windows, macOS, and Linux.
Hardware-agnostic execution with automated fallback guarantees.
"""

from __future__ import annotations

import os
import sys
import time
import math
import re
import subprocess
import urllib.request
import threading
from typing import Tuple, List, Optional
import cv2
import numpy as np

# Suppress TensorFlow / LiteRT logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.modules['tensorflow'] = None

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ================= CONFIGURATION =================
CONFIG = {
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "PROCESS_WIDTH": 384,          # Highly optimized inference resolution (16:9)
    "PROCESS_HEIGHT": 216,
    "TARGET_FPS": 30,              # Locked 30 FPS throttle
    "MIN_DETECTION_CONFIDENCE": 0.55,
    "MIN_TRACKING_CONFIDENCE": 0.55,
    "DEFAULT_BRUSH_THICKNESS": 6,
    "DEFAULT_ERASER_RADIUS": 35,
    "PINCH_THRESHOLD": 0.065,      # Normalized distance between thumb and index tip
    "PINCH_STABLE_FRAMES": 2,
    "DRAW_DELAY": 0.08,            # Quick debounce window
    "SHAPE_SNAP_PAUSE_TIME": 0.50, # Pause at end of stroke to trigger auto-snap
    "FILTER_MIN_CUTOFF": 1.5,      # 1Euro filter min cutoff frequency (Hz)
    "FILTER_BETA": 0.02,           # 1Euro filter responsiveness
    "FILTER_D_CUTOFF": 1.0,
    "BEZIER_STEPS": 10,
}

# 12-Color Professional Studio Palette
PALETTE = [
    ("Neon Cyan", (255, 230, 0)),
    ("Neon Green", (50, 255, 50)),
    ("Neon Pink", (220, 20, 255)),
    ("Flame Orange", (0, 140, 255)),
    ("Electric Blue", (255, 100, 0)),
    ("Lemon Yellow", (0, 240, 255)),
    ("Violet Purple", (255, 0, 160)),
    ("Crimson Red", (30, 30, 240)),
    ("Lime Yellow", (50, 255, 180)),
    ("Coral Peach", (160, 180, 255)),
    ("Sky Azure", (250, 190, 80)),
    ("Pure White", (255, 255, 255)),
]

MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe Standard 21 Hand Connections
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                               # Palm base
)


def ensure_model_asset(path: str = MODEL_PATH) -> str:
    """Verify presence of MediaPipe model asset; automatically download if absent."""
    if not os.path.exists(path):
        print(f"[MODEL] '{MODEL_FILENAME}' not found. Downloading model asset...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print(f"[MODEL] Download complete: {os.path.getsize(path)} bytes.")
    return path


# ================= UNIVERSAL HARDWARE & CAMERA DETECTION =================
def probe_cameras() -> List[Tuple[int, str]]:
    """
    Cross-platform camera discovery supporting Windows, macOS, and Linux.
    Never fails: guaranteed to return at least index 0.
    """
    cameras: List[Tuple[int, str]] = []

    # 1. Linux V4L2 probe
    if sys.platform.startswith("linux"):
        try:
            out = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True, stderr=subprocess.DEVNULL)
            blocks = out.split('\n\n')
            for block in blocks:
                lines = [l.strip() for l in block.split('\n') if l.strip()]
                if lines:
                    name = lines[0].split('(')[0].strip()
                    devs = re.findall(r'/dev/video(\d+)', block)
                    if devs:
                        cameras.append((int(devs[0]), name))
        except Exception:
            pass

    # 2. Windows / macOS / Generic index probing fallback
    if not cameras:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        for idx in range(4):
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cameras.append((idx, f"Camera {idx}"))
                    cap.release()
            except Exception:
                pass

    if not cameras:
        cameras.append((0, "Default System Camera"))

    return cameras


def probe_hardware_options() -> List[Tuple[str, str, str]]:
    """
    Detects OpenCL capable GPUs (NVIDIA, AMD, Intel Iris/UHD, Apple M-Series)
    with safe CPU fallback guarantees.
    """
    options: List[Tuple[str, str, str]] = []

    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                dev = cv2.ocl.Device.getDefault()
                d_name = dev.name()
                vendor = dev.vendorName()
                options.append(("GPU", f"GPU OpenCL Acceleration [{vendor} {d_name}]", f"GPU ({d_name})"))
    except Exception:
        pass

    options.append(("CPU", "CPU Processing [Optimized Multi-Core Host Pipeline]", "CPU (Host)"))
    return options


def run_startup_wizard() -> Tuple[int, str, str]:
    """
    Graphical on-screen dialog to choose camera and compute device.
    Robust fallback to CLI if headless or display environment fails.
    """
    cams = probe_cameras()
    hw_opts = probe_hardware_options()

    default_cam_idx = 0
    for i, (idx, name) in enumerate(cams):
        if "kreo" in name.lower():
            default_cam_idx = i
            break

    selected_cam_i = default_cam_idx
    selected_hw_i = 0
    launch_confirmed = False

    win_name = "Air Canvas - Hardware & Camera Setup"
    W, H = 680, 500

    try:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            nonlocal selected_cam_i, selected_hw_i, launch_confirmed
            if event == cv2.EVENT_LBUTTONDOWN:
                for i in range(len(cams)):
                    by = 100 + i * 44
                    if 40 <= x <= 640 and by <= y <= by + 36:
                        selected_cam_i = i
                        return
                for i in range(len(hw_opts)):
                    by = 260 + i * 44
                    if 40 <= x <= 640 and by <= y <= by + 36:
                        selected_hw_i = i
                        return
                if 200 <= x <= 480 and 420 <= y <= 468:
                    launch_confirmed = True

        cv2.setMouseCallback(win_name, on_mouse)

        while not launch_confirmed:
            canvas = np.full((H, W, 3), 20, dtype=np.uint8)

            # Header Plate
            cv2.rectangle(canvas, (0, 0), (W, 65), (28, 28, 28), -1)
            cv2.line(canvas, (0, 65), (W, 65), (55, 55, 55), 1)
            cv2.putText(canvas, "AIR CANVAS PRO", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Select Camera and Compute Device, then click Confirm & Launch", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            # Camera Selection Section
            cv2.putText(canvas, "1. SELECT CAMERA:", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
            for i, (idx, name) in enumerate(cams):
                by = 100 + i * 44
                is_sel = (i == selected_cam_i)
                bg_col = (45, 45, 45) if is_sel else (28, 28, 28)
                border_col = (0, 220, 255) if is_sel else (50, 50, 50)
                cv2.rectangle(canvas, (40, by), (640, by + 36), bg_col, -1)
                cv2.rectangle(canvas, (40, by), (640, by + 36), border_col, 1)

                rad_col = (0, 220, 255) if is_sel else (90, 90, 90)
                cv2.circle(canvas, (62, by + 18), 7, rad_col, -1 if is_sel else 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{name} (Device {idx})", (84, by + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255) if is_sel else (190, 190, 190), 1, cv2.LINE_AA)

            # Hardware Selection Section
            cv2.putText(canvas, "2. SELECT COMPUTE ACCELERATOR:", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
            for i, (_, label, _) in enumerate(hw_opts):
                by = 260 + i * 44
                is_sel = (i == selected_hw_i)
                bg_col = (45, 45, 45) if is_sel else (28, 28, 28)
                border_col = (0, 220, 255) if is_sel else (50, 50, 50)
                cv2.rectangle(canvas, (40, by), (640, by + 36), bg_col, -1)
                cv2.rectangle(canvas, (40, by), (640, by + 36), border_col, 1)

                rad_col = (0, 220, 255) if is_sel else (90, 90, 90)
                cv2.circle(canvas, (62, by + 18), 7, rad_col, -1 if is_sel else 1, cv2.LINE_AA)
                cv2.putText(canvas, label, (84, by + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255) if is_sel else (190, 190, 190), 1, cv2.LINE_AA)

            # Confirm & Launch Button
            cv2.rectangle(canvas, (200, 420), (480, 468), (0, 150, 220), -1)
            cv2.rectangle(canvas, (200, 420), (480, 468), (255, 255, 255), 1)
            cv2.putText(canvas, "CONFIRM & LAUNCH", (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):  # ENTER or SPACE to launch
                launch_confirmed = True
            elif key == 27:  # ESC to abort
                cv2.destroyAllWindows()
                print("[ABORT] Startup cancelled by user.")
                sys.exit(0)

        cv2.destroyWindow(win_name)

    except Exception:
        # Fallback to CLI wizard if GUI window initialization fails on headless/restricted hosts
        print("[SETUP] GUI setup fallback active. Using auto-discovered primary configuration.")
        selected_cam_i = default_cam_idx
        selected_hw_i = 0

    selected_cam_idx, selected_cam_name = cams[selected_cam_i]
    hw_type, hw_label, hw_tag = hw_opts[selected_hw_i]

    # Camera verification and fallback
    test_cap = cv2.VideoCapture(selected_cam_idx)
    if not test_cap.isOpened():
        print(f"[FALLBACK] Camera device {selected_cam_idx} failed. Falling back to default index 0.")
        selected_cam_idx = 0
        selected_cam_name = "Default Fallback Camera"
    else:
        test_cap.release()

    # Hardware verification and fallback
    if hw_type == "GPU":
        try:
            cv2.ocl.setUseOpenCL(True)
            if not cv2.ocl.useOpenCL():
                hw_tag = "CPU (Fallback)"
                cv2.ocl.setUseOpenCL(False)
        except Exception:
            hw_tag = "CPU (Fallback)"
            cv2.ocl.setUseOpenCL(False)
    else:
        cv2.ocl.setUseOpenCL(False)

    print("\n[ACTIVE SYSTEM CONFIGURATION]:")
    print(f"  • Video Ingest : {selected_cam_name} (Device {selected_cam_idx})")
    print(f"  • Acceleration : {hw_tag}")
    return selected_cam_idx, selected_cam_name, hw_tag


# ================= SMOOTHING & INTERPOLATION =================
class OneEuroFilter:
    """True 1€ (One Euro) adaptive low-pass filter."""
    __slots__ = ('min_cutoff', 'beta', 'd_cutoff', 'x_prev', 'dx_prev', 't_prev')

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.015, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: float) -> float:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return x

        dt = max(timestamp - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_filtered
        self.dx_prev = edx
        self.t_prev = timestamp
        return x_filtered

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


# Pre-allocated parameter vector for Bézier curve
_BEZIER_T = np.linspace(0.0, 1.0, CONFIG["BEZIER_STEPS"] + 1, dtype=np.float32)[:, None]
_BEZIER_1_MINUS_T = 1.0 - _BEZIER_T
_BEZIER_W0 = _BEZIER_1_MINUS_T ** 2
_BEZIER_W1 = 2.0 * _BEZIER_1_MINUS_T * _BEZIER_T
_BEZIER_W2 = _BEZIER_T ** 2


def get_bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Pre-computed weight matrix Bézier calculation."""
    curve = _BEZIER_W0 * p0 + _BEZIER_W1 * p1 + _BEZIER_W2 * p2
    return curve.astype(np.int32)


# ================= GEOMETRIC SHAPE RECOGNITION =================
def try_recognize_shape(points: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """Analyzes raw stroke coordinates and snaps to clean geometry."""
    if len(points) < 12:
        return None

    pts_arr = np.array(points, dtype=np.int32)
    start_pt = pts_arr[0]
    end_pt = pts_arr[-1]

    diffs = pts_arr[1:] - pts_arr[:-1]
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    total_length = float(np.sum(seg_lengths))
    if total_length < 25.0:
        return None

    endpoint_dist = float(math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]))
    straightness = endpoint_dist / max(total_length, 1e-5)

    # 1. Straight Line Snap
    if straightness > 0.92:
        return [tuple(start_pt), tuple(end_pt)]

    # Closed-loop checks
    if endpoint_dist / total_length < 0.22:
        peri = cv2.arcLength(pts_arr, True)
        if peri > 40:
            # 2. Circle Check
            (cx, cy), radius = cv2.minEnclosingCircle(pts_arr)
            circle_area = math.pi * (radius ** 2)
            contour_area = cv2.contourArea(pts_arr)
            area_ratio = contour_area / max(circle_area, 1e-5)

            if 0.70 < area_ratio < 1.15:
                theta = np.linspace(0, 2 * math.pi, 36)
                return [(int(cx + radius * math.cos(t)), int(cy + radius * math.sin(t))) for t in theta]

            # 3. Rectangle Check
            approx = cv2.approxPolyDP(pts_arr, 0.045 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(pts_arr)
                return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]

    return None


# ================= OBJECT-BASED VECTOR STROKES & HANDLES =================
class Stroke:
    """Individual digital ink stroke object with bounding box and scaling handle."""
    __slots__ = ('points', 'color', 'thickness', 'min_x', 'min_y', 'max_x', 'max_y', 'selected')

    def __init__(self, color: Tuple[int, int, int], thickness: int):
        self.points: List[Tuple[int, int]] = []
        self.color = color
        self.thickness = thickness
        self.min_x = 100000
        self.min_y = 100000
        self.max_x = -1
        self.max_y = -1
        self.selected = False

    def add_point(self, pt: Tuple[int, int]):
        self.points.append(pt)
        x, y = pt
        if x < self.min_x: self.min_x = x
        if y < self.min_y: self.min_y = y
        if x > self.max_x: self.max_x = x
        if y > self.max_y: self.max_y = y

    def add_points(self, pts: np.ndarray):
        for p in pts:
            self.add_point((int(p[0]), int(p[1])))

    def recalculate_bbox(self):
        if not self.points:
            self.min_x = self.min_y = self.max_x = self.max_y = 0
            return
        pts_arr = np.array(self.points, dtype=np.int32)
        mins = np.min(pts_arr, axis=0)
        maxs = np.max(pts_arr, axis=0)
        self.min_x, self.min_y = int(mins[0]), int(mins[1])
        self.max_x, self.max_y = int(maxs[0]), int(maxs[1])

    def translate(self, dx: int, dy: int, max_w: int, max_h: int):
        if not self.points:
            return
        self.points = [(max(0, min(max_w - 1, x + dx)), max(0, min(max_h - 1, y + dy))) for x, y in self.points]
        self.min_x += dx
        self.min_y += dy
        self.max_x += dx
        self.max_y += dy

    def scale_from_handle(self, new_handle_x: int, new_handle_y: int, max_w: int, max_h: int):
        anchor_x, anchor_y = self.min_x, self.min_y
        orig_w = max(15, self.max_x - anchor_x)
        orig_h = max(15, self.max_y - anchor_y)
        new_w = max(15, new_handle_x - anchor_x)
        new_h = max(15, new_handle_y - anchor_y)

        sx = new_w / orig_w
        sy = new_h / orig_h

        scaled_pts = []
        for x, y in self.points:
            nx = int(anchor_x + (x - anchor_x) * sx)
            ny = int(anchor_y + (y - anchor_y) * sy)
            scaled_pts.append((max(0, min(max_w - 1, nx)), max(0, min(max_h - 1, ny))))

        self.points = scaled_pts
        self.recalculate_bbox()

    def get_scale_handle_pos(self) -> Tuple[int, int]:
        return (self.max_x + 8, self.max_y + 8)

    def hits_scale_handle(self, x: int, y: int, radius: float = 24.0) -> bool:
        if not self.selected:
            return False
        hx, hy = self.get_scale_handle_pos()
        return math.hypot(x - hx, y - hy) <= radius

    def hits_point(self, x: int, y: int, threshold: float = 35.0) -> bool:
        pad = int(threshold + self.thickness)
        if (x < self.min_x - pad or x > self.max_x + pad or
            y < self.min_y - pad or y > self.max_y + pad):
            return False

        if not self.points:
            return False

        pts_arr = np.array(self.points, dtype=np.float32)
        dists = np.hypot(pts_arr[:, 0] - x, pts_arr[:, 1] - y)
        return float(np.min(dists)) <= threshold

    def erase_at(self, x: int, y: int, radius: float) -> List[Stroke]:
        if not self.hits_point(x, y, radius):
            return [self]

        new_strokes: List[Stroke] = []
        current: Optional[Stroke] = None

        for pt in self.points:
            d = math.hypot(pt[0] - x, pt[1] - y)
            if d > radius:
                if current is None:
                    current = Stroke(self.color, self.thickness)
                current.add_point(pt)
            else:
                if current is not None and len(current.points) > 1:
                    new_strokes.append(current)
                current = None

        if current is not None and len(current.points) > 1:
            new_strokes.append(current)

        return new_strokes


# ================= CROSS-PLATFORM CAMERA THREADING =================
class WebcamStream:
    """Universal cross-platform camera capture worker."""
    def __init__(self, src: int, width: int = CONFIG["CAMERA_WIDTH"], height: int = CONFIG["CAMERA_HEIGHT"]):
        # Backend selection by platform
        if sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2
        elif sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY

        self.stream = cv2.VideoCapture(src, backend)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = self.stream.get(cv2.CAP_PROP_FPS) or 30.0

    def start(self) -> WebcamStream:
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(0.005)
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        time.sleep(0.05)
        if self.stream.isOpened():
            self.stream.release()


# ================= MULTI-HAND DISAMBIGUATION & GESTURE ENGINE =================
class HandTracker:
    """Selects dominant foreground hand and tracks its spatial continuity."""
    __slots__ = ('locked_wrist_x', 'locked_wrist_y')

    def __init__(self):
        self.locked_wrist_x: Optional[float] = None
        self.locked_wrist_y: Optional[float] = None

    def select_dominant_hand(self, hand_landmarks_list: List[any]) -> any:
        if len(hand_landmarks_list) == 1:
            lm = hand_landmarks_list[0]
            self.locked_wrist_x, self.locked_wrist_y = lm[0].x, lm[0].y
            return lm

        best_score = -1.0
        best_lm = hand_landmarks_list[0]

        for lm in hand_landmarks_list:
            w_x, w_y = lm[0].x, lm[0].y
            m_x, m_y = lm[9].x, lm[9].y
            palm_size = math.hypot(m_x - w_x, m_y - w_y)

            continuity_bonus = 0.0
            if self.locked_wrist_x is not None:
                dist_prev = math.hypot(w_x - self.locked_wrist_x, w_y - self.locked_wrist_y)
                continuity_bonus = max(0.0, 0.25 - dist_prev) * 2.0

            total_score = palm_size + continuity_bonus
            if total_score > best_score:
                best_score = total_score
                best_lm = lm

        self.locked_wrist_x, self.locked_wrist_y = best_lm[0].x, best_lm[0].y
        return best_lm

    def reset(self):
        self.locked_wrist_x = None
        self.locked_wrist_y = None


class GestureEngine:
    """Orientation-invariant gesture classifier using palm coordinate basis."""
    @staticmethod
    def classify(landmarks) -> Tuple[str, dict]:
        w_x, w_y = landmarks[0].x, landmarks[0].y
        m_x, m_y = landmarks[9].x, landmarks[9].y

        v_long_x = m_x - w_x
        v_long_y = m_y - w_y
        palm_len = math.hypot(v_long_x, v_long_y)

        if palm_len < 1e-4:
            v_long_x, v_long_y = 0.0, -1.0
            palm_len = 0.2
        else:
            v_long_x /= palm_len
            v_long_y /= palm_len

        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            mcp = landmarks[mcp_idx]
            pip = landmarks[pip_idx]
            tip = landmarks[tip_idx]

            proj_tip = (tip.x - mcp.x) * v_long_x + (tip.y - mcp.y) * v_long_y
            proj_pip = (pip.x - mcp.x) * v_long_x + (pip.y - mcp.y) * v_long_y
            radial_tip = math.hypot(tip.x - w_x, tip.y - w_y)
            radial_pip = math.hypot(pip.x - w_x, pip.y - w_y)

            return (proj_tip > proj_pip * 1.15) and (radial_tip > radial_pip * 1.10)

        index_up = is_finger_extended(8, 6, 5)
        middle_up = is_finger_extended(12, 10, 9)
        ring_up = is_finger_extended(16, 14, 13)
        pinky_up = is_finger_extended(20, 18, 17)

        t_tip = landmarks[4]
        i_tip = landmarks[8]
        raw_pinch = math.hypot(t_tip.x - i_tip.x, t_tip.y - i_tip.y)
        pinch_ratio = raw_pinch / max(palm_len, 0.05)

        total_extended = index_up + middle_up + ring_up + pinky_up

        if total_extended == 0:
            mode = "ERASE"
        elif pinch_ratio < 0.35 or raw_pinch < CONFIG["PINCH_THRESHOLD"]:
            mode = "MOVE"
        elif index_up and not middle_up and not ring_up and not pinky_up:
            mode = "DRAW"
        elif index_up and middle_up and not ring_up and not pinky_up:
            mode = "HOVER_SELECTION"
        else:
            mode = "IDLE"

        return mode, {
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "pinch_ratio": pinch_ratio,
            "raw_pinch": raw_pinch,
        }


# ================= BARE-METAL C++ ZERO-ALLOCATION RENDERER =================
def render_canvas_strokes(strokes: List[Stroke], layer_bgr: np.ndarray, layer_mask: np.ndarray):
    """Renders vector strokes directly into pre-allocated static buffers."""
    layer_bgr[:] = 0
    layer_mask[:] = 0

    for stroke in strokes:
        n_pts = len(stroke.points)
        if n_pts < 2:
            if n_pts == 1:
                pt = stroke.points[0]
                cv2.circle(layer_bgr, pt, stroke.thickness // 2, stroke.color, -1, cv2.LINE_AA)
                cv2.circle(layer_mask, pt, stroke.thickness // 2, 255, -1, cv2.LINE_AA)
            continue

        pts_arr = np.array(stroke.points, dtype=np.int32).reshape((-1, 1, 2))
        col = stroke.color
        th = stroke.thickness

        cv2.polylines(layer_bgr, [pts_arr], False, col, th, cv2.LINE_AA)
        cv2.polylines(layer_mask, [pts_arr], False, 255, th, cv2.LINE_AA)

        cv2.circle(layer_bgr, stroke.points[0], th // 2, col, -1, cv2.LINE_AA)
        cv2.circle(layer_mask, stroke.points[0], th // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(layer_bgr, stroke.points[-1], th // 2, col, -1, cv2.LINE_AA)
        cv2.circle(layer_mask, stroke.points[-1], th // 2, 255, -1, cv2.LINE_AA)

        # Highlight if selected: Draw bounding box + corner scaling handle
        if stroke.selected:
            bx1, by1 = stroke.min_x - 6, stroke.min_y - 6
            bx2, by2 = stroke.max_x + 6, stroke.max_y + 6
            cv2.rectangle(layer_bgr, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(layer_mask, (bx1, by1), (bx2, by2), 255, 1, cv2.LINE_AA)

            hx, hy = stroke.get_scale_handle_pos()
            cv2.circle(layer_bgr, (hx, hy), 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(layer_bgr, (hx, hy), 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(layer_mask, (hx, hy), 8, 255, -1, cv2.LINE_AA)


def draw_cyberpunk_skeleton(frame: np.ndarray, coords: List[Tuple[int, int]]):
    """Render 21 hand landmarks with illuminated connective bones and nodes."""
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        pt1 = coords[p1_idx]
        pt2 = coords[p2_idx]
        cv2.line(frame, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    for idx, (x, y) in enumerate(coords):
        if idx in (4, 8, 12, 16, 20):  # Fingertips
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)
        elif idx == 0:  # Wrist
            cv2.circle(frame, (x, y), 7, (255, 120, 0), -1, cv2.LINE_AA)
        else:  # Intermediate knuckles
            cv2.circle(frame, (x, y), 4, (0, 255, 150), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, mode: str, color_name: str, color_bgr: tuple,
             brush_size: int, eraser_radius: int, fps: float, stroke_count: int,
             action_status: str, total_detected_hands: int, hw_tag: str):
    """Minimal, professional HUD overlay displaying telemetry and tool status."""
    h, w, _ = frame.shape

    # Status plate top-left
    cv2.rectangle(frame, (16, 16), (460, 116), (18, 18, 18), -1)
    cv2.rectangle(frame, (16, 16), (460, 116), (60, 60, 60), 1)

    mode_colors = {
        "DRAW": (0, 255, 0),
        "ERASE": (0, 0, 255),
        "MOVE": (255, 200, 0),
        "SCALE": (0, 255, 255),
        "HOVER_SELECTION": (255, 255, 0),
        "IDLE": (180, 180, 180),
    }
    badge_col = mode_colors.get(mode, (200, 200, 200))
    cv2.circle(frame, (36, 42), 6, badge_col, -1)
    status_label = f"MODE: {mode}"
    if action_status:
        status_label += f" ({action_status})"
    cv2.putText(frame, status_label, (52, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Active Color Swatch & Tool Info
    cv2.circle(frame, (36, 72), 8, color_bgr, -1)
    cv2.circle(frame, (36, 72), 9, (255, 255, 255), 1)
    tool_info = f"{color_name} | Size: {brush_size}px" if mode != "ERASE" else f"Eraser: {eraser_radius}px"
    cv2.putText(frame, tool_info, (54, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # Hardware & Performance Telemetry
    multi_info = f"Hands: {total_detected_hands} (Locked)" if total_detected_hands > 1 else "Hands: 1"
    cv2.putText(frame, f"FPS: {int(fps)} | {hw_tag} | {multi_info}", (36, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # Clean bottom controls guide
    controls_text = "[E/Q] Color | [W/S] Brush | [C] Clear | [Z] Undo | [X] Save | [ESC] Exit"
    cv2.putText(frame, controls_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)


# ================= MAIN ENGINE =================
def main():
    # 1. Universal Hardware & Camera Setup Wizard
    camera_idx, camera_name, hw_tag = run_startup_wizard()
    use_gpu_pipeline = "GPU" in hw_tag

    # 2. Ensure Model
    ensure_model_asset()

    # 3. Initialize MediaPipe HandLandmarker with 2 hands support
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=CONFIG["MIN_DETECTION_CONFIDENCE"],
        min_hand_presence_confidence=CONFIG["MIN_TRACKING_CONFIDENCE"]
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # 4. Start Camera Stream
    print(f"\n[CAMERA] Ingest pipeline booting on {camera_name}...")
    stream = WebcamStream(src=camera_idx).start()
    width, height = stream.width, stream.height
    print(f"[CAMERA] Pipeline active: {width}x{height} @ {stream.fps:.1f} FPS (Target: {CONFIG['TARGET_FPS']} FPS)")

    hand_tracker = HandTracker()

    # Static pre-allocated drawing canvas buffers
    stroke_layer_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    stroke_layer_mask = np.zeros((height, width), dtype=np.uint8)

    # Object-based stroke manager
    strokes: List[Stroke] = []
    undo_stack: List[List[Stroke]] = []
    MAX_UNDO = 15

    # Smoothing filters for fingertip cursor
    filter_x = OneEuroFilter(CONFIG["FILTER_MIN_CUTOFF"], CONFIG["FILTER_BETA"], CONFIG["FILTER_D_CUTOFF"])
    filter_y = OneEuroFilter(CONFIG["FILTER_MIN_CUTOFF"], CONFIG["FILTER_BETA"], CONFIG["FILTER_D_CUTOFF"])

    # State variables
    color_idx = 0
    brush_thickness = CONFIG["DEFAULT_BRUSH_THICKNESS"]
    eraser_radius = CONFIG["DEFAULT_ERASER_RADIUS"]

    current_stroke: Optional[Stroke] = None
    selected_stroke: Optional[Stroke] = None
    transform_mode = "NONE"

    xp, yp = 0, 0
    prev_mid_x, prev_mid_y = 0, 0
    prev_drag_x, prev_drag_y = 0, 0
    pinch_counter = 0
    draw_start_time: Optional[float] = None
    last_move_time = time.time()

    fps_tracker = 0.0
    last_frame_time = time.time()

    cv2.namedWindow("Air Canvas PRO UX", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Canvas PRO UX", width, height)

    while True:
        frame_start_time = time.time()

        success, raw_frame = stream.read()
        if not success or raw_frame is None:
            time.sleep(0.005)
            continue

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        current_fps = 1.0 / dt if dt > 0 else 30.0
        fps_tracker = 0.9 * fps_tracker + 0.1 * current_fps if fps_tracker > 0 else current_fps

        # Image Processing Pipeline: Dispatches to real GPU (UMat) or Host CPU
        proc_w = CONFIG["PROCESS_WIDTH"]
        proc_h = CONFIG["PROCESS_HEIGHT"]

        if use_gpu_pipeline:
            u_raw = cv2.UMat(raw_frame)
            u_small = cv2.resize(u_raw, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            u_small_flipped = cv2.flip(u_small, 1)
            u_rgb = cv2.cvtColor(u_small_flipped, cv2.COLOR_BGR2RGB)
            u_frame = cv2.flip(u_raw, 1)

            rgb_small = u_rgb.get()
            frame = u_frame.get()
        else:
            small = cv2.resize(raw_frame, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            small_flipped = cv2.flip(small, 1)
            rgb_small = cv2.cvtColor(small_flipped, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(raw_frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        detection_result = hand_landmarker.detect(mp_image)
        mode = "IDLE"
        action_status = ""
        all_coords: List[Tuple[int, int]] = []
        detected_hand_count = 0

        if detection_result.hand_landmarks:
            detected_hand_count = len(detection_result.hand_landmarks)

            dominant_landmarks = hand_tracker.select_dominant_hand(detection_result.hand_landmarks)

            for lm in dominant_landmarks:
                all_coords.append((int(lm.x * width), int(lm.y * height)))

            raw_x, raw_y = all_coords[8]

            x1 = int(filter_x.filter(raw_x, now))
            y1 = int(filter_y.filter(raw_y, now))

            mode, meta = GestureEngine.classify(dominant_landmarks)

            if mode == "MOVE":
                pinch_counter += 1
            else:
                pinch_counter = 0
            pinch_active = pinch_counter >= CONFIG["PINCH_STABLE_FRAMES"]

            # ================= GESTURE DISPATCH =================

            # ERASE MODE
            if mode == "ERASE":
                if current_stroke is not None:
                    current_stroke = None
                transform_mode = "NONE"
                if selected_stroke is not None:
                    selected_stroke.selected = False
                    selected_stroke = None
                xp, yp = 0, 0

                new_strokes_list: List[Stroke] = []
                erased_any = False
                for stk in strokes:
                    res = stk.erase_at(x1, y1, eraser_radius)
                    if len(res) != 1 or res[0] is not stk:
                        erased_any = True
                    new_strokes_list.extend(res)

                if erased_any:
                    if len(undo_stack) >= MAX_UNDO:
                        undo_stack.pop(0)
                    undo_stack.append([s for s in strokes])
                    strokes = new_strokes_list

                cv2.circle(frame, (x1, y1), eraser_radius, (0, 0, 255), 2, cv2.LINE_AA)

            # MOVE & SCALE (PINCH) MODE
            elif pinch_active:
                if current_stroke is not None:
                    current_stroke = None
                xp, yp = 0, 0

                if transform_mode == "NONE":
                    if selected_stroke is not None and selected_stroke.hits_scale_handle(x1, y1):
                        transform_mode = "SCALE"
                        action_status = "Scaling Object"
                        if len(undo_stack) >= MAX_UNDO:
                            undo_stack.pop(0)
                        undo_stack.append([s for s in strokes])
                    else:
                        target = None
                        for stk in reversed(strokes):
                            if stk.hits_point(x1, y1, threshold=45.0):
                                target = stk
                                break

                        if target is not None:
                            if selected_stroke is not None and selected_stroke is not target:
                                selected_stroke.selected = False

                            selected_stroke = target
                            selected_stroke.selected = True
                            transform_mode = "MOVE"
                            action_status = "Moving Object"
                            prev_drag_x, prev_drag_y = x1, y1

                            if len(undo_stack) >= MAX_UNDO:
                                undo_stack.pop(0)
                            undo_stack.append([s for s in strokes])

                elif transform_mode == "SCALE" and selected_stroke is not None:
                    mode = "SCALE"
                    action_status = "Scaling..."
                    selected_stroke.scale_from_handle(x1, y1, width, height)
                    cv2.circle(frame, (x1, y1), 9, (0, 255, 255), -1, cv2.LINE_AA)

                elif transform_mode == "MOVE" and selected_stroke is not None:
                    mode = "MOVE"
                    action_status = "Moving..."
                    dx = x1 - prev_drag_x
                    dy = y1 - prev_drag_y
                    if dx != 0 or dy != 0:
                        selected_stroke.translate(dx, dy, width, height)
                    prev_drag_x, prev_drag_y = x1, y1
                    cv2.circle(frame, (x1, y1), 8, (255, 200, 0), -1, cv2.LINE_AA)

            # DRAW MODE
            elif mode == "DRAW":
                if transform_mode != "NONE":
                    transform_mode = "NONE"
                if selected_stroke is not None:
                    selected_stroke.selected = False
                    selected_stroke = None

                if draw_start_time is None:
                    draw_start_time = now
                elif now - draw_start_time > CONFIG["DRAW_DELAY"]:
                    active_color = PALETTE[color_idx][1]

                    if current_stroke is None:
                        if len(undo_stack) >= MAX_UNDO:
                            undo_stack.pop(0)
                        undo_stack.append([s for s in strokes])

                        current_stroke = Stroke(active_color, brush_thickness)
                        current_stroke.add_point((x1, y1))
                        strokes.append(current_stroke)
                        xp, yp = x1, y1
                        prev_mid_x, prev_mid_y = x1, y1
                        last_move_time = now
                    else:
                        speed = math.hypot(x1 - xp, y1 - yp)
                        if speed > 4.0:
                            last_move_time = now
                            mid_x = (xp + x1) // 2
                            mid_y = (yp + y1) // 2

                            curve_pts = get_bezier_curve(
                                np.array([prev_mid_x, prev_mid_y], dtype=np.float32),
                                np.array([xp, yp], dtype=np.float32),
                                np.array([mid_x, mid_y], dtype=np.float32)
                            )
                            current_stroke.add_points(curve_pts)

                            xp, yp = x1, y1
                            prev_mid_x, prev_mid_y = mid_x, mid_y

                        elif now - last_move_time > CONFIG["SHAPE_SNAP_PAUSE_TIME"]:
                            snapped_points = try_recognize_shape(current_stroke.points)
                            if snapped_points:
                                current_stroke.points = snapped_points
                                current_stroke.recalculate_bbox()
                                action_status = "Shape Snapped!"
                                print("[SNAP] Stroke automatically snapped to clean geometry.")
                                last_move_time = now + 999.0

                cv2.circle(frame, (x1, y1), brush_thickness + 2, PALETTE[color_idx][1], -1, cv2.LINE_AA)
                cv2.circle(frame, (x1, y1), brush_thickness + 4, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                transform_mode = "NONE"
                current_stroke = None
                xp, yp = 0, 0
                draw_start_time = None
                cv2.circle(frame, (x1, y1), 5, (200, 200, 200), 1, cv2.LINE_AA)

            draw_cyberpunk_skeleton(frame, all_coords)

        else:
            hand_tracker.reset()
            filter_x.reset()
            filter_y.reset()
            transform_mode = "NONE"
            current_stroke = None
            xp, yp = 0, 0
            draw_start_time = None

        # BARE-METAL C++ RENDERING & COMPOSITING:
        render_canvas_strokes(strokes, stroke_layer_bgr, stroke_layer_mask)
        cv2.copyTo(stroke_layer_bgr, stroke_layer_mask, frame)

        # Minimalist Production HUD Overlay
        active_name, active_bgr = PALETTE[color_idx]
        draw_hud(frame, mode, active_name, active_bgr, brush_thickness, eraser_radius,
                 fps_tracker, len(strokes), action_status, detected_hand_count, hw_tag)

        cv2.imshow("Air Canvas PRO UX", frame)

        # Precise 30 FPS Throttling
        target_frame_time = 1.0 / CONFIG.get("TARGET_FPS", 30)
        elapsed = time.time() - frame_start_time
        wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        # Universal Key Controls (Works across 60%, 65%, 75%, TKL, and Full keyboards)
        if key == 27:  # ESC
            break
        elif key in (ord('c'), ord('C')):  # Clear
            if strokes:
                if len(undo_stack) >= MAX_UNDO:
                    undo_stack.pop(0)
                undo_stack.append([s for s in strokes])
                strokes.clear()
            selected_stroke = None
            current_stroke = None
            xp, yp = 0, 0
            print("[CANVAS] Cleared all strokes.")
        elif key in (ord('z'), ord('Z'), ord('u'), ord('U')):  # Undo
            if undo_stack:
                strokes = undo_stack.pop()
                selected_stroke = None
                current_stroke = None
                print(f"[UNDO] Restored previous stroke state. ({len(undo_stack)} in stack)")
        elif key in (ord('e'), ord('E'), ord(']'), ord('.')):  # Next color
            color_idx = (color_idx + 1) % len(PALETTE)
            print(f"[COLOR] Selected: {PALETTE[color_idx][0]}")
        elif key in (ord('q'), ord('Q'), ord('['), ord(',')):  # Prev color
            color_idx = (color_idx - 1) % len(PALETTE)
            print(f"[COLOR] Selected: {PALETTE[color_idx][0]}")
        elif key in (ord('w'), ord('W'), ord('+'), ord('=')):  # Increase brush
            brush_thickness = min(24, brush_thickness + 1)
            print(f"[BRUSH] Size: {brush_thickness}px")
        elif key in (ord('s'), ord('S'), ord('-'), ord('_')):  # Decrease brush
            brush_thickness = max(1, brush_thickness - 1)
            print(f"[BRUSH] Size: {brush_thickness}px")
        elif key in (ord('x'), ord('X'), ord('p'), ord('P')):  # Save artwork
            export_name = f"air_canvas_export_{int(time.time())}.png"
            export_bgr = np.zeros_like(frame)
            export_mask = np.zeros((height, width), dtype=np.uint8)
            render_canvas_strokes(strokes, export_bgr, export_mask)
            cv2.imwrite(export_name, export_bgr)
            print(f"[EXPORT] Saved canvas artwork to '{export_name}'")

    stream.stop()
    cv2.destroyAllWindows()
    print("[SYSTEM] Graceful shutdown completed.")


if __name__ == "__main__":
    main()
