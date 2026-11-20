# Air Canvas

A high-performance, low-latency computer vision application that turns your webcam into a gesture-controlled digital canvas using Google MediaPipe Tasks, OpenCV, and NumPy.

---

## Overview

Air Canvas provides a gesture-driven digital ink experience. The pipeline runs with bare-metal C++ compositing, orientation-invariant hand tracking, and adaptive jitter filtering.

### Key Capabilities
- **Bare-Metal Compositing**: Employs `cv2.copyTo` to merge vector strokes in ~0.13 ms, avoiding black fringe artifacts and non-contiguous memory copies.
- **Hardware & Camera Setup Wizard**: Scans and prompts for available video devices and compute acceleration backends (GPU via OpenCL or optimized multi-threaded CPU).
- **Multi-Hand Foreground Lock**: Tracks and isolates the dominant hand, discarding background hands to avoid jitter and gesture misclassifications.
- **Orientation-Invariant Hand Tracking**: Uses longitudinal and lateral palm vectors to classify gestures across a full 360-degree rotation range.
- **Vectorized Stroke Architecture**: Digital ink is managed as individual stroke objects, allowing specific strokes to be moved or scaled independently.
- **Shape Recognition and Auto-Snap**: Snaps rough freehand strokes into lines, circles, or rectangles when holding position at the end of a stroke.
- **Adaptive 1-Euro Filter**: Suppresses micro-tremors at low speeds while eliminating lag during fast movement.
- **Compact Keyboard Support**: Hotkeys mapped to primary letters (`W`/`S`, `E`/`Q`, `Z`, `C`, `X`) for 60%, 65%, 75%, and compact laptop keyboards.

---

## Technical Stack
- **Language**: Python 3.9+
- **Computer Vision**: OpenCV (`cv2`) with OpenCL Transparent API support
- **Hand Tracking**: Google MediaPipe Tasks (`HandLandmarker`)
- **Signal Filtering**: Adaptive 1-Euro Low-Pass Filter
- **Concurrency**: Native `threading` daemon with double-buffered frame capture

---

## Gestures

| Gesture | Mode | Description |
| :--- | :--- | :--- |
| **Index Finger Extended** | `DRAW` | Extrapolates movement into continuous C1 Bézier strokes. |
| **Pinch (Thumb & Index)** | `MOVE / SCALE` | Pinch stroke body to move; pinch corner handle to resize. |
| **Closed Fist** | `ERASE` | Clears markings within the eraser radius. |
| **Index & Middle Extended** | `HOVER` | Free cursor movement without drawing. |
| **Open Hand** | `IDLE` | Disengages drawing and transformation actions. |

---

## Controls

The control layout works across compact keyboards (60%, 65%, 75%, TKL) without requiring dedicated numpads or secondary function layers:

| Key | Action | Secondary Key |
| :--- | :--- | :--- |
| `E` | Next Color | `]` or `.` |
| `Q` | Previous Color | `[` or `,` |
| `W` | Increase Brush Size | `+` or `=` |
| `S` | Decrease Brush Size | `-` or `_` |
| `Z` | Undo Last Stroke | `U` |
| `C` | Clear All Strokes | - |
| `X` | Save Artwork to PNG | `P` |
| `ESC` | Exit Application | - |

---

## Setup & Execution

### 1. Requirements
Ensure Python 3.9+ is installed along with the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run
Launch the application:
```bash
python air_canvas.py
```

On first launch, the required `hand_landmarker.task` model file is verified and downloaded automatically if missing. Follow the on-screen terminal prompts to select your webcam and compute backend.
