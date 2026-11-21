# Air Canvas

A high-performance, real-time gesture-controlled drawing canvas powered by MediaPipe Hand Landmarker, WebAssembly, and OpenCV.

---

## Features

- **Dual-Interface Runtime**:
  - **Web Application (`app.py`)**: Runs in modern web browsers via Streamlit. Performs 100% of video processing, MediaPipe landmark detection, and vector rendering locally in-browser via WebAssembly and WebGL. Zero video bytes streamed over the network.
  - **Native Desktop Application (`air_canvas.py`)**: Hardware-accelerated Python OpenCV + MediaPipe standalone desktop runtime.
- **Hardware Acceleration Hierarchy**:
  - Automatically routes execution: **dGPU** (Discrete GPU) $\to$ **iGPU** (Integrated GPU) $\to$ **CPU** (WASM SIMD Multi-Core).
  - Minimal HUD displays active hardware and runtime telemetry.
  - Manual runtime override available via dropdown.
- **Rock-Solid 30 FPS Frame Pacing**:
  - Precision delta-accumulator loop eliminates micro-stutters and frame drops.
  - Dynamic load-adaptive inference resolution scaling automatically stabilizes frame rates.
  - Motion-optimized confidence thresholds maintain tracking lock during rapid sweeps.
- **Air Gestures**:
  - **Index Finger Alone**: Draw freehand vector strokes with Bézier smoothing.
  - **Pinch (Index + Thumb)**: Select and translate existing strokes across canvas without distortion.
  - **Closed Fist**: Eraser with dedicated adjustable radius.
  - **Index + Middle**: Hover cursor reticle.
- **Studio Canvas Tools**:
  - 6-swatch primary palette + native RGB color picker.
  - Brush thickness slider (1–24 px).
  - Dedicated eraser slider (15–80 px).
  - Fullscreen toggle, Undo, Redo, and Clear.
  - Dynamic multi-camera selector.

---

## Installation & Requirements

Ensure Python 3.9+ is installed:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Web Application (Recommended / Streamlit Cloud Ready)

```bash
streamlit run app.py
```

Opens an in-browser interface with full local WebGL/WASM acceleration, optimized for lightweight cloud containers and local laptops.

### 2. Standalone Desktop Application

```bash
python air_canvas.py
```

Runs the OpenCV desktop window with terminal camera and hardware backend selection wizard.

---

## Testing

Run the test suite:

```bash
pytest
```
