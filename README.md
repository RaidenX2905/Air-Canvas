<p align="center">
  <img src="assets/brand-mark-dark-512.png" alt="Air Canvas Brand Mark" width="128" height="128">
</p>

<div align="center">

# Air Canvas

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](#) [![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#) [![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](#) [![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-0078D4?style=flat-square&logo=google&logoColor=white)](#) [![WebAssembly](https://img.shields.io/badge/WebAssembly-WASM-654FF0?style=flat-square&logo=webassembly&logoColor=white)](#) [![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#)

<p align="center">
  A high-performance, real-time gesture-controlled drawing canvas powered by MediaPipe Hand Landmarker, OpenCV, and WebAssembly with Apple-inspired Liquid Glass UI and bi-directional hardware acceleration.
</p>

</div>

---

**Live Demo**: [https://aircanvasbyjsh.streamlit.app/](https://aircanvasbyjsh.streamlit.app/)

---

## Highlights

- **Apple-Inspired Liquid Glass UI**: Frosted translucent glassmorphism interface with backdrop saturation blur, floating pill controls, specular inner borders, and high-contrast dark canvas.
- **Bi-Directional Auto Hardware Stepping**: Dynamic hardware orchestration strictly prioritizing Discrete GPU (`dGPU`) -> Integrated GPU (`iGPU`) -> CPU WASM fallback, with real-time frame-pacing resolution scaling.
- **Dynamic Fullscreen Mode**: Seamless cross-context fullscreen synchronization across native browsers and iframe embeds.
- **Client-Side Privacy**: 100% of video processing, hand landmark estimation, and vector ink rendering occurs inside your browser. Zero video bytes streamed over the network.

> **Developer Recommendation**: For the best, lowest-latency drawing experience, run on a PC in **dGPU mode** (Discrete GPU).

---

## Gesture Controls

- **Index Finger**: Freehand vector drawing with Bezier smoothing.
- **Pinch (Index + Thumb)**: Select and move existing strokes across canvas.
- **Closed Fist**: Dynamic eraser with adjustable radius.
- **Index + Middle**: Hover reticle guide.

---

## Quick Start

### 1. Web Application (Recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Standalone Desktop Application

```bash
python air_canvas.py
```

### 3. Automated Tests

```bash
pytest -v
```

---

## License

MIT License
