/**
 * Air Canvas - Real-Time Browser Controller & MediaPipe Tracking
 * Uses @mediapipe/tasks-vision WebAssembly running 100% locally in browser.
 */

import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

(async function () {
  // Elements
  const video = document.getElementById('webcam');
  const cameraCanvas = document.getElementById('cameraCanvas');
  const paintCanvas = document.getElementById('paintCanvas');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const cameraCtx = cameraCanvas.getContext('2d', { alpha: false, desynchronized: true });
  const overlayCtx = overlayCanvas.getContext('2d', { alpha: true, desynchronized: true });

  const loaderOverlay = document.getElementById('loaderOverlay');
  const loaderText = document.getElementById('loaderText');
  const fpsCounter = document.getElementById('fpsCounter');
  const modeIndicator = document.getElementById('modeIndicator');
  const strokeCount = document.getElementById('strokeCount');
  const cameraSelect = document.getElementById('cameraSelect');
  const deviceSelect = document.getElementById('deviceSelect');
  const paletteContainer = document.getElementById('paletteContainer');
  const customColorPicker = document.getElementById('customColorPicker');
  const brushSlider = document.getElementById('brushSlider');
  const brushSizeVal = document.getElementById('brushSizeVal');
  const eraserSlider = document.getElementById('eraserSlider');
  const eraserSizeVal = document.getElementById('eraserSizeVal');

  let currentEraserRadius = 36;

  const btnUndo = document.getElementById('btnUndo');
  const btnRedo = document.getElementById('btnRedo');
  const btnClear = document.getElementById('btnClear');
  const btnFullscreen = document.getElementById('btnFullscreen');
  const hwBadge = document.getElementById('hwBadge');
  const hwActiveVal = document.getElementById('hwActiveVal');

  // Basic essential studio palette
  const PALETTE = [
    { name: 'Red', color: '#ff0000' },
    { name: 'Green', color: '#00ff00' },
    { name: 'Blue', color: '#0000ff' },
    { name: 'Yellow', color: '#ffff00' },
    { name: 'Cyan', color: '#00ffff' },
    { name: 'White', color: '#ffffff' }
  ];

  // Engine instance
  const engine = new CanvasEngine(paintCanvas);

  // Filters & State tracking
  const tipFilter = new PointFilter(1.0, 0.007);
  let handLandmarker = null;
  let visionResolver = null;
  let runningMode = 'VIDEO';
  let lastVideoTime = -1;
  let lastKnownResults = null;
  let currentStream = null;

  // Dynamic Load-Adaptive Resolution Scaling Engine
  // Automatically scales inference resolution (matching the exact display aspect ratio)
  // Low-end CPU / iGPU tiers:
  // 0: 640x360 (1.0), 1: 480x270 (0.75), 2: 320x180 (0.50), 3: 240x135 (0.375), 4: 160x90 (0.25)
  const ADAPTIVE_FACTORS = [1.0, 0.75, 0.50, 0.375, 0.25];
  let currentFactorIndex = 0;
  let consecutiveDropFrames = 0;
  let consecutiveGoodFrames = 0;
  let currentAvgFps = 30;

  // Offscreen inference canvases (zero DOM paint overhead)
  const adaptCanvas = document.createElement('canvas');
  adaptCanvas.width = 320;
  adaptCanvas.height = 180;
  const adaptCtx = adaptCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
  let activeDelegate = 'GPU';
  let isInferencing = false;

  // Gesture state machine
  let activeGesture = 'IDLE'; // 'DRAW', 'ERASE', 'MOVE', 'HOVER', 'IDLE'
  let gestureHoldCount = 0;
  let activeSelectedStroke = null;
  let initialPinchDist = 0;
  let initialPinchMid = [0, 0];

  // Performance tracking
  let frameTimes = [];
  let lastFrameTime = performance.now();

  // Setup Palette UI
  PALETTE.forEach((item, index) => {
    const swatch = document.createElement('div');
    swatch.className = 'color-swatch' + (index === 0 ? ' active' : '');
    swatch.style.backgroundColor = item.color;
    swatch.style.color = item.color;
    swatch.title = item.name;
    swatch.addEventListener('click', () => {
      document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
      swatch.classList.add('active');
      engine.currentColor = item.color;
    });
    paletteContainer.appendChild(swatch);
  });

  customColorPicker.addEventListener('input', (e) => {
    document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
    engine.currentColor = e.target.value;
  });

  brushSlider.addEventListener('input', (e) => {
    engine.brushSize = parseInt(e.target.value, 10);
    brushSizeVal.textContent = engine.brushSize + 'px';
  });

  if (eraserSlider) {
    eraserSlider.addEventListener('input', (e) => {
      currentEraserRadius = parseInt(e.target.value, 10);
      eraserSizeVal.textContent = currentEraserRadius + 'px';
    });
  }

  btnUndo.addEventListener('click', () => { engine.undo(); updateMetrics(); });
  btnRedo.addEventListener('click', () => { engine.redo(); updateMetrics(); });
  btnClear.addEventListener('click', () => { engine.clear(); updateMetrics(); });

  btnFullscreen.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  });

  function updateMetrics() {
    strokeCount.textContent = engine.strokes.length;
  }

  // Handle Aspect Ratio Preservation matching Video & Viewport
  function resizeCanvases() {
    const viewport = document.getElementById('viewport');
    const wrapper = document.getElementById('canvasWrapper');
    if (!viewport || !wrapper) return;

    const vpWidth = viewport.clientWidth;
    const vpHeight = viewport.clientHeight;
    if (vpWidth === 0 || vpHeight === 0) return;

    // Use native camera video stream aspect ratio if available, else 16:9
    const videoAspect = (video.videoWidth && video.videoHeight)
      ? (video.videoWidth / video.videoHeight)
      : (16 / 9);

    let displayW = vpWidth;
    let displayH = Math.round(vpWidth / videoAspect);

    if (displayH > vpHeight) {
      displayH = vpHeight;
      displayW = Math.round(vpHeight * videoAspect);
    }

    wrapper.style.width = displayW + 'px';
    wrapper.style.height = displayH + 'px';

    const renderW = video.videoWidth || displayW;
    const renderH = video.videoHeight || displayH;

    [cameraCanvas, paintCanvas, overlayCanvas].forEach(c => {
      if (c.width !== renderW || c.height !== renderH) {
        c.width = renderW;
        c.height = renderH;
      }
    });
    engine.resize(renderW, renderH);
  }
  window.addEventListener('resize', resizeCanvases);

  // Hardware GPU probe to distinguish discrete GPU (NVIDIA/AMD/Apple) vs integrated GPU (Intel/Mali)
  function detectGpuType() {
    const hostHw = window.__AIR_CANVAS_HOST_HW__ || {};
    let hostHasDgpu = false;
    if (hostHw.dgpu && hostHw.dgpu !== 'None Detected') {
      const dLow = hostHw.dgpu.toLowerCase();
      hostHasDgpu = dLow.includes('nvidia') || dLow.includes('geforce') || dLow.includes('rtx') || dLow.includes('gtx') || dLow.includes('radeon rx') || dLow.includes('apple');
    }

    try {
      const testCanvas = document.createElement('canvas');
      const gl = testCanvas.getContext('webgl2', { powerPreference: 'high-performance' }) ||
                 testCanvas.getContext('webgl', { powerPreference: 'high-performance' }) ||
                 testCanvas.getContext('webgl2') ||
                 testCanvas.getContext('webgl');
      if (!gl) return { hasGpu: hostHasDgpu, isDgpu: hostHasDgpu, renderer: hostHw.dgpu || '' };

      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = ext ? (gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '') : '';
      const rLower = renderer.toLowerCase();

      // Check for discrete GPU identifiers
      const isDgpuWebgl = rLower.includes('nvidia') ||
                          rLower.includes('geforce') ||
                          rLower.includes('rtx') ||
                          rLower.includes('gtx') ||
                          (rLower.includes('amd') && !rLower.includes('radeon graphics')) ||
                          rLower.includes('radeon rx') ||
                          rLower.includes('radeon pro') ||
                          rLower.includes('apple m');

      const isDgpu = isDgpuWebgl || hostHasDgpu;

      return { hasGpu: true, isDgpu, renderer: renderer || hostHw.dgpu || '' };
    } catch (e) {
      return { hasGpu: hostHasDgpu, isDgpu: hostHasDgpu, renderer: hostHw.dgpu || '' };
    }
  }

  // Client-side sanitizer for raw WebGL/CPU fallback strings
  function cleanClientHw(str, mode) {
    if (!str) return '';
    let s = str.replace(/ANGLE \((.*)\)/, '$1').replace(/Direct3D.*vs_\d_\d/, '').trim();
    s = s.replace(/Corporation/gi, '').replace(/Company/gi, '');
    s = s.replace(/\[.*?\]/g, '').replace(/\(.*?\)/g, '');
    if (mode === 'CPU') {
      s = s.replace(/\b\d+th\s+Gen\b/gi, '').replace(/Core\s+/gi, '').replace(/Processor/gi, '');
    } else {
      s = s.replace(/GeForce\s+/gi, '').replace(/Laptop GPU/gi, '').replace(/Mobile/gi, '');
    }
    return s.replace(/\s+/g, ' ').trim();
  }

  // Update Active Hardware HUD Display
  function updateHardwareDisplay(activeMode, gpuName, cpuName) {
    const hostHw = window.__AIR_CANVAS_HOST_HW__ || {};

    let details = '';
    if (activeMode === 'dGPU') {
      details = hostHw.dgpu || cleanClientHw(gpuName, 'dGPU');
    } else if (activeMode === 'CPU') {
      details = hostHw.cpu || cleanClientHw(cpuName, 'CPU');
    } else if (activeMode === 'iGPU') {
      details = hostHw.igpu || cleanClientHw(gpuName, 'iGPU');
    }

    if (hwBadge) {
      hwBadge.textContent = `${activeMode} • ACTIVE`;
      if (activeMode === 'dGPU') {
        hwBadge.style.color = '#00e5ff';
        hwBadge.style.borderColor = 'rgba(0, 229, 255, 0.4)';
      } else if (activeMode === 'CPU') {
        hwBadge.style.color = '#10b981';
        hwBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      } else {
        hwBadge.style.color = '#f59e0b';
        hwBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      }
    }

    if (hwActiveVal) {
      hwActiveVal.textContent = details ? `(${details})` : '';
      hwActiveVal.title = `${activeMode}: ${details}`;
    }
  }

  // Strict Hardware Hierarchy: 1. dGPU -> 2. iGPU -> 3. CPU (WASM)
  async function setupHandLandmarker(targetChoice = 'AUTO') {
    if (!visionResolver) {
      visionResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
    }

    const modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
    const gpuInfo = detectGpuType();

    const cores = navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency} Cores` : 'Multi-Core';
    const cpuName = `CPU (${cores} SIMD/WASM)`;
    const gpuName = gpuInfo.renderer ? gpuInfo.renderer.replace(/ANGLE \((.*)\)/, '$1').replace(/Direct3D.*vs_\d_\d/, '').trim() : 'WebGL Accelerator';

    // 1. dGPU attempt: If user requested dGPU, OR if AUTO and machine has dGPU
    const shouldTryDgpu = (targetChoice === 'dGPU') || (targetChoice === 'AUTO' && gpuInfo.isDgpu);
    if (shouldTryDgpu) {
      try {
        console.log('Attempting dGPU initialization...');
        handLandmarker = await HandLandmarker.createFromOptions(visionResolver, {
          baseOptions: {
            modelAssetPath: modelUrl,
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.35
        });
        activeDelegate = 'GPU';
        currentFactorIndex = 0; // High performance starting scale
        updateHardwareDisplay('dGPU', gpuName, cpuName);
        if (deviceSelect && targetChoice !== 'AUTO') deviceSelect.value = 'dGPU';
        return;
      } catch (dErr) {
        console.warn('dGPU init failed, falling back to iGPU:', dErr);
        if (targetChoice === 'dGPU') throw dErr; // If user explicitly selected dGPU, notify them
      }
    }

    // 2. iGPU attempt: If user requested iGPU, OR AUTO on systems without dGPU (or after dGPU fallback)
    const shouldTryIgpu = (targetChoice === 'iGPU') || (targetChoice === 'AUTO');
    if (shouldTryIgpu) {
      try {
        console.log('Attempting iGPU hardware acceleration initialization...');
        handLandmarker = await HandLandmarker.createFromOptions(visionResolver, {
          baseOptions: {
            modelAssetPath: modelUrl,
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.35
        });
        activeDelegate = 'GPU';
        currentFactorIndex = 1; // Start at 75% scale for smooth instant response on integrated chips
        updateHardwareDisplay('iGPU', gpuName, cpuName);
        if (deviceSelect && targetChoice !== 'AUTO') deviceSelect.value = 'iGPU';
        return;
      } catch (igpuErr) {
        console.warn('iGPU init failed, falling back to CPU:', igpuErr);
        if (targetChoice === 'iGPU') throw igpuErr;
      }
    }

    // 3. CPU fallback: If user requested CPU, OR AUTO fallback on systems without any working GPU
    if (targetChoice === 'CPU' || targetChoice === 'AUTO') {
      try {
        console.log('Attempting CPU WASM fallback initialization...');
        handLandmarker = await HandLandmarker.createFromOptions(visionResolver, {
          baseOptions: {
            modelAssetPath: modelUrl,
            delegate: 'CPU'
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.35
        });
        activeDelegate = 'CPU';
        // Auto-select starting tier for CPU based on core count
        const numCores = navigator.hardwareConcurrency || 2;
        currentFactorIndex = numCores <= 4 ? 3 : 2; // 35% scale on dual/quad-cores, 50% on higher
        updateHardwareDisplay('CPU', gpuName, cpuName);
        if (deviceSelect && targetChoice !== 'AUTO') deviceSelect.value = 'CPU';
        return;
      } catch (cpuErr) {
        console.error('All hardware initializations failed:', cpuErr);
        throw cpuErr;
      }
    }
  }

  async function initMediaPipe() {
    try {
      loaderText.textContent = 'Downloading MediaPipe Vision bundle...';
      loaderText.textContent = 'Probing hardware (dGPU -> iGPU -> CPU)...';
      await setupHandLandmarker('AUTO');

      loaderText.textContent = 'Accessing Camera...';
      await initCamera();
      loaderOverlay.classList.add('hidden');
    } catch (err) {
      console.error('MediaPipe Init Error:', err);
      loaderText.innerHTML = `<span style="color:#ef4444">Failed to initialize: ${err.message}</span>`;
    }
  }

  // Camera Management - Starts with highest capabilities of webcam
  async function initCamera(deviceId = null) {
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
    }

    const videoConstraints = {
      width: { ideal: 1920, min: 640 },
      height: { ideal: 1080, min: 480 },
      frameRate: { ideal: 30, max: 30 }
    };

    if (deviceId) {
      videoConstraints.deviceId = { exact: deviceId };
    } else {
      videoConstraints.facingMode = 'user';
    }

    const constraints = {
      video: videoConstraints,
      audio: false
    };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;
      await new Promise(r => video.onloadedmetadata = r);
      video.play();
      resizeCanvases();
      await populateCameraList(deviceId);
    } catch (err) {
      // Fallback to basic constraint if specific resolution was rejected
      try {
        const fallbackConstraints = {
          video: deviceId ? { deviceId: { exact: deviceId } } : true,
          audio: false
        };
        currentStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        video.srcObject = currentStream;
        await new Promise(r => video.onloadedmetadata = r);
        video.play();
        resizeCanvases();
        await populateCameraList(deviceId);
      } catch (err2) {
        console.error('Camera Access Error:', err2);
        loaderText.innerHTML = `<span style="color:#ef4444">Camera Error: ${err2.message}</span>`;
        throw err2;
      }
    }
  }

  async function populateCameraList(selectedDeviceId = null) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      cameraSelect.innerHTML = '';

      // Determine active deviceId from current active video track if not passed
      let activeTrackId = selectedDeviceId;
      if (!activeTrackId && currentStream) {
        const videoTrack = currentStream.getVideoTracks()[0];
        if (videoTrack) {
          const settings = videoTrack.getSettings ? videoTrack.getSettings() : {};
          activeTrackId = settings.deviceId || null;
        }
      }

      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Camera ${idx + 1}`;
        cameraSelect.appendChild(opt);
      });

      if (videoDevices.length > 0) {
        if (activeTrackId && videoDevices.some(d => d.deviceId === activeTrackId)) {
          cameraSelect.value = activeTrackId;
        } else {
          cameraSelect.value = videoDevices[0].deviceId;
        }
      }
    } catch (err) {
      console.warn('Could not enumerate cameras:', err);
    }
  }

  cameraSelect.addEventListener('change', () => {
    initCamera(cameraSelect.value);
  });

  if (deviceSelect) {
    deviceSelect.addEventListener('change', async () => {
      loaderOverlay.classList.remove('hidden');
      loaderText.textContent = `Switching processor to ${deviceSelect.value}...`;
      await setupHandLandmarker(deviceSelect.value);
      loaderOverlay.classList.add('hidden');
    });
  }

  // Orientation-invariant palm coordinate basis classifier matching desktop Python engine
  function classifyGesture(landmarks) {
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];

    let vLongX = middleMcp.x - wrist.x;
    let vLongY = middleMcp.y - wrist.y;
    let palmLen = Math.hypot(vLongX, vLongY);

    if (palmLen < 1e-4) {
      vLongX = 0.0;
      vLongY = -1.0;
      palmLen = 0.2;
    } else {
      vLongX /= palmLen;
      vLongY /= palmLen;
    }

    function isFingerExtended(tipIdx, pipIdx, mcpIdx) {
      const tip = landmarks[tipIdx];
      const pip = landmarks[pipIdx];
      const mcp = landmarks[mcpIdx];

      const projTip = (tip.x - mcp.x) * vLongX + (tip.y - mcp.y) * vLongY;
      const projPip = (pip.x - mcp.x) * vLongX + (pip.y - mcp.y) * vLongY;
      const radialTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
      const radialPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);

      return (projTip > projPip * 1.15) && (radialTip > radialPip * 1.10);
    }

    const indexUp = isFingerExtended(8, 6, 5);
    const middleUp = isFingerExtended(12, 10, 9);
    const ringUp = isFingerExtended(16, 14, 13);
    const pinkyUp = isFingerExtended(20, 18, 17);

    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const rawPinch = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
    const pinchRatio = rawPinch / Math.max(palmLen, 0.05);

    const totalExtended = (indexUp ? 1 : 0) + (middleUp ? 1 : 0) + (ringUp ? 1 : 0) + (pinkyUp ? 1 : 0);

    // 1. FIST / ERASE: All 4 fingers curled
    if (totalExtended === 0) {
      return 'ERASE';
    }

    // 2. PINCH / MOVE: Thumb and Index tips close
    if (pinchRatio < 0.38 || rawPinch < 0.055) {
      return 'MOVE';
    }

    // 3. DRAW: Index finger extended alone
    if (indexUp && !middleUp && !ringUp && !pinkyUp) {
      return 'DRAW';
    }

    // 4. HOVER / SELECTION: Index and Middle extended
    if (indexUp && middleUp && !ringUp && !pinkyUp) {
      return 'HOVER';
    }

    return 'IDLE';
  }

  // Precision 30 FPS Frame Pacing Engine with Delta Accumulator
  const TARGET_FPS = 30.0;
  const FRAME_INTERVAL = 1000.0 / TARGET_FPS; // 33.333ms per frame
  let framePacingDelta = 0;

  // Main Detection & Render Loop (Synchronous Direct-Feed for Rock-Solid Tracking)
  async function renderLoop(timestamp) {
    requestAnimationFrame(renderLoop);

    if (!timestamp) timestamp = performance.now();
    const elapsed = timestamp - lastFrameTime;

    // Guard against background tab suspension jumps
    if (elapsed > 1000.0) {
      lastFrameTime = timestamp;
      return;
    }

    // Accumulate elapsed delta and fire exactly when 33.33ms budget is satisfied
    framePacingDelta += elapsed;
    lastFrameTime = timestamp;

    if (framePacingDelta < FRAME_INTERVAL) {
      return; // Hold pacing until target 30 FPS boundary
    }

    // Deduct frame interval while capping accumulator drift
    framePacingDelta = Math.min(framePacingDelta % FRAME_INTERVAL, FRAME_INTERVAL * 0.5);

    const now = performance.now();

    // Track smoothed FPS
    frameTimes.push(now);
    if (frameTimes.length > 20) frameTimes.shift();
    if (frameTimes.length > 1) {
      currentAvgFps = Math.round((frameTimes.length - 1) / ((now - frameTimes[0]) / 1000));
      fpsCounter.textContent = Math.min(30, currentAvgFps);
    }

    const cw = cameraCanvas.width;
    const ch = cameraCanvas.height;

    // 1. Draw mirrored background video frame
    if (video.readyState >= 2) {
      cameraCtx.save();
      cameraCtx.translate(cw, 0);
      cameraCtx.scale(-1, 1);
      cameraCtx.drawImage(video, 0, 0, cw, ch);
      cameraCtx.restore();
    }

    // 2. Clear overlay layer
    overlayCtx.clearRect(0, 0, cw, ch);

    if (!handLandmarker || video.readyState < 2) return;

    // 3. MediaPipe video detection with Dynamic Load Adaptation
    let results = null;
    if (video.currentTime !== lastVideoTime && !isInferencing) {
      lastVideoTime = video.currentTime;
      isInferencing = true;
      const tInferenceStart = performance.now();

      try {
        const scale = ADAPTIVE_FACTORS[currentFactorIndex];
        const vW = video.videoWidth || 640;
        const vH = video.videoHeight || 360;

        // If on GPU with full scale, pass video directly; otherwise downscale to adaptCanvas
        if (scale === 1.0 && activeDelegate === 'GPU') {
          results = handLandmarker.detectForVideo(video, now);
        } else {
          const targetW = Math.max(160, Math.round(vW * scale));
          const targetH = Math.max(90, Math.round(vH * scale));

          if (adaptCanvas.width !== targetW || adaptCanvas.height !== targetH) {
            adaptCanvas.width = targetW;
            adaptCanvas.height = targetH;
          }
          adaptCtx.drawImage(video, 0, 0, targetW, targetH);
          results = handLandmarker.detectForVideo(adaptCanvas, now);
        }
        lastKnownResults = results;
      } catch (err) {
        console.warn('Tracking detect error:', err);
      } finally {
        isInferencing = false;
      }

      const inferenceDuration = performance.now() - tInferenceStart;

      // Ultra-Fast Recovery Dynamic Load Controller:
      // Drop prompt: 2 consecutive frame drops (< 24 FPS or > 26ms inference) steps down.
      // Fast recovery: only 6 consecutive healthy frames (<= 20ms inference & >= 25 FPS) immediately recovers resolution.
      if (inferenceDuration > 26.0 || currentAvgFps < 24) {
        consecutiveDropFrames++;
        consecutiveGoodFrames = 0;
        if (consecutiveDropFrames >= 2 && currentFactorIndex < ADAPTIVE_FACTORS.length - 1) {
          currentFactorIndex++;
          consecutiveDropFrames = 0;
        }
      } else if (inferenceDuration <= 20.0 && currentAvgFps >= 25) {
        consecutiveGoodFrames++;
        consecutiveDropFrames = 0;
        if (consecutiveGoodFrames >= 6 && currentFactorIndex > 0) {
          currentFactorIndex--;
          consecutiveGoodFrames = 0;
        }
      }
    } else {
      results = lastKnownResults;
    }

    if (results && results.landmarks && results.landmarks.length > 0) {
      // Pick dominant hand (largest palm)
      let bestHand = results.landmarks[0];
      if (results.landmarks.length > 1) {
        let maxPalm = 0;
        for (const h of results.landmarks) {
          const pSize = Math.hypot(h[9].x - h[0].x, h[9].y - h[0].y);
          if (pSize > maxPalm) {
            maxPalm = pSize;
            bestHand = h;
          }
        }
      }

      // Draw futuristic skeleton on overlay canvas (mirrored horizontally)
      drawHandSkeleton(bestHand, cw, ch);

      // Raw index tip mirrored
      const rawX = (1.0 - bestHand[8].x) * cw;
      const rawY = bestHand[8].y * ch;

      // Filter index tip point with high-speed response
      const [fx, fy] = tipFilter.filter(rawX, rawY, now / 1000);

      // Classify gesture
      const detected = classifyGesture(bestHand);

      // Instant activation on DRAW/ERASE, single frame hysteresis for transitions to eliminate fast-motion latency
      if (detected === 'DRAW' || detected === 'ERASE' || detected === activeGesture) {
        activeGesture = detected;
        gestureHoldCount = Math.min(gestureHoldCount + 1, 10);
      } else {
        gestureHoldCount++;
        if (gestureHoldCount >= 1) {
          activeGesture = detected;
          gestureHoldCount = 0;
        }
      }

      // Update UI Mode Tag
      modeIndicator.textContent = activeGesture;
      modeIndicator.className = 'mode-tag ' + activeGesture.toLowerCase();

      // Dispatch Gesture to Engine
      handleGestureAction(activeGesture, fx, fy, bestHand, cw, ch);
    } else if (results && results.landmarks && results.landmarks.length === 0) {
      // Hand lost: reset filters & release strokes
      tipFilter.reset();
      if (engine.currentStroke) {
        engine.endStroke();
        updateMetrics();
      }
      activeGesture = 'IDLE';
      modeIndicator.textContent = 'NO HAND';
      modeIndicator.className = 'mode-tag idle';
    }
  }

  function handleGestureAction(gesture, fx, fy, landmarks, cw, ch) {
    if (gesture === 'DRAW') {
      if (!engine.currentStroke) {
        engine.beginStroke(fx, fy);
      } else {
        engine.extendStroke(fx, fy);
      }
      drawPointerReticle(fx, fy, engine.currentColor, engine.brushSize);
    } else {
      if (engine.currentStroke) {
        engine.endStroke();
        updateMetrics();
      }
    }

    if (gesture === 'ERASE') {
      // Index finger MCP as center of fist
      const ex = (1.0 - landmarks[5].x) * cw;
      const ey = landmarks[5].y * ch;
      const eraseRadius = currentEraserRadius;
      if (engine.eraseAt(ex, ey, eraseRadius)) {
        updateMetrics();
      }
      drawEraserReticle(ex, ey, eraseRadius);
    } else {
      engine._lastErasePushTime = null;
    }

    if (gesture === 'MOVE') {
      // Pinch position: index tip (matching Python desktop engine x1, y1)
      const px = fx;
      const py = fy;

      if (!activeSelectedStroke) {
        // Find stroke within generous selection radius (60px)
        activeSelectedStroke = engine.findStrokeAt(px, py, 60);
        if (activeSelectedStroke) {
          engine.pushState(); // Save state before transform
          initialPinchMid = [px, py];
        }
      } else {
        // Move / drag only — preserves original stroke geometry and scale exactly
        const dx = px - initialPinchMid[0];
        const dy = py - initialPinchMid[1];
        if (dx !== 0 || dy !== 0) {
          activeSelectedStroke.translate(dx, dy);
          initialPinchMid = [px, py];
          engine.redraw();
        }
      }
      drawPinchReticle(px, py, 20);
    } else {
      activeSelectedStroke = null;
    }

    if (gesture === 'HOVER') {
      drawHoverReticle(fx, fy);
    }
  }

  // Overlay HUD Visualizations
  const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
  ];

  function drawHandSkeleton(landmarks, w, h) {
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
    overlayCtx.lineWidth = 1.5;

    for (const [i, j] of CONNECTIONS) {
      const x1 = (1.0 - landmarks[i].x) * w;
      const y1 = landmarks[i].y * h;
      const x2 = (1.0 - landmarks[j].x) * w;
      const y2 = landmarks[j].y * h;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x1, y1);
      overlayCtx.lineTo(x2, y2);
      overlayCtx.stroke();
    }

    for (let i = 0; i < landmarks.length; i++) {
      const x = (1.0 - landmarks[i].x) * w;
      const y = landmarks[i].y * h;
      overlayCtx.fillStyle = (i === 8) ? '#00e5ff' : 'rgba(255, 255, 255, 0.7)';
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, (i === 8) ? 4 : 2, 0, Math.PI * 2);
      overlayCtx.fill();
    }
    overlayCtx.restore();
  }

  function drawPointerReticle(x, y, color, size) {
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, size + 4, 0, Math.PI * 2);
    overlayCtx.stroke();

    overlayCtx.fillStyle = color;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.restore();
  }

  function drawEraserReticle(x, y, radius) {
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([4, 4]);
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, radius, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.restore();
  }

  function drawPinchReticle(x, y, dist) {
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(245, 158, 11, 0.8)';
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, Math.max(10, dist / 2), 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.restore();
  }

  function drawHoverReticle(x, y) {
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
    overlayCtx.lineWidth = 1.5;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 8, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.restore();
  }

  // Launch
  await initMediaPipe();
  renderLoop();
})();
