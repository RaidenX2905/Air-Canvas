/**
 * Air Canvas Engine - Client-side Geometry, Filtering & Stroke Management
 * Fully matches Python Air Canvas algorithms and guarantees.
 */

// 1. One Euro Filter for low-latency adaptive landmark filtering
class OneEuroFilter {
  constructor(freq = 30, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this.freq = freq;
    this.mincutoff = mincutoff;
    this.beta = beta;
    this.dcutoff = dcutoff;
    this.x_prev = null;
    this.dx_prev = 0.0;
    this.t_prev = null;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, t = null) {
    if (t === null) t = performance.now() / 1000.0;
    if (this.t_prev === null) {
      this.x_prev = x;
      this.dx_prev = 0.0;
      this.t_prev = t;
      return x;
    }
    const dt = Math.max(t - this.t_prev, 1e-4);
    this.t_prev = t;

    const dx = (x - this.x_prev) / dt;
    const a_d = this.alpha(this.dcutoff, dt);
    const dx_hat = a_d * dx + (1.0 - a_d) * this.dx_prev;
    this.dx_prev = dx_hat;

    const cutoff = this.mincutoff + this.beta * Math.abs(dx_hat);
    const a = this.alpha(cutoff, dt);
    const x_hat = a * x + (1.0 - a) * this.x_prev;
    this.x_prev = x_hat;
    return x_hat;
  }

  reset() {
    this.x_prev = null;
    this.dx_prev = 0.0;
    this.t_prev = null;
  }
}

// 2D Point Filter Pair
class PointFilter {
  constructor(mincutoff = 1.0, beta = 0.007) {
    this.fx = new OneEuroFilter(30, mincutoff, beta);
    this.fy = new OneEuroFilter(30, mincutoff, beta);
  }
  filter(x, y, t = null) {
    return [this.fx.filter(x, t), this.fy.filter(y, t)];
  }
  reset() {
    this.fx.reset();
    this.fy.reset();
  }
}

// 2. Math helpers: perpendicular point to segment distance
function distPointToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * dx + (py - y1) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return Math.hypot(px - projX, py - projY);
}

// 3. Vector Stroke Representation with Deep Clone Support
class Stroke {
  constructor(color = '#ff0000', thickness = 6, type = 'freehand') {
    this.color = color;
    this.thickness = thickness;
    this.type = type; // 'freehand', 'line', 'circle', 'rect'
    this.points = []; // array of [x, y]
    this.shapeData = null; // for geometric shapes: line {x1,y1,x2,y2}, circle {cx,cy,r}, rect {x,y,w,h}
    this.bbox = null; // [minX, minY, maxX, maxY]
  }

  addPoint(x, y) {
    this.points.push([x, y]);
    this.updateBbox();
  }

  updateBbox() {
    if (this.shapeData) {
      if (this.type === 'line') {
        const { x1, y1, x2, y2 } = this.shapeData;
        this.bbox = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
      } else if (this.type === 'circle') {
        const { cx, cy, r } = this.shapeData;
        this.bbox = [cx - r, cy - r, cx + r, cy + r];
      } else if (this.type === 'rect') {
        const { x, y, w, h } = this.shapeData;
        this.bbox = [x, y, x + w, y + h];
      }
      return;
    }
    if (this.points.length === 0) {
      this.bbox = null;
      return;
    }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of this.points) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    this.bbox = [minX, minY, maxX, maxY];
  }

  clone() {
    const copy = new Stroke(this.color, this.thickness, this.type);
    copy.points = this.points.map(p => [p[0], p[1]]);
    if (this.shapeData) {
      copy.shapeData = { ...this.shapeData };
    }
    if (this.bbox) {
      copy.bbox = [...this.bbox];
    }
    return copy;
  }

  translate(dx, dy) {
    for (let i = 0; i < this.points.length; i++) {
      this.points[i][0] += dx;
      this.points[i][1] += dy;
    }
    if (this.shapeData) {
      if (this.type === 'line') {
        this.shapeData.x1 += dx; this.shapeData.y1 += dy;
        this.shapeData.x2 += dx; this.shapeData.y2 += dy;
      } else if (this.type === 'circle') {
        this.shapeData.cx += dx; this.shapeData.cy += dy;
      } else if (this.type === 'rect') {
        this.shapeData.x += dx; this.shapeData.y += dy;
      }
    }
    this.updateBbox();
  }

  scale(factor, cx, cy) {
    for (let i = 0; i < this.points.length; i++) {
      this.points[i][0] = cx + (this.points[i][0] - cx) * factor;
      this.points[i][1] = cy + (this.points[i][1] - cy) * factor;
    }
    if (this.shapeData) {
      if (this.type === 'line') {
        this.shapeData.x1 = cx + (this.shapeData.x1 - cx) * factor;
        this.shapeData.y1 = cy + (this.shapeData.y1 - cy) * factor;
        this.shapeData.x2 = cx + (this.shapeData.x2 - cx) * factor;
        this.shapeData.y2 = cy + (this.shapeData.y2 - cy) * factor;
      } else if (this.type === 'circle') {
        this.shapeData.cx = cx + (this.shapeData.cx - cx) * factor;
        this.shapeData.cy = cy + (this.shapeData.cy - cy) * factor;
        this.shapeData.r *= factor;
      } else if (this.type === 'rect') {
        this.shapeData.x = cx + (this.shapeData.x - cx) * factor;
        this.shapeData.y = cy + (this.shapeData.y - cy) * factor;
        this.shapeData.w *= factor;
        this.shapeData.h *= factor;
      }
    }
    this.updateBbox();
  }

  // Segment-based collision detection for eraser
  hitsEraser(ex, ey, eraserRadius) {
    const threshold = eraserRadius + this.thickness / 2;
    if (this.bbox) {
      const [minX, minY, maxX, maxY] = this.bbox;
      if (ex < minX - threshold || ex > maxX + threshold ||
        ey < minY - threshold || ey > maxY + threshold) {
        return false;
      }
    }
    if (this.shapeData) {
      if (this.type === 'line') {
        const { x1, y1, x2, y2 } = this.shapeData;
        return distPointToSegment(ex, ey, x1, y1, x2, y2) <= threshold;
      } else if (this.type === 'circle') {
        const { cx, cy, r } = this.shapeData;
        const d = Math.abs(Math.hypot(ex - cx, ey - cy) - r);
        return d <= threshold;
      } else if (this.type === 'rect') {
        const { x, y, w, h } = this.shapeData;
        const top = distPointToSegment(ex, ey, x, y, x + w, y);
        const bot = distPointToSegment(ex, ey, x, y + h, x + w, y + h);
        const left = distPointToSegment(ex, ey, x, y, x, y + h);
        const right = distPointToSegment(ex, ey, x + w, y, x + w, y + h);
        return Math.min(top, bot, left, right) <= threshold;
      }
    }

    if (this.points.length === 1) {
      return Math.hypot(ex - this.points[0][0], ey - this.points[0][1]) <= threshold;
    }

    for (let i = 0; i < this.points.length - 1; i++) {
      const [x1, y1] = this.points[i];
      const [x2, y2] = this.points[i + 1];
      if (distPointToSegment(ex, ey, x1, y1, x2, y2) <= threshold) {
        return true;
      }
    }
    return false;
  }

  hitsPoint(x, y, threshold = 35.0) {
    return this.hitsEraser(x, y, threshold);
  }

  // Continuous segment collision erasure that splits the stroke into surviving sub-strokes
  eraseAt(ex, ey, eraserRadius) {
    const pad = eraserRadius + this.thickness / 2;
    if (this.bbox) {
      const [minX, minY, maxX, maxY] = this.bbox;
      if (ex < minX - pad || ex > maxX + pad ||
          ey < minY - pad || ey > maxY + pad) {
        return [this];
      }
    }

    if (this.shapeData) {
      if (this.hitsEraser(ex, ey, eraserRadius)) {
        return [];
      }
      return [this];
    }

    const n = this.points.length;
    if (n === 0) return [];
    if (n === 1) {
      if (Math.hypot(this.points[0][0] - ex, this.points[0][1] - ey) <= pad) {
        return [];
      }
      return [this];
    }

    const padSq = pad * pad;
    let hitCount = 0;
    const ptHit = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const dx = this.points[i][0] - ex;
      const dy = this.points[i][1] - ey;
      if (dx * dx + dy * dy <= padSq) {
        ptHit[i] = 1;
        hitCount++;
      }
    }

    // Early exit: if no points hit and bounding box is tight, do fast segment check
    let anySegHit = false;
    for (let i = 0; i < n - 1; i++) {
      if (ptHit[i] || ptHit[i + 1]) {
        anySegHit = true;
        break;
      }
      const [ax, ay] = this.points[i];
      const [bx, by] = this.points[i + 1];
      if (distPointToSegment(ex, ey, ax, ay, bx, by) <= pad) {
        anySegHit = true;
        break;
      }
    }

    if (!anySegHit && hitCount === 0) {
      return [this];
    }

    const newStrokes = [];
    let current = null;

    for (let i = 0; i < n; i++) {
      let isErased = (ptHit[i] === 1);
      if (!isErased && i > 0 && ptHit[i - 1] === 0) {
        const [ax, ay] = this.points[i - 1];
        const [bx, by] = this.points[i];
        if (distPointToSegment(ex, ey, ax, ay, bx, by) <= pad) {
          isErased = true;
        }
      }

      if (!isErased) {
        if (!current) {
          current = new Stroke(this.color, this.thickness, 'freehand');
        }
        current.points.push(this.points[i]);
      } else {
        if (current && current.points.length > 0) {
          current.updateBbox();
          newStrokes.push(current);
        }
        current = null;
      }
    }

    if (current && current.points.length > 0) {
      current.updateBbox();
      newStrokes.push(current);
    }

    return newStrokes;
  }

  // Draw onto HTML5 canvas with sub-pixel antialiasing
  render(ctx) {
    ctx.save();
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.thickness;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (this.shapeData) {
      ctx.beginPath();
      if (this.type === 'line') {
        ctx.moveTo(this.shapeData.x1, this.shapeData.y1);
        ctx.lineTo(this.shapeData.x2, this.shapeData.y2);
      } else if (this.type === 'circle') {
        ctx.arc(this.shapeData.cx, this.shapeData.cy, this.shapeData.r, 0, Math.PI * 2);
      } else if (this.type === 'rect') {
        ctx.rect(this.shapeData.x, this.shapeData.y, this.shapeData.w, this.shapeData.h);
      }
      ctx.stroke();
      ctx.restore();
      return;
    }

    if (this.points.length === 0) {
      ctx.restore();
      return;
    }
    if (this.points.length === 1) {
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.points[0][0], this.points[0][1], this.thickness / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    // Bezier curve smoothing across points
    ctx.beginPath();
    ctx.moveTo(this.points[0][0], this.points[0][1]);
    for (let i = 1; i < this.points.length - 1; i++) {
      const midX = (this.points[i][0] + this.points[i + 1][0]) / 2;
      const midY = (this.points[i][1] + this.points[i + 1][1]) / 2;
      ctx.quadraticCurveTo(this.points[i][0], this.points[i][1], midX, midY);
    }
    ctx.lineTo(this.points[this.points.length - 1][0], this.points[this.points.length - 1][1]);
    ctx.stroke();
    ctx.restore();
  }
}

// 4. Shape Recognition (Line, Circle, Rectangle Auto-Snap)
function tryRecognizeShape(stroke) {
  const pts = stroke.points;
  if (pts.length < 15) return null;

  const start = pts[0];
  const end = pts[pts.length - 1];
  const totalChord = Math.hypot(end[0] - start[0], end[1] - start[1]);

  let totalLen = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    totalLen += Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
  }
  if (totalLen < 30) return null;

  // 1. Line Test: ratio of chord distance to arc length close to 1
  if (totalChord / totalLen > 0.94) {
    return {
      type: 'line',
      shapeData: { x1: start[0], y1: start[1], x2: end[0], y2: end[1] }
    };
  }

  // 2. Closed Shape Test
  const isClosed = totalChord / totalLen < 0.18;
  if (isClosed) {
    // Compute centroid
    let sumX = 0, sumY = 0;
    for (const [x, y] of pts) {
      sumX += x;
      sumY += y;
    }
    const cx = sumX / pts.length;
    const cy = sumY / pts.length;

    // Distances from centroid
    const dists = pts.map(([x, y]) => Math.hypot(x - cx, y - cy));
    const meanR = dists.reduce((a, b) => a + b, 0) / dists.length;
    const variance = dists.reduce((a, b) => a + Math.pow(b - meanR, 2), 0) / dists.length;
    const stdDev = Math.sqrt(variance);

    // Circle: radial variance < 14% of mean radius
    if (stdDev / meanR < 0.14) {
      return {
        type: 'circle',
        shapeData: { cx, cy, r: meanR }
      };
    }

    // Rectangle Test: bounding box fill / corner density
    stroke.updateBbox();
    const [minX, minY, maxX, maxY] = stroke.bbox;
    const bw = maxX - minX;
    const bh = maxY - minY;
    if (bw > 25 && bh > 25) {
      const perimeter = 2 * (bw + bh);
      if (Math.abs(totalLen - perimeter) / perimeter < 0.28) {
        return {
          type: 'rect',
          shapeData: { x: minX, y: minY, w: bw, h: bh }
        };
      }
    }
  }

  return null;
}

// 5. Canvas Drawing Engine State Manager with Undo/Redo Snapshots
class CanvasEngine {
  constructor(paintCanvas) {
    this.canvas = paintCanvas;
    this.ctx = paintCanvas.getContext('2d', { alpha: true, desynchronized: true });
    this.strokes = [];
    this.undoStack = [];
    this.redoStack = [];
    this.currentStroke = null;
    this.currentColor = '#ff0000';
    this.brushSize = 6;
  }

  snapshot() {
    // Deep clone state snapshot
    return this.strokes.map(s => s.clone());
  }

  pushState() {
    this.undoStack.push(this.snapshot());
    if (this.undoStack.length > 50) this.undoStack.shift();
    this.redoStack = []; // Invalidate redo stack on new operation
  }

  beginStroke(x, y) {
    this.pushState();
    this.currentStroke = new Stroke(this.currentColor, this.brushSize, 'freehand');
    this.currentStroke.addPoint(x, y);
    this.redraw();
  }

  extendStroke(x, y) {
    if (!this.currentStroke) return;
    const n = this.currentStroke.points.length;
    if (n > 0) {
      const [lastX, lastY] = this.currentStroke.points[n - 1];
      if (Math.hypot(x - lastX, y - lastY) < 2.0) {
        return; // Ignore sub-pixel jitter to prevent stroke array bloat
      }
    }
    this.currentStroke.addPoint(x, y);

    // Incremental draw of latest segment for high performance, no full-canvas clearing during active draw
    const pts = this.currentStroke.points;
    const count = pts.length;
    if (count === 2) {
      this.ctx.save();
      this.ctx.strokeStyle = this.currentStroke.color;
      this.ctx.lineWidth = this.currentStroke.thickness;
      this.ctx.lineCap = 'round';
      this.ctx.lineJoin = 'round';
      this.ctx.beginPath();
      this.ctx.moveTo(pts[0][0], pts[0][1]);
      this.ctx.lineTo(pts[1][0], pts[1][1]);
      this.ctx.stroke();
      this.ctx.restore();
    } else if (count > 2) {
      const pPrev = pts[count - 3];
      const pMid = pts[count - 2];
      const pCurr = pts[count - 1];
      const mid1X = (pPrev[0] + pMid[0]) / 2;
      const mid1Y = (pPrev[1] + pMid[1]) / 2;
      const mid2X = (pMid[0] + pCurr[0]) / 2;
      const mid2Y = (pMid[1] + pCurr[1]) / 2;

      this.ctx.save();
      this.ctx.strokeStyle = this.currentStroke.color;
      this.ctx.lineWidth = this.currentStroke.thickness;
      this.ctx.lineCap = 'round';
      this.ctx.lineJoin = 'round';
      this.ctx.beginPath();
      this.ctx.moveTo(mid1X, mid1Y);
      this.ctx.quadraticCurveTo(pMid[0], pMid[1], mid2X, mid2Y);
      this.ctx.stroke();
      this.ctx.restore();
    }
  }

  endStroke() {
    if (!this.currentStroke) return;
    this.strokes.push(this.currentStroke);
    this.currentStroke = null;
    this.redraw();
  }

  eraseAt(x, y, radius = 28) {
    let erasedAny = false;
    const newStrokesList = [];
    for (const stroke of this.strokes) {
      const res = stroke.eraseAt(x, y, radius);
      if (res.length !== 1 || res[0] !== stroke) {
        erasedAny = true;
      }
      for (let i = 0; i < res.length; i++) {
        newStrokesList.push(res[i]);
      }
    }
    if (erasedAny) {
      // Debounce undo stack pushes during continuous erasing: only push snapshot on erase gesture start or periodically
      const now = performance.now();
      if (!this._lastErasePushTime || (now - this._lastErasePushTime) > 600) {
        this.pushState();
        this._lastErasePushTime = now;
      }
      this.strokes = newStrokesList;
      this.redraw();
    }
    return erasedAny;
  }

  findStrokeAt(x, y, threshold = 40) {
    // Search top-to-bottom
    for (let i = this.strokes.length - 1; i >= 0; i--) {
      if (this.strokes[i].hitsPoint(x, y, threshold)) {
        return this.strokes[i];
      }
    }
    return null;
  }

  undo() {
    if (this.undoStack.length === 0) return false;
    this.redoStack.push(this.snapshot());
    this.strokes = this.undoStack.pop();
    this.redraw();
    return true;
  }

  redo() {
    if (this.redoStack.length === 0) return false;
    this.undoStack.push(this.snapshot());
    this.strokes = this.redoStack.pop();
    this.redraw();
    return true;
  }

  clear() {
    if (this.strokes.length === 0) return;
    this.pushState();
    this.strokes = [];
    this.redraw();
  }

  redraw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const s of this.strokes) {
      s.render(this.ctx);
    }
    if (this.currentStroke) {
      this.currentStroke.render(this.ctx);
    }
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.redraw();
  }
}
