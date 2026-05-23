/**
 * 摄像头实时预览 —— 显示摄像头画面 + 叠加识别到的骨骼关键点
 */
const SKELETON_CONNECTIONS = [
  [0, 1], [0, 2],   // 鼻→双肩
  [1, 3], [3, 5],   // 左肩→左肘→左腕
  [2, 4], [4, 6],   // 右肩→右肘→右腕
  [1, 2],            // 肩→肩
  [1, 7], [2, 8],   // 肩→髋
  [7, 8],            // 髋→髋
  [7, 9], [9, 11],  // 左髋→左膝→左踝
  [8, 10], [10, 12], // 右髋→右膝→右踝
];

const JOINT_COLORS = [
  "#00ff88", // 0  nose
  "#ff4444", "#ff4444", // 1-2 shoulders
  "#ff8844", "#ff8844", // 3-4 elbows
  "#ffcc00", "#ffcc00", // 5-6 wrists
  "#44aaff", "#44aaff", // 7-8 hips
  "#4488ff", "#4488ff", // 9-10 knees
  "#44ccff", "#44ccff", // 11-12 ankles
];

const PREVIEW_W = 280;
const PREVIEW_H = 210;

export class CameraPreview {
  constructor() {
    this.video = document.getElementById("cam-video");
    this.canvas = document.getElementById("cam-overlay");
    this.ctx = this.canvas.getContext("2d");
    this.stream = null;
    this.landmarks = null;
    this.active = false;

    // Always set canvas size (no longer depends on video metadata)
    this.canvas.width = PREVIEW_W;
    this.canvas.height = PREVIEW_H;
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      this.video.srcObject = this.stream;
      this.video.style.display = "block";
      console.log("[Camera] 预览已启动");
    } catch (e) {
      console.warn("[Camera] 摄像头不可用:", e.message);
    }
    this.active = true;
    this._drawLoop();
  }

  startOverlay() {
    // 不打开摄像头, 仅渲染骨骼叠加层 (后端独占摄像头)
    this.active = true;
    this._drawLoop();
    console.log("[Camera] 骨骼叠加层已启动 (无摄像头)");
  }

  updateLandmarks(landmarks) {
    this.landmarks = landmarks;
  }

  _drawLoop() {
    if (!this.active) return;
    requestAnimationFrame(() => this._drawLoop());

    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (w === 0 || h === 0) return;

    // Dark background (useful when no video feed)
    ctx.fillStyle = "rgba(0,0,0,0.85)";
    ctx.fillRect(0, 0, w, h);

    if (!this.landmarks || this.landmarks.length < 13) return;

    const pts = this.landmarks.map(lm => ({
      x: lm.x * w,
      y: lm.y * h,
      v: lm.visibility ?? 1.0,
    }));

    // 骨骼连线
    ctx.lineWidth = 2;
    for (const [a, b] of SKELETON_CONNECTIONS) {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) continue;
      const avgV = (pa.v + pb.v) / 2;
      if (avgV < 0.3) continue;

      ctx.strokeStyle = `rgba(0, 255, 136, ${avgV.toFixed(2)})`;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    // 关键点
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p.v < 0.3) continue;

      const color = JOINT_COLORS[i] || "#ffffff";
      ctx.fillStyle = color;
      ctx.shadowColor = "rgba(0,0,0,0.6)";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  stop() {
    this.active = false;
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  }
}
