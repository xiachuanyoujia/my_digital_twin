import { createScene } from "./scene.js";
import { PoseWebSocket } from "./websocket.js";
import { MechaController } from "./mecha.js";
import { CameraPreview } from "./cameraPreview.js";

const container = document.getElementById("app");
const { scene, camera, renderer, resize, render } = createScene(container);

// 摄像头预览 —— 仅骨骼叠加层，不自行打开摄像头（后端 OpenCV 独占摄像头）
const camPreview = new CameraPreview();
camPreview.startOverlay();

// 加载机甲模型
const mecha = new MechaController();
scene.add(mecha.root);

// Attach 3D debug skeleton to scene for visualizing raw pose data
mecha.attachDebugSkeleton(scene);

mecha.createProcedural();
console.log("[Main] Mecha controller ready (procedural)");

// 连接 WebSocket
const ws = new PoseWebSocket("ws://localhost:8000/pose/stream");
let _wsMsgCount = 0;
let _frameCounter = 0;
let _lastFrameTimestamp = 0;

ws.onPoseFrame = (poseFrame) => {
  _wsMsgCount++;
  _frameCounter++;
  _lastFrameTimestamp = poseFrame.timestamp;

  if (_wsMsgCount <= 5 || _wsMsgCount % 30 === 0) {
    console.log(`[Main] WS msg #${_wsMsgCount} landmarks=${poseFrame.landmarks?.length || 0} backend=${poseFrame.backend || "?"}`);
  }
  mecha.updatePose(poseFrame.landmarks);
  camPreview.updateLandmarks(poseFrame.landmarks);

  // Every 10th frame: report model joint positions to calibration endpoint
  if (_frameCounter % 10 === 0 && _lastFrameTimestamp > 0) {
    _reportJointPositions();
  }
};
ws.connect();

// ---------------------------------------------------------------------------
// Calibration: report frontend 3D joint positions back to backend for
// comparison against detected landmarks
// ---------------------------------------------------------------------------
async function _reportJointPositions() {
  const positions = mecha.getJointWorldPositions();
  if (!positions) return;

  const joints = Object.entries(positions).map(([name, pos]) => ({
    name,
    x: Number(pos.x.toFixed(4)),
    y: Number(pos.y.toFixed(4)),
    z: Number(pos.z.toFixed(4)),
  }));

  try {
    await fetch("http://localhost:8000/calibrate/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timestamp: _lastFrameTimestamp, joints }),
    });
  } catch (e) {
    // Silently ignore network errors during calibration
  }
}

// Poll for updated calibration params every 5 seconds
setInterval(async () => {
  try {
    const resp = await fetch("http://localhost:8000/calibrate/params");
    if (!resp.ok) return;
    const params = await resp.json();
    if (params.sample_count > 0) {
      mecha.setMpToSceneParams(params.scale_x, params.scale_y, params.scale_z);
    }
  } catch (e) {
    // Silently ignore
  }
}, 5000);

// 窗口大小调整
window.addEventListener("resize", resize);

// 渲染循环
function animate() {
  requestAnimationFrame(animate);
  render();
}
animate();
