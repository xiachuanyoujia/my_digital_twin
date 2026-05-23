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

mecha.load("/机甲.glb").then(() => {
  console.log("[Main] Mecha controller ready");
});

// 连接 WebSocket
const ws = new PoseWebSocket("ws://localhost:8000/pose/stream");
let _wsMsgCount = 0;
ws.onPoseFrame = (poseFrame) => {
  _wsMsgCount++;
  if (_wsMsgCount <= 5 || _wsMsgCount % 30 === 0) {
    console.log(`[Main] WS msg #${_wsMsgCount} landmarks=${poseFrame.landmarks?.length || 0} backend=${poseFrame.backend || "?"}`);
  }
  mecha.updatePose(poseFrame.landmarks);
  camPreview.updateLandmarks(poseFrame.landmarks);
};
ws.connect();

// 窗口大小调整
window.addEventListener("resize", resize);

// 渲染循环
function animate() {
  requestAnimationFrame(animate);
  render();
}
animate();
