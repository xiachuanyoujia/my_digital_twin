import { createScene } from "./scene.js";
import { PoseWebSocket } from "./websocket.js";

const container = document.getElementById("app");
const { updatePose, resize, render } = createScene(container);

// 连接 WebSocket
const ws = new PoseWebSocket("ws://localhost:8000/pose/stream");
ws.onPoseFrame = (poseFrame) => {
  updatePose(poseFrame);
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
