import * as THREE from "three";

export function createScene(container) {
  // 渲染器
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  // 场景
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  scene.fog = new THREE.Fog(0x1a1a2e, 5, 30);

  // 相机
  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );
  camera.position.set(0, 1.2, 3.5);
  camera.lookAt(0, 0.9, 0);

  // 光照
  const ambient = new THREE.AmbientLight(0x404060, 1.2);
  scene.add(ambient);

  const directional = new THREE.DirectionalLight(0xffffff, 1.8);
  directional.position.set(2, 4, 3);
  directional.castShadow = true;
  directional.shadow.mapSize.set(2048, 2048);
  scene.add(directional);

  const backLight = new THREE.DirectionalLight(0x8888ff, 0.8);
  backLight.position.set(-1, 2, -2);
  scene.add(backLight);

  // 地面
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ color: 0x222244, roughness: 0.8 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  // 网格辅助线
  const grid = new THREE.GridHelper(20, 20, 0x444466, 0x222233);
  scene.add(grid);

  // 骨架可视化（占位：33 个关键点的球体）
  const landmarks = [];
  for (let i = 0; i < 33; i++) {
    const geometry = new THREE.SphereGeometry(0.03, 8, 8);
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(i / 33, 0.9, 0.6),
    });
    const sphere = new THREE.Mesh(geometry, material);
    sphere.visible = false;
    scene.add(sphere);
    landmarks.push(sphere);
  }

  // 骨骼连线（MediaPipe 连接关系）
  const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
    [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29],
    [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
  ];

  const boneLines = [];
  for (const [a, b] of POSE_CONNECTIONS) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([0, 0, 0, 0, 0, 0]);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x88ccff, linewidth: 1 })
    );
    line.visible = false;
    scene.add(line);
    boneLines.push({ line, from: a, to: b });
  }

  function updatePose(poseFrame) {
    if (!poseFrame || !poseFrame.landmarks) {
      landmarks.forEach((s) => (s.visible = false));
      boneLines.forEach((b) => (b.line.visible = false));
      return;
    }

    const pts = poseFrame.landmarks;
    landmarks.forEach((sphere, i) => {
      if (i < pts.length) {
        // 坐标系转换：MediaPipe XY (0-1) → 3D 世界空间
        sphere.position.set(
          (pts[i].x - 0.5) * 2.5,
          (1.0 - pts[i].y) * 3.5,
          pts[i].z * 2.5
        );
        sphere.visible = true;
      } else {
        sphere.visible = false;
      }
    });

    boneLines.forEach(({ line, from, to }) => {
      if (from < pts.length && to < pts.length) {
        const posAttr = line.geometry.getAttribute("position");
        const fa = landmarks[from].position;
        const tb = landmarks[to].position;
        posAttr.setXYZ(0, fa.x, fa.y, fa.z);
        posAttr.setXYZ(1, tb.x, tb.y, tb.z);
        posAttr.needsUpdate = true;
        line.visible = true;
      } else {
        line.visible = false;
      }
    });
  }

  function resize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  function render() {
    renderer.render(scene, camera);
  }

  return { scene, camera, renderer, updatePose, resize, render };
}
