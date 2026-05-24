import * as THREE from "three";

export function createScene(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  scene.fog = new THREE.Fog(0x1a1a2e, 8, 35);

  // 相机: 正面略俯视角, 完整展示 2.5 单位高的机甲模型
  const camera = new THREE.PerspectiveCamera(
    55,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );
  // 模型底部 y=0, 顶部 y≈2.5, 中心 y≈1.25
  // 相机放在模型前方, 略微俯视, 保证完整可见
  camera.position.set(0, 1.5, 5.5);
  camera.lookAt(0, 1.2, 0);

  // 光照 —— 三点光系统
  const ambient = new THREE.AmbientLight(0x808090, 2.0);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
  keyLight.position.set(3, 5, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 20;
  keyLight.shadow.camera.left = -4;
  keyLight.shadow.camera.right = 4;
  keyLight.shadow.camera.top = 4;
  keyLight.shadow.camera.bottom = -1;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xaaaaff, 1.0);
  fillLight.position.set(-2, 2, -1);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
  rimLight.position.set(0, 0.5, -3);
  scene.add(rimLight);

  // 地面参考线: 缩小为 4×4, 与模型尺寸匹配
  const grid = new THREE.GridHelper(4, 8, 0x444466, 0x222244);
  scene.add(grid);

  // 地面圆盘装饰 (半透明, 增强空间感)
  const floorGeo = new THREE.CircleGeometry(1.5, 32);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x333355,
    roughness: 0.8,
    metalness: 0.2,
    transparent: true,
    opacity: 0.3,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.001;
  floor.receiveShadow = true;
  scene.add(floor);

  function resize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  function render() {
    renderer.render(scene, camera);
  }

  return { scene, camera, renderer, resize, render };
}
