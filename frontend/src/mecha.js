import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// 13 个基础关键点 (与后端 BASIC_LANDMARK_INDICES 顺序一致)
//   nose(0), L-sh(1), R-sh(2), L-elbow(3), R-elbow(4),
//   L-wrist(5), R-wrist(6), L-hip(7), R-hip(8), L-knee(9), R-knee(10),
//   L-ankle(11), R-ankle(12)
const IDX = {
  NOSE: 0, L_SHOULDER: 1, R_SHOULDER: 2, L_ELBOW: 3, R_ELBOW: 4,
  L_WRIST: 5, R_WRIST: 6, L_HIP: 7, R_HIP: 8, L_KNEE: 9, R_KNEE: 10,
  L_ANKLE: 11, R_ANKLE: 12,
};

function mpToScene(lm) {
  return new THREE.Vector3(
    (lm.x - 0.5) * 2.5,
    (1.0 - lm.y) * 3.5,
    lm.z * 2.5
  );
}

// 13 个身体段 (新增 footL/footR)，归一化模型空间参考点用于 mesh 分类
const SEGMENT_REFS = [
  { name: "head",        nx: 0.50, ny: 0.88, nz: 0.50 },
  { name: "torso",       nx: 0.50, ny: 0.60, nz: 0.50 },
  { name: "hip",         nx: 0.50, ny: 0.30, nz: 0.50 },
  { name: "upperArmL",   nx: 0.20, ny: 0.62, nz: 0.50 },
  { name: "lowerArmL",   nx: 0.12, ny: 0.44, nz: 0.50 },
  { name: "upperArmR",   nx: 0.80, ny: 0.62, nz: 0.50 },
  { name: "lowerArmR",   nx: 0.88, ny: 0.44, nz: 0.50 },
  { name: "upperLegL",   nx: 0.44, ny: 0.20, nz: 0.50 },
  { name: "lowerLegL",   nx: 0.44, ny: 0.08, nz: 0.50 },
  { name: "footL",       nx: 0.42, ny: 0.02, nz: 0.45 },
  { name: "upperLegR",   nx: 0.56, ny: 0.20, nz: 0.50 },
  { name: "lowerLegR",   nx: 0.56, ny: 0.08, nz: 0.50 },
  { name: "footR",       nx: 0.58, ny: 0.02, nz: 0.45 },
];

const SEGMENT_PARENT = {
  head:        "torso",
  torso:       null,
  hip:         null,
  upperArmL:   "torso",
  lowerArmL:   "upperArmL",
  upperArmR:   "torso",
  lowerArmR:   "upperArmR",
  upperLegL:   "hip",
  lowerLegL:   "upperLegL",
  footL:       "lowerLegL",
  upperLegR:   "hip",
  lowerLegR:   "upperLegR",
  footR:       "lowerLegR",
};

const SMOOTH_ALPHA = 0.55;
const POS_SMOOTH = 0.65;

export class MechaController {
  constructor() {
    this.root = new THREE.Group();
    this.segments = {};
    this.restDirs = {};
    this.ready = false;
  }

  async load(url) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);

    // ---- 1. 模型空间包围盒 ----
    const bbox = new THREE.Box3().setFromObject(gltf.scene);
    const min = bbox.min.clone();
    const max = bbox.max.clone();
    const size = new THREE.Vector3().subVectors(max, min);

    // ---- 2. 缩放至约 2.5 单位高 ----
    const targetHeight = 2.5;
    const scale = targetHeight / size.y;

    // ---- 3. 收集所有 mesh 及归一化中心 ----
    const meshes = [];
    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        const cb = new THREE.Box3().setFromObject(child);
        const center = new THREE.Vector3();
        cb.getCenter(center);
        const nx = (center.x - min.x) / size.x;
        const ny = (center.y - min.y) / size.y;
        const nz = (center.z - min.z) / size.z;
        meshes.push({ mesh: child, nx, ny, nz, center: center.clone() });
      }
    });

    // ---- 4. 将 mesh 分配到最近的段 ----
    const assigned = {};
    for (const seg of SEGMENT_REFS) {
      assigned[seg.name] = [];
    }
    // 统计每个段的 mesh 数量
    for (const m of meshes) {
      let best = null;
      let bestDist = Infinity;
      for (const seg of SEGMENT_REFS) {
        const dx = m.nx - seg.nx;
        const dy = m.ny - seg.ny;
        const dz = m.nz - seg.nz;
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bestDist) { best = seg.name; bestDist = d; }
      }
      assigned[best].push(m);
    }

    // 日志：显示每个段分配了多少 mesh
    for (const [name, ms] of Object.entries(assigned)) {
      if (ms.length > 0) {
        console.log(`[Mecha] ${name}: ${ms.length} meshes`);
      }
    }

    // ---- 5. 计算每个段的中心 ----
    const segCenters = {};
    for (const [name, ms] of Object.entries(assigned)) {
      if (ms.length === 0) {
        segCenters[name] = new THREE.Vector3(0, 0, 0);
        continue;
      }
      const sum = new THREE.Vector3();
      for (const m of ms) sum.add(m.center);
      sum.divideScalar(ms.length);
      segCenters[name] = sum;
    }

    // ---- 6. 计算 rest 方向 ----
    const rd = {};
    rd.head       = new THREE.Vector3(0, 1, 0);
    rd.hip        = new THREE.Vector3(0, 1, 0);
    rd.torso      = new THREE.Vector3(0, 1, 0);

    // 手臂: 从段中心推算 rest 方向
    const uaL = new THREE.Vector3().subVectors(segCenters.lowerArmL, segCenters.upperArmL);
    rd.upperArmL  = uaL.length() > 0.01 ? uaL.normalize() : new THREE.Vector3(-0.7, -0.3, 0);
    const laL = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmL.x - 0.03, segCenters.lowerArmL.y - 0.08, segCenters.lowerArmL.z),
      segCenters.lowerArmL
    );
    rd.lowerArmL  = laL.length() > 0.01 ? laL.normalize() : new THREE.Vector3(0, -1, 0);

    const uaR = new THREE.Vector3().subVectors(segCenters.lowerArmR, segCenters.upperArmR);
    rd.upperArmR  = uaR.length() > 0.01 ? uaR.normalize() : new THREE.Vector3(0.7, -0.3, 0);
    const laR = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmR.x + 0.03, segCenters.lowerArmR.y - 0.08, segCenters.lowerArmR.z),
      segCenters.lowerArmR
    );
    rd.lowerArmR  = laR.length() > 0.01 ? laR.normalize() : new THREE.Vector3(0, -1, 0);

    // 腿: 默认朝下
    rd.upperLegL  = new THREE.Vector3(0, -1, 0);
    rd.lowerLegL  = new THREE.Vector3(0, -1, 0);
    rd.footL      = new THREE.Vector3(0, 0, 1);
    rd.upperLegR  = new THREE.Vector3(0, -1, 0);
    rd.lowerLegR  = new THREE.Vector3(0, -1, 0);
    rd.footR      = new THREE.Vector3(0, 0, 1);

    this.restDirs = rd;

    // ---- 7. 创建段层级 ----
    this._buildHierarchy(segCenters);

    // ---- 8. 将 mesh 从 GLTF 场景迁移到段组, 保留完整世界变换 ----
    gltf.scene.updateWorldMatrix(true, true);
    this.root.updateWorldMatrix(true, true);
    for (const [name, ms] of Object.entries(assigned)) {
      const group = this.segments[name];
      if (!group) continue;
      group.updateWorldMatrix(false, true);
      const invGroup = new THREE.Matrix4().copy(group.matrixWorld).invert();
      for (const m of ms) {
        // 保存网格的世界矩阵
        const worldMat = m.mesh.matrixWorld.clone();
        // 从旧父级移除，挂到段组
        m.mesh.parent?.remove(m.mesh);
        group.add(m.mesh);
        // 用逆矩阵算出局部变换，保留位置+旋转+缩放
        const localMat = new THREE.Matrix4().multiplyMatrices(invGroup, worldMat);
        localMat.decompose(m.mesh.position, m.mesh.quaternion, m.mesh.scale);
      }
    }

    // ---- 9. 关闭机甲 mesh 的阴影 (动画角色不需要, 大幅提升性能) ----
    for (const m of meshes) {
      m.mesh.castShadow = false;
      m.mesh.receiveShadow = false;
    }

    // ---- 10. 最终缩放和位移 ----
    this.root.scale.setScalar(scale);
    this.root.position.y = -min.y * scale;
    this.ready = true;
    console.log("[Mecha] Model loaded, segments:", Object.keys(this.segments).length,
                "meshes:", meshes.length);
  }

  _buildHierarchy(segCenters) {
    for (const seg of SEGMENT_REFS) {
      const g = new THREE.Group();
      g.name = seg.name;
      this.segments[seg.name] = g;
    }

    const setPivot = (name, pivot) => {
      const g = this.segments[name];
      if (g) g.position.copy(pivot);
    };

    const shoulderL = segCenters.upperArmL || new THREE.Vector3(-0.3, 1.3, 0);
    const shoulderR = segCenters.upperArmR || new THREE.Vector3(0.3, 1.3, 0);
    const elbowL = segCenters.lowerArmL || new THREE.Vector3(-0.35, 1.0, 0);
    const elbowR = segCenters.lowerArmR || new THREE.Vector3(0.35, 1.0, 0);
    const hipC = segCenters.hip || new THREE.Vector3(0, 0.6, 0);
    const kneeL = segCenters.lowerLegL
      ? new THREE.Vector3(segCenters.lowerLegL.x, segCenters.lowerLegL.y + 0.1, segCenters.lowerLegL.z)
      : new THREE.Vector3(-0.12, 0.35, 0);
    const kneeR = segCenters.lowerLegR
      ? new THREE.Vector3(segCenters.lowerLegR.x, segCenters.lowerLegR.y + 0.1, segCenters.lowerLegR.z)
      : new THREE.Vector3(0.12, 0.35, 0);
    const ankleL = segCenters.footL
      ? new THREE.Vector3(segCenters.footL.x, segCenters.footL.y + 0.03, segCenters.footL.z)
      : new THREE.Vector3(-0.12, 0.08, 0.05);
    const ankleR = segCenters.footR
      ? new THREE.Vector3(segCenters.footR.x, segCenters.footR.y + 0.03, segCenters.footR.z)
      : new THREE.Vector3(0.12, 0.08, 0.05);
    const headC = segCenters.head || new THREE.Vector3(0, 2.0, 0);

    setPivot("hip", hipC);
    setPivot("torso", hipC);
    setPivot("head", headC);

    setPivot("upperArmL", shoulderL);
    setPivot("lowerArmL", elbowL);
    setPivot("upperArmR", shoulderR);
    setPivot("lowerArmR", elbowR);

    setPivot("upperLegL", new THREE.Vector3(shoulderL.x * 0.4, hipC.y, hipC.z));
    setPivot("lowerLegL", kneeL);
    setPivot("footL", ankleL);
    setPivot("upperLegR", new THREE.Vector3(shoulderR.x * 0.4, hipC.y, hipC.z));
    setPivot("lowerLegR", kneeR);
    setPivot("footR", ankleR);

    // ---- 构建父子层级 ----
    this.root.add(this.segments.hip);
    this.root.add(this.segments.torso);

    for (const [name, parentName] of Object.entries(SEGMENT_PARENT)) {
      if (!parentName) continue;
      const parent = this.segments[parentName];
      const child = this.segments[name];
      if (!parent || !child) continue;
      parent.add(child);
      const rel = new THREE.Vector3().subVectors(child.position, parent.position);
      child.position.copy(rel);
    }

    this.segmentQuats = {};
    for (const seg of SEGMENT_REFS) {
      this.segmentQuats[seg.name] = new THREE.Quaternion();
    }
  }

  updatePose(landmarks) {
    // 诊断: 记录调用次数和状态
    if (!this._diagCount) this._diagCount = 0;
    this._diagCount++;
    if (this._diagCount <= 3 || this._diagCount % 60 === 0) {
      console.log(`[Mecha] updatePose #${this._diagCount} ready=${this.ready} hasLandmarks=${!!landmarks} len=${landmarks?.length || 0}`);
    }

    if (!this.ready || !landmarks || landmarks.length < 13) return;

    if (!this._frameCount) this._frameCount = 0;
    this._frameCount++;

    const lm = landmarks;
    const pts = {};
    for (const [key, idx] of Object.entries(IDX)) {
      pts[key] = mpToScene(lm[idx]);
    }

    const alpha = SMOOTH_ALPHA;

    // ---- 躯干 (髋→肩) ----
    const hipMid = new THREE.Vector3().addVectors(pts.L_HIP, pts.R_HIP).multiplyScalar(0.5);
    const shMid = new THREE.Vector3().addVectors(pts.L_SHOULDER, pts.R_SHOULDER).multiplyScalar(0.5);
    const torsoDir = new THREE.Vector3().subVectors(shMid, hipMid).normalize();
    if (torsoDir.length() > 0.01) {
      this._rotateSegment("torso", torsoDir, this.restDirs.torso, alpha);
      this._rotateSegment("hip", torsoDir, this.restDirs.hip, alpha);
    }

    // 每 30 帧打印一次调试信息
    if (this._frameCount % 30 === 0) {
      const count = this._frameCount;
      const hipStr = `(${hipMid.x.toFixed(2)},${hipMid.y.toFixed(2)},${hipMid.z.toFixed(2)})`;
      const torsoStr = `(${torsoDir.x.toFixed(2)},${torsoDir.y.toFixed(2)},${torsoDir.z.toFixed(2)})`;
      const rotNames = Object.entries(this.segmentQuats)
        .filter(([, q]) => Math.abs(q.w) < 0.999)
        .map(([name]) => name)
        .join(",");
      console.log(`[Mecha] f=${count} hip=${hipStr} torso=${torsoStr} root=(${this.root.position.x.toFixed(2)},${this.root.position.y.toFixed(2)},${this.root.position.z.toFixed(2)})${rotNames ? " rot:" + rotNames : ""}`);
    }

    // ---- 头部 (鼻→肩中点) ----
    const headDir = new THREE.Vector3().subVectors(pts.NOSE, shMid).normalize();
    if (headDir.length() > 0.01) {
      this._rotateSegment("head", headDir, this.restDirs.head, alpha);
    }

    // ---- 上臂 (肩→肘) ----
    const uaL = new THREE.Vector3().subVectors(pts.L_ELBOW, pts.L_SHOULDER);
    if (uaL.length() > 0.01) this._rotateSegment("upperArmL", uaL.normalize(), this.restDirs.upperArmL, alpha);
    const uaR = new THREE.Vector3().subVectors(pts.R_ELBOW, pts.R_SHOULDER);
    if (uaR.length() > 0.01) this._rotateSegment("upperArmR", uaR.normalize(), this.restDirs.upperArmR, alpha);

    // ---- 前臂 (肘→腕) ----
    const laL = new THREE.Vector3().subVectors(pts.L_WRIST, pts.L_ELBOW);
    if (laL.length() > 0.01) this._rotateSegment("lowerArmL", laL.normalize(), this.restDirs.lowerArmL, alpha);
    const laR = new THREE.Vector3().subVectors(pts.R_WRIST, pts.R_ELBOW);
    if (laR.length() > 0.01) this._rotateSegment("lowerArmR", laR.normalize(), this.restDirs.lowerArmR, alpha);

    // ---- 大腿 (髋→膝) ----
    const ulL = new THREE.Vector3().subVectors(pts.L_KNEE, pts.L_HIP);
    if (ulL.length() > 0.01) this._rotateSegment("upperLegL", ulL.normalize(), this.restDirs.upperLegL, alpha);
    const ulR = new THREE.Vector3().subVectors(pts.R_KNEE, pts.R_HIP);
    if (ulR.length() > 0.01) this._rotateSegment("upperLegR", ulR.normalize(), this.restDirs.upperLegR, alpha);

    // ---- 小腿 (膝→踝) ----
    const llL = new THREE.Vector3().subVectors(pts.L_ANKLE, pts.L_KNEE);
    if (llL.length() > 0.01) this._rotateSegment("lowerLegL", llL.normalize(), this.restDirs.lowerLegL, alpha);
    const llR = new THREE.Vector3().subVectors(pts.R_ANKLE, pts.R_KNEE);
    if (llR.length() > 0.01) this._rotateSegment("lowerLegR", llR.normalize(), this.restDirs.lowerLegR, alpha);

    // ---- 脚部 (踝→脚尖) ----
    const footDirL = new THREE.Vector3(0, -0.3, 1).normalize();
    this._rotateSegment("footL", footDirL, this.restDirs.footL, alpha * 0.6);
    const footDirR = new THREE.Vector3(0, -0.3, 1).normalize();
    this._rotateSegment("footR", footDirR, this.restDirs.footR, alpha * 0.6);

    // ---- 位移: 整个模型跟随髋部 (X/Y/Z 三维) ----
    const posAlpha = POS_SMOOTH;
    this.root.position.x += (hipMid.x - this.root.position.x) * posAlpha;
    this.root.position.y += (hipMid.y - this.root.position.y) * posAlpha;
    this.root.position.z += (hipMid.z - this.root.position.z) * posAlpha;
  }

  _rotateSegment(name, currentDir, restDir, alpha) {
    const g = this.segments[name];
    if (!g || currentDir.length() < 0.001) return;

    const from = restDir.clone().normalize();
    const to = currentDir.clone().normalize();

    if (Math.abs(from.dot(to)) > 0.9999) return;

    const targetQ = new THREE.Quaternion().setFromUnitVectors(from, to);

    const prevQ = this.segmentQuats[name] || new THREE.Quaternion();
    const smoothQ = new THREE.Quaternion().slerpQuaternions(prevQ, targetQ, alpha);
    this.segmentQuats[name] = smoothQ;

    g.quaternion.copy(smoothQ);
  }
}
