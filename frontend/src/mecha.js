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

// 身体段名称列表 (保持与旧代码兼容)
const ALL_SEGMENTS = [
  "head", "torso", "hip",
  "upperArmL", "lowerArmL", "upperArmR", "lowerArmR",
  "upperLegL", "lowerLegL", "footL",
  "upperLegR", "lowerLegR", "footR",
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

/**
 * 基于模型归一化坐标 (0-1) 的区域分类器
 *
 * 决策逻辑: 先按 Y (高度) 分带, 再按 X (左右) 细分
 *   Y >= 0.75        → head
 *   Y 0.55-0.75      → X<0.35=upperArmL, X>0.65=upperArmR, 其余=torso
 *   Y 0.35-0.55      → X<0.25=lowerArmL, X>0.75=lowerArmR, 其余=torso
 *   Y 0.22-0.35      → hip
 *   Y 0.10-0.22      → X<0.45=upperLegL, X>0.55=upperLegR, 其余=hip
 *   Y 0.03-0.10      → X<0.45=lowerLegL, X>0.55=lowerLegR, 其余=hip
 *   Y < 0.03         → X<0.45=footL, X>0.55=footR, 其余=hip
 */
function classifyMesh(nx, ny, nz) {
  if (ny >= 0.75) {
    return "head";
  }
  if (ny >= 0.55) {
    if (nx < 0.35) return "upperArmL";
    if (nx > 0.65) return "upperArmR";
    return "torso";
  }
  if (ny >= 0.35) {
    if (nx < 0.25) return "lowerArmL";
    if (nx > 0.75) return "lowerArmR";
    return "torso";
  }
  if (ny >= 0.22) {
    return "hip";
  }
  if (ny >= 0.10) {
    if (nx < 0.45) return "upperLegL";
    if (nx > 0.55) return "upperLegR";
    return "hip";
  }
  if (ny >= 0.03) {
    if (nx < 0.45) return "lowerLegL";
    if (nx > 0.55) return "lowerLegR";
    return "hip";
  }
  // ny < 0.03
  if (nx < 0.45) return "footL";
  if (nx > 0.55) return "footR";
  return "hip";
}

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
    console.log(`[Mecha] 包围盒: min=(${min.x.toFixed(2)},${min.y.toFixed(2)},${min.z.toFixed(2)}) max=(${max.x.toFixed(2)},${max.y.toFixed(2)},${max.z.toFixed(2)}) size=(${size.x.toFixed(2)},${size.y.toFixed(2)},${size.z.toFixed(2)})`);

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

    // ---- 4. 区域分类: 将每个 mesh 分配到身体段 ----
    const assigned = {};
    for (const seg of ALL_SEGMENTS) {
      assigned[seg] = [];
    }

    // 诊断: 打印所有 mesh 位置分布
    console.log(`[Mecha] 共 ${meshes.length} 个 mesh, 归一化坐标分布:`);
    const yBands = {};
    for (const m of meshes) {
      const seg = classifyMesh(m.nx, m.ny, m.nz);
      assigned[seg].push(m);
      const yBand = Math.floor(m.ny * 10) / 10;
      if (!yBands[yBand]) yBands[yBand] = { total: 0, left: 0, center: 0, right: 0 };
      yBands[yBand].total++;
      if (m.nx < 0.4) yBands[yBand].left++;
      else if (m.nx > 0.6) yBands[yBand].right++;
      else yBands[yBand].center++;
    }

    // 打印 Y 高度带分布
    console.log("[Mecha] Y高度带分布 (左/中/右):");
    for (const band of Object.keys(yBands).sort((a, b) => b - a)) {
      const b = yBands[band];
      console.log(`  y≈${band}: ${b.total} meshes (左${b.left} 中${b.center} 右${b.right})`);
    }

    // 打印每个段的 mesh 数量
    console.log("[Mecha] 段分配结果:");
    for (const seg of ALL_SEGMENTS) {
      const n = assigned[seg].length;
      if (n > 0) {
        // 打印该段 mesh 的中心范围
        let yMin = 1, yMax = 0, xMin = 1, xMax = 0;
        for (const m of assigned[seg]) {
          if (m.ny < yMin) yMin = m.ny;
          if (m.ny > yMax) yMax = m.ny;
          if (m.nx < xMin) xMin = m.nx;
          if (m.nx > xMax) xMax = m.nx;
        }
        console.log(`  ${seg}: ${n} meshes (x:${xMin.toFixed(2)}-${xMax.toFixed(2)}, y:${yMin.toFixed(2)}-${yMax.toFixed(2)})`);
      } else {
        console.warn(`  ${seg}: 0 meshes ⚠️`);
      }
    }

    // ---- 5. 计算每个段的中心 (模型空间) ----
    const segCenters = {};
    for (const seg of ALL_SEGMENTS) {
      const ms = assigned[seg];
      if (ms.length === 0) {
        // 无 mesh 的段使用推测位置
        segCenters[seg] = this._estimateCenter(seg, assigned, min, size);
        continue;
      }
      const sum = new THREE.Vector3();
      for (const m of ms) sum.add(m.center);
      sum.divideScalar(ms.length);
      segCenters[seg] = sum;
    }

    console.log("[Mecha] 段中心 (模型空间):");
    for (const [name, c] of Object.entries(segCenters)) {
      console.log(`  ${name}: (${c.x.toFixed(3)},${c.y.toFixed(3)},${c.z.toFixed(3)})`);
    }

    // ---- 6. 计算 rest 方向 ----
    const rd = {};
    rd.head  = new THREE.Vector3(0, 1, 0);
    rd.hip   = new THREE.Vector3(0, 1, 0);
    rd.torso = new THREE.Vector3(0, 1, 0);

    // 手臂: 从段中心推算
    const uaL = new THREE.Vector3().subVectors(segCenters.lowerArmL, segCenters.upperArmL);
    rd.upperArmL = uaL.length() > 0.01 ? uaL.normalize() : new THREE.Vector3(-0.7, -0.3, 0);
    const laL = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmL.x - 0.03, segCenters.lowerArmL.y - 0.08, segCenters.lowerArmL.z),
      segCenters.lowerArmL
    );
    rd.lowerArmL = laL.length() > 0.01 ? laL.normalize() : new THREE.Vector3(0, -1, 0);

    const uaR = new THREE.Vector3().subVectors(segCenters.lowerArmR, segCenters.upperArmR);
    rd.upperArmR = uaR.length() > 0.01 ? uaR.normalize() : new THREE.Vector3(0.7, -0.3, 0);
    const laR = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmR.x + 0.03, segCenters.lowerArmR.y - 0.08, segCenters.lowerArmR.z),
      segCenters.lowerArmR
    );
    rd.lowerArmR = laR.length() > 0.01 ? laR.normalize() : new THREE.Vector3(0, -1, 0);

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

    // ---- 8. 将 mesh 从 GLTF 场景迁移到段组 ----
    gltf.scene.updateWorldMatrix(true, true);
    this.root.updateWorldMatrix(true, true);
    for (const seg of ALL_SEGMENTS) {
      const group = this.segments[seg];
      if (!group) continue;
      group.updateWorldMatrix(false, true);
      const invGroup = new THREE.Matrix4().copy(group.matrixWorld).invert();
      for (const m of assigned[seg]) {
        const worldMat = m.mesh.matrixWorld.clone();
        m.mesh.parent?.remove(m.mesh);
        group.add(m.mesh);
        const localMat = new THREE.Matrix4().multiplyMatrices(invGroup, worldMat);
        localMat.decompose(m.mesh.position, m.mesh.quaternion, m.mesh.scale);
      }
    }

    // ---- 9. 关闭阴影 (提升性能) ----
    for (const m of meshes) {
      m.mesh.castShadow = false;
      m.mesh.receiveShadow = false;
    }

    // ---- 10. 最终缩放和位移 ----
    this.root.scale.setScalar(scale);
    this.root.position.y = -min.y * scale;
    this.ready = true;

    const segWithMeshes = Object.values(assigned).filter(ms => ms.length > 0).length;
    console.log(`[Mecha] 加载完成: ${ALL_SEGMENTS.length} 段, ${segWithMeshes} 段有mesh, 共${meshes.length} mesh`);
  }

  /** 为没有 mesh 的段推测位置 (基于相邻段) */
  _estimateCenter(segName, assigned, min, size) {
    // 使用相邻段的位置推算
    const neighbors = {
      upperArmL: "lowerArmL",
      lowerArmL: "upperArmL",
      upperArmR: "lowerArmR",
      lowerArmR: "upperArmR",
      upperLegL: "lowerLegL",
      lowerLegL: "upperLegL",
      footL: "lowerLegL",
      upperLegR: "lowerLegR",
      lowerLegR: "upperLegR",
      footR: "lowerLegR",
    };

    const neighbor = neighbors[segName];
    if (neighbor && assigned[neighbor]?.length > 0) {
      const nc = new THREE.Vector3();
      for (const m of assigned[neighbor]) nc.add(m.center);
      nc.divideScalar(assigned[neighbor].length);

      if (segName.startsWith("foot")) {
        return new THREE.Vector3(nc.x, nc.y - 0.06, nc.z + 0.02);
      }
      if (segName.startsWith("lowerLeg")) {
        return new THREE.Vector3(nc.x, nc.y + 0.06, nc.z);
      }
      if (segName.startsWith("upperLeg")) {
        return new THREE.Vector3(nc.x, nc.y + 0.06, nc.z);
      }
      if (segName.startsWith("upperArm")) {
        return new THREE.Vector3(nc.x, nc.y + 0.08, nc.z);
      }
      if (segName.startsWith("lowerArm")) {
        return new THREE.Vector3(nc.x, nc.y - 0.08, nc.z);
      }
    }

    // 完全推测的兜底位置
    const defaults = {
      upperArmL: new THREE.Vector3(min.x + size.x * 0.2, min.y + size.y * 0.68, min.z + size.z * 0.5),
      lowerArmL: new THREE.Vector3(min.x + size.x * 0.12, min.y + size.y * 0.52, min.z + size.z * 0.5),
      upperArmR: new THREE.Vector3(min.x + size.x * 0.8, min.y + size.y * 0.68, min.z + size.z * 0.5),
      lowerArmR: new THREE.Vector3(min.x + size.x * 0.88, min.y + size.y * 0.52, min.z + size.z * 0.5),
      upperLegL: new THREE.Vector3(min.x + size.x * 0.44, min.y + size.y * 0.18, min.z + size.z * 0.5),
      lowerLegL: new THREE.Vector3(min.x + size.x * 0.44, min.y + size.y * 0.08, min.z + size.z * 0.5),
      footL:     new THREE.Vector3(min.x + size.x * 0.42, min.y + size.y * 0.02, min.z + size.z * 0.45),
      upperLegR: new THREE.Vector3(min.x + size.x * 0.56, min.y + size.y * 0.18, min.z + size.z * 0.5),
      lowerLegR: new THREE.Vector3(min.x + size.x * 0.56, min.y + size.y * 0.08, min.z + size.z * 0.5),
      footR:     new THREE.Vector3(min.x + size.x * 0.58, min.y + size.y * 0.02, min.z + size.z * 0.45),
    };
    return defaults[segName] || new THREE.Vector3(0, 0, 0);
  }

  _buildHierarchy(segCenters) {
    for (const seg of ALL_SEGMENTS) {
      const g = new THREE.Group();
      g.name = seg;
      this.segments[seg] = g;
    }

    const setPivot = (name, pivot) => {
      const g = this.segments[name];
      if (g) g.position.copy(pivot);
    };

    // 关节位置 (使用段中心)
    const shoulderL = segCenters.upperArmL || new THREE.Vector3(-0.3, 1.3, 0);
    const shoulderR = segCenters.upperArmR || new THREE.Vector3(0.3, 1.3, 0);
    const elbowL = segCenters.lowerArmL || new THREE.Vector3(-0.35, 1.0, 0);
    const elbowR = segCenters.lowerArmR || new THREE.Vector3(0.35, 1.0, 0);
    const hipC = segCenters.hip || new THREE.Vector3(0, 0.6, 0);

    // 膝盖: 使用 upperLeg 和 lowerLeg 之间的中点
    const kneeL = segCenters.lowerLegL
      ? new THREE.Vector3(segCenters.lowerLegL.x, segCenters.lowerLegL.y + 0.08, segCenters.lowerLegL.z)
      : new THREE.Vector3(-0.12, 0.35, 0);
    const kneeR = segCenters.lowerLegR
      ? new THREE.Vector3(segCenters.lowerLegR.x, segCenters.lowerLegR.y + 0.08, segCenters.lowerLegR.z)
      : new THREE.Vector3(0.12, 0.35, 0);

    // 脚踝: 使用 foot 段中心
    const ankleL = segCenters.footL
      ? new THREE.Vector3(segCenters.footL.x, segCenters.footL.y + 0.03, segCenters.footL.z)
      : new THREE.Vector3(-0.12, 0.08, 0.05);
    const ankleR = segCenters.footR
      ? new THREE.Vector3(segCenters.footR.x, segCenters.footR.y + 0.03, segCenters.footR.z)
      : new THREE.Vector3(0.12, 0.08, 0.05);

    const headC = segCenters.head || new THREE.Vector3(0, 2.0, 0);

    // 设置 pivot
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
    for (const seg of ALL_SEGMENTS) {
      this.segmentQuats[seg] = new THREE.Quaternion();
    }
  }

  updatePose(landmarks) {
    if (!this._diagCount) this._diagCount = 0;
    this._diagCount++;
    if (this._diagCount <= 5) {
      console.log(`[Mecha] updatePose #${this._diagCount} ready=${this.ready} hasLandmarks=${!!landmarks} len=${landmarks?.length || 0}`);
    }
    if (this._diagCount % 60 === 0) {
      console.log(`[Mecha] updatePose #${this._diagCount} ready=${this.ready} len=${landmarks?.length || 0}`);
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

    // ---- 脚部 ----
    const footDirL = new THREE.Vector3(0, -0.3, 1).normalize();
    this._rotateSegment("footL", footDirL, this.restDirs.footL, alpha * 0.6);
    const footDirR = new THREE.Vector3(0, -0.3, 1).normalize();
    this._rotateSegment("footR", footDirR, this.restDirs.footR, alpha * 0.6);

    // ---- 位移: 整个模型跟随髋部 ----
    const posAlpha = POS_SMOOTH;
    this.root.position.x += (hipMid.x - this.root.position.x) * posAlpha;
    this.root.position.y += (hipMid.y - this.root.position.y) * posAlpha;
    this.root.position.z += (hipMid.z - this.root.position.z) * posAlpha;

    // ---- 诊断日志 (每 30 帧) ----
    if (this._frameCount % 30 === 0) {
      const dirs = {};
      for (const [name, seg] of Object.entries(this.segments)) {
        if (seg.children.length > 0) {
          dirs[name] = true;
        }
      }
      const activeSegs = Object.keys(dirs).join(",");
      const rotNames = Object.entries(this.segmentQuats)
        .filter(([, q]) => Math.abs(q.w) < 0.95)
        .map(([name]) => name)
        .join(",");

      console.log(`[Mecha] f=${this._frameCount} hip=(${hipMid.x.toFixed(2)},${hipMid.y.toFixed(2)},${hipMid.z.toFixed(2)}) ` +
        `torso=(${torsoDir.x.toFixed(2)},${torsoDir.y.toFixed(2)},${torsoDir.z.toFixed(2)}) ` +
        `shMid=(${shMid.x.toFixed(2)},${shMid.y.toFixed(2)},${shMid.z.toFixed(2)}) ` +
        `nose=(${pts.NOSE.x.toFixed(2)},${pts.NOSE.y.toFixed(2)},${pts.NOSE.z.toFixed(2)}) ` +
        `activeSegs=[${activeSegs}]${rotNames ? " rotated:[" + rotNames + "]" : ""}`);

      // 打印手臂方向用于调试
      console.log(`[Mecha]   uaL=(${uaL.x.toFixed(2)},${uaL.y.toFixed(2)},${uaL.z.toFixed(2)}) ` +
        `uaR=(${uaR.x.toFixed(2)},${uaR.y.toFixed(2)},${uaR.z.toFixed(2)}) ` +
        `ulL=(${ulL.x.toFixed(2)},${ulL.y.toFixed(2)},${ulL.z.toFixed(2)}) ` +
        `ulR=(${ulR.x.toFixed(2)},${ulR.y.toFixed(2)},${ulR.z.toFixed(2)})`);
    }
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
