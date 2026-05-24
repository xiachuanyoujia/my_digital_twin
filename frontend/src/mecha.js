import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// 13 basic keypoints (consistent with backend BASIC_LANDMARK_INDICES order)
//   nose(0), L-sh(1), R-sh(2), L-elbow(3), R-elbow(4),
//   L-wrist(5), R-wrist(6), L-hip(7), R-hip(8), L-knee(9), R-knee(10),
//   L-ankle(11), R-ankle(12)
const IDX = {
  NOSE: 0, L_SHOULDER: 1, R_SHOULDER: 2, L_ELBOW: 3, R_ELBOW: 4,
  L_WRIST: 5, R_WRIST: 6, L_HIP: 7, R_HIP: 8, L_KNEE: 9, R_KNEE: 10,
  L_ANKLE: 11, R_ANKLE: 12,
};

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

const SMOOTH_ALPHA = 0.60;
const POS_SMOOTH = 0.75;

// ---------------------------------------------------------------------------
// Logging utility
// ---------------------------------------------------------------------------
const LOG_PREFIX = "[Mecha]";
let _logSeq = 0;
function log(level, msg) {
  const seq = String(++_logSeq).padStart(4, "0");
  const ts = (performance.now() / 1000).toFixed(2);
  console[level](`${LOG_PREFIX} [${seq}|${ts}s] ${msg}`);
}

// ---------------------------------------------------------------------------
// Coordinate conversion (params updated by calibration loop)
// ---------------------------------------------------------------------------
let _mpScale = { x: 2.5, y: 3.5, z: 2.5 };

function mpToScene(lm) {
  return new THREE.Vector3(
    (lm.x - 0.5) * _mpScale.x,
    (1.0 - lm.y) * _mpScale.y,
    lm.z * _mpScale.z
  );
}

// ---------------------------------------------------------------------------
// Adaptive vertex-level body part classifier
// Thresholds are computed from actual vertex distribution — not hard-coded
// ---------------------------------------------------------------------------
let _CLASSIFY_THRESHOLDS = null;

function computeAdaptiveThresholds(vertexSamples) {
  // vertexSamples: array of {nx, ny, nz}
  const n = vertexSamples.length;
  const ys = vertexSamples.map(v => v.ny).sort((a, b) => a - b);
  const xs = vertexSamples.map(v => v.nx).sort((a, b) => a - b);

  const yPct = (p) => ys[Math.floor(n * p)];
  const xPct = (p) => xs[Math.floor(n * p)];

  const t = {
    headY: yPct(0.85),
    upperY: yPct(0.55),
    midY: yPct(0.32),
    hipY: yPct(0.18),
    upperLegY: yPct(0.08),
    lowerLegY: yPct(0.03),
    // X separation: use quartiles for arm/leg lateral split
    leftArmX: xPct(0.30),
    rightArmX: xPct(0.70),
    leftLegX: xPct(0.38),
    rightLegX: xPct(0.62),
  };

  log("info", `自适应阈值: headY>=${t.headY.toFixed(2)} upperY>=${t.upperY.toFixed(2)} midY>=${t.midY.toFixed(2)} hipY>=${t.hipY.toFixed(2)} upperLegY>=${t.upperLegY.toFixed(2)} lowerLegY>=${t.lowerLegY.toFixed(2)}`);
  log("info", `  水平分割: leftArm<${t.leftArmX.toFixed(2)} rightArm>${t.rightArmX.toFixed(2)} leftLeg<${t.leftLegX.toFixed(2)} rightLeg>${t.rightLegX.toFixed(2)}`);
  return t;
}

function classifyVertex(nx, ny, nz) {
  const t = _CLASSIFY_THRESHOLDS;
  if (!t) return "torso";

  if (ny >= t.headY) return "head";

  if (ny >= t.upperY) {
    if (nx < t.leftArmX) return "upperArmL";
    if (nx > t.rightArmX) return "upperArmR";
    return "torso";
  }
  if (ny >= t.midY) {
    if (nx < t.leftArmX) return "lowerArmL";
    if (nx > t.rightArmX) return "lowerArmR";
    return "torso";
  }
  if (ny >= t.hipY) return "hip";

  if (ny >= t.upperLegY) {
    if (nx < t.leftLegX) return "upperLegL";
    if (nx > t.rightLegX) return "upperLegR";
    return "hip";
  }
  if (ny >= t.lowerLegY) {
    if (nx < t.leftLegX) return "lowerLegL";
    if (nx > t.rightLegX) return "lowerLegR";
    return "hip";
  }
  // Feet
  if (nx < t.leftLegX) return "footL";
  if (nx > t.rightLegX) return "footR";
  return "hip";
}

// ---------------------------------------------------------------------------
// Debug skeleton: 3D spheres + lines showing raw pose data
// ---------------------------------------------------------------------------
class DebugSkeleton {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.root.name = "DebugSkeleton";
    this.root.visible = false;
    scene.add(this.root);

    // Joint spheres
    const sphereGeo = new THREE.SphereGeometry(0.04, 8, 8);
    this.jointMeshes = [];
    const colors = [
      0x00ff88, // nose
      0xff4444, 0xff4444, // shoulders
      0xff8844, 0xff8844, // elbows
      0xffcc00, 0xffcc00, // wrists
      0x44aaff, 0x44aaff, // hips
      0x4488ff, 0x4488ff, // knees
      0x44ccff, 0x44ccff, // ankles
    ];
    for (let i = 0; i < 13; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: colors[i] || 0xffffff });
      const sphere = new THREE.Mesh(sphereGeo, mat);
      sphere.visible = false;
      this.root.add(sphere);
      this.jointMeshes.push(sphere);
    }

    // Bone lines
    this.boneConnections = [
      [0, 1], [0, 2], [1, 2],       // nose→shoulders, shoulder→shoulder
      [1, 3], [3, 5],                 // L arm
      [2, 4], [4, 6],                 // R arm
      [1, 7], [2, 8], [7, 8],        // shoulder→hip, hip→hip
      [7, 9], [9, 11],                // L leg
      [8, 10], [10, 12],              // R leg
    ];
    this.boneLines = [];
    const lineMat = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 1 });
    for (let i = 0; i < this.boneConnections.length; i++) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(), new THREE.Vector3()
      ]);
      const line = new THREE.Line(lineGeo, lineMat);
      line.visible = false;
      this.root.add(line);
      this.boneLines.push(line);
    }
  }

  update(landmarks) {
    if (!landmarks || landmarks.length < 13) {
      this.root.visible = false;
      return;
    }
    this.root.visible = true;

    const pts = landmarks.map(lm => mpToScene(lm));

    // Update joint spheres
    for (let i = 0; i < 13; i++) {
      const s = this.jointMeshes[i];
      s.position.copy(pts[i]);
      s.visible = true;
    }

    // Update bone lines
    for (let i = 0; i < this.boneConnections.length; i++) {
      const [a, b] = this.boneConnections[i];
      const line = this.boneLines[i];
      const positions = line.geometry.attributes.position;
      positions.setXYZ(0, pts[a].x, pts[a].y, pts[a].z);
      positions.setXYZ(1, pts[b].x, pts[b].y, pts[b].z);
      positions.needsUpdate = true;
      line.visible = true;
    }
  }
}

// ---------------------------------------------------------------------------
// MechaController
// ---------------------------------------------------------------------------
export class MechaController {
  constructor() {
    this.root = new THREE.Group();
    this.segments = {};
    this.restDirs = {};
    this.ready = false;
    this.debugSkeleton = null;
  }

  /**
   * Attach a debug skeleton to the same scene after load completes.
   */
  attachDebugSkeleton(scene) {
    if (!this.debugSkeleton) {
      this.debugSkeleton = new DebugSkeleton(scene);
    }
  }

  /**
   * Update mpToScene() scale parameters (called by calibration loop).
   */
  setMpToSceneParams(sx, sy, sz) {
    // Reject extreme aspect ratios that break model proportions.
    // No single axis may exceed 2x any other axis.
    const vals = [sx, sy, sz];
    const vMax = Math.max(...vals);
    const vMin = Math.min(...vals);
    if (vMin < 0.5 || vMax / vMin > 2.0) {
      if (!this._warnedBadParams) {
        console.warn(`[Mecha] Rejecting bad mpToScene params: (${sx}, ${sy}, ${sz}) ratio=${(vMax/vMin).toFixed(2)}`);
        this._warnedBadParams = true;
      }
      return;
    }
    this._warnedBadParams = false;
    _mpScale.x = sx;
    _mpScale.y = sy;
    _mpScale.z = sz;
    console.log(`[Mecha] mpToScene params updated: (${sx}, ${sy}, ${sz})`);
  }

  /**
   * Compute world-space positions of all 13 key joints by traversing the
   * segment hierarchy. Returns {joint_name: {x, y, z}} or null if not ready.
   */
  getJointWorldPositions() {
    if (!this.ready) return null;

    // Ensure world matrices are up to date before querying positions
    this.root.updateWorldMatrix(true, true);

    const v = new THREE.Vector3();
    const positions = {};

    // Joints that map directly to segment group world positions
    const jointFromSegment = {
      left_shoulder: "upperArmL",
      right_shoulder: "upperArmR",
      left_elbow: "lowerArmL",
      right_elbow: "lowerArmR",
      left_hip: "upperLegL",
      right_hip: "upperLegR",
      left_knee: "lowerLegL",
      right_knee: "lowerLegR",
    };

    for (const [joint, seg] of Object.entries(jointFromSegment)) {
      const g = this.segments[seg];
      if (g) {
        g.getWorldPosition(v);
        positions[joint] = { x: v.x, y: v.y, z: v.z };
      }
    }

    // Ankles from foot segments
    for (const side of ["L", "R"]) {
      const footSeg = this.segments[`foot${side}`];
      if (footSeg) {
        footSeg.getWorldPosition(v);
        positions[`${side === "L" ? "left" : "right"}_ankle`] = { x: v.x, y: v.y + 0.03, z: v.z };
      }
    }

    // Wrists from hand meshes (nested in lowerArm segments)
    if (this._handMeshes) {
      if (this._handMeshes.handL) {
        this._handMeshes.handL.getWorldPosition(v);
        positions.left_wrist = { x: v.x, y: v.y, z: v.z };
      }
      if (this._handMeshes.handR) {
        this._handMeshes.handR.getWorldPosition(v);
        positions.right_wrist = { x: v.x, y: v.y, z: v.z };
      }
    }

    // Nose from head segment (sphere center + radius offset)
    const headG = this.segments.head;
    if (headG) {
      headG.getWorldPosition(v);
      positions.nose = { x: v.x, y: v.y + 0.12, z: v.z + 0.04 };
    }

    return positions;
  }

  /**
   * Build a procedural humanoid model from Three.js primitives.
   * Each body segment is created as a separate mesh with correct pivot placement
   * — no GLB loading or vertex splitting needed.
   */
  createProcedural() {
    // Material palette
    const matHead = new THREE.MeshStandardMaterial({ color: 0xffcc88, roughness: 0.4, metalness: 0.2 });
    const matTorso = new THREE.MeshStandardMaterial({ color: 0x4488cc, roughness: 0.5, metalness: 0.1 });
    const matArm = new THREE.MeshStandardMaterial({ color: 0xff6644, roughness: 0.5, metalness: 0.3 });
    const matArmL = new THREE.MeshStandardMaterial({ color: 0xff8855, roughness: 0.5, metalness: 0.3 });
    const matLeg = new THREE.MeshStandardMaterial({ color: 0x4466aa, roughness: 0.5, metalness: 0.3 });
    const matLegL = new THREE.MeshStandardMaterial({ color: 0x5588cc, roughness: 0.5, metalness: 0.3 });
    const matFoot = new THREE.MeshStandardMaterial({ color: 0x333344, roughness: 0.7, metalness: 0.1 });
    const matHip = new THREE.MeshStandardMaterial({ color: 0x336699, roughness: 0.5, metalness: 0.2 });
    const matHand = new THREE.MeshStandardMaterial({ color: 0xffcc88, roughness: 0.4, metalness: 0.1 });

    // ---- Segment centers (model space, y-up, 2.5 units tall) ----
    const C = {
      head:        new THREE.Vector3(0, 2.15, 0),
      torso:       new THREE.Vector3(0, 1.30, 0),
      hip:         new THREE.Vector3(0, 0.85, 0),
      upperArmL:   new THREE.Vector3(-0.42, 1.55, 0),
      lowerArmL:   new THREE.Vector3(-0.42, 1.10, 0),
      upperArmR:   new THREE.Vector3(0.42, 1.55, 0),
      lowerArmR:   new THREE.Vector3(0.42, 1.10, 0),
      upperLegL:   new THREE.Vector3(-0.12, 0.62, 0),
      lowerLegL:   new THREE.Vector3(-0.12, 0.30, 0),
      footL:       new THREE.Vector3(-0.12, 0.04, 0.05),
      upperLegR:   new THREE.Vector3(0.12, 0.62, 0),
      lowerLegR:   new THREE.Vector3(0.12, 0.30, 0),
      footR:       new THREE.Vector3(0.12, 0.04, 0.05),
    };

    // ---- Rest directions (same as GLTF path) ----
    this.restDirs = {
      head: new THREE.Vector3(0, 1, 0),
      torso: new THREE.Vector3(0, 1, 0),
      hip: new THREE.Vector3(0, 1, 0),
      upperArmL: new THREE.Vector3(0, -1, 0),
      lowerArmL: new THREE.Vector3(0, -1, 0),
      upperArmR: new THREE.Vector3(0, -1, 0),
      lowerArmR: new THREE.Vector3(0, -1, 0),
      upperLegL: new THREE.Vector3(0, -1, 0),
      lowerLegL: new THREE.Vector3(0, -1, 0),
      footL: new THREE.Vector3(0, 0, 1),
      upperLegR: new THREE.Vector3(0, -1, 0),
      lowerLegR: new THREE.Vector3(0, -1, 0),
      footR: new THREE.Vector3(0, 0, 1),
    };

    // ---- Build geometry for each segment ----
    const geometries = {};

    // Head: sphere
    geometries.head = (() => {
      const g = new THREE.Group();
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 12), matHead);
      sphere.position.y = 0;
      sphere.castShadow = true;
      g.add(sphere);
      // Eyes
      const eyeGeo = new THREE.SphereGeometry(0.03, 6, 6);
      const eyeMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
      [-0.06, 0.06].forEach(ex => {
        const eye = new THREE.Mesh(eyeGeo, eyeMat);
        eye.position.set(ex, 0.04, 0.16);
        g.add(eye);
      });
      return g;
    })();

    // Torso: tapered box
    geometries.torso = (() => {
      const g = new THREE.Group();
      const h = 0.70;
      const geo = new THREE.CylinderGeometry(0.22, 0.18, h, 8, 4);
      const mesh = new THREE.Mesh(geo, matTorso);
      mesh.position.y = 0;
      mesh.castShadow = true;
      g.add(mesh);
      return g;
    })();

    // Hip: flattened box
    geometries.hip = (() => {
      const g = new THREE.Group();
      const geo = new THREE.BoxGeometry(0.30, 0.12, 0.18);
      const mesh = new THREE.Mesh(geo, matHip);
      mesh.castShadow = true;
      g.add(mesh);
      return g;
    })();

    // Helper: create limb (cylinder between two joint centers)
    // Returns { group, mesh, restLength }
    const makeLimb = (start, end, radius, material) => {
      const g = new THREE.Group();
      const dir = new THREE.Vector3().subVectors(end, start);
      const len = dir.length();
      if (len < 0.001) return { group: g, mesh: null, restLength: 0 };
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const geo = new THREE.CylinderGeometry(radius, radius, 1, 8, 4);
      const mesh = new THREE.Mesh(geo, material);
      mesh.position.copy(mid);
      mesh.scale.y = len;  // cylinder height=1, scale to desired length
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      mesh.quaternion.copy(quat);
      mesh.castShadow = true;
      g.add(mesh);
      return { group: g, mesh, restLength: len };
    };

    // Arms: shoulder→elbow, elbow→wrist
    const wristL = new THREE.Vector3(C.lowerArmL.x - 0.02, 0.70, 0.02);
    const wristR = new THREE.Vector3(C.lowerArmR.x + 0.02, 0.70, 0.02);
    const uaL_obj = makeLimb(C.upperArmL, C.lowerArmL, 0.07, matArm);
    const laL_obj = makeLimb(C.lowerArmL, wristL, 0.055, matArmL);
    const uaR_obj = makeLimb(C.upperArmR, C.lowerArmR, 0.07, matArm);
    const laR_obj = makeLimb(C.lowerArmR, wristR, 0.055, matArmL);
    geometries.upperArmL = uaL_obj.group;
    geometries.lowerArmL = laL_obj.group;
    geometries.upperArmR = uaR_obj.group;
    geometries.lowerArmR = laR_obj.group;

    // Hand spheres (attached at wrist endpoints)
    const makeHand = (center) => {
      const g = new THREE.Group();
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), matHand);
      sphere.position.copy(center);
      sphere.castShadow = true;
      g.add(sphere);
      return { group: g };
    };
    const handL = makeHand(wristL);
    const handR = makeHand(wristR);

    // Legs: hip→knee, knee→ankle
    const ulL_obj = makeLimb(
      new THREE.Vector3(C.upperLegL.x, C.hip.y - 0.04, C.upperLegL.z), C.lowerLegL, 0.10, matLeg);
    const llL_obj = makeLimb(C.lowerLegL, C.footL, 0.075, matLegL);
    const ulR_obj = makeLimb(
      new THREE.Vector3(C.upperLegR.x, C.hip.y - 0.04, C.upperLegR.z), C.lowerLegR, 0.10, matLeg);
    const llR_obj = makeLimb(C.lowerLegR, C.footR, 0.075, matLegL);
    geometries.upperLegL = ulL_obj.group;
    geometries.lowerLegL = llL_obj.group;
    geometries.upperLegR = ulR_obj.group;
    geometries.lowerLegR = llR_obj.group;

    // Store limb config for dynamic scaling in updatePose
    this._limbConfig = {
      upperArmL:  { mesh: uaL_obj.mesh,  restLen: uaL_obj.restLength },
      lowerArmL:  { mesh: laL_obj.mesh,  restLen: laL_obj.restLength },
      upperArmR:  { mesh: uaR_obj.mesh,  restLen: uaR_obj.restLength },
      lowerArmR:  { mesh: laR_obj.mesh,  restLen: laR_obj.restLength },
      upperLegL:  { mesh: ulL_obj.mesh,  restLen: ulL_obj.restLength },
      lowerLegL:  { mesh: llL_obj.mesh,  restLen: llL_obj.restLength },
      upperLegR:  { mesh: ulR_obj.mesh,  restLen: ulR_obj.restLength },
      lowerLegR:  { mesh: llR_obj.mesh,  restLen: llR_obj.restLength },
    };
    // Hand meshes (stored separately for direct positioning)
    this._handMeshes = {
      handL: handL.group,
      handR: handR.group,
    };

    // Feet: small boxes
    const makeFoot = (center, material) => {
      const g = new THREE.Group();
      const geo = new THREE.BoxGeometry(0.08, 0.05, 0.14);
      const mesh = new THREE.Mesh(geo, material);
      mesh.position.copy(center);
      mesh.position.y -= 0.02;
      mesh.castShadow = true;
      g.add(mesh);
      return g;
    };
    geometries.footL = makeFoot(C.footL, matFoot);
    geometries.footR = makeFoot(C.footR, matFoot);

    // ---- Pivot positions (joint centers) ----
    const pivots = {
      head:        C.head.clone(),
      torso:       C.hip.clone(),  // torso rotates from hip up
      hip:         C.hip.clone(),
      upperArmL:   C.upperArmL.clone(),
      lowerArmL:   C.lowerArmL.clone(),
      upperArmR:   C.upperArmR.clone(),
      lowerArmR:   C.lowerArmR.clone(),
      upperLegL:   new THREE.Vector3(C.upperLegL.x, C.hip.y - 0.04, C.upperLegL.z),
      lowerLegL:   C.lowerLegL.clone(),
      footL:       C.footL.clone(),
      upperLegR:   new THREE.Vector3(C.upperLegR.x, C.hip.y - 0.04, C.upperLegR.z),
      lowerLegR:   C.lowerLegR.clone(),
      footR:       C.footR.clone(),
    };

    // ---- Build hierarchy ----
    // Compute pivot-local offsets for each geometry.
    // Limb meshes (built by makeLimb) store their world-space center in mesh.position,
    // so the pivot-local offset is mid − pivot = −pivot (since mid = mesh.position).
    // Head/torso/hip geometries are built with content at local origin, so the
    // geometry group needs to be placed at the geometry's intended world center
    // relative to the pivot.
    const _geometryWorldCenters = {
      head:  C.head.clone(),
      torso: C.torso.clone(),
      hip:   C.hip.clone(),
    };

    for (const seg of ALL_SEGMENTS) {
      const g = new THREE.Group();
      g.name = seg;
      g.position.copy(pivots[seg]);
      if (geometries[seg]) {
        let offset;
        if (_geometryWorldCenters[seg]) {
          // Head/torso/hip: geometry content is at local origin
          offset = new THREE.Vector3().subVectors(_geometryWorldCenters[seg], pivots[seg]);
        } else {
          // Limbs: mesh.position stores world-space midpoint
          offset = pivots[seg].clone().negate();
        }
        geometries[seg].position.copy(offset);
        g.add(geometries[seg]);
      }
      this.segments[seg] = g;
    }

    // Attach hands to lower arm segments (position at wrist relative to elbow pivot)
    for (const side of ["L", "R"]) {
      const wristPos = side === "L" ? wristL : wristR;
      const elbowPivot = pivots[`lowerArm${side}`];
      const handLocal = new THREE.Vector3().subVectors(wristPos, elbowPivot);
      this._handMeshes[`hand${side}`].position.copy(handLocal);
      this.segments[`lowerArm${side}`].add(this._handMeshes[`hand${side}`]);
    }

    this.root.add(this.segments.hip);
    this.root.add(this.segments.torso);

    for (const [name, parentName] of Object.entries(SEGMENT_PARENT)) {
      if (!parentName) continue;
      const parent = this.segments[parentName];
      const child = this.segments[name];
      if (!parent || !child) continue;
      const rel = new THREE.Vector3().subVectors(child.position, parent.position);
      child.position.copy(rel);
      parent.add(child);
    }

    this.segmentQuats = {};
    for (const seg of ALL_SEGMENTS) {
      this.segmentQuats[seg] = new THREE.Quaternion();
    }

    // Position model: bottom at y=0
    this.root.position.set(0, 0, 0);
    this.ready = true;
    log("info", "程序化人体模型创建完成");
  }

  async load(url) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);

    // ---- 0. Check for armature/skeleton in the model ----
    const bones = [];
    gltf.scene.traverse((child) => {
      if (child.isBone) {
        bones.push(child);
      }
    });
    log("info", `骨架检测: ${bones.length} 根骨骼${bones.length > 0 ? ", 模型自带骨骼可用" : ", 使用顶点级分割"}`);
    if (bones.length > 0) {
      for (const b of bones) {
        log("info", `  骨骼: ${b.name}`);
      }
    }

    // ---- 1. Model-space bounding box ----
    const bbox = new THREE.Box3().setFromObject(gltf.scene);
    const min = bbox.min.clone();
    const max = bbox.max.clone();
    const size = new THREE.Vector3().subVectors(max, min);
    log("info", `包围盒: min=(${min.x.toFixed(2)},${min.y.toFixed(2)},${min.z.toFixed(2)}) max=(${max.x.toFixed(2)},${max.y.toFixed(2)},${max.z.toFixed(2)}) size=(${size.x.toFixed(2)},${size.y.toFixed(2)},${size.z.toFixed(2)})`);

    // ---- 2. Scale to ~2.5 units tall ----
    const targetHeight = 2.5;
    const scale = targetHeight / size.y;

    // ---- 3. Collect all meshes with vertex data ----
    const meshEntries = [];
    gltf.scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        meshEntries.push(child);
      }
    });
    log("info", `共 ${meshEntries.length} 个 mesh`);

    // ---- 4. Per-vertex classification and face-level mesh splitting ----
    // For each mesh, classify every vertex, then assign each triangle face
    // to the body segment its center falls into.
    const segmentFaces = {};  // segName → [{mesh, material, indices: []}]
    for (const seg of ALL_SEGMENTS) {
      segmentFaces[seg] = [];
    }

    // ---- 4a. Sampling pass: collect vertex positions for adaptive thresholds ----
    const vertexSamples = [];
    for (const mesh of meshEntries) {
      mesh.updateWorldMatrix(true, true);
      const worldMat = mesh.matrixWorld;
      const posAttr = mesh.geometry.getAttribute("position");
      if (!posAttr) continue;
      const vertCount = posAttr.count;
      const stride = Math.max(1, Math.floor(vertCount / 2000));
      for (let v = 0; v < vertCount; v += stride) {
        const pt = new THREE.Vector3(posAttr.getX(v), posAttr.getY(v), posAttr.getZ(v));
        pt.applyMatrix4(worldMat);
        vertexSamples.push({
          nx: (pt.x - min.x) / size.x,
          ny: (pt.y - min.y) / size.y,
          nz: (pt.z - min.z) / size.z,
        });
      }
    }
    log("info", `收集了 ${vertexSamples.length} 个顶点样本用于自适应阈值计算`);
    _CLASSIFY_THRESHOLDS = computeAdaptiveThresholds(vertexSamples);

    const vertexStats = {}; // for diagnostic
    for (const seg of ALL_SEGMENTS) {
      vertexStats[seg] = 0;
    }

    for (const mesh of meshEntries) {
      const geo = mesh.geometry;
      const posAttr = geo.getAttribute("position");
      if (!posAttr) continue;

      // Compute vertex world positions
      mesh.updateWorldMatrix(true, true);
      const worldMat = mesh.matrixWorld;

      // Classify each vertex
      const vertCount = posAttr.count;
      const vertSegs = new Array(vertCount);
      for (let v = 0; v < vertCount; v++) {
        const pt = new THREE.Vector3(posAttr.getX(v), posAttr.getY(v), posAttr.getZ(v));
        pt.applyMatrix4(worldMat);
        const nx = (pt.x - min.x) / size.x;
        const ny = (pt.y - min.y) / size.y;
        const nz = (pt.z - min.z) / size.z;
        vertSegs[v] = classifyVertex(nx, ny, nz);
        vertexStats[vertSegs[v]]++;
      }

      // Group faces by segment
      const indexAttr = geo.getIndex();
      const faceMap = {}; // segName → [face indices]
      for (const seg of ALL_SEGMENTS) {
        faceMap[seg] = [];
      }

      if (indexAttr) {
        const idxCount = indexAttr.count;
        for (let i = 0; i < idxCount; i += 3) {
          const a = indexAttr.getX(i);
          const b = indexAttr.getX(i + 1);
          const c = indexAttr.getX(i + 2);
          // Face center determines segment
          const cx = (posAttr.getX(a) + posAttr.getX(b) + posAttr.getX(c)) / 3;
          const cy = (posAttr.getY(a) + posAttr.getY(b) + posAttr.getY(c)) / 3;
          const cz = (posAttr.getZ(a) + posAttr.getZ(b) + posAttr.getZ(c)) / 3;
          const pt = new THREE.Vector3(cx, cy, cz).applyMatrix4(worldMat);
          const nx = (pt.x - min.x) / size.x;
          const ny = (pt.y - min.y) / size.y;
          const nz = (pt.z - min.z) / size.z;
          const seg = classifyVertex(nx, ny, nz);
          faceMap[seg].push(a, b, c);
        }
      } else {
        // Non-indexed: every 3 vertices = 1 triangle
        for (let i = 0; i < vertCount; i += 3) {
          const cx = (posAttr.getX(i) + posAttr.getX(i+1) + posAttr.getX(i+2)) / 3;
          const cy = (posAttr.getY(i) + posAttr.getY(i+1) + posAttr.getY(i+2)) / 3;
          const cz = (posAttr.getZ(i) + posAttr.getZ(i+1) + posAttr.getZ(i+2)) / 3;
          const pt = new THREE.Vector3(cx, cy, cz).applyMatrix4(worldMat);
          const nx = (pt.x - min.x) / size.x;
          const ny = (pt.y - min.y) / size.y;
          const nz = (pt.z - min.z) / size.z;
          const seg = classifyVertex(nx, ny, nz);
          faceMap[seg].push(i, i+1, i+2);
        }
      }

      // Build sub-geometry for each non-empty segment
      for (const seg of ALL_SEGMENTS) {
        const faces = faceMap[seg];
        if (faces.length === 0) continue;

        const subGeo = new THREE.BufferGeometry();
        // Copy position attribute
        const newPos = new Float32Array(faces.length * 3);
        for (let f = 0; f < faces.length; f++) {
          const vi = faces[f];
          newPos[f * 3] = posAttr.getX(vi);
          newPos[f * 3 + 1] = posAttr.getY(vi);
          newPos[f * 3 + 2] = posAttr.getZ(vi);
        }
        subGeo.setAttribute("position", new THREE.BufferAttribute(newPos, 3));

        // Copy normal if present
        const normalAttr = geo.getAttribute("normal");
        if (normalAttr && normalAttr.count === vertCount) {
          const newNorm = new Float32Array(faces.length * 3);
          for (let f = 0; f < faces.length; f++) {
            const vi = faces[f];
            newNorm[f * 3] = normalAttr.getX(vi);
            newNorm[f * 3 + 1] = normalAttr.getY(vi);
            newNorm[f * 3 + 2] = normalAttr.getZ(vi);
          }
          subGeo.setAttribute("normal", new THREE.BufferAttribute(newNorm, 3));
        }

        // Copy UV if present
        const uvAttr = geo.getAttribute("uv");
        if (uvAttr && uvAttr.count === vertCount) {
          const newUV = new Float32Array(faces.length * 2);
          for (let f = 0; f < faces.length; f++) {
            const vi = faces[f];
            newUV[f * 2] = uvAttr.getX(vi);
            newUV[f * 2 + 1] = uvAttr.getY(vi);
          }
          subGeo.setAttribute("uv", new THREE.BufferAttribute(newUV, 2));
        }

        subGeo.computeVertexNormals();

        // Clone material
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const mat = materials[0].clone();

        segmentFaces[seg].push({ geometry: subGeo, material: mat });
      }
    }

    // ---- 5. Diagnostic: vertex distribution ----
    log("info", "顶点分布:");
    for (const seg of ALL_SEGMENTS) {
      const n = vertexStats[seg];
      if (n > 0) {
        log("info", `  ${seg}: ${n} vertices`);
      } else {
        log("warn", `  ${seg}: 0 vertices`);
      }
    }

    // ---- 6. Compute segment centers ----
    const segCenters = {};
    for (const seg of ALL_SEGMENTS) {
      const entries = segmentFaces[seg];
      if (entries.length === 0) {
        segCenters[seg] = this._estimateCenter(seg, segmentFaces, min, size);
        continue;
      }
      // Compute center from all face vertices
      const sum = new THREE.Vector3();
      let count = 0;
      for (const entry of entries) {
        const pos = entry.geometry.getAttribute("position");
        for (let v = 0; v < pos.count; v++) {
          sum.x += pos.getX(v);
          sum.y += pos.getY(v);
          sum.z += pos.getZ(v);
          count++;
        }
      }
      sum.divideScalar(count);
      segCenters[seg] = sum;
    }

    log("info", "段中心 (模型空间):");
    for (const [name, c] of Object.entries(segCenters)) {
      const hasGeo = segmentFaces[name].length > 0;
      log("info", `  ${name}: (${c.x.toFixed(3)},${c.y.toFixed(3)},${c.z.toFixed(3)}) geo=${hasGeo ? "Y" : "N"}`);
    }

    // ---- 7. Compute rest directions ----
    const rd = {};
    rd.head  = new THREE.Vector3(0, 1, 0);
    rd.hip   = new THREE.Vector3(0, 1, 0);
    rd.torso = new THREE.Vector3(0, 1, 0);

    // Arms: from segment centers, with fallback
    const uaL = new THREE.Vector3().subVectors(segCenters.lowerArmL, segCenters.upperArmL);
    rd.upperArmL = uaL.length() > 0.01 ? uaL.clone().normalize() : new THREE.Vector3(-0.7, -0.3, 0);
    const laL = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmL.x - 0.05, segCenters.lowerArmL.y - 0.1, segCenters.lowerArmL.z),
      segCenters.lowerArmL
    );
    rd.lowerArmL = laL.length() > 0.01 ? laL.normalize() : new THREE.Vector3(0, -1, 0);

    const uaR = new THREE.Vector3().subVectors(segCenters.lowerArmR, segCenters.upperArmR);
    rd.upperArmR = uaR.length() > 0.01 ? uaR.clone().normalize() : new THREE.Vector3(0.7, -0.3, 0);
    const laR = new THREE.Vector3().subVectors(
      new THREE.Vector3(segCenters.lowerArmR.x + 0.05, segCenters.lowerArmR.y - 0.1, segCenters.lowerArmR.z),
      segCenters.lowerArmR
    );
    rd.lowerArmR = laR.length() > 0.01 ? laR.normalize() : new THREE.Vector3(0, -1, 0);

    // Legs: downward from hip
    rd.upperLegL = new THREE.Vector3(0, -1, 0);
    rd.lowerLegL = new THREE.Vector3(0, -1, 0);
    rd.footL     = new THREE.Vector3(0, 0, 1);
    rd.upperLegR = new THREE.Vector3(0, -1, 0);
    rd.lowerLegR = new THREE.Vector3(0, -1, 0);
    rd.footR     = new THREE.Vector3(0, 0, 1);

    this.restDirs = rd;

    // ---- 8. Build segment hierarchy ----
    this._buildHierarchy(segCenters);

    // ---- 9. Add sub-meshes to segment groups ----
    for (const seg of ALL_SEGMENTS) {
      const group = this.segments[seg];
      if (!group) continue;
      group.updateWorldMatrix(false, true);
      const invGroup = new THREE.Matrix4().copy(group.matrixWorld).invert();

      for (const entry of segmentFaces[seg]) {
        const subMesh = new THREE.Mesh(entry.geometry, entry.material);
        subMesh.castShadow = false;
        subMesh.receiveShadow = false;

        // Transform sub-mesh into group local space
        const localMat = new THREE.Matrix4().multiplyMatrices(invGroup, this.root.matrixWorld);
        subMesh.applyMatrix4(localMat);

        group.add(subMesh);
      }

      // For segments with no geometry, create procedural cylinder between joints
      if (segmentFaces[seg].length === 0) {
        this._createLimbGeometry(seg, group, invGroup, segCenters);
      }
    }

    // Remove original meshes from GLTF scene
    for (const mesh of meshEntries) {
      mesh.parent?.remove(mesh);
      if (mesh.geometry) mesh.geometry.dispose();
      if (mesh.material) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const m of mats) m.dispose();
      }
    }

    // ---- 10. Final scale and position ----
    // Center model in XZ plane, bottom at y=0
    const cx = (min.x + max.x) / 2;
    const cz = (min.z + max.z) / 2;
    this.root.scale.setScalar(scale);
    this.root.position.set(-cx * scale, -min.y * scale, -cz * scale);
    this.ready = true;
    log("info", `模型位置: (${this.root.position.x.toFixed(2)},${this.root.position.y.toFixed(2)},${this.root.position.z.toFixed(2)}) scale=${scale.toFixed(3)}`);

    const segsWithGeo = Object.values(segmentFaces).filter(arr => arr.length > 0).length;
    log("info", `加载完成: ${ALL_SEGMENTS.length} 段, ${segsWithGeo} 段有几何, 共${meshEntries.length} 原始mesh`);
  }

  /** Create procedural cylinder geometry for a limb segment between two joints */
  _createLimbGeometry(segName, group, invGroup, segCenters) {
    // Determine joint pairs for each segment
    const jointPairs = {
      upperArmL: ["upperArmL", "lowerArmL"],
      lowerArmL: ["lowerArmL", null],   // null → extend downward
      upperArmR: ["upperArmR", "lowerArmR"],
      lowerArmR: ["lowerArmR", null],
      upperLegL: ["upperLegL", "lowerLegL"],
      lowerLegL: ["lowerLegL", "footL"],
      footL: ["footL", null],
      upperLegR: ["upperLegR", "lowerLegR"],
      lowerLegR: ["lowerLegR", "footR"],
      footR: ["footR", null],
    };

    const pair = jointPairs[segName];
    if (!pair) return;

    const startName = pair[0];
    const endName = pair[1];

    const start = segCenters[startName]?.clone() || new THREE.Vector3();
    let end;
    if (endName && segCenters[endName]) {
      end = segCenters[endName].clone();
    } else {
      // Extend from start in the segment's natural direction
      end = start.clone();
      if (segName.startsWith("lowerArm")) {
        end.y -= 0.3;
      } else if (segName.startsWith("foot")) {
        end.y -= 0.1;
        end.z += 0.15;
      }
    }

    // Transform to group local space
    const localMat = new THREE.Matrix4().multiplyMatrices(invGroup, this.root.matrixWorld);
    const localStart = start.clone().applyMatrix4(localMat);
    const localEnd = end.clone().applyMatrix4(localMat);

    // Create cylinder between start and end
    const dir = new THREE.Vector3().subVectors(localEnd, localStart);
    const length = dir.length();
    if (length < 0.001) return;

    const mid = new THREE.Vector3().addVectors(localStart, localEnd).multiplyScalar(0.5);

    const radius = segName.includes("Arm") ? 0.08 :
                   segName.includes("Leg") ? 0.10 :
                   segName.includes("foot") ? 0.06 : 0.06;

    const cylGeo = new THREE.CylinderGeometry(radius, radius, length, 8, 4);
    const cylColor = segName.includes("upperArm") ? 0xff6644 :
                     segName.includes("lowerArm") ? 0xff8844 :
                     segName.includes("upperLeg") ? 0x4488ff :
                     segName.includes("lowerLeg") ? 0x44aaff :
                     segName.includes("foot") ? 0x44ccff : 0x888888;

    const cylMat = new THREE.MeshStandardMaterial({
      color: cylColor,
      roughness: 0.6,
      metalness: 0.4,
      transparent: true,
      opacity: 0.85,
    });
    const cyl = new THREE.Mesh(cylGeo, cylMat);
    cyl.name = `${segName}_limb`;
    cyl.castShadow = false;
    cyl.receiveShadow = false;

    // Position at midpoint, orient along direction
    cyl.position.copy(mid);

    // Align cylinder (Y-up by default) to the direction vector
    const yAxis = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.normalize());
    cyl.quaternion.copy(quat);

    group.add(cyl);
    log("info", `  程序化肢体 ${segName}: len=${length.toFixed(2)} r=${radius}`);
  }

  /** Estimate position for segments with zero geometry (based on neighbors) */
  _estimateCenter(segName, segmentFaces, min, size) {
    const neighbors = {
      upperArmL: "lowerArmL", lowerArmL: "upperArmL",
      upperArmR: "lowerArmR", lowerArmR: "upperArmR",
      upperLegL: "lowerLegL", lowerLegL: "upperLegL",
      footL: "lowerLegL", upperLegR: "lowerLegR",
      lowerLegR: "upperLegR", footR: "lowerLegR",
    };

    const neighbor = neighbors[segName];
    if (neighbor && segmentFaces[neighbor]?.length > 0) {
      const entries = segmentFaces[neighbor];
      const nc = new THREE.Vector3();
      let count = 0;
      for (const entry of entries) {
        const pos = entry.geometry.getAttribute("position");
        for (let v = 0; v < pos.count; v++) {
          nc.x += pos.getX(v); nc.y += pos.getY(v); nc.z += pos.getZ(v);
          count++;
        }
      }
      nc.divideScalar(count);

      if (segName.startsWith("foot")) return new THREE.Vector3(nc.x, nc.y - 0.06, nc.z + 0.02);
      if (segName.startsWith("lowerLeg")) return new THREE.Vector3(nc.x, nc.y + 0.06, nc.z);
      if (segName.startsWith("upperLeg")) return new THREE.Vector3(nc.x, nc.y + 0.06, nc.z);
      if (segName.startsWith("upperArm")) return new THREE.Vector3(nc.x * 0.5, nc.y + 0.08, nc.z);
      if (segName.startsWith("lowerArm")) return new THREE.Vector3(nc.x * 0.5, nc.y - 0.08, nc.z);
    }

    // Hardcoded fallbacks
    const defaults = {
      upperArmL: new THREE.Vector3(min.x + size.x * 0.15, min.y + size.y * 0.60, min.z + size.z * 0.4),
      lowerArmL: new THREE.Vector3(min.x + size.x * 0.10, min.y + size.y * 0.42, min.z + size.z * 0.4),
      upperArmR: new THREE.Vector3(min.x + size.x * 0.85, min.y + size.y * 0.60, min.z + size.z * 0.4),
      lowerArmR: new THREE.Vector3(min.x + size.x * 0.90, min.y + size.y * 0.42, min.z + size.z * 0.4),
      upperLegL: new THREE.Vector3(min.x + size.x * 0.42, min.y + size.y * 0.18, min.z + size.z * 0.5),
      lowerLegL: new THREE.Vector3(min.x + size.x * 0.42, min.y + size.y * 0.08, min.z + size.z * 0.5),
      footL:     new THREE.Vector3(min.x + size.x * 0.40, min.y + size.y * 0.02, min.z + size.z * 0.45),
      upperLegR: new THREE.Vector3(min.x + size.x * 0.58, min.y + size.y * 0.18, min.z + size.z * 0.5),
      lowerLegR: new THREE.Vector3(min.x + size.x * 0.58, min.y + size.y * 0.08, min.z + size.z * 0.5),
      footR:     new THREE.Vector3(min.x + size.x * 0.60, min.y + size.y * 0.02, min.z + size.z * 0.45),
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

    // Joint positions from segment centers
    const shoulderL = segCenters.upperArmL || new THREE.Vector3(-0.3, 1.3, 0);
    const shoulderR = segCenters.upperArmR || new THREE.Vector3(0.3, 1.3, 0);
    const elbowL = segCenters.lowerArmL || new THREE.Vector3(-0.35, 1.0, 0);
    const elbowR = segCenters.lowerArmR || new THREE.Vector3(0.35, 1.0, 0);
    const hipC = segCenters.hip || new THREE.Vector3(0, 0.6, 0);
    const headC = segCenters.head || new THREE.Vector3(0, 2.0, 0);

    // Knee/ankle from leg centers
    const kneeL = segCenters.lowerLegL
      ? new THREE.Vector3(segCenters.lowerLegL.x, segCenters.lowerLegL.y + 0.08, segCenters.lowerLegL.z)
      : new THREE.Vector3(-0.12, 0.35, 0);
    const kneeR = segCenters.lowerLegR
      ? new THREE.Vector3(segCenters.lowerLegR.x, segCenters.lowerLegR.y + 0.08, segCenters.lowerLegR.z)
      : new THREE.Vector3(0.12, 0.35, 0);
    const ankleL = segCenters.footL
      ? new THREE.Vector3(segCenters.footL.x, segCenters.footL.y + 0.03, segCenters.footL.z)
      : new THREE.Vector3(-0.12, 0.08, 0.05);
    const ankleR = segCenters.footR
      ? new THREE.Vector3(segCenters.footR.x, segCenters.footR.y + 0.03, segCenters.footR.z)
      : new THREE.Vector3(0.12, 0.08, 0.05);

    // Set pivots
    setPivot("hip", hipC);
    setPivot("torso", hipC);
    setPivot("head", headC);
    setPivot("upperArmL", shoulderL);
    setPivot("lowerArmL", elbowL);
    setPivot("upperArmR", shoulderR);
    setPivot("lowerArmR", elbowR);
    setPivot("upperLegL", new THREE.Vector3(hipC.x - 0.10, hipC.y, hipC.z));
    setPivot("lowerLegL", kneeL);
    setPivot("footL", ankleL);
    setPivot("upperLegR", new THREE.Vector3(hipC.x + 0.10, hipC.y, hipC.z));
    setPivot("lowerLegR", kneeR);
    setPivot("footR", ankleR);

    // ---- Build parent-child hierarchy ----
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

  // -----------------------------------------------------------------------
  // Per-frame pose update
  // -----------------------------------------------------------------------
  updatePose(landmarks) {
    if (!this._diagCount) this._diagCount = 0;
    this._diagCount++;

    if (!this.ready || !landmarks || landmarks.length < 13) {
      if (this._diagCount <= 3) {
        log("warn", `updatePose #${this._diagCount} SKIP ready=${this.ready} nLandmarks=${landmarks?.length || 0}`);
      }
      return;
    }

    if (!this._frameCount) this._frameCount = 0;
    this._frameCount++;

    const lm = landmarks;
    const pts = {};
    for (const [key, idx] of Object.entries(IDX)) {
      pts[key] = mpToScene(lm[idx]);
    }

    const alpha = SMOOTH_ALPHA;

    // ---- Torso (hip → shoulder midpoint) ----
    const hipMid = new THREE.Vector3().addVectors(pts.L_HIP, pts.R_HIP).multiplyScalar(0.5);
    const shMid = new THREE.Vector3().addVectors(pts.L_SHOULDER, pts.R_SHOULDER).multiplyScalar(0.5);
    const torsoDir = new THREE.Vector3().subVectors(shMid, hipMid).normalize();
    if (torsoDir.length() > 0.01) {
      this._rotateSegment("torso", torsoDir, this.restDirs.torso, alpha);
      this._rotateSegment("hip", torsoDir, this.restDirs.hip, alpha);
    }

    // ---- Head (nose → shoulder midpoint) ----
    const headDir = new THREE.Vector3().subVectors(pts.NOSE, shMid).normalize();
    if (headDir.length() > 0.01) {
      this._rotateSegment("head", headDir, this.restDirs.head, alpha);
    }

    // ---- Upper arms (shoulder → elbow) ----
    const uaL = new THREE.Vector3().subVectors(pts.L_ELBOW, pts.L_SHOULDER);
    if (uaL.length() > 0.01) {
      this._rotateSegment("upperArmL", uaL.normalize(), this.restDirs.upperArmL, alpha);
      this._scaleLimb("upperArmL", uaL.length());
      this.segments.lowerArmL.position.copy(
        this.restDirs.upperArmL.clone().normalize().multiplyScalar(uaL.length())
      );
    }
    const uaR = new THREE.Vector3().subVectors(pts.R_ELBOW, pts.R_SHOULDER);
    if (uaR.length() > 0.01) {
      this._rotateSegment("upperArmR", uaR.normalize(), this.restDirs.upperArmR, alpha);
      this._scaleLimb("upperArmR", uaR.length());
      this.segments.lowerArmR.position.copy(
        this.restDirs.upperArmR.clone().normalize().multiplyScalar(uaR.length())
      );
    }

    // ---- Forearms (elbow → wrist) ----
    const laL = new THREE.Vector3().subVectors(pts.L_WRIST, pts.L_ELBOW);
    if (laL.length() > 0.01) {
      this._rotateSegment("lowerArmL", laL.normalize(), this.restDirs.lowerArmL, alpha);
      this._scaleLimb("lowerArmL", laL.length());
      this._handMeshes.handL.position.copy(
        this.restDirs.lowerArmL.clone().normalize().multiplyScalar(laL.length())
      );
    }
    const laR = new THREE.Vector3().subVectors(pts.R_WRIST, pts.R_ELBOW);
    if (laR.length() > 0.01) {
      this._rotateSegment("lowerArmR", laR.normalize(), this.restDirs.lowerArmR, alpha);
      this._scaleLimb("lowerArmR", laR.length());
      this._handMeshes.handR.position.copy(
        this.restDirs.lowerArmR.clone().normalize().multiplyScalar(laR.length())
      );
    }

    // ---- Upper legs (hip → knee) ----
    const ulL = new THREE.Vector3().subVectors(pts.L_KNEE, pts.L_HIP);
    if (ulL.length() > 0.01) {
      this._rotateSegment("upperLegL", ulL.normalize(), this.restDirs.upperLegL, alpha);
      this._scaleLimb("upperLegL", ulL.length());
      this.segments.lowerLegL.position.copy(
        this.restDirs.upperLegL.clone().normalize().multiplyScalar(ulL.length())
      );
    }
    const ulR = new THREE.Vector3().subVectors(pts.R_KNEE, pts.R_HIP);
    if (ulR.length() > 0.01) {
      this._rotateSegment("upperLegR", ulR.normalize(), this.restDirs.upperLegR, alpha);
      this._scaleLimb("upperLegR", ulR.length());
      this.segments.lowerLegR.position.copy(
        this.restDirs.upperLegR.clone().normalize().multiplyScalar(ulR.length())
      );
    }

    // ---- Lower legs (knee → ankle) ----
    const llL = new THREE.Vector3().subVectors(pts.L_ANKLE, pts.L_KNEE);
    if (llL.length() > 0.01) {
      this._rotateSegment("lowerLegL", llL.normalize(), this.restDirs.lowerLegL, alpha);
      this._scaleLimb("lowerLegL", llL.length());
      this.segments.footL.position.copy(
        this.restDirs.lowerLegL.clone().normalize().multiplyScalar(llL.length())
      );
    }
    const llR = new THREE.Vector3().subVectors(pts.R_ANKLE, pts.R_KNEE);
    if (llR.length() > 0.01) {
      this._rotateSegment("lowerLegR", llR.normalize(), this.restDirs.lowerLegR, alpha);
      this._scaleLimb("lowerLegR", llR.length());
      this.segments.footR.position.copy(
        this.restDirs.lowerLegR.clone().normalize().multiplyScalar(llR.length())
      );
    }

    // ---- Feet: derive direction from ankle-to-knee + forward component ----
    // Project lower leg direction onto horizontal plane for foot pointing
    const footFromLeg = (legDir, anklePt) => {
      const horiz = new THREE.Vector3(legDir.x, 0, legDir.z).normalize();
      const forward = new THREE.Vector3(0, 0, 1);
      // Blend: mostly forward, slight tilt based on leg horizontal direction
      const fwd = new THREE.Vector3().addVectors(
        forward.clone().multiplyScalar(0.7),
        horiz.clone().multiplyScalar(0.3)
      );
      // Add slight downward tilt
      fwd.y = -0.15;
      return fwd.normalize();
    };
    const footDirL = footFromLeg(llL, pts.L_ANKLE);
    this._rotateSegment("footL", footDirL, this.restDirs.footL, alpha * 0.7);
    const footDirR = footFromLeg(llR, pts.R_ANKLE);
    this._rotateSegment("footR", footDirR, this.restDirs.footR, alpha * 0.7);

    // ---- Translation: root follows hip midpoint ----
    const posAlpha = POS_SMOOTH;
    this.root.position.x += (hipMid.x - this.root.position.x) * posAlpha;
    this.root.position.y += (hipMid.y - this.root.position.y) * posAlpha;
    this.root.position.z += (hipMid.z - this.root.position.z) * posAlpha;

    // ---- Update debug skeleton ----
    if (this.debugSkeleton) {
      this.debugSkeleton.update(landmarks);
    }

    // ---- Standardized per-frame log (every 30 frames) ----
    if (this._frameCount % 30 === 0) {
      const f = this._frameCount;
      log("info", `frame=${f} hip=(${hipMid.x.toFixed(2)},${hipMid.y.toFixed(2)},${hipMid.z.toFixed(2)}) ` +
        `torsoDir=(${torsoDir.x.toFixed(2)},${torsoDir.y.toFixed(2)},${torsoDir.z.toFixed(2)}) ` +
        `headDir=(${headDir.x.toFixed(2)},${headDir.y.toFixed(2)},${headDir.z.toFixed(2)})`);

      log("info", `  ARM_L: sh→el=(${uaL.x.toFixed(2)},${uaL.y.toFixed(2)},${uaL.z.toFixed(2)}) ` +
        `el→wr=(${laL.x.toFixed(2)},${laL.y.toFixed(2)},${laL.z.toFixed(2)})`);
      log("info", `  ARM_R: sh→el=(${uaR.x.toFixed(2)},${uaR.y.toFixed(2)},${uaR.z.toFixed(2)}) ` +
        `el→wr=(${laR.x.toFixed(2)},${laR.y.toFixed(2)},${laR.z.toFixed(2)})`);
      log("info", `  LEG_L: hp→kn=(${ulL.x.toFixed(2)},${ulL.y.toFixed(2)},${ulL.z.toFixed(2)}) ` +
        `kn→an=(${llL.x.toFixed(2)},${llL.y.toFixed(2)},${llL.z.toFixed(2)})`);
      log("info", `  LEG_R: hp→kn=(${ulR.x.toFixed(2)},${ulR.y.toFixed(2)},${ulR.z.toFixed(2)}) ` +
        `kn→an=(${llR.x.toFixed(2)},${llR.y.toFixed(2)},${llR.z.toFixed(2)})`);

      // Log all 13 landmark positions
      const lmNames = ["nose", "L-sh", "R-sh", "L-elb", "R-elb", "L-wr", "R-wr", "L-hip", "R-hip", "L-knee", "R-knee", "L-ank", "R-ank"];
      const lmStr = lmNames.map((name, i) => {
        return `${name}=(${lm[i].x.toFixed(3)},${lm[i].y.toFixed(3)},${lm[i].z.toFixed(3)},v${lm[i].visibility?.toFixed(2) || "?"})`;
      }).join(" ");
      log("info", `  LANDMARKS: ${lmStr}`);
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

  /** Scale limb cylinder to match actual landmark distance */
  _scaleLimb(name, targetLen) {
    const cfg = this._limbConfig?.[name];
    if (!cfg || !cfg.mesh || cfg.restLen <= 0 || targetLen <= 0) return;
    const scale = targetLen / cfg.restLen;
    // Clamp to reasonable range to avoid extreme stretching
    const clamped = Math.max(0.3, Math.min(3.0, scale));
    const smooth = cfg.mesh.scale.y + (clamped - cfg.mesh.scale.y) * 0.6;
    cfg.mesh.scale.y = smooth;
  }
}
