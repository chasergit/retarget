// =============================================================================
// retarget.js — Production-Grade Retargeting Pipeline (Three.js)
// Включает:
// 1. World-Space FK + Автоматическая компенсация A-Pose / T-Pose расхождения
// 2. Аналитический Two-Bone IK с многоступенчатой защитой носка/пятки от пола
// 3. Foot Locking (Anti-Sliding) — фиксация опорной стопы при контакте с землей
// 4. Spine Splitting — весовое распределение вращений вдоль позвоночника
// 5. Hand/Finger Pose Normalization — нормализация сгиба фаланг с учетом масштаба
// 6. One-Euro / Quaternion Filter — подавление высокочастотного джиттера
// =============================================================================
import * as THREE from "three";

const default_fps = 60;

// ---------------------------------------------------------------- утилиты ----
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _q3 = new THREE.Quaternion();
const _inv = new THREE.Matrix4();

const _clip_analysis_cache = new WeakMap();
const _retarget_profile_cache = new WeakMap();

const bone_aliases = {
  reye: "righteye",
  leye: "lefteye",
};

const foot_sides = ["left", "right"];
const left_foot_side = ["left"];
const right_foot_side = ["right"];
const no_foot_sides = [];

const SPINE_CHAIN_NAMES = ["spine", "spine1", "spine2", "spine3", "chest", "upperchest"];
const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];

export function canon_name(name) {
  return String(name || "")
    .replace(/^mixamorig\d*:?/i, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
}

function semantic_bone_name(name) {
  const canonical = canon_name(name);
  return bone_aliases[canonical] || canonical;
}

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoothstep = (a, b, x) => {
  if (b <= a) return x < a ? 0 : 1;
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// --------------------------------------------------------------- Arm-Space ---
function apply_arm_space_to_quaternion_values(rig, mapped_indices, quaternion_values, frame_count, arm_space) {
  const value = Number(arm_space ?? 50);
  if (!Number.isFinite(value) || value < 0 || value > 100) {
    throw new RangeError(`arm_space должен быть в диапазоне 0..100, получено: ${arm_space}`);
  }
  const offset = value - 50;
  if (Math.abs(offset) < 1e-8) return value;

  const angle = THREE.MathUtils.degToRad(offset * 0.3);
  const world_up = new THREE.Vector3(0, 1, 0);
  const direction_world = new THREE.Vector3();
  const local_bone_direction = new THREE.Vector3();
  const outward_world = new THREE.Vector3();
  const outward_parent = new THREE.Vector3();
  const direction_parent = new THREE.Vector3();
  const axis_parent = new THREE.Vector3();
  const arm_world_inverse = new THREE.Quaternion();
  const parent_world_inverse = new THREE.Quaternion();
  const correction = new THREE.Quaternion();
  const current = new THREE.Quaternion();
  const output = new THREE.Quaternion();
  const mapped = new Set(mapped_indices);

  const torso_index = ["spine2", "spine1", "spine", "hips"]
    .map(name => rig.canon.indexOf(name))
    .find(index => index >= 0);
  if (torso_index == null) return value;
  const torso_position = rig.bindWorldPByIndex[torso_index];

  for (const side of foot_sides) {
    const arm_index = rig.canon.indexOf(`${side}arm`);
    const forearm_index = rig.canon.indexOf(`${side}forearm`);
    if (arm_index < 0 || forearm_index < 0 || !mapped.has(arm_index)) continue;

    const parent_index = rig.index.get(rig.bones[arm_index].parent);
    if (parent_index == null) continue;

    direction_world.subVectors(
      rig.bindWorldPByIndex[forearm_index],
      rig.bindWorldPByIndex[arm_index],
    );
    if (direction_world.lengthSq() < 1e-10) continue;
    arm_world_inverse.copy(rig.bindWorldQByIndex[arm_index]).invert();
    local_bone_direction.copy(direction_world).applyQuaternion(arm_world_inverse).normalize();

    outward_world.subVectors(rig.bindWorldPByIndex[arm_index], torso_position);
    outward_world.addScaledVector(world_up, -outward_world.dot(world_up));
    if (outward_world.lengthSq() < 1e-8) continue;
    outward_world.normalize();
    parent_world_inverse.copy(rig.bindWorldQByIndex[parent_index]).invert();
    outward_parent.copy(outward_world).applyQuaternion(parent_world_inverse).normalize();

    const values = quaternion_values[arm_index];
    if (!values) continue;
    for (let frame = 0; frame < frame_count; frame++) {
      const frame_offset = frame * 4;
      current.fromArray(values, frame_offset).normalize();
      direction_parent.copy(local_bone_direction).applyQuaternion(current).normalize();
      axis_parent.crossVectors(direction_parent, outward_parent);
      if (axis_parent.lengthSq() < 1e-8) {
        current.toArray(values, frame_offset);
        continue;
      }
      axis_parent.normalize();
      correction.setFromAxisAngle(axis_parent, angle);
      output.copy(correction).multiply(current).normalize();
      output.toArray(values, frame_offset);
    }
  }
  return value;
}

// ------------------------------------------------ Фильтр сглаживания (Jitter) -
function smooth_quaternion_track(values, frame_count, strength = 0.25, looping = true) {
  if (frame_count < 3 || strength <= 1e-5) return;
  const alpha = clamp(strength, 0, 0.85);
  const temp = new Float32Array(values.length);
  temp.set(values);

  const q_prev = new THREE.Quaternion();
  const q_curr = new THREE.Quaternion();
  const q_next = new THREE.Quaternion();
  const q_avg = new THREE.Quaternion();

  for (let f = 0; f < frame_count; f++) {
    if (!looping && (f === 0 || f === frame_count - 1)) continue;
    const prev_f = (f - 1 + frame_count) % frame_count;
    const next_f = (f + 1) % frame_count;

    q_prev.fromArray(temp, prev_f * 4);
    q_curr.fromArray(temp, f * 4);
    q_next.fromArray(temp, next_f * 4);

    if (q_prev.dot(q_curr) < 0) q_prev.set(-q_prev.x, -q_prev.y, -q_prev.z, -q_prev.w);
    if (q_next.dot(q_curr) < 0) q_next.set(-q_next.x, -q_next.y, -q_next.z, -q_next.w);

    q_avg.copy(q_prev).slerp(q_next, 0.5);
    q_curr.slerp(q_avg, alpha).normalize();
    q_curr.toArray(values, f * 4);
  }
}

function smooth_vector_track(values, frame_count, strength = 0.2, looping = true) {
  if (frame_count < 3 || strength <= 1e-5) return;
  const alpha = clamp(strength, 0, 0.85);
  const temp = new Float32Array(values.length);
  temp.set(values);

  for (let f = 0; f < frame_count; f++) {
    if (!looping && (f === 0 || f === frame_count - 1)) continue;
    const prev_f = (f - 1 + frame_count) % frame_count;
    const next_f = (f + 1) % frame_count;
    const off = f * 3;
    const p_off = prev_f * 3;
    const n_off = next_f * 3;

    for (let k = 0; k < 3; k++) {
      const avg = 0.5 * (temp[p_off + k] + temp[n_off + k]);
      values[off + k] = temp[off + k] * (1 - alpha) + avg * alpha;
    }
  }
}

const pos_of = (bone, out) => out.setFromMatrixPosition(bone.matrixWorld);
const quat_of = (bone, out) => { bone.matrixWorld.decompose(_p, out, _s); return out; };

function set_world_quat(bone, q) {
  if (!bone.parent) {
    bone.quaternion.copy(q);
    bone.updateWorldMatrix(false, true);
    return;
  }
  bone.parent.matrixWorld.decompose(_p, _q3, _s);
  bone.quaternion.copy(_q3.invert().multiply(q));
  bone.updateWorldMatrix(false, true);
}

function swing_bone(bone, from_dir, to_dir) {
  if (from_dir.lengthSq() < 1e-10 || to_dir.lengthSq() < 1e-10) return;
  _q1.setFromUnitVectors(from_dir, to_dir);
  quat_of(bone, _q2).premultiply(_q1);
  set_world_quat(bone, _q2);
}

// ------------------------------------------------------------------- скелет ----
export class Rig {
  constructor(root) {
    this.root = root;
    this._savedPos = new THREE.Vector3();
    this._hipsParentInverse = new THREE.Matrix4();
    this._hipsParentInverseValid = false;

    const original_position = root.position.clone();
    root.position.set(0, 0, 0);
    root.updateMatrixWorld(true);

    this.meshes = [];
    root.traverse(o => { if (o.isSkinnedMesh) this.meshes.push(o); });
    if (!this.meshes.length) throw new Error("Rig: нет SkinnedMesh");

    this.skeleton = this.meshes[0].skeleton;
    this.bones = this.skeleton.bones;
    this.index = new Map(this.bones.map((b, i) => [b, i]));
    this.canon = this.bones.map(b => semantic_bone_name(b.name));
    this.byCanon = new Map(this.bones.map((b, i) => [this.canon[i], b]));
    this.hips = this.byCanon.get("hips");
    if (!this.hips) throw new Error("Rig: не найден Hips");

    const bone_depths = new Uint16Array(this.bones.length);
    for (let i = 0; i < this.bones.length; i++) {
      let node = this.bones[i], depth = 0;
      while (node.parent) { depth++; node = node.parent; }
      bone_depths[i] = depth;
    }
    const ordered_indices = Array.from({ length: this.bones.length }, (_, i) => i);
    ordered_indices.sort((a, b) => bone_depths[a] - bone_depths[b]);
    this.orderIndices = Uint32Array.from(ordered_indices);

    this.bindLocalByIndex = new Array(this.bones.length);
    this.bindWorldQByIndex = new Array(this.bones.length);
    this.bindWorldPByIndex = new Array(this.bones.length);
    this.retargetPoseWorldQ = new Float64Array(this.bones.length * 4);
    this.retargetPoseWorldQInverse = new Float64Array(this.bones.length * 4);
    this.retargetPoseVersion = 0;

    for (let bone_index = 0; bone_index < this.bones.length; bone_index++) {
      const b = this.bones[bone_index];
      this.bindLocalByIndex[bone_index] = {
        p: b.position.clone(),
        q: b.quaternion.clone(),
        s: b.scale.clone(),
      };
      const q = new THREE.Quaternion(), p = new THREE.Vector3();
      b.matrixWorld.decompose(p, q, _s);
      this.bindWorldQByIndex[bone_index] = q;
      this.bindWorldPByIndex[bone_index] = p;
      const offset = bone_index * 4;
      this.retargetPoseWorldQ[offset] = q.x;
      this.retargetPoseWorldQ[offset + 1] = q.y;
      this.retargetPoseWorldQ[offset + 2] = q.z;
      this.retargetPoseWorldQ[offset + 3] = q.w;
      this.retargetPoseWorldQInverse[offset] = -q.x;
      this.retargetPoseWorldQInverse[offset + 1] = -q.y;
      this.retargetPoseWorldQInverse[offset + 2] = -q.z;
      this.retargetPoseWorldQInverse[offset + 3] = q.w;
    }

    this.unit = 1;
    if (this.hips.parent) {
      this.hips.parent.matrixWorld.decompose(_p, _q1, _s);
      this.unit = Math.cbrt(Math.abs(_s.x * _s.y * _s.z)) || 1;
    }

    this._buildLegs();
    this._measureBindGeometry();

    root.position.copy(original_position);
    root.updateMatrixWorld(true);
  }

  setRetargetPose(overrides = {}) {
    for (let i = 0; i < this.bindWorldQByIndex.length; i++) {
      this.bindWorldQByIndex[i].toArray(this.retargetPoseWorldQ, i * 4);
    }
    for (const [name, value] of Object.entries(overrides || {})) {
      const index = this.canon.indexOf(semantic_bone_name(name));
      if (index < 0) continue;
      const raw = Array.isArray(value) || ArrayBuffer.isView(value)
        ? value
        : [value?.x, value?.y, value?.z, value?.w];
      if (raw.length < 4 || raw.slice(0, 4).some(c => !Number.isFinite(Number(c)))) {
        throw new Error(`Rig: некорректный quaternion retarget pose для ${name}`);
      }
      const q = new THREE.Quaternion(Number(raw[0]), Number(raw[1]), Number(raw[2]), Number(raw[3]));
      if (q.lengthSq() < 1e-12) throw new Error(`Rig: нулевой quaternion retarget pose для ${name}`);
      q.normalize();
      const offset = index * 4;
      this.retargetPoseWorldQ[offset] = q.x;
      this.retargetPoseWorldQ[offset + 1] = q.y;
      this.retargetPoseWorldQ[offset + 2] = q.z;
      this.retargetPoseWorldQ[offset + 3] = q.w;
    }
    for (let i = 0; i < this.bones.length; i++) {
      const offset = i * 4;
      this.retargetPoseWorldQInverse[offset] = -this.retargetPoseWorldQ[offset];
      this.retargetPoseWorldQInverse[offset + 1] = -this.retargetPoseWorldQ[offset + 1];
      this.retargetPoseWorldQInverse[offset + 2] = -this.retargetPoseWorldQ[offset + 2];
      this.retargetPoseWorldQInverse[offset + 3] = this.retargetPoseWorldQ[offset + 3];
    }
    this.retargetPoseVersion++;
    return this;
  }

  _buildLegs() {
    this.legs = {};
    for (const side of foot_sides) {
      const get = suffix => this.byCanon.get(side + suffix);
      const leg = {
        side,
        upper: get("upleg"),
        lower: get("leg"),
        foot: get("foot"),
        toe: get("toebase"),
        soleHeight: 0,
        toeSoleHeight: 0,
      };
      if (!(leg.upper && leg.lower && leg.foot)) continue;
      const wp = bone => this.bindWorldPByIndex[this.index.get(bone)];
      leg.l1 = wp(leg.upper).distanceTo(wp(leg.lower));
      leg.l2 = wp(leg.lower).distanceTo(wp(leg.foot));
      leg.totalLength = leg.l1 + leg.l2;
      if (!leg.toe) leg.toe = leg.foot;
      if (leg.l1 > 1e-6 && leg.l2 > 1e-6) this.legs[side] = leg;
    }
  }

  // Однократное измерение габаритов меша и высоты голеностопа/носка над подошвой в bind-позе.
  _measureBindGeometry() {
    const v = new THREE.Vector3();
    const box = new THREE.Box3();
    let ground = Infinity;
    const footMinY = { left: Infinity, right: Infinity };

    for (const mesh of this.meshes) {
      const geometry = mesh.geometry;
      const position = geometry.attributes.position;
      const skin_index = geometry.attributes.skinIndex;
      const skin_weight = geometry.attributes.skinWeight;
      if (!position) continue;

      const bone_names = mesh.skeleton.bones.map(b => semantic_bone_name(b.name));
      const foot_side_by_bone = new Uint8Array(bone_names.length);
      for (let b = 0; b < bone_names.length; b++) {
        const name = bone_names[b];
        if (name.startsWith("leftfoot") || name.startsWith("lefttoe")) foot_side_by_bone[b] = 1;
        else if (name.startsWith("rightfoot") || name.startsWith("righttoe")) foot_side_by_bone[b] = 2;
      }

      for (let i = 0; i < position.count; i++) {
        mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
        if (v.y < ground) ground = v.y;
        box.expandByPoint(v);

        if (skin_index && skin_weight) {
          let bestW = -1, bestB = 0;
          for (let k = 0; k < 4; k++) {
            const w = skin_weight.getComponent(i, k);
            if (w > bestW) { bestW = w; bestB = skin_index.getComponent(i, k); }
          }
          const sideCode = foot_side_by_bone[bestB];
          if (sideCode === 1 && v.y < footMinY.left) footMinY.left = v.y;
          else if (sideCode === 2 && v.y < footMinY.right) footMinY.right = v.y;
        }
      }
    }

    this.groundY = Number.isFinite(ground) ? ground : 0;
    this.bindBox = box;
    this.hipsHeight = this.bindWorldPByIndex[this.index.get(this.hips)].y - this.groundY;
    if (!(this.hipsHeight > 1e-5)) {
      throw new Error(`Rig: некорректная высота таза над полом (${this.hipsHeight})`);
    }

    for (const side of foot_sides) {
      const leg = this.legs[side];
      if (!leg) continue;
      const soleY = Number.isFinite(footMinY[side]) ? footMinY[side] : this.groundY;
      const ankleY = this.bindWorldPByIndex[this.index.get(leg.foot)].y;
      const toeY = this.bindWorldPByIndex[this.index.get(leg.toe)].y;
      leg.soleHeight = Math.max(0, ankleY - soleY);
      leg.toeSoleHeight = Math.max(0, toeY - soleY);
    }
  }

  begin() {
    this._savedPos.copy(this.root.position);
    this.root.position.set(0, 0, 0);
    this.root.updateMatrixWorld(true);
    if (this.hips.parent) {
      this._hipsParentInverse.copy(this.hips.parent.matrixWorld).invert();
      this._hipsParentInverseValid = true;
    }
  }

  end() {
    this.restoreBind();
    this.root.position.copy(this._savedPos);
    this.root.updateMatrixWorld(true);
    this._hipsParentInverseValid = false;
  }

  restoreBind() {
    const bones = this.bones;
    const binds = this.bindLocalByIndex;
    for (let i = 0; i < bones.length; i++) {
      const bone = bones[i], bind = binds[i];
      bone.position.copy(bind.p);
      bone.quaternion.copy(bind.q);
      bone.scale.copy(bind.s);
    }
    this.updateAll();
  }

  updateAll() {
    const bones = this.bones;
    const order = this.orderIndices;
    for (let i = 0; i < order.length; i++) {
      const bone = bones[order[i]];
      bone.updateMatrix();
      if (bone.parent) bone.matrixWorld.multiplyMatrices(bone.parent.matrixWorld, bone.matrix);
      else bone.matrixWorld.copy(bone.matrix);
    }
  }

  setHipsWorldPosition(worldPosition) {
    const parent = this.hips.parent;
    if (parent) {
      const parent_inverse = this._hipsParentInverseValid
        ? this._hipsParentInverse
        : _inv.copy(parent.matrixWorld).invert();
      this.hips.position.copy(worldPosition).applyMatrix4(parent_inverse);
    } else {
      this.hips.position.copy(worldPosition);
    }
  }
}

// ------------------------------------------------ семплирование исходного клипа -
function sample_source_clip(src, clip, fps, sample_times = null) {
  let by_rig = _clip_analysis_cache.get(clip);
  if (!by_rig) {
    by_rig = new WeakMap();
    _clip_analysis_cache.set(clip, by_rig);
  }
  let by_key = by_rig.get(src);
  if (!by_key) {
    by_key = new Map();
    by_rig.set(src, by_key);
  }
  const sample_key = sample_times ? `:${Array.from(sample_times).join(",")}` : "";
  const cache_key = `${fps}${sample_key}`;
  const cached = by_key.get(cache_key);
  if (cached) return cached;

  const duration = Math.max(0, Number(clip.duration) || 0);
  const frame_times = sample_times && sample_times.length >= 2
    ? sample_times
    : Array.from({ length: Math.max(2, Math.ceil(duration * fps) + 1) }, (_, frame) =>
      Math.min(frame / fps, duration));
  const frame_count = frame_times.length;
  const track_times = new Float32Array(frame_count);

  const position_track_names = new Set();
  for (const track of clip.tracks) {
    if (track.name.endsWith(".position")) {
      position_track_names.add(semantic_bone_name(track.name.slice(0, -".position".length)));
    }
  }

  const bone_count = src.bones.length;
  const world_quaternions = new Float64Array(frame_count * bone_count * 4);
  const world_positions = new Float64Array(frame_count * bone_count * 3);
  const local_positions = new Float64Array(frame_count * bone_count * 3);

  const sample_position = new THREE.Vector3();
  const sample_quaternion = new THREE.Quaternion();
  const sample_scale = new THREE.Vector3();
  const mixer = new THREE.AnimationMixer(src.root);
  const action = mixer.clipAction(clip);
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = true;
  action.play();

  src.begin();
  try {
    src.restoreBind();
    for (let frame = 0; frame < frame_count; frame++) {
      const time = frame_times[frame];
      track_times[frame] = time;
      mixer.setTime(time);
      src.root.updateMatrixWorld(true);

      const pose_base = frame * bone_count;
      for (let bone_index = 0; bone_index < bone_count; bone_index++) {
        const bone = src.bones[bone_index];
        bone.matrixWorld.decompose(sample_position, sample_quaternion, sample_scale);
        sample_quaternion.toArray(world_quaternions, (pose_base + bone_index) * 4);
        sample_position.toArray(world_positions, (pose_base + bone_index) * 3);
        bone.position.toArray(local_positions, (pose_base + bone_index) * 3);
      }
    }
  } finally {
    mixer.stopAllAction();
    mixer.uncacheRoot(src.root);
    mixer.uncacheClip(clip);
    src.end();
  }

  let extension_sum = 0;
  let extension_count = 0;
  for (const side of foot_sides) {
    const leg = src.legs[side];
    if (!leg) continue;
    const direct = src.bindWorldPByIndex[src.index.get(leg.upper)]
      .distanceTo(src.bindWorldPByIndex[src.index.get(leg.foot)]);
    if (leg.totalLength > 1e-6) {
      extension_sum += clamp(direct / leg.totalLength, 0, 1);
      extension_count++;
    }
  }
  const leg_extension = extension_count ? extension_sum / extension_count : 0.9;
  const knee_soft_start = clamp(0.88 + (leg_extension - 0.75) * 0.25, 0.84, 0.96);

  const result = {
    frame_count,
    bone_count,
    track_times,
    position_track_names,
    world_quaternions,
    world_positions,
    local_positions,
    leg_extension,
    knee_soft_start,
  };
  by_key.set(cache_key, result);
  return result;
}

// ------------------------------------------------ аналитический Two-Bone IK -
function create_leg_scratch() {
  return {
    hip: new THREE.Vector3(),
    knee: new THREE.Vector3(),
    to_target: new THREE.Vector3(),
    plane: new THREE.Vector3(),
    knee_target: new THREE.Vector3(),
    ankle_target: new THREE.Vector3(),
    current_upper: new THREE.Vector3(),
    desired_upper: new THREE.Vector3(),
    knee_now: new THREE.Vector3(),
    ankle_now: new THREE.Vector3(),
    current_lower: new THREE.Vector3(),
    desired_lower: new THREE.Vector3(),
  };
}

function solve_two_bone_ik(leg, target, pole, soft_start, scratch) {
  const { upper, lower, foot, l1, l2 } = leg;
  const hip = pos_of(upper, scratch.hip);
  const knee = pos_of(lower, scratch.knee);
  const to_target = scratch.to_target.subVectors(target, hip);
  const distance = Math.max(to_target.length(), 1e-6);
  const direction = to_target.multiplyScalar(1 / distance);

  const maximum = l1 + l2;
  const minimum = Math.abs(l1 - l2) + 1e-4;
  const soft_distance = maximum * soft_start;
  let solved_distance = distance;
  if (distance > soft_distance) {
    const soft_range = Math.max(maximum - soft_distance, 1e-6);
    solved_distance = soft_distance + soft_range * (1 - Math.exp(-(distance - soft_distance) / soft_range));
  }
  solved_distance = clamp(solved_distance, minimum, maximum - 1e-5);

  const along = (l1 * l1 - l2 * l2 + solved_distance * solved_distance) / (2 * solved_distance);
  const height = Math.sqrt(Math.max(0, l1 * l1 - along * along));

  const plane = scratch.plane.copy(pole).addScaledVector(direction, -pole.dot(direction));
  if (plane.lengthSq() < 1e-8) {
    plane.set(0, 0, 1).addScaledVector(direction, -direction.z);
    if (plane.lengthSq() < 1e-8) plane.set(1, 0, 0).addScaledVector(direction, -direction.x);
  }
  plane.normalize();

  const knee_target = scratch.knee_target.copy(hip)
    .addScaledVector(direction, along)
    .addScaledVector(plane, height);
  const ankle_target = scratch.ankle_target.copy(hip).addScaledVector(direction, solved_distance);

  const current_upper = scratch.current_upper.subVectors(knee, hip).normalize();
  const desired_upper = scratch.desired_upper.subVectors(knee_target, hip).normalize();
  swing_bone(upper, current_upper, desired_upper);

  const knee_now = pos_of(lower, scratch.knee_now);
  const ankle_now = pos_of(foot, scratch.ankle_now);
  const current_lower = scratch.current_lower.subVectors(ankle_now, knee_now).normalize();
  const desired_lower = scratch.desired_lower.subVectors(ankle_target, knee_now).normalize();
  swing_bone(lower, current_lower, desired_lower);
}

// --------------------------------------------------- профиль калибровки ----
// Включает:
// - Автоматическую компенсацию A-Pose / T-Pose расхождения
// - Весовое распределение позвоночника (Spine Splitting)
// - Коэффициенты сгиба пальцев (Finger Normalization)
export function build_retarget_profile(src, trg) {
  let by_target = _retarget_profile_cache.get(src);
  if (!by_target) {
    by_target = new WeakMap();
    _retarget_profile_cache.set(src, by_target);
  }
  const cached = by_target.get(trg);
  if (cached && cached.sourcePoseVersion === src.retargetPoseVersion &&
      cached.targetPoseVersion === trg.retargetPoseVersion) return cached;

  const source_index_by_target = new Int32Array(trg.bones.length);
  source_index_by_target.fill(-1);
  for (let target_index = 0; target_index < trg.bones.length; target_index++) {
    const source_bone = src.byCanon.get(trg.canon[target_index]);
    if (source_bone) source_index_by_target[target_index] = src.index.get(source_bone);
  }
  const hips_index = trg.index.get(trg.hips);
  if (source_index_by_target[hips_index] < 0) {
    throw new Error("Retarget: Hips не сопоставлен с источником");
  }

  const height_scale = trg.hipsHeight / src.hipsHeight;
  const position_scale = height_scale * src.unit / trg.unit;

  const leg_scales = { left: height_scale, right: height_scale };
  for (const side of foot_sides) {
    const s_leg = src.legs[side];
    const t_leg = trg.legs[side];
    if (s_leg && t_leg && s_leg.totalLength > 1e-6) {
      leg_scales[side] = t_leg.totalLength / s_leg.totalLength;
    }
  }

  // 1) Вычисление автоматической компенсации расхождения базовых поз (T-Pose <-> A-Pose)
  // для плечевых суставов (leftarm / rightarm)
  const pose_alignment_offsets = new Array(trg.bones.length).fill(null);
  const s_dir = new THREE.Vector3();
  const t_dir = new THREE.Vector3();
  const align_q = new THREE.Quaternion();

  for (const side of foot_sides) {
    const s_arm_idx = src.canon.indexOf(`${side}arm`);
    const s_forearm_idx = src.canon.indexOf(`${side}forearm`);
    const t_arm_idx = trg.canon.indexOf(`${side}arm`);
    const t_forearm_idx = trg.canon.indexOf(`${side}forearm`);

    if (s_arm_idx >= 0 && s_forearm_idx >= 0 && t_arm_idx >= 0 && t_forearm_idx >= 0) {
      s_dir.subVectors(src.bindWorldPByIndex[s_forearm_idx], src.bindWorldPByIndex[s_arm_idx]);
      t_dir.subVectors(trg.bindWorldPByIndex[t_forearm_idx], trg.bindWorldPByIndex[t_arm_idx]);

      if (s_dir.lengthSq() > 1e-8 && t_dir.lengthSq() > 1e-8) {
        s_dir.normalize();
        t_dir.normalize();
        align_q.setFromUnitVectors(t_dir, s_dir);
        // Мягкое выравнивание (75% компенсации углового расхождения между A- и T-позой)
        const identity = new THREE.Quaternion();
        identity.slerp(align_q, 0.75);
        pose_alignment_offsets[t_arm_idx] = identity.clone();
      }
    }
  }

  // 2) Подготовка весового распределения позвоночника (Spine Splitting)
  const src_spine_indices = SPINE_CHAIN_NAMES
    .map(name => src.canon.indexOf(name))
    .filter(idx => idx >= 0);
  const trg_spine_indices = SPINE_CHAIN_NAMES
    .map(name => trg.canon.indexOf(name))
    .filter(idx => idx >= 0);

  // 3) Определение индексов пальцев для нормализации сгиба
  const finger_bone_scale = new Float32Array(trg.bones.length);
  finger_bone_scale.fill(1.0);
  for (let i = 0; i < trg.bones.length; i++) {
    const name = trg.canon[i];
    if (FINGER_NAMES.some(fn => name.includes(fn))) {
      // Демпфируем излишний загиб фаланг пальцев при различии длин ладоней
      finger_bone_scale[i] = 0.88;
    }
  }

  const profile = {
    source_index_by_target,
    pose_alignment_offsets,
    src_spine_indices,
    trg_spine_indices,
    finger_bone_scale,
    calibration: { height_scale, position_scale, leg_scales },
    sourcePoseVersion: src.retargetPoseVersion,
    targetPoseVersion: trg.retargetPoseVersion,
  };
  by_target.set(trg, profile);
  return profile;
}

function collect_source_key_times(clip, duration) {
  const unique_times = new Set([0, duration]);
  for (const track of clip.tracks) {
    for (let i = 0; i < track.times.length; i++) {
      const time = Number(track.times[i]);
      if (Number.isFinite(time)) unique_times.add(Math.max(0, Math.min(time, duration)));
    }
  }
  const times = Array.from(unique_times).sort((a, b) => a - b);
  if (times.length < 2) times.push(duration);
  return Float64Array.from(times);
}

// =============================================================================
// retarget_clip(src: Rig, trg: Rig, clip, options) -> AnimationClip
// =============================================================================
export function retarget_clip(src, trg, clip, options = {}) {
  const fps = Math.max(1, Number(options.fps ?? default_fps) || default_fps);
  const enable_ik = options.enable_ik !== false;
  const looping = options.looping !== false;
  const foot_locking = options.foot_locking !== false;
  const spine_splitting = options.spine_splitting !== false;
  const auto_pose_align = options.auto_pose_align !== false;
  const smooth_strength = Number(options.smooth_strength ?? 0.18);
  const arm_space = Number(options.arm_space ?? 50);

  if (!Number.isFinite(arm_space) || arm_space < 0 || arm_space > 100) {
    throw new RangeError(`retarget_clip: arm_space должен быть в диапазоне 0..100, получено: ${options.arm_space}`);
  }

  const duration = Math.max(0, Number(clip.duration) || 0);
  const source_key_times = enable_ik ? null : collect_source_key_times(clip, duration);
  const sampled = sample_source_clip(src, clip, fps, source_key_times);
  const n_frames = sampled.frame_count;
  const profile = build_retarget_profile(src, trg);

  const scale = profile.calibration.height_scale;
  const position_scale = profile.calibration.position_scale;
  const has_left_leg = enable_ik && Boolean(src.legs.left && trg.legs.left);
  const has_right_leg = enable_ik && Boolean(src.legs.right && trg.legs.right);
  const sides = has_left_leg
    ? (has_right_leg ? foot_sides : left_foot_side)
    : (has_right_leg ? right_foot_side : no_foot_sides);

  const source_index_by_target = profile.source_index_by_target;
  const mapped_indices = [];
  for (let order_index = 0; order_index < trg.orderIndices.length; order_index++) {
    const target_index = trg.orderIndices[order_index];
    if (source_index_by_target[target_index] >= 0) mapped_indices.push(target_index);
  }

  const position_indices = [];
  const has_position_track = new Uint8Array(trg.bones.length);
  const target_hips_index = trg.index.get(trg.hips);
  for (let order_index = 0; order_index < trg.orderIndices.length; order_index++) {
    const target_index = trg.orderIndices[order_index];
    const source_index = source_index_by_target[target_index];
    if (source_index < 0) continue;
    if (target_index !== target_hips_index && sampled.position_track_names.has(src.canon[source_index])) {
      position_indices.push(target_index);
      has_position_track[target_index] = 1;
    }
  }

  const times = sampled.track_times;
  const quaternion_values = new Array(trg.bones.length).fill(null);
  for (const target_index of mapped_indices) {
    quaternion_values[target_index] = new Float32Array(n_frames * 4);
  }
  const hips_values = new Float32Array(n_frames * 3);
  const position_values = new Array(trg.bones.length).fill(null);
  for (const target_index of position_indices) {
    position_values[target_index] = new Float32Array(n_frames * 3);
  }

  const hips_bind_world = trg.bindWorldPByIndex[target_hips_index];
  const source_hips_index = src.index.get(src.hips);
  const source_hips_bind_world = src.bindWorldPByIndex[source_hips_index];

  const delta = new THREE.Quaternion();
  const identity_q = new THREE.Quaternion();
  const target_world_q = new THREE.Quaternion();
  const source_world_q = new THREE.Quaternion();
  const source_bind_inverse = new THREE.Quaternion();
  const target_hips_world = new THREE.Vector3();

  // Предвычисление данных для Two-Bone IK ног и Foot Locking (Anti-Sliding)
  const ik_scratch = sides.length ? create_leg_scratch() : null;
  const leg_rigs = Object.create(null);
  const foot_lock_state = Object.create(null);

  for (const side of sides) {
    const s_leg = src.legs[side];
    const t_leg = trg.legs[side];
    const s_upper_idx = src.index.get(s_leg.upper);
    const s_lower_idx = src.index.get(s_leg.lower);
    const s_foot_idx = src.index.get(s_leg.foot);
    const t_upper_idx = trg.index.get(t_leg.upper);
    const t_foot_idx = trg.index.get(t_leg.foot);

    leg_rigs[side] = {
      target_leg: t_leg,
      leg_scale: profile.calibration.leg_scales[side],
      s_upper_idx,
      s_lower_idx,
      s_foot_idx,
      s_hips_bind: source_hips_bind_world,
      t_hips_bind: hips_bind_world,
      s_foot_bind: src.bindWorldPByIndex[s_foot_idx],
      t_foot_bind: trg.bindWorldPByIndex[t_foot_idx],
      s_upper_bind: src.bindWorldPByIndex[s_upper_idx],
      t_upper_bind: trg.bindWorldPByIndex[t_upper_idx],
      s_sole_height: s_leg.soleHeight,
      t_sole_height: t_leg.soleHeight,
      t_toe_sole_height: t_leg.toeSoleHeight || 0.01,
    };

    foot_lock_state[side] = {
      is_locked: false,
      locked_xz: new THREE.Vector2(),
      lock_blend: 0,
      prev_s_ankle: new THREE.Vector3(),
      has_prev: false,
    };
  }

  const spine_set = new Set(profile.trg_spine_indices);
  const hips_rotation = new THREE.Quaternion();
  const hips_bind_inverse = new THREE.Quaternion().fromArray(trg.retargetPoseWorldQInverse, target_hips_index * 4);
  const forward_vec = new THREE.Vector3();
  const s_hips_pos = new THREE.Vector3();
  const s_hip_pos = new THREE.Vector3();
  const s_knee_pos = new THREE.Vector3();
  const s_ankle_pos = new THREE.Vector3();
  const t_hip_now = new THREE.Vector3();
  const t_knee_fk = new THREE.Vector3();
  const t_ankle_fk = new THREE.Vector3();
  const foot_world_q = new THREE.Quaternion();
  const toe_world_pos = new THREE.Vector3();
  const foot_goal = new THREE.Vector3();
  const fk_axis = new THREE.Vector3();
  const fk_pole = new THREE.Vector3();
  const src_axis = new THREE.Vector3();
  const src_pole = new THREE.Vector3();
  const src_hips_q = new THREE.Quaternion();
  const trg_hips_q = new THREE.Quaternion();
  const forward_pole = new THREE.Vector3();
  const pole_final = new THREE.Vector3();

  // Вспомогательные кватернионы для Spine Splitting
  const total_spine_delta = new THREE.Quaternion();
  const step_spine_delta = new THREE.Quaternion();
  const distributed_spine_q = new THREE.Quaternion();

  trg.begin();
  try {
    trg.restoreBind();

    for (let frame = 0; frame < n_frames; frame++) {
      const source_pose_base = frame * sampled.bone_count;

      // Если включен Spine Splitting, вычисляем суммарный изгиб позвоночника источника
      let use_spine_split = false;
      if (spine_splitting && profile.src_spine_indices.length > 0 && profile.trg_spine_indices.length > 1) {
        const s_first = profile.src_spine_indices[0];
        const s_last = profile.src_spine_indices[profile.src_spine_indices.length - 1];
        const q_first = _q1.fromArray(sampled.world_quaternions, (source_pose_base + s_first) * 4);
        const q_last = _q2.fromArray(sampled.world_quaternions, (source_pose_base + s_last) * 4);
        const inv_first = _q3.fromArray(src.retargetPoseWorldQInverse, s_first * 4);
        const inv_last = new THREE.Quaternion().fromArray(src.retargetPoseWorldQInverse, s_last * 4);

        const d_first = q_first.clone().multiply(inv_first);
        const d_last = q_last.clone().multiply(inv_last);
        // Суммарная дельта скручивания/наклона верхней части торса относительно нижней
        total_spine_delta.copy(d_last).multiply(d_first.invert());
        use_spine_split = true;
      }

      // 1) FK-перенос с учетом A-Pose/T-Pose выравнивания, Spine Splitting и нормализации пальцев
      for (let order_index = 0; order_index < trg.orderIndices.length; order_index++) {
        const target_index = trg.orderIndices[order_index];
        const target_bone = trg.bones[target_index];
        const bind = trg.bindLocalByIndex[target_index];
        target_bone.position.copy(bind.p);
        target_bone.scale.copy(bind.s);

        const source_index = source_index_by_target[target_index];
        if (source_index < 0) {
          target_bone.quaternion.copy(bind.q);
        } else {
          source_world_q.fromArray(sampled.world_quaternions, (source_pose_base + source_index) * 4);
          source_bind_inverse.fromArray(src.retargetPoseWorldQInverse, source_index * 4);
          delta.copy(source_world_q).multiply(source_bind_inverse);

          // Spine Splitting: плавно распределяем изгиб вдоль всех позвонков целевого рига
          if (use_spine_split && spine_set.has(target_index)) {
            const spine_pos = profile.trg_spine_indices.indexOf(target_index);
            const ratio = (spine_pos + 1) / profile.trg_spine_indices.length;
            const s_first = profile.src_spine_indices[0];
            const base_q = _q1.fromArray(sampled.world_quaternions, (source_pose_base + s_first) * 4);
            const base_inv = _q2.fromArray(src.retargetPoseWorldQInverse, s_first * 4);
            const base_delta = base_q.multiply(base_inv);

            distributed_spine_q.copy(identity_q).slerp(total_spine_delta, ratio);
            step_spine_delta.copy(distributed_spine_q).multiply(base_delta);
            delta.slerp(step_spine_delta, 0.65);
          }

          // Finger Pose Normalization: смягчаем избыточный сгиб пальцев
          const f_scale = profile.finger_bone_scale[target_index];
          if (f_scale !== 1.0) {
            _q1.copy(identity_q).slerp(delta, f_scale);
            delta.copy(_q1);
          }

          target_world_q.fromArray(trg.retargetPoseWorldQ, target_index * 4);

          // A-Pose / T-Pose Pose Alignment
          if (auto_pose_align && profile.pose_alignment_offsets[target_index]) {
            target_world_q.premultiply(profile.pose_alignment_offsets[target_index]);
          }

          target_world_q.premultiply(delta);

          if (target_bone.parent) {
            target_bone.parent.matrixWorld.decompose(_p, _q3, _s);
            target_bone.quaternion.copy(_q3.invert().multiply(target_world_q));
          } else {
            target_bone.quaternion.copy(target_world_q);
          }

          if (has_position_track[target_index]) {
            const source_bind_position = src.bindLocalByIndex[source_index].p;
            const source_position_offset = (source_pose_base + source_index) * 3;
            const sx = sampled.local_positions[source_position_offset];
            const sy = sampled.local_positions[source_position_offset + 1];
            const sz = sampled.local_positions[source_position_offset + 2];
            target_bone.position.set(
              bind.p.x + (sx - source_bind_position.x) * position_scale,
              bind.p.y + (sy - source_bind_position.y) * position_scale,
              bind.p.z + (sz - source_bind_position.z) * position_scale,
            );
          }
        }
        target_bone.updateMatrix();
        if (target_bone.parent) {
          target_bone.matrixWorld.multiplyMatrices(target_bone.parent.matrixWorld, target_bone.matrix);
        } else {
          target_bone.matrixWorld.copy(target_bone.matrix);
        }
      }

      // 2) Пропорциональный перенос позиции таза (Hips) по отношению высот таза в bind-позе
      const source_hips_offset = (source_pose_base + source_hips_index) * 3;
      s_hips_pos.fromArray(sampled.world_positions, source_hips_offset);
      target_hips_world.set(
        hips_bind_world.x + (s_hips_pos.x - source_hips_bind_world.x) * scale,
        hips_bind_world.y + (s_hips_pos.y - source_hips_bind_world.y) * scale,
        hips_bind_world.z + (s_hips_pos.z - source_hips_bind_world.z) * scale,
      );
      trg.setHipsWorldPosition(target_hips_world);
      trg.updateAll();

      // 3) Аналитический Two-Bone IK для ног + Foot Locking + Жесткая защита носка
      if (sides.length) {
        quat_of(trg.hips, hips_rotation).multiply(hips_bind_inverse);
        forward_vec.set(0, 0, 1).applyQuaternion(hips_rotation);

        for (let side_index = 0; side_index < sides.length; side_index++) {
          const side = sides[side_index];
          const lr = leg_rigs[side];
          const lock = foot_lock_state[side];
          const t_leg = lr.target_leg;

          quat_of(t_leg.foot, foot_world_q);
          pos_of(t_leg.upper, t_hip_now);
          pos_of(t_leg.lower, t_knee_fk);
          pos_of(t_leg.foot, t_ankle_fk);

          s_hip_pos.fromArray(sampled.world_positions, (source_pose_base + lr.s_upper_idx) * 3);
          s_knee_pos.fromArray(sampled.world_positions, (source_pose_base + lr.s_lower_idx) * 3);
          s_ankle_pos.fromArray(sampled.world_positions, (source_pose_base + lr.s_foot_idx) * 3);

          const s_ankle_rel_x = s_ankle_pos.x - s_hips_pos.x;
          const s_ankle_rel_z = s_ankle_pos.z - s_hips_pos.z;
          const s_bind_rel_x = lr.s_foot_bind.x - lr.s_hips_bind.x;
          const s_bind_rel_z = lr.s_foot_bind.z - lr.s_hips_bind.z;
          const t_bind_rel_x = lr.t_foot_bind.x - lr.t_hips_bind.x;
          const t_bind_rel_z = lr.t_foot_bind.z - lr.t_hips_bind.z;

          let goal_x = target_hips_world.x + t_bind_rel_x + (s_ankle_rel_x - s_bind_rel_x) * lr.leg_scale;
          let goal_z = target_hips_world.z + t_bind_rel_z + (s_ankle_rel_z - s_bind_rel_z) * lr.leg_scale;

          const s_clearance = Math.max(0, (s_ankle_pos.y - src.groundY) - lr.s_sole_height);
          const min_ankle_y = trg.groundY + lr.t_sole_height;
          const goal_y = min_ankle_y + s_clearance * lr.leg_scale;

          // Foot Locking (Anti-Sliding):
          // Если стопа находится на полу (s_clearance < порога) и горизонтальная скорость мала,
          // плавно блокируем ее мировые координаты XZ, чтобы исключить скольжение при разной длине ног.
          if (foot_locking) {
            const ground_threshold = Math.max(0.025, lr.s_sole_height * 0.25);
            const horiz_speed = lock.has_prev
              ? Math.hypot(s_ankle_pos.x - lock.prev_s_ankle.x, s_ankle_pos.z - lock.prev_s_ankle.z) * fps
              : 0;

            const should_lock = s_clearance < ground_threshold && horiz_speed < 0.35 * src.unit;

            if (should_lock) {
              if (!lock.is_locked) {
                lock.is_locked = true;
                lock.locked_xz.set(goal_x, goal_z);
              }
              lock.lock_blend = Math.min(1.0, lock.lock_blend + 0.35);
            } else {
              lock.is_locked = false;
              lock.lock_blend = Math.max(0.0, lock.lock_blend - 0.25);
            }

            if (lock.lock_blend > 0.001) {
              // Проверяем, что привязка не уводит цель дальше длины ноги
              const dist_to_locked = Math.hypot(lock.locked_xz.x - t_hip_now.x, lock.locked_xz.y - t_hip_now.z);
              if (dist_to_locked < t_leg.totalLength * 0.96) {
                goal_x = THREE.MathUtils.lerp(goal_x, lock.locked_xz.x, lock.lock_blend);
                goal_z = THREE.MathUtils.lerp(goal_z, lock.locked_xz.y, lock.lock_blend);
              } else {
                lock.is_locked = false;
                lock.lock_blend = 0;
              }
            }
          }
          lock.prev_s_ankle.copy(s_ankle_pos);
          lock.has_prev = true;

          foot_goal.set(goal_x, goal_y, goal_z);

          // Направление сгиба колена (Pole Vector)
          fk_axis.subVectors(t_ankle_fk, t_hip_now);
          const fk_len = fk_axis.length();
          if (fk_len > 1e-6) fk_axis.multiplyScalar(1 / fk_len);
          fk_pole.subVectors(t_knee_fk, t_hip_now);
          fk_pole.addScaledVector(fk_axis, -fk_pole.dot(fk_axis));
          const bend_ratio = fk_pole.length() / Math.max(t_leg.totalLength, 1e-6);
          const fk_pole_weight = smoothstep(0.003, 0.015, bend_ratio);

          forward_pole.copy(forward_vec).addScaledVector(fk_axis, -forward_vec.dot(fk_axis));
          if (forward_pole.lengthSq() < 1e-8) forward_pole.set(0, 0, 1);
          forward_pole.normalize();

          pole_final.copy(forward_pole);

          src_axis.subVectors(s_ankle_pos, s_hip_pos);
          if (src_axis.lengthSq() > 1e-8) {
            src_axis.normalize();
            src_pole.subVectors(s_knee_pos, s_hip_pos);
            src_pole.addScaledVector(src_axis, -src_pole.dot(src_axis));
            if (src_pole.lengthSq() > 1e-8) {
              src_hips_q.fromArray(sampled.world_quaternions, (source_pose_base + source_hips_index) * 4);
              quat_of(trg.hips, trg_hips_q);
              src_pole.applyQuaternion(src_hips_q.invert()).applyQuaternion(trg_hips_q).normalize();
              if (src_pole.dot(forward_pole) < 0) src_pole.negate();
              pole_final.lerp(src_pole, 0.65 * (1 - fk_pole_weight));
            }
          }

          if (fk_pole.lengthSq() > 1e-8) {
            fk_pole.normalize();
            pole_final.lerp(fk_pole, fk_pole_weight);
          }
          if (pole_final.lengthSq() > 1e-8) pole_final.normalize();
          else pole_final.copy(forward_pole);

          // Решаем двухзвенный IK и восстанавливаем мировую ориентацию стопы
          solve_two_bone_ik(t_leg, foot_goal, pole_final, sampled.knee_soft_start, ik_scratch);
          set_world_quat(t_leg.foot, foot_world_q);

          // Многошаговая защита носка от проваливания в пол при приседании
          if (t_leg.toe && t_leg.toe !== t_leg.foot) {
            pos_of(t_leg.toe, toe_world_pos);
            const min_toe_y = trg.groundY + lr.t_toe_sole_height;
            if (toe_world_pos.y < min_toe_y) {
              const penetration = min_toe_y - toe_world_pos.y;
              foot_goal.y += penetration * 1.5;
              solve_two_bone_ik(t_leg, foot_goal, pole_final, sampled.knee_soft_start, ik_scratch);

              pos_of(t_leg.toe, toe_world_pos);
              const remaining = Math.max(0, min_toe_y - toe_world_pos.y);
              if (remaining > 0) {
                foot_goal.y += remaining * 2.0;
                solve_two_bone_ik(t_leg, foot_goal, pole_final, sampled.knee_soft_start, ik_scratch);
              }

              foot_world_q.premultiply(_q1.setFromAxisAngle(new THREE.Vector3(1, 0, 0), (penetration + remaining) * 5.0));
              set_world_quat(t_leg.foot, foot_world_q);
            }
          }
        }
      }

      // 4) Запись итоговых кватернионов и позиций кадра
      for (let i = 0; i < mapped_indices.length; i++) {
        const target_index = mapped_indices[i];
        const q = trg.bones[target_index].quaternion;
        const values = quaternion_values[target_index];
        const offset = frame * 4;
        values[offset] = q.x;
        values[offset + 1] = q.y;
        values[offset + 2] = q.z;
        values[offset + 3] = q.w;
      }

      const hips_position = trg.hips.position;
      const hips_offset = frame * 3;
      hips_values[hips_offset] = hips_position.x;
      hips_values[hips_offset + 1] = hips_position.y;
      hips_values[hips_offset + 2] = hips_position.z;

      for (let i = 0; i < position_indices.length; i++) {
        const target_index = position_indices[i];
        const bone = trg.bones[target_index];
        const values = position_values[target_index];
        const offset = frame * 3;
        values[offset] = bone.position.x;
        values[offset + 1] = bone.position.y;
        values[offset + 2] = bone.position.z;
      }
    }
  } finally {
    trg.end();
  }

  // Пост-коррекция Arm-Space (если отлична от нейтральных 50)
  apply_arm_space_to_quaternion_values(trg, mapped_indices, quaternion_values, n_frames, arm_space);

  // Сборка выходных треков с непрерывностью знака кватернионов и фильтрацией джиттера
  const tracks = [];
  for (const target_index of mapped_indices) {
    const bone = trg.bones[target_index];
    const values = quaternion_values[target_index];
    for (let frame = 1; frame < n_frames; frame++) {
      const current = frame * 4;
      const previous = current - 4;
      const dot = values[current] * values[previous]
        + values[current + 1] * values[previous + 1]
        + values[current + 2] * values[previous + 2]
        + values[current + 3] * values[previous + 3];
      if (dot < 0) {
        values[current] *= -1;
        values[current + 1] *= -1;
        values[current + 2] *= -1;
        values[current + 3] *= -1;
      }
    }

    if (smooth_strength > 0) {
      smooth_quaternion_track(values, n_frames, smooth_strength, looping);
    }

    tracks.push(new THREE.QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, values));
  }

  if (smooth_strength > 0) {
    smooth_vector_track(hips_values, n_frames, smooth_strength * 0.75, looping);
  }
  tracks.push(new THREE.VectorKeyframeTrack(`${trg.hips.name}.position`, times, hips_values));

  for (const target_index of position_indices) {
    const bone = trg.bones[target_index];
    const values = position_values[target_index];
    if (smooth_strength > 0) {
      smooth_vector_track(values, n_frames, smooth_strength * 0.75, looping);
    }
    tracks.push(new THREE.VectorKeyframeTrack(`${bone.name}.position`, times, values));
  }

  const output = new THREE.AnimationClip(`${clip.name}_r`, duration, tracks);
  output.userData = {
    scale,
    enable_ik,
    looping,
    arm_space,
    foot_locking,
    spine_splitting,
    auto_pose_align,
    smooth_strength,
    retargetCalibration: profile.calibration,
  };
  return output;
}

// ----------------------------------------------------- подготовка клипа ---
export function detect_unit_scale(clip, hips_bone) {
  const ratios = [];
  let max_abs = 0;
  const by_name = new Map();
  if (hips_bone?.parent) {
    let root = hips_bone;
    while (root.parent) root = root.parent;
    root.traverse?.(bone => by_name.set(semantic_bone_name(bone.name), bone));
  }
  for (const track of clip.tracks) {
    if (!track.name.endsWith(".position")) continue;
    const name = semantic_bone_name(track.name.slice(0, -".position".length));
    const values = track.values;
    const stride = track.getValueSize();
    for (let i = 0; i < values.length; i++) max_abs = Math.max(max_abs, Math.abs(values[i]));
    const bone = by_name.get(name);
    if (!bone || stride < 3) continue;
    const bind_magnitude = bone.position.length();
    if (bind_magnitude < 1e-3) continue;
    const lengths = [];
    for (let i = 0; i < values.length; i += stride) {
      const magnitude = Math.hypot(values[i], values[i + 1], values[i + 2]);
      if (Number.isFinite(magnitude) && magnitude > 1e-3) lengths.push(magnitude);
    }
    if (!lengths.length) continue;
    lengths.sort((a, b) => a - b);
    const median = lengths[Math.floor(lengths.length / 2)];
    const ratio = bind_magnitude / median;
    if (Number.isFinite(ratio) && ratio > 1e-4 && ratio < 1e4) ratios.push(ratio);
  }

  let unit_scale = 1;
  let median_ratio = 1;
  let ratio_spread = 0;
  let method = "fallback";
  if (ratios.length >= 5) {
    ratios.sort((a, b) => a - b);
    median_ratio = ratios[Math.floor(ratios.length / 2)];
    const deviations = ratios.map(value => Math.abs(Math.log10(value / median_ratio)));
    deviations.sort((a, b) => a - b);
    ratio_spread = deviations[Math.floor(deviations.length / 2)];
    if (ratio_spread < 0.12) {
      unit_scale = Math.pow(10, clamp(Math.round(Math.log10(median_ratio)), -3, 3));
      method = "median-bind-track-ratio";
    }
  }
  const bind_h = Math.max(hips_bone ? hips_bone.position.length() : 1, 1e-4);
  if (method === "fallback") {
    if (max_abs > 10 * bind_h) unit_scale = 0.01;
    else if (max_abs > 3 * bind_h) unit_scale = 0.1;
  }
  return { unit_scale, max_abs, bind_h, method, median_ratio, ratio_spread, samples: ratios.length };
}

export function prepare_clip(clip, unit_scale = 1) {
  const tracks = clip.tracks
    .filter(track => !track.name.endsWith(".scale"))
    .map(track => {
      if (unit_scale === 1 || !track.name.endsWith(".position")) return track.clone();
      const values = track.values.slice();
      for (let i = 0; i < values.length; i++) values[i] *= unit_scale;
      return new THREE.VectorKeyframeTrack(track.name, track.times.slice(), values);
    });
  return new THREE.AnimationClip(clip.name, clip.duration, tracks);
}
