// =============================================================================
// retarget.js — перенос анимации в мировом пространстве (FK) и двухзвенный IK ног.
// Исходный и целевой скелеты могут иметь разные пропорции, ориентации привязки
// и различающиеся наборы костей Mixamo. Сопоставление выполняется по каноническим именам.
// =============================================================================
import * as THREE from "three";

// 60 FPS обеспечивает более частое семплирование IK и более плотную итоговую анимацию.
// Дополнительные ключи практически не влияют на производительность воспроизведения.
const default_fps = 60;

// ---------------------------------------------------------------- утилиты ----
const _p = new THREE.Vector3(), _s = new THREE.Vector3();
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _q3 = new THREE.Quaternion();
const _inv = new THREE.Matrix4();
const _skin_outer = new THREE.Matrix4();
const _skin_inner = new THREE.Matrix4();
const _skin_combined = new THREE.Matrix4();
const _clip_analysis_cache = new WeakMap();
const _retarget_profile_cache = new WeakMap();

// Семантические цепочки Mixamo делают сопоставление явным и удобным для проверки.
// Сначала используются точные канонические совпадения; псевдонимы учитывают различия экспорта.
const bone_aliases = {
  reye: "righteye",
  leye: "lefteye",
};
const foot_contact_enter = 0.58;
const foot_contact_exit = 0.28;
const foot_support_speed_lo = 0.10;
const foot_support_speed_hi = 0.38;
const foot_release_seconds = 0.065;
const foot_phase = { SWING: 0, CONTACT: 1, PLANTED: 2, RELEASE: 3 };
const foot_pivot = { HEEL: 0, TOE: 1 };
const foot_pivot_handoff_frames = 4;
// Опорная нога «на колене»: когда колено ближе к полу, чем kneel_enter_height (доля высоты таза),
// вес держит колено/голень, а стопа лишь касается пола носком (пятка высоко). Якорь на носке
// в этом случае тянет колено вверх от пола, поэтому опорной точкой остаётся голеностоп.
// Гистерезис (kneel_exit_height) исключает переключения на границе. 0 отключает правило.
const kneel_enter_height = 0.20;
const kneel_exit_height = 0.32;
// Упреждающее освобождение опоры включается только для циклических клипов.
const foot_release_lead_seconds = 0.033;

function semantic_bone_name(name) {
  const canonical = canon_name(name);
  return bone_aliases[canonical] || canonical;
}
const foot_sides = ["left", "right"];
const left_foot_side = ["left"];
const right_foot_side = ["right"];
const no_foot_sides = [];

export function canon_name(name) {
  return String(name || "")
    .replace(/^mixamorig\d*:?/i, "")
    .replace(/[^a-z0-9]/gi, "")
    .toLowerCase();
}

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoothstep = (a, b, x) => {
  if (b <= a) return x < a ? 0 : 1;
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const median_sorted = (values, length = values.length) => {
  if (!length) return 0;
  const middle = Math.floor(length / 2);
  return length & 1 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
};

// Arm-Space — постоянная поправка к положению верхних частей рук, а не
// масштабирование всей анимации. Нейтральное значение 50 не меняет исходные ключи.
// Ось разведения вычисляется по bind-позе конкретного скелета: направление плечевой
// кости и вертикаль задают плоскость движения. Это позволяет не угадывать локальные
// оси Mixamo и не затрагивать локти, кисти и анимационную амплитуду.
function apply_arm_space_to_quaternion_values(rig, mapped_indices, quaternion_values, frame_count, arm_space) {
  const value = Number(arm_space ?? 50);
  if (!Number.isFinite(value) || value < 0 || value > 100) {
    throw new RangeError(`arm_space должен быть в диапазоне 0..100, получено: ${arm_space}`);
  }
  const offset = value - 50;
  if (Math.abs(offset) < 1e-8) return value;

  // Знак arm_space имеет единый смысл для всех клипов: выше 50 — наружу,
  // ниже 50 — к корпусу. Ось пересчитывается для каждого кадра по текущему
  // направлению плечевой кости, поэтому анимации с разной исходной позой
  // не получают противоположную поправку.
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

  for (const side of ["left", "right"]) {
    const arm_index = rig.canon.indexOf(`${side}arm`);
    const forearm_index = rig.canon.indexOf(`${side}forearm`);
    if (arm_index < 0 || forearm_index < 0 || !mapped.has(arm_index)) continue;

    const parent_index = rig.index.get(rig.bones[arm_index].parent);
    if (parent_index == null) continue;

    // Направление плечевой кости в её локальном пространстве.
    direction_world.subVectors(
      rig.bindWorldPByIndex[forearm_index],
      rig.bindWorldPByIndex[arm_index],
    );
    if (direction_world.lengthSq() < 1e-10) continue;
    arm_world_inverse.copy(rig.bindWorldQByIndex[arm_index]).invert();
    local_bone_direction.copy(direction_world).applyQuaternion(arm_world_inverse).normalize();

    // Наружное направление от центра корпуса к плечу, в координатах родителя.
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

      // Текущее направление локтя в пространстве родителя. Ось выбирается
      // так, чтобы производная вращения всегда имела компоненту наружу.
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

// Построение консервативных пространственных кластеров для точного поиска минимальной высоты.
// Сохраняются все вершины. Кластеры пропускаются только тогда, когда аффинная нижняя граница
// доказывает, что ни одна их вершина не может оказаться ниже текущего минимума.
const min_y_cluster_size = 24;
function build_min_yclusters(set) {
  const { indices, skin_cache } = set;
  if (!skin_cache.fastSkinning || indices.length < min_y_cluster_size * 2) return null;

  const ordered = Array.from(indices);
  const dominant = skin_cache.dominant_bone;
  const base = skin_cache.base_positions;
  // Эта граница учитывает три измерения и предполагает аффинную матрицу привязки
  // (однородная координата w = 1). Для нестандартных данных используется полный перебор.
  for (let i = 0; i < ordered.length; i++) {
    const offset = ordered[i] * 4;
    if (!Number.isFinite(base[offset]) || !Number.isFinite(base[offset + 1]) ||
        !Number.isFinite(base[offset + 2]) || Math.abs(base[offset + 3] - 1) > 1e-9) {
      return null;
    }
  }
  ordered.sort((a, b) => {
    const d = dominant[a] - dominant[b];
    return d || base[a * 4 + 1] - base[b * 4 + 1];
  });

  const cluster_indices = new Int32Array(ordered.length);
  const starts = [0], centers = [], radii = [], boneStarts = [0], bones = [];
  const min_weight_sums = [], maxWeightSums = [];
  const influence_indices = skin_cache.influence_indices;
  const influence_weights = skin_cache.influence_weights;
  // Буфер меток позволяет не создавать отдельный Set для каждого кластера
  // (при инициализации скелета в больших сетках могут быть тысячи кластеров).
  const seen_bone_stamps = new Uint32Array(skin_cache.mesh.skeleton.bones.length);
  let cluster_stamp = 0;

  for (let start = 0; start < ordered.length;) {
    cluster_stamp++;
    let end = start + 1;
    const dominant_bone = dominant[ordered[start]];
    while (end < ordered.length && end - start < min_y_cluster_size &&
           dominant[ordered[end]] === dominant_bone) end++;

    let cx = 0, cy = 0, cz = 0;
    for (let j = start; j < end; j++) {
      const i = ordered[j] * 4;
      cx += base[i]; cy += base[i + 1]; cz += base[i + 2];
    }
    const count = end - start;
    cx /= count; cy /= count; cz /= count;

    let radius = 0, minWeightSum = Infinity, maxWeightSum = 0;
    for (let j = start; j < end; j++) {
      const vertex = ordered[j], offset = vertex * 4;
      const dx = base[offset] - cx;
      const dy = base[offset + 1] - cy;
      const dz = base[offset + 2] - cz;
      radius = Math.max(radius, Math.hypot(dx, dy, dz));
      let weight_sum = 0;
      for (let k = 0; k < 4; k++) {
        const weight = influence_weights[offset + k];
        weight_sum += weight;
        if (weight > 0) {
          const bone_index = influence_indices[offset + k];
          if (seen_bone_stamps[bone_index] !== cluster_stamp) {
            seen_bone_stamps[bone_index] = cluster_stamp;
            bones.push(bone_index);
          }
        }
      }
      minWeightSum = Math.min(minWeightSum, weight_sum);
      maxWeightSum = Math.max(maxWeightSum, weight_sum);
      cluster_indices[j] = vertex;
    }

    centers.push(cx, cy, cz);
    radii.push(radius);
    min_weight_sums.push(minWeightSum);
    maxWeightSums.push(maxWeightSum);
    boneStarts.push(bones.length);
    starts.push(end);
    start = end;
  }

  return {
    indices: cluster_indices,
    starts: Int32Array.from(starts),
    centers: Float64Array.from(centers),
    radii: Float64Array.from(radii),
    boneStarts: Int32Array.from(boneStarts),
    bones: Uint32Array.from(bones),
    min_weight_sums: Float64Array.from(min_weight_sums),
    maxWeightSums: Float64Array.from(maxWeightSums),
    best: -1,
  };
}

function evaluate_min_ycluster(set, cluster_index, coeff, outer_y, current_minimum) {
  const cluster = set.minYClusters;
  const skin_cache = set.skin_cache;
  const base = skin_cache.base_positions;
  const bone_indices = skin_cache.influence_indices;
  const weights = skin_cache.influence_weights;
  let minimum = current_minimum;
  const start = cluster.starts[cluster_index];
  const end = cluster.starts[cluster_index + 1];

  for (let j = start; j < end; j++) {
    const offset = cluster.indices[j] * 4;
    const x = base[offset], y = base[offset + 1], z = base[offset + 2], w = base[offset + 3];
    let world_y = outer_y;
    for (let k = 0; k < 4; k++) {
      const weight = weights[offset + k];
      if (weight === 0) continue;
      const c = bone_indices[offset + k] * 4;
      world_y += weight * (coeff[c] * x + coeff[c + 1] * y + coeff[c + 2] * z + coeff[c + 3] * w);
    }
    if (world_y < minimum) minimum = world_y;
  }
  if (minimum < current_minimum) cluster.best = cluster_index;
  return minimum;
}

const pos_of = (bone, out) => out.setFromMatrixPosition(bone.matrixWorld);
const quat_of = (bone, out) => { bone.matrixWorld.decompose(_p, out, _s); return out; };

function set_world_quat(bone, q) {
  if (!bone.parent) { bone.quaternion.copy(q); bone.updateWorldMatrix(false, true); return; }
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
    // Повторно используется резервным путём для morph targets в minY(); этот метод вызывается
    // многократно в цикле сходимости коррекции пола для IK.
    this._minYVertex = new THREE.Vector3();

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
    // Опорные кватернионы ретаргетинга хранятся в индексированных массивах;
    // после создания скелета дополнительная таблица обратных матриц привязки не нужна.
    this.retargetPoseWorldQ = new Float64Array(this.bones.length * 4);
    this.retargetPoseWorldQInverse = new Float64Array(this.bones.length * 4);
    this.retargetPoseVersion = 0;
    for (let bone_index = 0; bone_index < this.bones.length; bone_index++) {
      const b = this.bones[bone_index];
      this.bindLocalByIndex[bone_index] = {
        p: b.position.clone(), q: b.quaternion.clone(), s: b.scale.clone(),
      };
      const q = new THREE.Quaternion(), p = new THREE.Vector3();
      b.matrixWorld.decompose(p, q, _s);
      this.bindWorldQByIndex[bone_index] = q;
      this.bindWorldPByIndex[bone_index] = p;
      const inverse_offset = bone_index * 4;
      this.retargetPoseWorldQ[inverse_offset] = q.x;
      this.retargetPoseWorldQ[inverse_offset + 1] = q.y;
      this.retargetPoseWorldQ[inverse_offset + 2] = q.z;
      this.retargetPoseWorldQ[inverse_offset + 3] = q.w;
      this.retargetPoseWorldQInverse[inverse_offset] = -q.x;
      this.retargetPoseWorldQInverse[inverse_offset + 1] = -q.y;
      this.retargetPoseWorldQInverse[inverse_offset + 2] = -q.z;
      this.retargetPoseWorldQInverse[inverse_offset + 3] = q.w;
    }

    // Масштаб скелета в мировых координатах. Используемые скелеты Mixamo имеют
    // равномерный масштаб; среднее геометрическое устойчиво при небольшой неравномерности модели.
    this.unit = 1;
    if (this.hips.parent) {
      this.hips.parent.matrixWorld.decompose(_p, _q1, _s);
      this.unit = Math.cbrt(Math.abs(_s.x * _s.y * _s.z)) || 1;
    }

    this._buildLegs();
    this._buildVertexSets();

    root.position.copy(original_position);
    root.updateMatrixWorld(true);
  }

  // Необязательная виртуальная опорная поза для калибровки. Если переопределения не заданы,
  // в качестве опорной используется импортированная поза привязки/покоя, что сохраняет
  // исходный результат ретаргетинга. Значения — мировые кватернионы, заданные для костей
  // по каноническим именам или именам экспортёра: [x, y, z, w] либо {x, y, z, w}.
  setRetargetPose(overrides = {}) {
    for (let i = 0; i < this.bindWorldQByIndex.length; i++) this.bindWorldQByIndex[i].toArray(this.retargetPoseWorldQ, i * 4);
    for (const [name, value] of Object.entries(overrides || {})) {
      const index = this.canon.indexOf(semantic_bone_name(name));
      if (index < 0) continue;
      const raw = Array.isArray(value) || ArrayBuffer.isView(value)
        ? value
        : [value?.x, value?.y, value?.z, value?.w];
      if (raw.length < 4 || raw.slice(0, 4).some(component => !Number.isFinite(Number(component)))) {
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
      };
      if (!(leg.upper && leg.lower && leg.foot)) continue;
      const wp = bone => this.bindWorldPByIndex[this.index.get(bone)];
      leg.l1 = wp(leg.upper).distanceTo(wp(leg.lower));
      leg.l2 = wp(leg.lower).distanceTo(wp(leg.foot));
      if (!leg.toe) leg.toe = leg.foot;
      if (leg.l1 > 1e-6 && leg.l2 > 1e-6) this.legs[side] = leg;
    }
  }

  // Кэширование всех данных вершин, необходимых для точного вычисления высоты CPU-скиннингом.
  // Вместо тысяч вызовов SkinnedMesh.getVertexPosition() на каждом
  // кадре один раз сохраняются вершины в пространстве привязки и четыре влияния костей.
  // При каждом запросе обновляется одна строка Y для каждой кости, затем каждая вершина
  // вычисляется по тому же уравнению линейного смешивания скиннинга, что и в Three.js.
  _buildVertexSets() {
    const v = new THREE.Vector3();
    const base = new THREE.Vector4();
    const box = new THREE.Box3();
    let ground = Infinity;
    const per_mesh = [];

    for (const mesh of this.meshes) {
      const geometry = mesh.geometry;
      const position = geometry.attributes.position;
      const skin_index = geometry.attributes.skinIndex;
      const skin_weight = geometry.attributes.skinWeight;
      if (!position || !skin_index || !skin_weight || skin_index.itemSize < 4 || skin_weight.itemSize < 4) {
        throw new Error(`Rig: ${mesh.name || "SkinnedMesh"} должен содержать position, skinIndex и skinWeight с четырьмя влияниями`);
      }
      const bone_names = mesh.skeleton.bones.map(b => semantic_bone_name(b.name));
      const foot_side_by_bone = new Uint8Array(bone_names.length); // геометрия подошвы: 0=тело, 1=левая сторона, 2=правая сторона
      const floor_excluded_side_by_bone = new Uint8Array(bone_names.length); // геометрия голени и стопы обрабатывается IK ног
      for (let bone_index = 0; bone_index < bone_names.length; bone_index++) {
        const name = bone_names[bone_index];
        if (name.startsWith("leftfoot") || name.startsWith("lefttoe")) foot_side_by_bone[bone_index] = 1;
        else if (name.startsWith("rightfoot") || name.startsWith("righttoe")) foot_side_by_bone[bone_index] = 2;
        if (name.startsWith("leftleg") || name.startsWith("leftfoot") || name.startsWith("lefttoe")) floor_excluded_side_by_bone[bone_index] = 1;
        else if (name.startsWith("rightleg") || name.startsWith("rightfoot") || name.startsWith("righttoe")) floor_excluded_side_by_bone[bone_index] = 2;
      }
      const count = position.count;
      const dominant_bone = new Int32Array(count);
      const base_positions = new Float64Array(count * 4);
      const influence_indices = new Uint32Array(count * 4);
      const influence_weights = new Float64Array(count * 4);

      for (let i = 0; i < count; i++) {
        base.set(position.getX(i), position.getY(i), position.getZ(i), 1)
          .applyMatrix4(mesh.bindMatrix);
        const offset = i * 4;
        base_positions[offset] = base.x;
        base_positions[offset + 1] = base.y;
        base_positions[offset + 2] = base.z;
        base_positions[offset + 3] = base.w;

        let best_weight = -1, bestIndex = 0;
        for (let k = 0; k < 4; k++) {
          const weight = skin_weight.getComponent(i, k);
          const bone_index = skin_index.getComponent(i, k);
          if (!Number.isFinite(weight) || weight < 0) {
            throw new Error(`Rig: некорректный skin_weight у ${mesh.name || "mesh"}, vertex ${i}`);
          }
          if (weight > 0 && (!Number.isInteger(bone_index) || bone_index < 0 || bone_index >= mesh.skeleton.bones.length)) {
            throw new Error(`Rig: skin_index вне диапазона у ${mesh.name || "mesh"}, vertex ${i}`);
          }
          influence_indices[offset + k] = bone_index;
          influence_weights[offset + k] = weight;
          if (weight > best_weight) {
            best_weight = weight;
            bestIndex = bone_index;
          }
        }
        dominant_bone[i] = bestIndex;

        // Границы исходной позы привязки вычисляются средствами самого Three.js;
        // это однократная операция, которая также проверяет базис ускоренного алгоритма.
        mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
        if (v.y < ground) ground = v.y;
        box.expandByPoint(v);
      }

      const skin_cache = {
        mesh,
        base_positions,
        influence_indices,
        influence_weights,
        dominant_bone,
        // Позиции morph targets меняются динамически и не входят в кэшированные
        // данные пространства привязки; для таких сеток сохраняется точный путь Three.js.
        fastSkinning: !(geometry.morphAttributes?.position?.length),
        yCoefficients: new Float64Array(mesh.skeleton.bones.length * 4),
        yRowNorms: new Float64Array(mesh.skeleton.bones.length),
      };
      per_mesh.push({ mesh, dominant_bone, foot_side_by_bone, floor_excluded_side_by_bone, skin_cache });
    }

    this.groundY = Number.isFinite(ground) ? ground : 0;
    this.hipsHeight = this.bindWorldPByIndex[this.index.get(this.hips)].y - this.groundY;
    if (!(this.hipsHeight > 1e-5)) {
      throw new Error(`Rig: некорректная высота таза над полом (${this.hipsHeight})`);
    }

    this.footSets = { left: [], right: [] };
    this.bodySets = [];

    for (const { mesh, dominant_bone, foot_side_by_bone, floor_excluded_side_by_bone, skin_cache } of per_mesh) {
      const foot = { left: [], right: [] };
      const body = [];
      for (let i = 0; i < dominant_bone.length; i++) {
        const foot_side = foot_side_by_bone[dominant_bone[i]];
        if (foot_side === 1) foot.left.push(i);
        else if (foot_side === 2) foot.right.push(i);
        else if (!floor_excluded_side_by_bone[dominant_bone[i]]) body.push(i);
      }
      for (const side of foot_sides) {
        if (foot[side].length) {
          this.footSets[side].push({
            mesh, indices: Int32Array.from(foot[side]), skin_cache,
          });
        }
      }
      if (body.length) {
        this.bodySets.push({
          mesh, indices: Int32Array.from(body), skin_cache,
        });
      }
    }

    // Кластеризуются все точные наборы данных скиннинга. Вершины не прореживаются:
    // граница учитывает матрицу каждой кости, её линейный масштаб и сумму весов скиннинга.
    for (const side of foot_sides) {
      for (const set of this.footSets[side]) set.minYClusters = build_min_yclusters(set);
    }
    for (const set of this.bodySets) set.minYClusters = build_min_yclusters(set);

    this.bindBox = box;

    // Автоматическая калибровка скелета выполняется по импортированной позе привязки
    // и геометрии после скиннинга. Сначала она только описывает модель: на решение IK влияют
    // лишь надёжные признаки, подтверждённые геометрией (например, чрезмерно крупная обувь).
    // Измерения роста, единиц и длины конечностей также используются в профиле ретаргетинга
    // между исходным и целевым скелетами, поэтому не нужны ручные константы для каждой модели.
    this.soleSupportRequired = this._detectSoleSupport();

    // Если измеренная длина обуви необычно велика относительно голени, начала костей
    // плохо подходят в качестве опорных точек контакта с полом. Кэшируются реальные
    // точки пятки и передней части стопы только для такой обуви; для обычной обуви
    // сохраняются исходные опорные точки Foot/ToeBase. Имя файла модели не проверяется.
    this.bootSoleSupport = null;
    if (foot_sides.some(side => this.soleSupportRequired[side])) {
      this.bootSoleSupport = Object.create(null);
      const world_point = new THREE.Vector3();
      const local_point = new THREE.Vector3();
      const forward_local = new THREE.Vector3();
      const foot_world_inverse = new THREE.Matrix4();
      const toe_world_inverse = new THREE.Matrix4();
      const toe_world = new THREE.Vector3();
      const foot_world = new THREE.Vector3();
      for (const side of foot_sides) {
        const leg = this.legs[side];
        const sets = this.footSets[side] || [];
        if (!leg || !sets.length || !this.soleSupportRequired[side]) continue;
        const foot_bone = leg.foot;
        const toe_bone = leg.toe;
        foot_world_inverse.copy(foot_bone.matrixWorld).invert();
        toe_world_inverse.copy(toe_bone.matrixWorld).invert();
        forward_local.subVectors(toe_bone.getWorldPosition(toe_world), foot_bone.getWorldPosition(foot_world))
          .transformDirection(foot_world_inverse);
        if (forward_local.lengthSq() < 1e-10) continue;
        forward_local.normalize();

        // Хранение выборок вершин в упакованных числовых буферах. Прежнее представление
        // с отдельным объектом на вершину создавало два клона Vector3 и
        // один объект для каждой вершины стопы при инициализации скелета.
        const candidate_count = sets.reduce((count, set) => count + set.indices.length, 0);
        const world_x = new Float64Array(candidate_count);
        const world_y = new Float64Array(candidate_count);
        const world_z = new Float64Array(candidate_count);
        const local_x = new Float64Array(candidate_count);
        const local_y = new Float64Array(candidate_count);
        const local_z = new Float64Array(candidate_count);
        const projections = new Float64Array(candidate_count);
        let sample_count = 0, minY = Infinity, maxY = -Infinity;
        for (const set of sets) {
          const mesh = set.mesh;
          for (let j = 0; j < set.indices.length; j++) {
            mesh.getVertexPosition(set.indices[j], world_point).applyMatrix4(mesh.matrixWorld);
            if (world_point.y < minY) minY = world_point.y;
            if (world_point.y > maxY) maxY = world_point.y;
            local_point.copy(world_point).applyMatrix4(foot_world_inverse);
            world_x[sample_count] = world_point.x;
            world_y[sample_count] = world_point.y;
            world_z[sample_count] = world_point.z;
            local_x[sample_count] = local_point.x;
            local_y[sample_count] = local_point.y;
            local_z[sample_count] = local_point.z;
            projections[sample_count] = local_point.dot(forward_local);
            sample_count++;
          }
        }
        if (sample_count < 8 || !Number.isFinite(minY)) continue;
        const sole_band = Math.max(0.008 * this.unit, (maxY - minY) * 0.16);
        let sole_count = 0, minProjection = Infinity, maxProjection = -Infinity;
        for (let i = 0; i < sample_count; i++) {
          if (world_y[i] > minY + sole_band) continue;
          sole_count++;
          minProjection = Math.min(minProjection, projections[i]);
          maxProjection = Math.max(maxProjection, projections[i]);
        }
        if (sole_count < 4) continue;
        const span = Math.max(maxProjection - minProjection, 1e-6);
        const heel_band = span * 0.18;
        const toe_band = span * 0.18;
        const heel = new THREE.Vector3(), forefootWorldPoint = new THREE.Vector3();
        let heel_count = 0, forefootCount = 0;
        for (let i = 0; i < sample_count; i++) {
          if (world_y[i] > minY + sole_band) continue;
          const projection = projections[i];
          if (projection <= minProjection + heel_band) {
            heel.x += local_x[i]; heel.y += local_y[i]; heel.z += local_z[i]; heel_count++;
          }
          if (projection >= maxProjection - toe_band) {
            forefootWorldPoint.x += world_x[i];
            forefootWorldPoint.y += world_y[i];
            forefootWorldPoint.z += world_z[i];
            forefootCount++;
          }
        }
        if (!heel_count || !forefootCount) continue;
        heel.multiplyScalar(1 / heel_count);
        forefootWorldPoint.multiplyScalar(1 / forefootCount);
        // Задняя часть подошвы следует за Foot, передняя — за ToeBase.
        // Хранение каждой точки в пространстве своей кости сохраняет перекат ботинка.
        const forefoot = forefootWorldPoint.clone().applyMatrix4(toe_world_inverse);
        this.bootSoleSupport[side] = {
          heel, forefoot,
          heelBind: heel.clone().applyMatrix4(foot_bone.matrixWorld),
          forefootBind: forefoot.clone().applyMatrix4(toe_bone.matrixWorld),
        };
      }
      if (!Object.keys(this.bootSoleSupport).length) this.bootSoleSupport = null;
    }
  }

  _detectSoleSupport() {
    const required = Object.create(null);
    const world_point = new THREE.Vector3();
    const foot_position = new THREE.Vector3();
    const toe_position = new THREE.Vector3();
    const axis = new THREE.Vector3();

    for (const side of foot_sides) {
      const leg = this.legs[side];
      const sets = this.footSets[side] || [];
      if (!leg || !sets.length) continue;
      leg.foot.getWorldPosition(foot_position);
      leg.toe.getWorldPosition(toe_position);
      axis.subVectors(toe_position, foot_position);
      const bone_span = axis.length();
      if (bone_span > 1e-8) axis.multiplyScalar(1 / bone_span);
      else axis.set(1, 0, 0);
      const across = new THREE.Vector3(-axis.z, 0, axis.x);
      if (across.lengthSq() < 1e-10) across.set(1, 0, 0);
      else across.normalize();

      // Веса скиннинга кости Foot могут включать манжеты и края брюк. Размер обуви определяется
      // только по нижней полосе подошвы, а не по всему набору вершин с весами этой кости.
      // Упакованные числовые выборки не позволяют создавать краткоживущий объект
      // для каждой вершины стопы при инициализации скелета.
      const sample_count = sets.reduce((count, set) => count + set.indices.length, 0);
      const along_samples = new Float64Array(sample_count);
      const lateral_samples = new Float64Array(sample_count);
      const height_samples = new Float64Array(sample_count);
      let sample_index = 0, minY = Infinity, maxY = -Infinity;
      for (const set of sets) {
        for (let i = 0; i < set.indices.length; i++) {
          set.mesh.getVertexPosition(set.indices[i], world_point).applyMatrix4(set.mesh.matrixWorld);
          const y = world_point.y;
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
          along_samples[sample_index] = world_point.dot(axis);
          lateral_samples[sample_index] = world_point.dot(across);
          height_samples[sample_index] = y;
          sample_index++;
        }
      }
      const sole_band = Math.max(0.008 * this.unit, (maxY - minY) * 0.16);
      let min_along = Infinity, maxAlong = -Infinity;
      let min_across = Infinity, maxAcross = -Infinity;
      let sole_vertex_count = 0;
      for (let i = 0; i < sample_count; i++) {
        if (height_samples[i] > minY + sole_band) continue;
        min_along = Math.min(min_along, along_samples[i]);
        maxAlong = Math.max(maxAlong, along_samples[i]);
        min_across = Math.min(min_across, lateral_samples[i]);
        maxAcross = Math.max(maxAcross, lateral_samples[i]);
        sole_vertex_count++;
      }
      const sole_length = Number.isFinite(min_along) ? maxAlong - min_along : 0;
      const sole_width = Number.isFinite(min_across) ? maxAcross - min_across : 0;
      const length_to_lower_leg = sole_length / Math.max(leg.l2, 1e-8);
      const width_to_lower_leg = sole_width / Math.max(leg.l2, 1e-8);
      // Характерный профиль чрезмерно крупного ботинка определяется по ОБОИМ размерам.
      // У некоторых обычных ботинок проекция подошвы велика из-за вершин манжеты,
      // но в этом наборе моделей ни у одной нет одновременно длины более 72% длины голени
      // и ширины более 45%. У Gentleman 4 значения с обеих сторон составляют примерно 76% / 57%.
      const requires_sole_support = sole_vertex_count >= 8 &&
        length_to_lower_leg > 0.72 && width_to_lower_leg > 0.45;
      required[side] = requires_sole_support;
    }

    return required;
  }

  minY(sets) {
    let minimum = Infinity;
    for (let set_index = 0; set_index < sets.length; set_index++) {
      const set = sets[set_index];
      const indices = set.indices;
      const skin_cache = set.skin_cache;
      const mesh = skin_cache.mesh;
      const skeleton = mesh.skeleton;

      if (!skin_cache.fastSkinning) {
        const vertex = this._minYVertex;
        for (let i = 0; i < indices.length; i++) {
          mesh.getVertexPosition(indices[i], vertex).applyMatrix4(mesh.matrixWorld);
          if (vertex.y < minimum) minimum = vertex.y;
        }
        continue;
      }

      // M = meshWorld * bindMatrixInverse. Перенос матрицы M применяется
      // один раз после смешивания — точно так же, как в SkinnedMesh.applyBoneTransform().
      _skin_outer.multiplyMatrices(mesh.matrixWorld, mesh.bindMatrixInverse);
      const outer = _skin_outer.elements;
      const coeff = skin_cache.yCoefficients;
      const row_norms = skin_cache.yRowNorms;
      const outer_y = outer[13];

      for (let bone_index = 0; bone_index < skeleton.bones.length; bone_index++) {
        _skin_inner.multiplyMatrices(
          skeleton.bones[bone_index].matrixWorld,
          skeleton.boneInverses[bone_index],
        );
        _skin_combined.multiplyMatrices(_skin_outer, _skin_inner);
        const e = _skin_combined.elements;
        const offset = bone_index * 4;
        coeff[offset] = e[1];
        coeff[offset + 1] = e[5];
        coeff[offset + 2] = e[9];
        row_norms[bone_index] = Math.hypot(e[1], e[5], e[9]);
        // Внешний перенос удаляется здесь и восстанавливается один раз после
        // суммирования с весами, в том числе если сумма импортированных весов неточна.
        coeff[offset + 3] = e[13] - outer_y;
      }

      const clusters = set.minYClusters;
      if (!clusters) {
        const base = skin_cache.base_positions;
        const bone_indices = skin_cache.influence_indices;
        const weights = skin_cache.influence_weights;
        for (let i = 0; i < indices.length; i++) {
          const offset = indices[i] * 4;
          const x = base[offset], y = base[offset + 1], z = base[offset + 2], w = base[offset + 3];
          let world_y = outer_y;
          for (let k = 0; k < 4; k++) {
            const weight = weights[offset + k];
            if (weight === 0) continue;
            const c = bone_indices[offset + k] * 4;
            world_y += weight * (coeff[c] * x + coeff[c + 1] * y + coeff[c + 2] * z + coeff[c + 3] * w);
          }
          if (world_y < minimum) minimum = world_y;
        }
        continue;
      }

      // Сначала проверяется кластер, который дал предыдущий минимум. Следующая граница
      // остаётся консервативной даже при неравномерном масштабе и ненормированных весах:
      // преобразованная координата Y ограничена выражением center_y - |rowXYZ| * radius;
      // затем минимальная и максимальная суммы весов дают безопасную границу смешивания.
      const previous_best = clusters.best;
      if (previous_best >= 0) {
        minimum = evaluate_min_ycluster(set, previous_best, coeff, outer_y, minimum);
      }
      const cluster_count = clusters.starts.length - 1;
      for (let cluster_index = 0; cluster_index < cluster_count; cluster_index++) {
        if (cluster_index === previous_best) continue;
        const center_offset = cluster_index * 3;
        const cx = clusters.centers[center_offset];
        const cy = clusters.centers[center_offset + 1];
        const cz = clusters.centers[center_offset + 2];
        const radius = clusters.radii[cluster_index];
        let min_bone_bound = Infinity;
        for (let q = clusters.boneStarts[cluster_index]; q < clusters.boneStarts[cluster_index + 1]; q++) {
          const c = clusters.bones[q] * 4;
          const row_norm = row_norms[clusters.bones[q]];
          const center_y = coeff[c] * cx + coeff[c + 1] * cy + coeff[c + 2] * cz + coeff[c + 3];
          const bone_bound = center_y - row_norm * radius;
          if (bone_bound < min_bone_bound) min_bone_bound = bone_bound;
        }
        const weight_bound = min_bone_bound >= 0
          ? min_bone_bound * clusters.min_weight_sums[cluster_index]
          : min_bone_bound * clusters.maxWeightSums[cluster_index];
        const lower_bound = outer_y + weight_bound;
        const safety_margin = Math.max(1e-8, Math.abs(lower_bound) * 1e-8, radius * 1e-8);
        if (Number.isFinite(lower_bound) && lower_bound - safety_margin >= minimum) continue;
        minimum = evaluate_min_ycluster(set, cluster_index, coeff, outer_y, minimum);
      }
    }
    return minimum;
  }

  footMinY(side) { return this.minY(this.footSets[side] || []); }
  bodyMinY() { return this.minY(this.bodySets); }

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

// ------------------------------------------------ анализ анимационного клипа -
// Пороговые значения контакта один раз оцениваются для каждого исходного клипа по его
// собственным измерениям относительно пола. Результат кэшируется для всех целевых моделей.
function analyse_clip(src, clip, fps, looping = true, sample_times = null) {
  let by_rig = _clip_analysis_cache.get(clip);
  if (!by_rig) {
    by_rig = new WeakMap();
    _clip_analysis_cache.set(clip, by_rig);
  }
  let by_fps = by_rig.get(src);
  if (!by_fps) {
    by_fps = new Map();
    by_rig.set(src, by_fps);
  }
  const sample_key = sample_times ? `:${Array.from(sample_times).join(",")}` : "";
  const cache_key = `${fps}:${looping ? 1 : 0}${sample_key}`;
  const cached = by_fps.get(cache_key);
  if (cached) return cached;

  const duration = Math.max(0, Number(clip.duration) || 0);
  const frame_times = sample_times && sample_times.length >= 2
    ? sample_times
    : Array.from({ length: Math.max(2, Math.ceil(duration * fps) + 1) }, (_, frame) =>
      Math.min(frame / fps, duration));
  const frame_count = frame_times.length;
  const times = new Float64Array(frame_count);
  const track_times = new Float32Array(frame_count);
  const position_track_names = new Set();
  for (const track of clip.tracks) {
    if (track.name.endsWith(".position")) {
      position_track_names.add(semantic_bone_name(track.name.slice(0, -".position".length)));
    }
  }
  const heights = {
    left: new Float64Array(frame_count),
    right: new Float64Array(frame_count),
  };
  const velocities = {
    left: new Float64Array(frame_count),
    right: new Float64Array(frame_count),
  };
  const toe_support_velocities = {
    left: new Float64Array(frame_count),
    right: new Float64Array(frame_count),
  };
  const foot_support_velocities = {
    left: new Float64Array(frame_count),
    right: new Float64Array(frame_count),
  };
  const plant_weights = {
    left: new Float32Array(frame_count),
    right: new Float32Array(frame_count),
  };
  const plant_states = {
    left: new Uint8Array(frame_count),
    right: new Uint8Array(frame_count),
  };
  const plant_pivots = {
    left: new Uint8Array(frame_count),
    right: new Uint8Array(frame_count),
  };
  const plant_anchors = {
    left: new Float64Array(frame_count * 3),
    right: new Float64Array(frame_count * 3),
  };
  const source_foot_indices = {
    left: src.legs.left ? src.index.get(src.legs.left.foot) : -1,
    right: src.legs.right ? src.index.get(src.legs.right.foot) : -1,
  };
  const source_toe_indices = {
    left: src.legs.left ? src.index.get(src.legs.left.toe) : -1,
    right: src.legs.right ? src.index.get(src.legs.right.toe) : -1,
  };
  // Исходный клип анализируется только один раз. Эти неизменяемые буферы поз
  // повторно используются для всех целевых скелетов, исключая шесть лишних проходов AnimationMixer.
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
      times[frame] = time;
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

      for (let side_index = 0; side_index < foot_sides.length; side_index++) {
        const side = foot_sides[side_index];
        const y = src.footMinY(side);
        heights[side][frame] = Number.isFinite(y) ? y - src.groundY : NaN;
      }
    }
  } finally {
    mixer.stopAllAction();
    mixer.uncacheRoot(src.root);
    mixer.uncacheClip(clip);
    src.end();
  }

  // Шум пола: устойчивая медиана нижних 15% абсолютных значений высоты подошвы.
  // Фиксированный числовой буфер не создаёт растущие массивы и упакованные числа.
  let absolute_sample_count = 0;
  for (const side of foot_sides) {
    for (const h of heights[side]) if (Number.isFinite(h)) absolute_sample_count++;
  }
  const absolute_samples = new Float64Array(absolute_sample_count);
  let absolute_sample_index = 0;
  for (const side of foot_sides) {
    for (const h of heights[side]) {
      if (Number.isFinite(h)) absolute_samples[absolute_sample_index++] = Math.abs(h);
    }
  }
  absolute_samples.sort();
  const near_ground_count = absolute_sample_count
    ? Math.max(1, Math.ceil(absolute_sample_count * 0.15))
    : 0;
  const near_median = median_sorted(absolute_samples, near_ground_count);
  const deviations = new Float64Array(near_ground_count);
  for (let i = 0; i < near_ground_count; i++) deviations[i] = Math.abs(absolute_samples[i] - near_median);
  deviations.sort();
  const noise_mad = median_sorted(deviations);
  const floor_noise = clamp(near_median + 1.4826 * noise_mad, src.hipsHeight * 0.0005, src.hipsHeight * 0.015);

  const contact_lo = clamp(
    Math.max(src.hipsHeight * 0.008, floor_noise * 2.5),
    src.hipsHeight * 0.005,
    src.hipsHeight * 0.025,
  );
  const contact_hi = clamp(
    Math.max(contact_lo + src.hipsHeight * 0.022, floor_noise * 6 + src.hipsHeight * 0.025),
    contact_lo + src.hipsHeight * 0.015,
    src.hipsHeight * 0.10,
  );

  // Контакт анализируется заранее. При перекате с пятки на носок одна из двух
  // опорных точек (приближение голеностопа/стопы или ToeBase) остаётся стабильнее
  // другой. Использование только скорости ToeBase ошибочно определяет опорную стопу как движущуюся.
  for (let side_index = 0; side_index < foot_sides.length; side_index++) {
    const side = foot_sides[side_index];
    const foot_index = source_foot_indices[side];
    const toe_index = source_toe_indices[side];
    for (let frame = 0; frame < frame_count; frame++) {
      const a = Math.max(0, frame - 1);
      const b = Math.min(frame_count - 1, frame + 1);
      const dt = times[b] - times[a];
      const ha = heights[side][a], hb = heights[side][b];
      velocities[side][frame] = dt > 1e-8 && Number.isFinite(ha) && Number.isFinite(hb)
        ? (hb - ha) / dt : 0;
      if (dt > 1e-8 && foot_index >= 0 && toe_index >= 0) {
        const foot_a = (a * bone_count + foot_index) * 3;
        const foot_b = (b * bone_count + foot_index) * 3;
        const toe_a = (a * bone_count + toe_index) * 3;
        const toe_b = (b * bone_count + toe_index) * 3;
        const fdx = world_positions[foot_b] - world_positions[foot_a];
        const fdy = world_positions[foot_b + 1] - world_positions[foot_a + 1];
        const fdz = world_positions[foot_b + 2] - world_positions[foot_a + 2];
        const tdx = world_positions[toe_b] - world_positions[toe_a];
        const tdy = world_positions[toe_b + 1] - world_positions[toe_a + 1];
        const tdz = world_positions[toe_b + 2] - world_positions[toe_a + 2];
        // Полная трёхмерная скорость конечной точки: отрыв стопы должен обнаруживаться,
        // даже если сначала она поднимается почти вертикально и почти не смещается по XZ.
        foot_support_velocities[side][frame] = Math.hypot(fdx, fdy, fdz) / dt;
        toe_support_velocities[side][frame] = Math.hypot(tdx, tdy, tdz) / dt;
      }
    }

    const contact_candidates = new Float32Array(frame_count);
    const preferred_pivots = new Uint8Array(frame_count);
    let preferred_pivot = foot_pivot.HEEL;
    const pivot_dead_band = src.hipsHeight * 0.012;
    const knee_index = src.legs[side] ? src.index.get(src.legs[side].lower) : -1;
    const kneel_enter = src.hipsHeight * kneel_enter_height;
    const kneel_exit = src.hipsHeight * kneel_exit_height;
    let kneeling = false;
    for (let frame = 0; frame < frame_count; frame++) {
      const h = heights[side][frame];
      const vertical_speed = Math.abs(velocities[side][frame]);
      const toe_speed = toe_support_velocities[side][frame];
      const heel_speed = foot_support_velocities[side][frame];
      const height_contact = Number.isFinite(h)
        ? 1 - smoothstep(contact_lo, contact_hi, Math.max(0, h)) : 0;
      const vertical_contact = 1 - smoothstep(src.hipsHeight * 0.06, src.hipsHeight * 0.28, vertical_speed);
      // При перекате с пятки на носок одна из опорных точек может на мгновение остановиться,
      // поэтому сигналом скорости опоры служит меньшая из двух скоростей. В отличие от
      // прежней проверки только по XZ, здесь учитывается и вертикальная скорость отрыва.
      const support_speed = Math.min(toe_speed, heel_speed);
      const speed_contact = 1 - smoothstep(
        src.hipsHeight * foot_support_speed_lo,
        src.hipsHeight * foot_support_speed_hi,
        support_speed,
      );
      contact_candidates[frame] = height_contact * vertical_contact * speed_contact;

      if (looping) {
        // Для циклических клипов сохраняем улучшенную опору на ToeBase.
        preferred_pivots[frame] = foot_pivot.TOE;
      } else {
        // Для разовых клипов сохраняем адаптивный выбор пятки/носка и опору на пятку при kneeling.
        if (toe_speed + pivot_dead_band < heel_speed) preferred_pivot = foot_pivot.TOE;
        else if (heel_speed + pivot_dead_band < toe_speed) preferred_pivot = foot_pivot.HEEL;
        if (kneel_enter > 0 && knee_index >= 0) {
          const knee_height = world_positions[(frame * bone_count + knee_index) * 3 + 1] - src.groundY;
          if (kneeling ? knee_height > kneel_exit : knee_height < kneel_enter) kneeling = !kneeling;
          if (kneeling) preferred_pivot = foot_pivot.HEEL;
        }
        preferred_pivots[frame] = preferred_pivot;
      }
    }

    const first_foot_offset = Math.max(0, foot_index) * 3;
    const first_toe_offset = Math.max(0, toe_index) * 3;
    const first_pivot = preferred_pivots[0];
    const last_pivot = preferred_pivots[frame_count - 1];
    const first_pivot_index = first_pivot === foot_pivot.TOE ? toe_index : foot_index;
    const last_pivot_index = last_pivot === foot_pivot.TOE ? toe_index : foot_index;
    const first_pivot_offset = Math.max(0, first_pivot_index) * 3;
    const last_pivot_offset = ((frame_count - 1) * bone_count + Math.max(0, last_pivot_index)) * 3;
    const seam_pivot_distance = first_pivot_index >= 0 && last_pivot_index >= 0 && first_pivot === last_pivot
      ? Math.hypot(
        world_positions[last_pivot_offset] - world_positions[first_pivot_offset],
        world_positions[last_pivot_offset + 2] - world_positions[first_pivot_offset + 2],
      ) : Infinity;
    const first_height = heights[side][0];
    const last_height = heights[side][frame_count - 1];
    const seam_height_delta = Number.isFinite(first_height) && Number.isFinite(last_height)
      ? Math.abs(first_height - last_height) : Infinity;
    const seam_tolerance = Math.max(src.hipsHeight * 0.002, floor_noise * 4);
    const opening_contact = foot_index >= 0 && toe_index >= 0 && frame_count > 2 &&
      contact_candidates[0] >= foot_contact_enter && contact_candidates[1] >= foot_contact_enter;
    const loop_seam_contact = opening_contact &&
      contact_candidates[frame_count - 1] >= foot_contact_exit &&
      seam_pivot_distance <= seam_tolerance && seam_height_delta <= seam_tolerance;
    const starts_in_stable_contact = looping ? loop_seam_contact : opening_contact;

    let planted = starts_in_stable_contact;
    let enter_count = starts_in_stable_contact ? 2 : 0;
    let release_count = 0;
    let anchor_pivot = preferred_pivots[0];
    let anchor_x = world_positions[anchor_pivot === foot_pivot.TOE ? first_toe_offset : first_foot_offset];
    let anchor_y = world_positions[(anchor_pivot === foot_pivot.TOE ? first_toe_offset : first_foot_offset) + 1];
    let anchor_z = world_positions[(anchor_pivot === foot_pivot.TOE ? first_toe_offset : first_foot_offset) + 2];
    let plant_weight = starts_in_stable_contact ? 1 : 0;
    let contact_start_weight = plant_weight;
    let phase = starts_in_stable_contact ? foot_phase.PLANTED : foot_phase.SWING;
    let phase_age = 0;

    for (let frame = 0; frame < frame_count; frame++) {
      const candidate = contact_candidates[frame];
      const pivot = preferred_pivots[frame];
      const point_index = pivot === foot_pivot.TOE ? toe_index : foot_index;
      const point_offset = (frame * bone_count + Math.max(0, point_index)) * 3;
      const point_x = world_positions[point_offset];
      const point_y = world_positions[point_offset + 1];
      const point_z = world_positions[point_offset + 2];
      const dt_frame = frame > 0 ? times[frame] - times[frame - 1] : 1 / fps;

      if (!planted) {
        enter_count = candidate >= foot_contact_enter ? enter_count + 1 : 0;
        if (enter_count >= 2 && point_index >= 0) {
          planted = true;
          phase = foot_phase.CONTACT;
          phase_age = 0;
          contact_start_weight = Math.max(plant_weight, candidate);
          release_count = 0;
          anchor_pivot = preferred_pivots[Math.max(0, frame - 1)];
          const capture_index = anchor_pivot === foot_pivot.TOE ? toe_index : foot_index;
          const capture_offset = (Math.max(0, frame - 1) * bone_count + capture_index) * 3;
          anchor_x = world_positions[capture_offset];
          anchor_y = world_positions[capture_offset + 1];
          anchor_z = world_positions[capture_offset + 2];
        }
      } else {
        if (candidate <= foot_contact_exit) release_count++;
        else release_count = 0;
        if (release_count >= 2) {
          planted = false;
          enter_count = 0;
          phase = foot_phase.RELEASE;
          phase_age = 0;
        } else if (pivot !== anchor_pivot && candidate >= foot_contact_exit) {
          // Перенос опоры с приближения пятки/голеностопа на переднюю часть стопы выполняется
          // в момент, когда передняя точка становится стабильнее. Захват выполняется на текущем кадре,
          // поэтому сама передача опоры не вызывает скачка положения.
          anchor_pivot = pivot;
          anchor_x = point_x;
          anchor_y = point_y;
          anchor_z = point_z;
        }
      }

      if (planted) {
        phase_age += Math.max(0, dt_frame);
        if (phase === foot_phase.CONTACT && (plant_weight >= 0.92 || phase_age >= 0.07)) phase = foot_phase.PLANTED;
      } else if (phase === foot_phase.RELEASE) {
        phase_age += Math.max(0, dt_frame);
      }

      // Не следует сглаживать пружиной освобождённую мировую опорную точку: это сохраняет большой вес IK
      // уже после того, как исходная стопа начала движение. Освобождение — короткое плавное затухание,
      // привязанное к событию; затем стопа снова следует исходной FK-анимации.
      if (planted && phase === foot_phase.CONTACT) {
        plant_weight = THREE.MathUtils.lerp(
          contact_start_weight,
          1,
          smoothstep(0, 0.05, phase_age),
        );
      } else if (planted) {
        plant_weight = 1;
      } else if (phase === foot_phase.RELEASE) {
        plant_weight = 1 - smoothstep(0, foot_release_seconds, phase_age);
        if (phase_age >= foot_release_seconds) {
          plant_weight = 0;
          phase = foot_phase.SWING;
          phase_age = 0;
        }
      } else {
        plant_weight = candidate;
      }
      plant_weight = clamp(plant_weight, 0, 1);

      plant_weights[side][frame] = plant_weight;
      plant_states[side][frame] = phase;
      const keep_anchor = planted || phase === foot_phase.RELEASE;
      plant_pivots[side][frame] = keep_anchor ? anchor_pivot : pivot;
      const anchor_offset = frame * 3;
      if (keep_anchor) {
        plant_anchors[side][anchor_offset] = anchor_x;
        plant_anchors[side][anchor_offset + 1] = anchor_y;
        plant_anchors[side][anchor_offset + 2] = anchor_z;
      } else {
        plant_anchors[side][anchor_offset] = point_x;
        plant_anchors[side][anchor_offset + 1] = point_y;
        plant_anchors[side][anchor_offset + 2] = point_z;
      }
    }

    // Если конечная опора циклического клипа несовместима с начальной, её необходимо
    // освободить до зацикливания, а не переносить всё ещё зафиксированную точку на кадр 0.
    if (looping && !loop_seam_contact && frame_count > 3) {
      const start_weight = plant_weights[side][0];
      const end_weight = plant_weights[side][frame_count - 1];
      if (end_weight > start_weight + 0.04) {
        const fade_frames = Math.min(frame_count - 1, Math.max(3, Math.ceil(fps * 0.12)));
        const fade_start = frame_count - fade_frames;
        for (let frame = fade_start; frame < frame_count; frame++) {
          const t = (frame - fade_start) / Math.max(1, fade_frames - 1);
          const fade = smoothstep(0, 1, t);
          plant_weights[side][frame] = clamp(THREE.MathUtils.lerp(plant_weights[side][frame], start_weight, fade), 0, 1);
          if (plant_weights[side][frame] <= 0.05) {
            plant_states[side][frame] = foot_phase.SWING;
            const pivot = preferred_pivots[frame];
            plant_pivots[side][frame] = pivot;
            const idx = pivot === foot_pivot.TOE ? toe_index : foot_index;
            const offset = (frame * bone_count + Math.max(0, idx)) * 3;
            plant_anchors[side][frame * 3] = world_positions[offset];
            plant_anchors[side][frame * 3 + 1] = world_positions[offset + 1];
            plant_anchors[side][frame * 3 + 2] = world_positions[offset + 2];
          } else {
            plant_states[side][frame] = foot_phase.RELEASE;
          }
        }
      }
    }

    // Упреждающее освобождение опоры применяется только к циклическим клипам.
    // Снижается только вес перед отрывом; набор веса опоры не изменяется.
    if (looping) {
      const release_lead = Math.round(foot_release_lead_seconds * fps);
      if (release_lead > 0) {
        const weights = plant_weights[side];
        const original_weights = Float32Array.from(weights);
        for (let frame = 0; frame < frame_count; frame++) {
          const later = original_weights[Math.min(frame_count - 1, frame + release_lead)];
          if (later < weights[frame]) weights[frame] = later;
        }
      }
    }

  }

  // Оценка выпрямления ноги в опорной позе солдата. Это подстраивает мягкое включение
  // IK под фактическую геометрию исходной ноги, а не под фиксированную константу.
  let extension_sum = 0;
  let extension_count = 0;
  for (const side of foot_sides) {
    const leg = src.legs[side];
    if (!leg) continue;
    const direct = src.bindWorldPByIndex[src.index.get(leg.upper)].distanceTo(src.bindWorldPByIndex[src.index.get(leg.foot)]);
    const total = leg.l1 + leg.l2;
    if (total > 1e-6) {
      extension_sum += clamp(direct / total, 0, 1);
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
    heights,
    plant_weights,
    plant_states,
    plant_pivots,
    plant_anchors,
    floor_noise,
    contact_lo,
    contact_hi,
    leg_extension,
    knee_soft_start,
  };
  by_fps.set(cache_key, result);
  return result;
}

// ---------------------------------------------------------- двухзвенный IK ----
// Временные векторы передаются вызывающим кодом и повторно используются при каждом решении.
// Решатель выполняется внутри цикла сходимости коррекции пола на каждом кадре, поэтому
// даже несколько новых векторов здесь создавали бы заметную нагрузку на сборщик мусора.
function solve_leg(leg, target, pole, soft_start, scratch) {
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
  const height_sq = Math.max(0, l1 * l1 - along * along);
  const height = Math.sqrt(height_sq);

  const plane = scratch.plane.copy(pole).addScaledVector(direction, -pole.dot(direction));
  if (plane.lengthSq() < 1e-8) {
    plane.set(0, 0, 1).addScaledVector(direction, -direction.z);
    if (plane.lengthSq() < 1e-8) plane.set(1, 0, 0).addScaledVector(direction, -direction.x);
  }
  plane.normalize();

  const knee_target = scratch.knee_target.copy(hip)
    .addScaledVector(direction, along).addScaledVector(plane, height);
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
    current_foot_q: new THREE.Quaternion(),
    toeCorrection: new THREE.Vector3(),
  };
}

function create_frame_leg_data() {
  const vector = () => new THREE.Vector3();
  return {
    contact: 0,
    pivot: foot_pivot.HEEL,
    weight: 0,
    footRotationWeight: 0,
    minimumAnkleY: 0,
    ankleIK: vector(),
    supportFK: vector(),
    toeTarget: vector(),
    effectiveToeTarget: vector(),
    currentToe: vector(),
    source_pole: vector(),
    sourcePoleAligned: vector(),
    sourcePoleWeight: 0,
    ankle_fk: vector(),
    hip_fk: vector(),
    knee_fk: vector(),
    toe_fk: vector(),
    fkAxis: vector(),
    kneeOffset: vector(),
    footWorldQ: new THREE.Quaternion(),
    previousPivot: -1,
    previousAnkleIK: vector(),
    previousToeTarget: vector(),
    handoffAnkleOffset: vector(),
    handoffToeOffset: vector(),
    pivotHandoffStartFrame: -1,
    desired: vector(),
    hip_now: vector(),
    fk_now: vector(),
    target: vector(),
    direction: vector(),
    pole_from_fk: vector(),
    forward_pole: vector(),
    pole: vector(),
  };
}

function apply_frame_ik(sides, frame_legs, forward, target_rig, soft_start, solve_scratch) {
  for (let side_index = 0; side_index < sides.length; side_index++) {
    const side = sides[side_index];
    const data = frame_legs[side];
    const target_leg = target_rig.legs[side];
    const hip_now = pos_of(target_leg.upper, data.hip_now);
    const fk_now = data.fk_now.copy(data.ankle_fk);
    const pelvis_delta_y = hip_now.y - data.hip_fk.y;
    fk_now.y += pelvis_delta_y;
    // Достижимость цели пересчитывается после каждой итерации коррекции таза/пола. Зафиксированная
    // стопа может быть частично освобождена, если для сохранения зазора над поверхностью телу
    // пришлось бы вытянуть ногу за пределы её физически возможной длины.
    const max_reach = target_leg.l1 + target_leg.l2;
    const current_distance = hip_now.distanceTo(data.ankleIK);
    const air_weight = 1 - smoothstep(max_reach * soft_start, max_reach * 1.005, current_distance);
    const contact_reach = 1 - smoothstep(max_reach * 0.97, max_reach * 1.005, current_distance);
    const effective_contact = data.contact * contact_reach;
    data.weight = effective_contact + (1 - effective_contact) * air_weight;
    // Поворот стопы — независимый канал: в опоре он полностью зафиксирован, а в полёте
    // немного смягчён, чтобы избежать резкого щелчка при достижении ногой предела длины.
    data.footRotationWeight = effective_contact + (1 - effective_contact) * air_weight * 0.9;
    const target = data.target.lerpVectors(fk_now, data.ankleIK, data.weight);
    const unclamped_ankle_y = target.y;
    target.y = Math.max(target.y, data.minimumAnkleY);
    // Цель опорной точки носка должна участвовать в том же смешивании FK/IK, что и голеностоп.
    // Ветка FK следует за коррекцией таза; ветка IK остаётся мировой целью контакта.
    // Ограничение по полу должно перемещать обе опорные точки на одинаковую величину.
    data.effectiveToeTarget.lerpVectors(data.toe_fk, data.toeTarget, data.weight);
    data.effectiveToeTarget.y += pelvis_delta_y * (1 - data.weight) + (target.y - unclamped_ankle_y);

    const direction = data.direction.subVectors(target, hip_now).normalize();
    const pole_from_fk = data.pole_from_fk.copy(data.kneeOffset);
    const pole_length = pole_from_fk.length();
    // Плоскость естественного сгиба целевой ноги оценивается относительно фактической длины ноги,
    // а не роста всего тела. Когда колено согнуто примерно на 1% длины ноги,
    // собственная плоскость FK уже надёжнее задаёт направление, чем опорное направление источника.
    const bend_ratio = pole_length / Math.max(target_leg.l1 + target_leg.l2, 1e-6);
    const pole_weight = smoothstep(0.002, 0.01, bend_ratio);
    if (pole_length > 1e-8) {
      pole_from_fk.multiplyScalar(1 / pole_length);
      pole_from_fk.applyQuaternion(_q1.setFromUnitVectors(data.fkAxis, direction));
    }
    const forward_pole = data.forward_pole.copy(forward)
      .addScaledVector(direction, -forward.dot(direction));
    if (forward_pole.lengthSq() < 1e-8) forward_pole.set(0, 0, 1);
    forward_pole.normalize();

    // Плоскость сгиба целевой FK-анимации имеет приоритет, если она определена достаточно надёжно.
    // Плоскость колена солдата используется только как запасной вариант почти при прямой ноге;
    // это не позволяет переносить изгиб или скручивание исходного скелета в уже корректную целевую ногу.
    const pole = data.pole.copy(forward_pole);
    if (data.sourcePoleWeight > 1e-5 && data.source_pole.lengthSq() > 1e-8) {
      const fallback_weight = data.sourcePoleWeight * (1 - pole_weight);
      if (fallback_weight > 1e-5) {
        const aligned_source_pole = data.sourcePoleAligned.copy(data.source_pole);
        if (aligned_source_pole.dot(forward_pole) < 0) aligned_source_pole.negate();
        pole.lerp(aligned_source_pole, fallback_weight);
      }
    }
    pole.lerp(pole_from_fk, pole_weight).normalize();

    solve_leg(target_leg, target, pole, soft_start, solve_scratch);

    // Поворот стопы смешивается независимо. Компенсация опоры носка нужна только
    // в фазе контакта: коррекция всей цепочки ноги во время движения может внести
    // лишнее осевое скручивание бедра и голени, хотя стопа свободна.
    const current_foot_q = quat_of(target_leg.foot, solve_scratch.current_foot_q);
    current_foot_q.slerp(data.footWorldQ, data.footRotationWeight);
    set_world_quat(target_leg.foot, current_foot_q);

    if (data.pivot === foot_pivot.TOE && data.contact > 1e-4 && data.footRotationWeight > 1e-4) {
      // Единое непрерывное ограничение по опоре передней части стопы для всех клипов.
      // Без отдельных режимов для стояния/ходьбы и без резкого порога длины ноги, дёргающего колено.
      pos_of(target_leg.toe, data.currentToe);
      const correction_weight = smoothstep(0.72, 0.92, data.weight) * data.contact;
      const correction = solve_scratch.toeCorrection
        .subVectors(data.effectiveToeTarget, data.currentToe)
        .multiplyScalar(correction_weight);
      if (correction.lengthSq() > 1e-12) {
        target.add(correction);
        solve_leg(target_leg, target, pole, soft_start, solve_scratch);
        set_world_quat(target_leg.foot, current_foot_q);
      }
    }
  }
}

// Построение и кэширование канонической карты костей целевого и исходного скелетов и профиля масштаба.
// Опорные кватернионы привязки нормализуют ориентацию покоя каждого скелета перед переносом движения.
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
  if (source_index_by_target[hips_index] < 0) throw new Error("Retarget: Hips не сопоставлен с источником");

  const height_scale = trg.hipsHeight / src.hipsHeight;
  const position_scale = height_scale * src.unit / trg.unit;
  const profile = {
    source_index_by_target,
    calibration: { height_scale, position_scale },
    sourcePoseVersion: src.retargetPoseVersion,
    targetPoseVersion: trg.retargetPoseVersion,
  };
  by_target.set(trg, profile);
  return profile;
}

// Для режима без IK достаточно вычислять FK-позу в моменты исходных ключей.
// Объединение временных отметок всех треков сохраняет общую синхронизацию скелета.
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
// Функция retarget_clip(src: Rig, trg: Rig, clip, { enable_ik?, fps? }) возвращает AnimationClip
// =============================================================================
export function retarget_clip(src, trg, clip, options = {}) {
  const fps = Math.max(1, Number(options.fps ?? default_fps) || default_fps);
  const enable_ik = options.enable_ik !== false;
  const looping = options.looping !== false;
  const arm_space = Number(options.arm_space ?? 50);
  if (!Number.isFinite(arm_space) || arm_space < 0 || arm_space > 100) {
    throw new RangeError(`retarget_clip: arm_space должен быть в диапазоне 0..100, получено: ${options.arm_space}`);
  }
  const duration = Math.max(0, Number(clip.duration) || 0);
  const source_key_times = enable_ik ? null : collect_source_key_times(clip, duration);
  const n_frames = source_key_times ? source_key_times.length : Math.max(2, Math.ceil(duration * fps) + 1);
  const analysis = analyse_clip(src, clip, fps, looping, source_key_times);
  const profile = build_retarget_profile(src, trg);

  const scale = profile.calibration.height_scale;
  const position_scale = profile.calibration.position_scale;
  const has_left_leg = enable_ik && src.legs.left && trg.legs.left && src.footSets.left.length && trg.footSets.left.length;
  const has_right_leg = enable_ik && src.legs.right && trg.legs.right && src.footSets.right.length && trg.footSets.right.length;
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
  for (let order_index = 0; order_index < trg.orderIndices.length; order_index++) {
    const target_index = trg.orderIndices[order_index];
    const source_index = source_index_by_target[target_index];
    if (source_index < 0) continue;
    if (target_index !== trg.index.get(trg.hips) &&
        analysis.position_track_names.has(src.canon[source_index])) {
      position_indices.push(target_index);
      has_position_track[target_index] = 1;
    }
  }

  const times = analysis.track_times;
  const quaternion_values = new Array(trg.bones.length).fill(null);
  for (const target_index of mapped_indices) quaternion_values[target_index] = new Float32Array(n_frames * 4);
  const hips_values = new Float32Array(n_frames * 3);
  const position_values = new Array(trg.bones.length).fill(null);
  for (const target_index of position_indices) position_values[target_index] = new Float32Array(n_frames * 3);
  const target_hips_index = trg.index.get(trg.hips);
  const hips_bind_world = trg.bindWorldPByIndex[target_hips_index];
  const source_hips_index = src.index.get(src.hips);
  const source_hips_bind_world = src.bindWorldPByIndex[source_hips_index];

  const delta = new THREE.Quaternion();
  const target_world_q = new THREE.Quaternion();
  const source_world_q = new THREE.Quaternion();
  const source_bind_inverse = new THREE.Quaternion();
  const target_hips_world = new THREE.Vector3();
  // Все временные данные IK для кадров и итераций выделяются один раз на клип.
  const ik_scratch = sides.length ? create_leg_scratch() : null;
  const frame_legs = sides.length ? Object.create(null) : null;
  if (frame_legs) for (const side of sides) frame_legs[side] = create_frame_leg_data();
  const hips_rotation = sides.length ? new THREE.Quaternion() : null;
  const hips_bind_inverse = sides.length
    ? new THREE.Quaternion().fromArray(trg.retargetPoseWorldQInverse, target_hips_index * 4)
    : null;
  const leg_rigs = Object.create(null);
  for (const side of sides) {
    const source_leg = src.legs[side];
    const target_leg = trg.legs[side];
    const source_toe_index = src.index.get(source_leg.toe);
    const target_toe_index = trg.index.get(target_leg.toe);
    leg_rigs[side] = {
      target_leg,
      source_toe_index,
      sourceUpperIndex: src.index.get(source_leg.upper),
      sourceLowerIndex: src.index.get(source_leg.lower),
      sourceFootIndex: src.index.get(source_leg.foot),
      sourceFootBind: src.bindWorldPByIndex[src.index.get(source_leg.foot)],
      targetFootBind: trg.bindWorldPByIndex[trg.index.get(target_leg.foot)],
      sourceToeBind: src.bindWorldPByIndex[source_toe_index],
      targetToeBind: trg.bindWorldPByIndex[target_toe_index],
    };
  }
  const forward = sides.length ? new THREE.Vector3() : null;
  const source_toe = sides.length ? new THREE.Vector3() : null;
  const source_foot = sides.length ? new THREE.Vector3() : null;
  const source_hip_q = sides.length ? new THREE.Quaternion() : null;
  const target_hip_q = sides.length ? new THREE.Quaternion() : null;
  const source_pole = sides.length ? new THREE.Vector3() : null;
  const source_pole_local = sides.length ? new THREE.Vector3() : null;
  const source_hips_inverse = sides.length ? new THREE.Quaternion() : null;
  const source_hip_position = sides.length ? new THREE.Vector3() : null;
  const source_knee_position = sides.length ? new THREE.Vector3() : null;
  const source_ankle_position = sides.length ? new THREE.Vector3() : null;
  const boot_plant_states = Object.create(null);
  if (trg.bootSoleSupport) for (const side of sides) {
    if (trg.bootSoleSupport[side]) boot_plant_states[side] = {
      active: false,
      pivot: -1,
      anchor: new THREE.Vector3(),
    };
  }
  const floor_tolerance = Math.max(1e-5, trg.hipsHeight * 1e-5);
  // Коррекция пола выполняется до сходимости; достаточно большой предельный счётчик защищает от
  // преждевременной остановки при необычно низких аксессуарах или широких элементах одежды.
  const max_floor_iterations = 64;

  trg.begin();
  try {
    trg.restoreBind();

    for (let frame = 0; frame < n_frames; frame++) {
      const source_pose_base = frame * analysis.bone_count;

      // 1) Проход FK: перенос запечённых дельт мирового вращения относительно позы привязки.
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
          source_world_q.fromArray(analysis.world_quaternions, (source_pose_base + source_index) * 4);
          source_bind_inverse.fromArray(src.retargetPoseWorldQInverse, source_index * 4);
          delta.copy(source_world_q).multiply(source_bind_inverse);
          target_world_q.fromArray(trg.retargetPoseWorldQ, target_index * 4).premultiply(delta);
          if (target_bone.parent) {
            target_bone.parent.matrixWorld.decompose(_p, _q3, _s);
            target_bone.quaternion.copy(_q3.invert().multiply(target_world_q));
          } else {
            target_bone.quaternion.copy(target_world_q);
          }

          // Перенос положения всегда активен для анимируемых исходных костей.
          if (has_position_track[target_index]) {
            const source_bind_position = src.bindLocalByIndex[source_index].p;
            const source_position_offset = (source_pose_base + source_index) * 3;
            const source_x = analysis.local_positions[source_position_offset];
            const source_y = analysis.local_positions[source_position_offset + 1];
            const source_z = analysis.local_positions[source_position_offset + 2];
            target_bone.position.set(
              bind.p.x + (source_x - source_bind_position.x) * position_scale,
              bind.p.y + (source_y - source_bind_position.y) * position_scale,
              bind.p.z + (source_z - source_bind_position.z) * position_scale,
            );
          }
        }
        target_bone.updateMatrix();
        if (target_bone.parent) target_bone.matrixWorld.multiplyMatrices(target_bone.parent.matrixWorld, target_bone.matrix);
        else target_bone.matrixWorld.copy(target_bone.matrix);
      }

      // 2) Перенос смещения таза/корня всегда выполняется с учётом масштаба по высоте.
      const source_hips_offset = (source_pose_base + source_hips_index) * 3;
      target_hips_world.set(
        hips_bind_world.x + (analysis.world_positions[source_hips_offset] - source_hips_bind_world.x) * scale,
        hips_bind_world.y + (analysis.world_positions[source_hips_offset + 1] - source_hips_bind_world.y) * scale,
        hips_bind_world.z + (analysis.world_positions[source_hips_offset + 2] - source_hips_bind_world.z) * scale,
      );
      trg.setHipsWorldPosition(target_hips_world);
      trg.updateAll();

      // 3) Адаптивный контакт стопы и IK с учётом достижимости цели.
      if (sides.length) {
        quat_of(trg.hips, hips_rotation).multiply(hips_bind_inverse);
        forward.set(0, 0, 1).applyQuaternion(hips_rotation);

        for (let side_index = 0; side_index < sides.length; side_index++) {
          const side = sides[side_index];
          const data = frame_legs[side];
          const leg_rig = leg_rigs[side];
          const target_leg = leg_rig.target_leg;
          const sample_index = Math.min(frame, analysis.frame_count - 1);
          const sampled_height = analysis.heights[side][sample_index];
          const sole_height = Number.isFinite(sampled_height) ? Math.max(0, sampled_height) : src.hipsHeight;
          let contact = analysis.plant_weights[side][sample_index];

          const source_toe_offset = (source_pose_base + leg_rig.source_toe_index) * 3;
          const source_foot_offset = (source_pose_base + leg_rig.sourceFootIndex) * 3;
          source_toe.fromArray(analysis.world_positions, source_toe_offset);
          source_foot.fromArray(analysis.world_positions, source_foot_offset);
          const pivot = analysis.plant_pivots[side][sample_index];
          const source_pivot = pivot === foot_pivot.TOE ? source_toe : source_foot;
          const source_pivot_bind = pivot === foot_pivot.TOE ? leg_rig.sourceToeBind : leg_rig.sourceFootBind;
          const boot_support = trg.bootSoleSupport?.[side];
          const target_pivot_bind = boot_support
            ? (pivot === foot_pivot.TOE ? boot_support.forefootBind : boot_support.heelBind)
            : (pivot === foot_pivot.TOE ? leg_rig.targetToeBind : leg_rig.targetFootBind);
          const target_support_fk = boot_support
            ? data.supportFK.copy(pivot === foot_pivot.TOE ? boot_support.forefoot : boot_support.heel)
              .applyMatrix4(pivot === foot_pivot.TOE ? target_leg.toe.matrixWorld : target_leg.foot.matrixWorld)
            : (pivot === foot_pivot.TOE ? data.toe_fk : data.ankle_fk);
          const boot_plant = boot_plant_states[side];
          if (boot_support && boot_plant) {
            const phase = analysis.plant_states[side][sample_index];
            // Не начинаем фиксацию стопы во время предварительного затухания фазы SWING. Захватываем
            // собственную точку подошвы целевой FK-позы только после подтверждения CONTACT, чтобы избежать
            // несовпадения опоры в пространстве привязки и исходной опорной точки, способного резко сместить голеностоп.
            if (phase === foot_phase.SWING) {
              boot_plant.active = false;
              boot_plant.pivot = pivot;
              contact = 0;
            } else {
              if (!boot_plant.active || boot_plant.pivot !== pivot) {
                boot_plant.anchor.copy(target_support_fk);
                boot_plant.active = true;
                boot_plant.pivot = pivot;
              }
            }
          }
          const anchor_offset = sample_index * 3;
          const anchor_x = analysis.plant_anchors[side][anchor_offset];
          const anchor_y = analysis.plant_anchors[side][anchor_offset + 1];
          const anchor_z = analysis.plant_anchors[side][anchor_offset + 2];

          const hip_fk = pos_of(target_leg.upper, data.hip_fk);
          const knee_fk = pos_of(target_leg.lower, data.knee_fk);
          const ankle_fk = pos_of(target_leg.foot, data.ankle_fk);
          const toe_fk = pos_of(target_leg.toe, data.toe_fk);
          const sole_offset = trg.footMinY(side) - ankle_fk.y;

          // Приближение пятки/голеностопа и ToeBase — разные опорные точки. Переносим
          // захваченную исходную опору на соответствующую точку целевого скелета, затем
          // восстанавливаем цель голеностопа по текущему смещению от голеностопа до опорной точки.
          // Во время движения следует за опорной точкой, преобразованной костью-владельцем.
          // Перенос только смещения исходной кости игнорирует вращение Foot/ToeBase
          // и может создать большое боковое смещение голеностопа у длинных ботинок.
          const dynamic_pivot_x = boot_support
            ? target_support_fk.x
            : target_pivot_bind.x + scale * (source_pivot.x - source_pivot_bind.x);
          const dynamic_pivot_z = boot_support
            ? target_support_fk.z
            : target_pivot_bind.z + scale * (source_pivot.z - source_pivot_bind.z);
          const planted_pivot_x = boot_support && boot_plant?.active
            ? boot_plant.anchor.x
            : target_pivot_bind.x + scale * (anchor_x - source_pivot_bind.x);
          const planted_pivot_z = boot_support && boot_plant?.active
            ? boot_plant.anchor.z
            : target_pivot_bind.z + scale * (anchor_z - source_pivot_bind.z);
          const target_pivot_x = THREE.MathUtils.lerp(dynamic_pivot_x, planted_pivot_x, contact);
          const target_pivot_z = THREE.MathUtils.lerp(dynamic_pivot_z, planted_pivot_z, contact);
          const support_offset_x = target_support_fk.x - ankle_fk.x;
          const support_offset_z = target_support_fk.z - ankle_fk.z;
          const toe_offset_x = toe_fk.x - ankle_fk.x;
          const toe_offset_y = toe_fk.y - ankle_fk.y;
          const toe_offset_z = toe_fk.z - ankle_fk.z;
          data.pivot = pivot;
          // Голеностоп рассчитывается по реальной опорной точке подошвы, а не по
          // началу кости Foot/ToeBase. Цель носка остаётся связанной с FK-смещением
          // голеностопа, поэтому исходная артикуляция стопы сохраняется.
          data.ankleIK.set(target_pivot_x - support_offset_x,
            trg.groundY + scale * sole_height - sole_offset,
            target_pivot_z - support_offset_z);
          data.toeTarget.set(data.ankleIK.x + toe_offset_x,
            data.ankleIK.y + toe_offset_y,
            data.ankleIK.z + toe_offset_z);
          // У опорных точек пятки и носка разные смещения относительно позы привязки. Смешиваем обе мировые
          // цели во время передачи опоры, чтобы цель голеностопа для IK не прыгала за один кадр.
          if (data.previousPivot !== -1 && pivot !== data.previousPivot) {
            data.handoffAnkleOffset.subVectors(data.previousAnkleIK, data.ankleIK);
            data.handoffToeOffset.subVectors(data.previousToeTarget, data.toeTarget);
            data.pivotHandoffStartFrame = frame;
          }
          if (data.pivotHandoffStartFrame >= 0) {
            const handoff_t = (frame - data.pivotHandoffStartFrame) / foot_pivot_handoff_frames;
            const remaining = 1 - smoothstep(0, 1, handoff_t);
            data.ankleIK.addScaledVector(data.handoffAnkleOffset, remaining);
            data.toeTarget.addScaledVector(data.handoffToeOffset, remaining);
            if (handoff_t >= 1) data.pivotHandoffStartFrame = -1;
          }
          data.previousPivot = pivot;
          data.previousAnkleIK.copy(data.ankleIK);
          data.previousToeTarget.copy(data.toeTarget);
          // Плоскость колена солдата переносится через опорные системы координат таза
          // исходного и целевого скелетов. Сгиб целевой FK-анимации сохраняется как стабилизирующая составляющая.
          const source_hip_offset = (source_pose_base + leg_rigs[side].sourceUpperIndex) * 3;
          const source_knee_offset = (source_pose_base + leg_rigs[side].sourceLowerIndex) * 3;
          const source_ankle_offset = (source_pose_base + leg_rigs[side].sourceFootIndex) * 3;
          source_hip_position.fromArray(analysis.world_positions, source_hip_offset);
          source_knee_position.fromArray(analysis.world_positions, source_knee_offset);
          source_ankle_position.fromArray(analysis.world_positions, source_ankle_offset);
          source_pole.subVectors(source_ankle_position, source_hip_position);
          const source_leg_distance_sq = source_pole.lengthSq();
          if (source_leg_distance_sq > 1e-10) {
            source_pole.multiplyScalar(1 / Math.sqrt(source_leg_distance_sq));
            source_pole_local.subVectors(source_knee_position, source_hip_position);
            source_pole_local.addScaledVector(source_pole, -source_pole_local.dot(source_pole));
            source_hip_q.fromArray(analysis.world_quaternions, (source_pose_base + source_hips_index) * 4);
            source_hips_inverse.copy(source_hip_q).invert();
            quat_of(trg.hips, target_hip_q);
            data.source_pole.copy(source_pole_local).applyQuaternion(source_hips_inverse).applyQuaternion(target_hip_q);
            const source_pole_length = data.source_pole.length() * scale;
            data.sourcePoleWeight = smoothstep(0.004 * trg.hipsHeight, 0.025 * trg.hipsHeight, source_pole_length);
            if (data.source_pole.lengthSq() > 1e-10) data.source_pole.normalize();
          } else {
            data.source_pole.set(0, 0, 0);
            data.sourcePoleWeight = 0;
          }

          // Вес IK для стопы в воздухе на каждом кадре: ослабляем IK до того, как целевая нога
          // превысит комфортную длину, вместо того чтобы растягивать конечность.
          const max_reach = target_leg.l1 + target_leg.l2;
          const target_distance = hip_fk.distanceTo(data.ankleIK);
          const air_ik = 1 - smoothstep(max_reach * analysis.knee_soft_start, max_reach * 1.005, target_distance);
          const weight = contact + (1 - contact) * air_ik;

          data.fkAxis.subVectors(ankle_fk, hip_fk).normalize();
          data.kneeOffset.subVectors(knee_fk, hip_fk);
          data.kneeOffset.addScaledVector(data.fkAxis, -data.kneeOffset.dot(data.fkAxis));
          data.contact = contact;
          data.weight = weight;
          data.footRotationWeight = weight;
          data.minimumAnkleY = trg.groundY - sole_offset;
          quat_of(target_leg.foot, data.footWorldQ);

        }

        // Опускаем таз только тогда, когда зафиксированная стопа физически не может достичь
        // заданной цели. Стопы в воздухе не должны тянуть таз вниз.
        let pelvis_drop = 0;
        for (let side_index = 0; side_index < sides.length; side_index++) {
          const side = sides[side_index];
          const data = frame_legs[side];
          const target_leg = trg.legs[side];
          const desired = data.desired.lerpVectors(data.ankle_fk, data.ankleIK, data.weight);
          desired.y = Math.max(desired.y, data.minimumAnkleY);
          const reach = (target_leg.l1 + target_leg.l2) * 0.985;
          const dx = desired.x - data.hip_fk.x;
          const dz = desired.z - data.hip_fk.z;
          const horizontal_sq = dx * dx + dz * dz;
          if (horizontal_sq < reach * reach) {
            const vertical = data.hip_fk.y - desired.y;
            const allowed_vertical = Math.sqrt(reach * reach - horizontal_sq);
            const needed = vertical - allowed_vertical;
            if (needed > 0) pelvis_drop = Math.max(pelvis_drop, needed * data.contact);
          }
        }

        if (pelvis_drop > 0) {
          target_hips_world.y -= pelvis_drop;
          trg.setHipsWorldPosition(target_hips_world);
          trg.updateAll();
        }

        // Повторно используем те же буферы ноги и временных векторов на протяжении всей
        // сходимости коррекции пола; меняются только преобразования костей и скалярные значения.
        let floor_iters = 0;
        while (floor_iters < max_floor_iterations) {
          apply_frame_ik(sides, frame_legs, forward, trg, analysis.knee_soft_start, ik_scratch);
          floor_iters++;
          const body_minimum = trg.bodyMinY();
          const deficit = Number.isFinite(body_minimum) ? trg.groundY - body_minimum : 0;
          if (deficit <= floor_tolerance) break;
          target_hips_world.y += deficit;
          trg.setHipsWorldPosition(target_hips_world);
          trg.updateAll();
        }
      }

      // Сохранение преобразований после ретаргетинга для текущего обработанного кадра.
      for (let bone_index = 0; bone_index < mapped_indices.length; bone_index++) {
        const target_index = mapped_indices[bone_index];
        const bone = trg.bones[target_index];
        const q = bone.quaternion;
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
      for (let bone_index = 0; bone_index < position_indices.length; bone_index++) {
        const bone = trg.bones[position_indices[bone_index]];
        const values = position_values[position_indices[bone_index]];
        const offset = frame * 3;
        values[offset] = bone.position.x;
        values[offset + 1] = bone.position.y;
        values[offset + 2] = bone.position.z;
      }
    }
  } finally {
    trg.end();
  }

  // При IK сохраняем каждый шаг 60-FPS-сетки, чтобы не терять короткие фазы контакта стопы.
  // Без IK временная сетка состоит только из исходных ключей и границ клипа.
  // Коррекция Arm-Space применяется к готовым ключам после FK/IK-ретаргетинга,
  // поэтому параметр одинаково действует в обоих режимах и не вмешивается в IK ног.
  apply_arm_space_to_quaternion_values(trg, mapped_indices, quaternion_values, n_frames, arm_space);

  const tracks = [];
  for (const target_index of mapped_indices) {
    const bone = trg.bones[target_index];
    const values = quaternion_values[target_index];
    // Знаки кватернионов приводятся к непрерывной последовательности, чтобы избежать вращательных скачков.
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
    // KeyframeTrack сохраняет ссылку на Float32Array, поэтому все треки могут безопасно использовать
    // одну неизменяемую временную шкалу вместо создания отдельной копии для каждой кости.
    tracks.push(new THREE.QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, values));
  }
  tracks.push(new THREE.VectorKeyframeTrack(`${trg.hips.name}.position`, times, hips_values));
  for (const target_index of position_indices) {
    const bone = trg.bones[target_index];
    const values = position_values[target_index];
    tracks.push(new THREE.VectorKeyframeTrack(`${bone.name}.position`, times, values));
  }

  const output = new THREE.AnimationClip(`${clip.name}_r`, duration, tracks);
  output.userData = {
    scale,
    enable_ik,
    looping,
    arm_space,
    retargetCalibration: profile.calibration,
    autoParams: {
      floor_noise: analysis.floor_noise,
      contact_lo: analysis.contact_lo,
      contact_hi: analysis.contact_hi,
      leg_extension: analysis.leg_extension,
      knee_soft_start: analysis.knee_soft_start,
    },
  };
  return output;
}

// ----------------------------------------------------- подготовка клипа ---
export function detect_unit_scale(clip, hips_bone) {
  const ratios = [];
  let max_abs = 0;
  // Треки положения содержат локальные смещения костей. Сравнение их устойчивой медианной величины
  // с соответствующим локальным смещением в позе привязки позволяет обнаружить несоответствие единиц
  // (см/м и подобные различия в десять раз), не искажая оценку скачками корневой кости.
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
    // Единицы измерения в экспорте из DCC обычно отличаются степенями десяти. Устойчивая
    // оценка округляется до ближайшей степени десяти только при согласованных измерениях.
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