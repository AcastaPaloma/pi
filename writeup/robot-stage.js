// The inference page's 3D stage: two SO-101 followers, an overhead gantry camera and two wrist
// cameras, in the same coordinates as pi_sim (meters, z up, table top at z = 0). Robot motion is
// scripted from IK waypoints; nothing here runs a model. play(spec) choreographs one timeline block.
import * as THREE from './vendor/three.module.min.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const ACCENT = new THREE.Color('#1f3a8a'), PENDING = new THREE.Color('#d9480f');
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = t => t * t * (3 - 2 * t);
const easeInOut = t => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const OVERLAY = 1; // layer for helpers the sensor cameras must not see

// ---- Scene layout (pi_sim/scene.py and so101_camera_rig.json) ----
const BASES = { left: V(-0.18, -0.22, 0), right: V(0.18, -0.22, 0) };
// pi_sim's home pose: gripper about 6 cm above the table, pointing down. Arms rest here.
const HOME = [0, -0.8, 0.8, 1.57, -0.0487];
const OVERHEAD = { pos: V(0, -0.03, 0.7), fovy: 47.8 };
const SINK = { x: 0, y: 0, w: 0.24, d: 0.15, h: 0.055, t: 0.008 };
// The faucet sits at the back-right corner, where the right arm can work its lever (IK error under 1 mm).
const FAUCET = { x: 0.04, y: 0.09 }, LEVER = { pivot: V(0.0525, 0.09, 0.06), length: 0.05, open: 0.6 };
const PLATE = V(-0.025, -0.005, 0.0065), TRAY = V(-0.175, -0.02, 0), SPONGE_REST = V(-0.175, -0.02, 0.016);
const NOZZLE = V(0.04, 0.036, 0.098);
const forward = side => side === 'right' ? V(0.1025 - 0.18, 0.09 + 0.22, -0.3).normalize() : V(0, 1, -1).normalize();

// ---- Scripted subtasks: waypoints for the active arm, solved by IK at load ----
function circle(center, radius, loops, count, z) {
  return Array.from({ length: count }, (_, i) => {
    const a = 2 * Math.PI * loops * (i + 1) / count;
    return { p: [center.x + radius * Math.sin(a), center.y + radius * Math.cos(a), z], axis: 'down', grip: 0.42, d: 0.075, ease: 'linear' };
  });
}
const SCRIPTS = [
  { arm: 'right', start: { lever: 0, sponge: 'tray' }, way: [
    { home: true, grip: 0.08 },
    { p: [0.1225, 0.04, 0.06], axis: 'fwd', grip: 0.08, d: 1.4 },
    { p: [0.1005, 0.083, 0.047], axis: 'fwd', grip: 0.08, d: 0.9 },
    { p: [0.0925, 0.083, 0.08], axis: 'fwd', grip: 0.08, d: 1.3, lever: [0, LEVER.open] },
    { p: [0.1225, 0.02, 0.105], axis: 'fwd', grip: 0.08, d: 0.9 },
    { home: true, grip: 0.08, d: 1.3 }] },
  { arm: 'left', start: { lever: LEVER.open, sponge: 'tray' }, way: [
    { home: true, grip: 1.25 },
    { p: [TRAY.x, TRAY.y, 0.085], axis: 'down', grip: 1.25, d: 1.3 },
    { p: [TRAY.x, TRAY.y, 0.026], axis: 'down', grip: 1.25, d: 0.8 },
    { p: [TRAY.x, TRAY.y, 0.026], axis: 'down', grip: 0.42, d: 0.5, event: 'grasp' },
    { p: [TRAY.x, TRAY.y, 0.085], axis: 'down', grip: 0.42, d: 0.7 },
    { p: [PLATE.x, PLATE.y, 0.07], axis: 'down', grip: 0.42, d: 1.2 },
    { p: [PLATE.x, PLATE.y, 0.034], axis: 'down', grip: 0.42, d: 0.7 },
    ...circle(PLATE, 0.022, 2.5, 40, 0.034),
    { p: [PLATE.x, PLATE.y, 0.075], axis: 'down', grip: 0.42, d: 0.7 }] },
  { arm: 'left', start: { lever: LEVER.open, sponge: 'held' }, way: [
    { p: [PLATE.x, PLATE.y, 0.075], axis: 'down', grip: 0.42 },
    { p: [TRAY.x, TRAY.y, 0.09], axis: 'down', grip: 0.42, d: 1.4 },
    { p: [TRAY.x, TRAY.y, 0.028], axis: 'down', grip: 0.42, d: 0.9 },
    { p: [TRAY.x, TRAY.y, 0.028], axis: 'down', grip: 1.25, d: 0.5, event: 'release' },
    { p: [TRAY.x, TRAY.y, 0.09], axis: 'down', grip: 1.25, d: 0.7 },
    { home: true, grip: 1.25, d: 1.2 }] },
  { arm: 'right', start: { lever: LEVER.open, sponge: 'tray' }, way: [
    { home: true, grip: 0.08 },
    { p: [0.0985, 0.07, 0.13], axis: 'fwd', grip: 0.08, d: 1.4 },
    { p: [0.0925, 0.086, 0.1], axis: 'fwd', grip: 0.08, d: 0.8 },
    { p: [0.1005, 0.086, 0.072], axis: 'fwd', grip: 0.08, d: 1.2, lever: [LEVER.open, 0] },
    { p: [0.1225, 0.04, 0.12], axis: 'fwd', grip: 0.08, d: 0.8 },
    { home: true, grip: 0.08, d: 1.3 }] }
];

// ---- Asset loading ----
async function loadSO101(base, fetcher = fetch) {
  const [meta, bin] = await Promise.all([
    fetcher(base + 'so101.json').then(r => { if (!r.ok) throw new Error('so101.json'); return r.json(); }),
    fetcher(base + 'so101.bin').then(r => { if (!r.ok) throw new Error('so101.bin'); return r.arrayBuffer(); })]);
  const geometries = {};
  for (const [name, m] of Object.entries(meta.meshes)) {
    const q = new Int16Array(bin, m.position[0], m.vertices * 3), position = new Float32Array(m.vertices * 3);
    for (let i = 0; i < position.length; i++) position[i] = (q[i] + 32768) * m.scale[i % 3] + m.min[i % 3];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(position, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Int8Array(bin, m.normal[0], m.vertices * 3), 3, true));
    g.setIndex(new THREE.BufferAttribute(m.wideIndex ? new Uint32Array(bin, m.index[0], m.triangles * 3) : new Uint16Array(bin, m.index[0], m.triangles * 3), 1));
    g.computeBoundingSphere();
    geometries[name] = g;
  }
  return { meta, geometries };
}
const quatWXYZ = ([w, x, y, z]) => new THREE.Quaternion(x, y, z, w);

function buildArm(asset, side, materials) {
  const joints = {}, servoMaterial = materials.servo.clone(), parts = { servoMaterial };
  const root = new THREE.Group(); root.name = `${side}-pedestal`;
  root.position.copy(BASES[side]); root.rotation.z = Math.PI / 2;
  (function body(b, parent) {
    const group = new THREE.Group(); group.position.fromArray(b.pos); group.quaternion.copy(quatWXYZ(b.quat)); parent.add(group);
    let holder = group;
    if (b.joint) {
      holder = new THREE.Group(); group.add(holder);
      joints[b.joint.name] = { node: holder, axis: V(...b.joint.axis), range: b.joint.range, value: 0 };
    }
    for (const geom of b.geoms) {
      const dark = geom.rgba[0] < .3;
      const mesh = new THREE.Mesh(asset.geometries[geom.mesh], dark ? servoMaterial : materials.plastic);
      mesh.position.fromArray(geom.pos); mesh.quaternion.copy(quatWXYZ(geom.quat));
      mesh.castShadow = mesh.receiveShadow = true; holder.add(mesh);
    }
    if (b.sites.gripperframe) { parts.tip = new THREE.Object3D(); parts.tip.position.fromArray(b.sites.gripperframe.pos); holder.add(parts.tip); parts.hand = holder; }
    if (b.cameras.wrist) {
      const c = b.cameras.wrist, cam = new THREE.PerspectiveCamera(c.fovy, 4 / 3, 0.005, 4);
      cam.position.fromArray(c.pos); cam.quaternion.copy(quatWXYZ(c.quat)); holder.add(cam);
      cam.add(cameraBody(0.038, 0.007, 0.0043, 0.0095, materials));
      parts.camera = cam;
    }
    for (const child of b.children) body(child, holder);
  })(asset.meta.root, root);
  return { side, root, joints, ...parts, q: [0, 0, 0, 0, 0], grip: 1 };
}
function cameraBody(board, lensRadius, lensZ, boardZ, materials) {
  // A 32 mm Arducam board with its lens; the optical center is the parent camera's origin.
  const group = new THREE.Group();
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(lensRadius, lensRadius, 0.0095, 20), materials.lens);
  lens.rotation.x = Math.PI / 2; lens.position.z = lensZ;
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(board, board, 0.0016), materials.pcb);
  pcb.position.z = boardZ;
  const glass = new THREE.Mesh(new THREE.CircleGeometry(lensRadius * .62, 20), materials.glass);
  glass.position.z = lensZ - 0.0048; glass.rotation.y = Math.PI;
  for (const m of [lens, pcb]) m.castShadow = true;
  group.add(lens, pcb, glass);
  return group;
}
const JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'];
function setArm(arm, q, grip) {
  JOINTS.forEach((name, i) => { const j = arm.joints[name]; j.node.quaternion.setFromAxisAngle(j.axis, q[i]); });
  const g = arm.joints.gripper; g.node.quaternion.setFromAxisAngle(g.axis, grip);
  arm.q = q; arm.grip = grip;
}
const _m = new THREE.Vector3();
function toolAxis(arm, out) { const e = arm.hand.matrixWorld.elements; return out.set(-e[8], -e[9], -e[10]).normalize(); }

// Damped least squares on position plus a soft tool-axis constraint. wrist_roll is held fixed.
function solveIK(arm, target, axis, seed, w = .35) {
  const q = seed.slice(), want = axis, h = 1e-4, lambda = .012;
  const p = V(), a = V(), r = new Float64Array(6), J = Array.from({ length: 6 }, () => new Float64Array(4));
  const fk = qq => { setArm(arm, qq, arm.grip); arm.root.updateMatrixWorld(true); p.setFromMatrixPosition(arm.tip.matrixWorld); toolAxis(arm, a); };
  for (let it = 0; it < 160; it++) {
    fk(q);
    r[0] = target.x - p.x; r[1] = target.y - p.y; r[2] = target.z - p.z;
    r[3] = w * (want.x - a.x); r[4] = w * (want.y - a.y); r[5] = w * (want.z - a.z);
    if (Math.hypot(r[0], r[1], r[2]) < 2e-4 && Math.hypot(r[3], r[4], r[5]) < 4e-3) break;
    const p0 = p.clone(), a0 = a.clone();
    for (let j = 0; j < 4; j++) {
      q[j] += h; fk(q); q[j] -= h;
      J[0][j] = (p.x - p0.x) / h; J[1][j] = (p.y - p0.y) / h; J[2][j] = (p.z - p0.z) / h;
      J[3][j] = w * (a.x - a0.x) / h; J[4][j] = w * (a.y - a0.y) / h; J[5][j] = w * (a.z - a0.z) / h;
    }
    // (JᵀJ + λ²I) dq = Jᵀ r, solved by Gaussian elimination on the 4 × 4 system.
    const A = Array.from({ length: 4 }, (_, i) => { const row = new Float64Array(5); for (let j = 0; j < 4; j++) { let s = i === j ? lambda * lambda : 0; for (let k = 0; k < 6; k++) s += J[k][i] * J[k][j]; row[j] = s; } let s = 0; for (let k = 0; k < 6; k++) s += J[k][i] * r[k]; row[4] = s; return row; });
    for (let i = 0; i < 4; i++) { let best = i; for (let k = i + 1; k < 4; k++) if (Math.abs(A[k][i]) > Math.abs(A[best][i])) best = k; [A[i], A[best]] = [A[best], A[i]]; for (let k = i + 1; k < 4; k++) { const f = A[k][i] / A[i][i]; for (let j = i; j < 5; j++) A[k][j] -= f * A[i][j]; } }
    const dq = new Float64Array(4); for (let i = 3; i >= 0; i--) { let s = A[i][4]; for (let j = i + 1; j < 4; j++) s -= A[i][j] * dq[j]; dq[i] = s / A[i][i]; }
    for (let j = 0; j < 4; j++) { const range = arm.joints[JOINTS[j]].range; q[j] = clamp(q[j] + clamp(dq[j], -.25, .25), range[0], range[1]); }
  }
  fk(q);
  return { q, error: p.distanceTo(target) };
}

// ---- Props ----
function buildProps(materials) {
  const group = new THREE.Group(), add = (mesh, x, y, z) => { mesh.position.set(x, y, z); mesh.castShadow = mesh.receiveShadow = true; group.add(mesh); return mesh; };
  const { w, d, h, t } = SINK;
  add(new THREE.Mesh(new THREE.BoxGeometry(w, d, t * .75), materials.ceramic), SINK.x, SINK.y, t * .375);
  add(new THREE.Mesh(new THREE.BoxGeometry(w, t, h), materials.ceramic), SINK.x, SINK.y - d / 2 + t / 2, h / 2);
  add(new THREE.Mesh(new THREE.BoxGeometry(w, t, h), materials.ceramic), SINK.x, SINK.y + d / 2 - t / 2, h / 2);
  add(new THREE.Mesh(new THREE.BoxGeometry(t, d, h), materials.ceramic), SINK.x - w / 2 + t / 2, SINK.y, h / 2);
  add(new THREE.Mesh(new THREE.BoxGeometry(t, d, h), materials.ceramic), SINK.x + w / 2 - t / 2, SINK.y, h / 2);
  add(new THREE.Mesh(new THREE.CylinderGeometry(.008, .008, .001, 20).rotateX(Math.PI / 2), materials.drain), SINK.x - .085, SINK.y + .035, t * .75 + .0006);
  // Faucet: a chrome pipe arching over the basin, with a side lever.
  add(new THREE.Mesh(new THREE.CylinderGeometry(.015, .017, .016, 28).rotateX(Math.PI / 2), materials.chrome), FAUCET.x, FAUCET.y, .008);
  const fx = FAUCET.x, pipe = new THREE.CatmullRomCurve3([V(fx, .09, .014), V(fx, .09, .09), V(fx, .085, .118), V(fx, .065, .128), V(fx, .045, .118), V(fx, .036, .104)]);
  add(new THREE.Mesh(new THREE.TubeGeometry(pipe, 48, .0065, 14), materials.chrome), 0, 0, 0);
  add(new THREE.Mesh(new THREE.CylinderGeometry(.0085, .0075, .008, 18).rotateX(Math.PI / 2), materials.chrome), NOZZLE.x, NOZZLE.y, NOZZLE.z + .002);
  const lever = new THREE.Group(); lever.position.copy(LEVER.pivot); group.add(lever);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(.009, .009, .012, 20), materials.chrome); lever.add(hub);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(LEVER.length, .007, .006), materials.chrome); arm.position.x = LEVER.length / 2; lever.add(arm);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(.0065, 16, 12), materials.chrome); knob.position.x = LEVER.length; lever.add(knob);
  for (const m of [hub, arm, knob]) m.castShadow = true;
  // Plate and sponge tray.
  const plateProfile = [[0, 0], [.03, 0], [.034, .0015], [.046, .006], [.05, .0075], [.0505, .0068], [.046, .0049], [.034, .0006], [0, .0006]].map(([x, y]) => new THREE.Vector2(x, y));
  const plate = add(new THREE.Mesh(new THREE.LatheGeometry(plateProfile, 48).rotateX(Math.PI / 2), materials.plate), PLATE.x, PLATE.y, PLATE.z);
  const trayProfile = [[0, 0], [.036, 0], [.042, .008], [.0425, .0078], [.037, .0012], [0, .0012]].map(([x, y]) => new THREE.Vector2(x, y));
  add(new THREE.Mesh(new THREE.LatheGeometry(trayProfile, 40).rotateX(Math.PI / 2), materials.tray), TRAY.x, TRAY.y, 0);
  const sponge = new THREE.Group();
  const foam = new THREE.Mesh(new THREE.BoxGeometry(.048, .03, .015), materials.sponge); foam.position.z = .0025;
  const scrub = new THREE.Mesh(new THREE.BoxGeometry(.048, .03, .005), materials.scrub); scrub.position.z = -.0075;
  for (const m of [foam, scrub]) { m.castShadow = true; sponge.add(m); }
  sponge.position.copy(SPONGE_REST); group.add(sponge);
  // Water: a slightly wavering stream and a ripple ring where it lands.
  const stream = new THREE.Mesh(new THREE.CylinderGeometry(.0038, .005, NOZZLE.z - .012, 14, 1, true).rotateX(Math.PI / 2), materials.water);
  stream.position.set(NOZZLE.x, NOZZLE.y, (NOZZLE.z + .012) / 2); group.add(stream);
  const ripple = new THREE.Mesh(new THREE.RingGeometry(.006, .009, 32), materials.ripple); ripple.position.set(NOZZLE.x, NOZZLE.y, .0122); group.add(ripple);
  return { group, lever, sponge, stream, ripple, plate };
}

function buildRig(materials) {
  const group = new THREE.Group(), r = .01;
  const bar = (a, b, radius = r) => { const dir = b.clone().sub(a), mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, dir.length(), 16), materials.gantry); mesh.position.copy(a).add(b).multiplyScalar(.5); mesh.quaternion.setFromUnitVectors(V(0, 1, 0), dir.normalize()); mesh.castShadow = true; group.add(mesh); return mesh; };
  const barY = OVERHEAD.pos.y + .18, barZ = OVERHEAD.pos.z + .055;
  bar(V(-.56, barY, -.74), V(-.56, barY, barZ)); bar(V(.56, barY, -.74), V(.56, barY, barZ));
  bar(V(-.56, barY, barZ), V(.56, barY, barZ));
  bar(V(0, OVERHEAD.pos.y, OVERHEAD.pos.z + .025), V(0, barY, barZ), .008);
  const camera = new THREE.PerspectiveCamera(OVERHEAD.fovy, 4 / 3, .02, 4);
  camera.position.copy(OVERHEAD.pos); group.add(camera);
  camera.add(cameraBody(.032, .006, .009, .017, materials));
  // Table, legs, and a floor that fades into the backdrop.
  const top = new THREE.Mesh(new THREE.BoxGeometry(1, .8, .05), materials.table); top.position.z = -.025; top.receiveShadow = true; group.add(top);
  for (const [x, y] of [[-.46, -.36], [.46, -.36], [-.46, .36], [.46, .36]]) { const leg = new THREE.Mesh(new THREE.BoxGeometry(.04, .04, .7), materials.leg); leg.position.set(x, y, -.4); group.add(leg); }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), materials.floor); floor.position.z = -.75; floor.receiveShadow = true; group.add(floor);
  return { group, camera };
}

function frustum(camera, depth, aspect = 4 / 3) {
  const t = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * depth, s = t * aspect;
  const c = [V(-s, -t, -depth), V(s, -t, -depth), V(s, t, -depth), V(-s, t, -depth)], o = V(0, 0, 0);
  const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([o, c[0], o, c[1], o, c[2], o, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]),
    new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0, depthWrite: false }));
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([o, c[0], c[1], o, c[1], c[2], o, c[2], c[3], o, c[3], c[0]]),
    new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  const group = new THREE.Group(); group.add(lines, faces);
  group.traverse(o => o.layers.set(OVERLAY));
  camera.add(group);
  return { group, set(v) { lines.material.opacity = .85 * v; faces.material.opacity = .09 * v; group.visible = v > .01; } };
}

function materialsFor() {
  const m = (o) => new THREE.MeshStandardMaterial(o);
  return {
    plastic: m({ color: '#f2b81c', roughness: .5, metalness: 0 }),
    servo: m({ color: '#1c1c1e', roughness: .45, metalness: .15, emissive: '#000000' }),
    lens: m({ color: '#121314', roughness: .3, metalness: .4 }),
    glass: m({ color: '#0b1a33', roughness: .05, metalness: .9 }),
    pcb: m({ color: '#145230', roughness: .6 }),
    ceramic: m({ color: '#f7f6f2', roughness: .22, metalness: 0 }),
    drain: m({ color: '#8d9196', roughness: .3, metalness: .9 }),
    chrome: m({ color: '#e9ecef', roughness: .16, metalness: 1 }),
    plate: m({ color: '#fbfaf7', roughness: .18 }),
    tray: m({ color: '#c9ced6', roughness: .4 }),
    sponge: m({ color: '#f2a65a', roughness: .95 }),
    scrub: m({ color: '#2f7d4f', roughness: .9 }),
    water: m({ color: '#b9dcff', roughness: .05, metalness: 0, transparent: true, opacity: .78, emissive: '#7fb6ee', emissiveIntensity: .45, depthWrite: false }),
    ripple: new THREE.MeshBasicMaterial({ color: '#a8d1f5', transparent: true, opacity: .6, depthWrite: false }),
    gantry: m({ color: '#3b3f45', roughness: .45, metalness: .6 }),
    table: m({ color: '#6e5f4b', roughness: .78 }),
    leg: m({ color: '#3a3633', roughness: .9 }),
    floor: m({ color: '#242322', roughness: 1 })
  };
}

function studioEnvironment(renderer) {
  // A soft room: bright panels overhead and to the sides, for the chrome and ceramic to reflect.
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#e9e5de');
  const panel = (w, h, pos, look, intensity) => { const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(intensity, intensity, intensity), side: THREE.DoubleSide })); mesh.position.copy(pos); mesh.lookAt(look); scene.add(mesh); };
  panel(6, 6, V(0, 0, 5), V(0, 0, 0), 2.2); panel(4, 3, V(5, -2, 2), V(0, 0, 0), 1.4); panel(4, 3, V(-5, 2, 2), V(0, 0, 0), 1.1); panel(8, 2, V(0, 6, 1), V(0, 0, 0), .8);
  const pmrem = new THREE.PMREMGenerator(renderer), texture = pmrem.fromScene(scene, .04).texture; pmrem.dispose();
  return texture;
}

// Solve every waypoint once, warm-starting from the previous one; home waypoints use HOME directly.
function solveScripts(arms) {
  const axisFor = (spec, side) => spec === 'down' ? V(0, 0, -1) : forward(side);
  for (const script of SCRIPTS) {
    let q = HOME;
    script.errors = [];
    script.solved = script.way.map(w => {
      if (w.home) { script.errors.push(0); return (q = HOME); }
      const result = solveIK(arms[script.arm], V(...w.p), axisFor(w.axis, script.arm), q, w.axis === 'down' ? .35 : .06);
      script.errors.push(result.error);
      return (q = result.q);
    });
    const total = script.way.reduce((s, w) => s + (w.d ?? 0), 0);
    let t = 0; script.times = script.way.map(w => (t += (w.d ?? 0)) / total);
  }
}

// Kinematics without a renderer, for tests and tooling.
export const kinematics = { loadSO101, buildArm, materialsFor, solveIK, solveScripts, setArm, toolAxis, SCRIPTS, HOME, BASES };

// ---- Stage ----
export async function createStage(container, { subtasks, task, base = './assets/so101/' }) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const canvas = document.createElement('canvas'); canvas.className = 'stage-canvas';
  container.prepend(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = .95;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const small = () => Math.min(container.clientWidth, container.clientHeight) < 420;
  const scene = new THREE.Scene();
  // A dark studio so the yellow arms, white sink, and chrome stand out.
  scene.background = new THREE.Color('#2a2928'); scene.fog = new THREE.Fog('#2a2928', 2.2, 5.5);
  scene.environment = studioEnvironment(renderer); scene.environmentIntensity = .4;
  scene.add(new THREE.HemisphereLight('#fff4e0', '#3a342c', .5));
  const key = new THREE.DirectionalLight('#fff1dc', 3.1); key.position.set(.6, .9, 1.5); key.castShadow = true;
  Object.assign(key.shadow.camera, { left: -.6, right: .6, top: .6, bottom: -.6, near: .2, far: 4 });
  key.shadow.mapSize.setScalar(small() ? 1024 : 2048); key.shadow.bias = -.0004; key.shadow.normalBias = .01; key.shadow.radius = 4;
  const fill = new THREE.DirectionalLight('#dfe8ff', .35); fill.position.set(-1, .6, .7);
  const rim = new THREE.DirectionalLight('#fff6ea', 1.6); rim.position.set(-.3, -1.1, .9);
  scene.add(key, fill, rim);

  const materials = materialsFor();
  const asset = await loadSO101(base);
  const arms = { left: buildArm(asset, 'left', materials), right: buildArm(asset, 'right', materials) };
  const rig = buildRig(materials), props = buildProps(materials);
  scene.add(rig.group, props.group, arms.left.root, arms.right.root);
  const sensors = { overhead: rig.camera, left: arms.left.camera, right: arms.right.camera };
  const frusta = { overhead: frustum(rig.camera, .68), left: frustum(arms.left.camera, .16), right: frustum(arms.right.camera, .16) };

  // Solve every waypoint once. Each script starts from the other arm resting at READY.
  const ready = { left: HOME, right: HOME };
  solveScripts(arms);
  for (const script of SCRIPTS) {
  }
  // Pose of everything at absolute task time T (subtask index + progress).
  function stateAt(T) {
    const k = Math.floor(T), s = ((k % 4) + 4) % 4, p = clamp(T - k, 0, 1), script = SCRIPTS[s];
    let i = 1; while (i < script.times.length - 1 && script.times[i] < p) i++;
    const t0 = script.times[i - 1], t1 = script.times[i], w = script.way[i];
    const u = t1 > t0 ? clamp((p - t0) / (t1 - t0), 0, 1) : 1, e = w.ease === 'linear' ? u : smooth(u);
    const qa = script.solved[i - 1], qb = script.solved[i];
    const q = qa.map((v, j) => v + (qb[j] - v) * e), grip = script.way[i - 1].grip + (w.grip - script.way[i - 1].grip) * e;
    let lever = script.start.lever, sponge = script.start.sponge;
    for (let j = 1; j < script.way.length; j++) {
      const wj = script.way[j], done = j < i || (j === i && u >= 1);
      if (wj.lever) lever = j < i ? wj.lever[1] : j === i ? wj.lever[0] + (wj.lever[1] - wj.lever[0]) * e : lever;
      if (wj.event && done) sponge = wj.event === 'grasp' ? 'held' : 'tray';
    }
    const other = script.arm === 'left' ? 'right' : 'left';
    return { s, p, active: script.arm, arms: { [script.arm]: { q, grip }, [other]: { q: ready[other], grip: other === 'left' ? 1.25 : .08 } }, lever, sponge };
  }
  const _q = new THREE.Quaternion();
  function apply(state) {
    for (const side of ['left', 'right']) setArm(arms[side], state.arms[side].q, state.arms[side].grip);
    props.lever.rotation.y = -state.lever;
    const flowing = state.lever > .3;
    props.stream.visible = props.ripple.visible = flowing;
    scene.updateMatrixWorld(true);
    if (state.sponge === 'held') {
      const arm = arms.left; arm.tip.getWorldPosition(props.sponge.position); arm.hand.getWorldQuaternion(_q);
      props.sponge.quaternion.copy(_q); props.sponge.position.addScaledVector(toolAxis(arm, _m), -.004);
    } else { props.sponge.position.copy(SPONGE_REST); props.sponge.quaternion.identity(); }
  }
  // Dense table of the active gripper's path, for drawing action chunks.
  const pathCache = new Map();
  function tipAt(T) {
    const key = Math.round(T * 600);
    if (!pathCache.has(key)) { const st = stateAt(key / 600); apply(st); pathCache.set(key, arms[st.active].tip.getWorldPosition(V())); }
    return pathCache.get(key);
  }

  // ---- HUD ----
  const hud = document.createElement('div'); hud.className = 'stage-hud';
  hud.innerHTML = `<div class="hud-top"><div class="hud-caption"><span class="hud-label">Task · ${task}</span><span class="hud-subtask" aria-live="polite"></span><span class="hud-note"></span></div><div class="hud-clock"><span class="hud-time"></span><span class="hud-lane"></span><span class="hud-joints"></span></div></div><div class="hud-callout" hidden></div><div class="hud-strip"><div class="hud-group" data-group="frames"><span class="hud-label"><span class="long">Camera frames</span><span class="short">Frames</span></span><div class="hud-thumbs"></div></div><div class="hud-group" data-group="goals"><span class="hud-label"><span class="long">Subgoal images · BAGEL</span><span class="short">BAGEL</span> <b class="goal-status"></b></span><div class="hud-thumbs"></div></div><div class="hud-group" data-group="chunk"><span class="hud-label"><span class="long">Action chunk [50, d]</span><span class="short">Chunk</span> <b class="chunk-status"></b></span><div class="hud-ticks"></div></div></div><div class="shutter" aria-hidden="true"><i></i><i></i><b></b><s></s></div>`;
  container.append(hud);
  const $ = sel => hud.querySelector(sel);
  const thumbs = (group, names) => names.map(name => { const c = document.createElement('canvas'); c.width = 160; c.height = 120; c.title = name; c.dataset.name = name; hud.querySelector(`[data-group=${group}] .hud-thumbs`).append(c); return c; });
  const frameThumbs = thumbs('frames', ['overhead', 'left wrist', 'right wrist']);
  const goalThumbs = thumbs('goals', ['overhead goal', 'left wrist goal', 'right wrist goal']);
  const ticks = Array.from({ length: 50 }, () => { const t = document.createElement('i'); $('.hud-ticks').append(t); return t; });
  for (const c of [...frameThumbs, ...goalThumbs]) { const x = c.getContext('2d'); x.fillStyle = '#e6e2da'; x.fillRect(0, 0, 160, 120); }

  // ---- Rendering helpers ----
  const view = new THREE.PerspectiveCamera(32, 1, .01, 20); view.layers.enable(OVERLAY);
  view.position.set(.62, -.78, .5); view.lookAt(0, -.04, .06);
  function resize() {
    const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio, small() ? 1.75 : 2)); renderer.setSize(w, h, false);
    view.aspect = w / h; view.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container); resize();
  // Draw what a sensor sees into a 4:3 thumbnail: render it to the main canvas and copy it in the same task.
  function snapshot(camera, target) {
    const w = canvas.width, h = canvas.height, aspect = w / h, fov = camera.fov;
    const hTan = Math.tan(THREE.MathUtils.degToRad(fov) / 2) * 4 / 3;
    camera.aspect = aspect; camera.fov = aspect >= 4 / 3 ? fov : THREE.MathUtils.radToDeg(2 * Math.atan(hTan / aspect)); camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    const sw = aspect >= 4 / 3 ? h * 4 / 3 : w, sh = aspect >= 4 / 3 ? h : w * 3 / 4;
    target.getContext('2d').drawImage(canvas, (w - sw) / 2, (h - sh) / 2, sw, sh, 0, 0, target.width, target.height);
    camera.fov = fov; camera.aspect = 4 / 3; camera.updateProjectionMatrix();
  }
  const goalImages = new Map();
  function goalsFor(s) {
    if (!goalImages.has(s)) {
      apply(stateAt(s + .999));
      goalImages.set(s, ['overhead', 'left', 'right'].map(name => { const c = document.createElement('canvas'); c.width = 320; c.height = 240; snapshot(sensors[name], c); return c; }));
      apply(stateAt(T));
    }
    return goalImages.get(s);
  }
  const noise = Array.from({ length: 6 }, () => { const c = document.createElement('canvas'); c.width = 80; c.height = 60; const x = c.getContext('2d'), d = x.createImageData(80, 60); for (let i = 0; i < d.data.length; i += 4) { const v = Math.random() * 255; d.data[i] = v; d.data[i + 1] = v * (.8 + Math.random() * .3); d.data[i + 2] = v * (.8 + Math.random() * .4); d.data[i + 3] = 255; } x.putImageData(d, 0, 0); return c; });
  function drawGoals(s, amount) {
    // amount in [0, 1]: 0 is pure noise, 1 the finished image (25 denoising steps).
    const images = s == null ? null : goalsFor(s), step = Math.round(amount * 25);
    goalThumbs.forEach((thumb, i) => {
      const x = thumb.getContext('2d');
      if (!images) { x.fillStyle = '#e6e2da'; x.fillRect(0, 0, 160, 120); return; }
      x.globalAlpha = 1; x.drawImage(images[i], 0, 0, 160, 120);
      if (step < 25) { x.globalAlpha = (1 - step / 25) ** .7; x.imageSmoothingEnabled = false; x.drawImage(noise[(step + i) % noise.length], 0, 0, 160, 120); x.imageSmoothingEnabled = true; x.globalAlpha = 1; }
    });
    $('.goal-status').textContent = !images ? '' : step >= 25 ? '· ready' : `· step ${step}/25`;
  }

  // ---- Action chunk dots ----
  const dotGeometry = new THREE.SphereGeometry(.0042, 12, 8);
  const dots = new THREE.InstancedMesh(dotGeometry, new THREE.MeshBasicMaterial({ toneMapped: false }), 100);
  dots.layers.set(OVERLAY); dots.frustumCulled = false; dots.count = 0; scene.add(dots);
  const pathLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 50 }, () => V())), new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: .35, depthWrite: false }));
  pathLine.layers.set(OVERLAY); pathLine.frustumCulled = false; scene.add(pathLine);
  const chunks = { active: null, pending: null };
  function makeChunk(t0, t1, seed) {
    const rand = mulberry(seed), times = Array.from({ length: 50 }, (_, i) => t0 + (t1 - t0) * i / 49);
    // Noise positions sit on a loose spiral around the path, like the drawing's swirl.
    return { t0, t1, times, target: times.map(t => tipAt(t).clone()), noise: times.map((_, i) => { const a = i * .9 + rand() * 6, r = .03 + rand() * .05; return V(Math.cos(a) * r, Math.sin(a) * r, (rand() - .5) * .06); }), level: 0, from: 0, start: 0, dur: 0 };
  }
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const _mat = new THREE.Matrix4(), _p = V(), _c = new THREE.Color();
  function drawChunks(now) {
    let n = 0;
    for (const [kind, chunk] of Object.entries(chunks)) {
      if (!chunk) continue;
      const level = chunk.dur ? chunk.from + (chunk.level - chunk.from) * easeInOut(clamp((now - chunk.start) / chunk.dur, 0, 1)) : chunk.level;
      chunk.shown = level;
      chunk.times.forEach((t, i) => {
        const done = kind === 'active' && T >= t - 1e-4, stale = kind === 'active' && chunk.skip && i < chunk.skip;
        _p.copy(chunk.target[i]).addScaledVector(chunk.noise[i], 1 - level);
        const scale = done ? .45 : stale ? .7 : 1;
        _mat.makeScale(scale, scale, scale).setPosition(_p); dots.setMatrixAt(n, _mat);
        _c.copy(kind === 'pending' ? PENDING : ACCENT).lerp(new THREE.Color('#b9b9b2'), done || stale ? .75 : (1 - level) * .6);
        dots.setColorAt(n, _c); n++;
      });
      if (kind === 'active') { const pos = pathLine.geometry.attributes.position; chunk.target.forEach((p, i) => pos.setXYZ(i, p.x, p.y, p.z)); pos.needsUpdate = true; pathLine.material.opacity = .4 * level; }
    }
    pathLine.visible = Boolean(chunks.active);
    dots.count = n; dots.instanceMatrix.needsUpdate = true; if (dots.instanceColor) dots.instanceColor.needsUpdate = true;
    const active = chunks.active;
    ticks.forEach((tick, i) => { tick.className = !active ? '' : T >= active.times[i] - 1e-4 ? 'done' : active.skip && i < active.skip ? 'skip' : 'ready'; });
    $('.chunk-status').textContent = !active ? '' : active.shown < .999 ? `· denoising ${Math.min(5, Math.floor(active.shown * 5) + 1)}/5` : `· ${ticks.filter(t => t.className === 'done').length} of 50 executed`;
  }

  // ---- Camera shots ----
  const look = (pos, target, fov) => { const c = new THREE.PerspectiveCamera(); c.up.set(0, 0, 1); c.position.copy(pos); c.lookAt(target); return { pos: pos.clone(), quat: c.quaternion.clone(), fov }; };
  const portrait = () => container.clientWidth / container.clientHeight < 1.1;
  // Most shots look from the far side of the table, so the arms face the viewer and never hide the sink.
  const far = () => portrait() ? 1.3 : 1;
  const shots = {
    wide: t => { const a = .55 + Math.sin(t * .07) * .1, r = .95 * far(); return look(V(Math.sin(a) * r, Math.cos(a) * r, .62 * far()), V(0, -.09, .05), 30); },
    front: () => look(V(.06, .78 * far(), .5 * far()), V(0, -.09, .07), 30),
    high: () => look(V(.04, .5 * far(), .98 * far()), V(0, -.06, .02), 32),
    rigTop: () => look(V(.24, .36, .98), V(0, -.03, .69), 30),
    sensor: name => () => { const cam = sensors[name]; cam.updateWorldMatrix(true, false); const pos = V(), quat = new THREE.Quaternion(); cam.getWorldPosition(pos); cam.getWorldQuaternion(quat); return { pos, quat, fov: cam.fov * (portrait() ? 1.25 : 1) }; },
    behind: name => () => { const cam = sensors[name]; cam.updateWorldMatrix(true, false); const pos = V(), back = V(0, 0, 1).applyQuaternion(cam.getWorldQuaternion(new THREE.Quaternion())); cam.getWorldPosition(pos); return look(pos.clone().addScaledVector(back, .2).add(V(name === 'left' ? -.05 : .05, .02, .07)), pos, 34); },
    arm: side => () => { const tip = arms[side].tip.getWorldPosition(V()), focus = tip.clone().lerp(V(side === 'left' ? -.08 : .08, -.04, .06), .4); return look(focus.clone().add(V(side === 'left' ? -.38 : .38, .52, .34).multiplyScalar(far())), focus, 32); },
    servo: side => () => { const j = arms[side].joints.elbow_flex.node.getWorldPosition(V()); return look(j.clone().add(V(side === 'left' ? -.13 : .13, .16, .07)), j, 30); }
  };
  let shot = shots.wide, blend = null;
  const current = () => ({ pos: view.position.clone(), quat: view.quaternion.clone(), fov: view.fov });
  function cut(next, dur = 1.3) { blend = reduced.matches || !dur ? null : { from: current(), start: now, dur }; shot = next; }
  function updateView(dt) {
    const s = shot(now / 1000);
    if (blend) {
      const k = easeInOut(clamp((now - blend.start) / (blend.dur * 1000), 0, 1));
      view.position.copy(blend.from.pos).lerp(s.pos, k); view.quaternion.copy(blend.from.quat).slerp(s.quat, k); view.fov = blend.from.fov + (s.fov - blend.from.fov) * k;
      if (k >= 1) blend = null;
    } else {
      const k = 1 - Math.exp(-dt * 6);
      view.position.lerp(s.pos, k); view.quaternion.slerp(s.quat, k); view.fov += (s.fov - view.fov) * k;
    }
    view.updateProjectionMatrix();
  }

  // ---- Effects ----
  const shutter = hud.querySelector('.shutter');
  function click(name, fly = true) {
    // Freeze what the sensor sees, close the shutter, and file the frame in memory.
    const index = { overhead: 0, left: 1, right: 2 }[name], thumb = frameThumbs[index];
    snapshot(sensors[name], thumb);
    if (reduced.matches) return;
    shutter.classList.remove('is-firing'); void shutter.offsetWidth; shutter.classList.add('is-firing');
    if (!fly) { thumb.classList.remove('is-new'); void thumb.offsetWidth; thumb.classList.add('is-new'); return; }
    const ghost = document.createElement('canvas'); ghost.className = 'frame-ghost'; ghost.width = thumb.width; ghost.height = thumb.height;
    ghost.getContext('2d').drawImage(thumb, 0, 0);
    const box = container.getBoundingClientRect(), to = thumb.getBoundingClientRect();
    const w = Math.min(box.width * .62, box.height * .62 * 4 / 3), h = w * 3 / 4;
    Object.assign(ghost.style, { width: `${w}px`, height: `${h}px`, left: `${(box.width - w) / 2}px`, top: `${(box.height - h) / 2}px` });
    hud.append(ghost);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ghost.style.transform = `translate(${to.left - box.left - (box.width - w) / 2}px, ${to.top - box.top - (box.height - h) / 2}px) scale(${to.width / w})`;
      ghost.style.opacity = '.2';
    }));
    setTimeout(() => { ghost.remove(); thumb.classList.remove('is-new'); void thumb.offsetWidth; thumb.classList.add('is-new'); }, 900);
  }
  function callout(text, ms = 2200) {
    const el = $('.hud-callout'); el.textContent = text; el.hidden = !text; clearTimeout(el.timer);
    if (text) el.timer = setTimeout(() => { el.hidden = true; }, ms);
  }
  let frustumGlow = { overhead: 0, left: 0, right: 0 }, jointPulse = 0;
  const flash = (names, v = 1) => names.forEach(n => { frustumGlow[n] = v; });
  function updateJoints() {
    const deg = q => q.slice(0, 5).map(v => Math.round(THREE.MathUtils.radToDeg(v))).join(' ');
    $('.hud-joints').innerHTML = `<span>q<sub>L</sub> ${deg(arms.left.q)}</span><span>q<sub>R</sub> ${deg(arms.right.q)}</span>`;
    $('.hud-joints').classList.toggle('is-pulsing', jointPulse > 0);
  }

  // ---- Beats ----
  let T = 0, motion = null, sequence = [], beatStart = 0, now = performance.now(), typing = null, spec = null, goalAnim = null;
  const at = (seconds, fn) => sequence.push({ at: seconds * 1000, fn, done: false });
  const BEATS = {
    establish() { cut(shots.wide, 2); },
    'capture-full'() {
      cut(shots.rigTop, 1.4);
      at(1.3, () => { flash(['overhead'], 1); callout('Overhead camera · Arducam IMX708 · 640 × 480, resized to 448 × 448'); });
      at(1.6, () => cut(shots.sensor('overhead'), 1)); at(2.8, () => click('overhead'));
      at(3.5, () => cut(shots.behind('left'), 1.1)); at(4.4, () => { flash(['left'], 1); callout('Left wrist camera'); });
      at(4.7, () => cut(shots.sensor('left'), .8)); at(5.7, () => click('left'));
      at(6.3, () => cut(shots.behind('right'), 1.1)); at(7.2, () => { flash(['right'], 1); callout('Right wrist camera'); });
      at(7.5, () => cut(shots.sensor('right'), .8)); at(8.5, () => click('right'));
      at(9.1, () => { cut(shots.front, 1.5); jointPulse = 2.5; callout('Proprioception · 5 joints + gripper per arm · metadata: quality 5, no mistake'); });
    },
    'capture-quick'() {
      cut(shots.wide, 1.2);
      at(1.2, () => { flash(['overhead', 'left', 'right'], 1); jointPulse = 1.6; callout('New observation · 3 camera frames + joint positions'); });
      at(1.5, () => click('overhead', false)); at(1.6, () => click('left', false)); at(1.7, () => click('right', false));
    },
    subtask() { cut(shots.front, 2); },
    subgoal() { cut(shots.high, 1.6); at(.6, () => callout('BAGEL · 25 denoising steps · 1.25 s on 4 × H100 in the paper')); },
    'subgoal-slow'() { cut(shots.wide, 1.6); at(.6, () => callout('BAGEL restarts for the new subtask · it will take 1.5 s here')); },
    'subgoal-land'() { cut(shots.high, 1.2); at(.4, () => callout('New subgoal images arrive · the next chunk uses them')); at(2.8, () => cut(shots.arm('left'), 1.6)); },
    chunk() { cut(shots.arm(activeSide()), 1.5); at(.8, () => callout('Action expert · noise → 50 actions in 5 denoising steps')); },
    'chunk-overlap'() { cut(shots.arm(activeSide()), 1.4); at(.8, () => callout('Next chunk computing (orange) while the arm finishes the current one')); },
    execute() { cut(shots.arm(activeSide()), 1.4); },
    'execute-close'() { cut(shots.servo(activeSide()), 1.3); at(.9, () => callout('a₀ → servos: one action every 20 ms')); at(3.2, () => cut(shots.arm(activeSide()), 1.6)); },
    splice() { cut(shots.arm(activeSide()), 1.2); at(.6, () => callout('Switch: skip the actions already executed, enter the new chunk at a₆')); }
  };
  const activeSide = () => SCRIPTS[((Math.floor(spec?.T ?? T) % 4) + 4) % 4].arm;

  function play(next) {
    // next: { beat, T, dwell, caption, typing, goal: [s, from, to] | null, active, pending, splice, time, lane, laneTitle }
    spec = next; sequence = []; beatStart = now; callout('');
    const jump = Math.abs(next.T - T) > 1.05 || reduced.matches;
    const dur = next.motion ?? Math.max(1.2, next.dwell * .72);
    motion = jump ? null : { from: T, to: next.T, start: now, dur: dur * 1000 };
    if (jump) T = next.T;
    // Caption: type it out when the high-level policy writes it.
    const caption = $('.hud-subtask');
    $('.hud-note').textContent = next.caption == null ? 'waiting for the high-level policy' : next.typing ? 'the high-level policy is writing…' : 'subtask from the high-level policy';
    typing = null;
    if (next.caption == null) caption.textContent = '—';
    else if (next.typing && !reduced.matches) { caption.textContent = ''; typing = { words: next.caption.split(' '), shown: 0, start: now + 700 }; }
    else caption.textContent = next.caption;
    // Subgoal images.
    goalAnim = next.goal ? { s: next.goal[0], from: next.goal[1], to: next.goal[2], start: now + 500, dur: (next.beat === 'subgoal' ? 3.8 : next.beat === 'subgoal-land' ? 1 : next.dwell * .8) * 1000 } : null;
    if (!goalAnim) drawGoals(null, 0);
    // Action chunks: build deterministically from the walkthrough's state.
    const build = (c, kind) => c ? makeChunk(c.t0, c.t1, Math.round(c.t0 * 997) + (kind === 'pending' ? 7 : 0)) : null;
    const animateChunk = (chunk, delay) => { if (!chunk) return; if (reduced.matches) { chunk.level = 1; return; } chunk.level = 0; chunk.from = 0; at(delay, () => { chunk.from = 0; chunk.level = 1; chunk.start = now; chunk.dur = 3600; }); };
    chunks.active = build(next.active, 'active'); chunks.pending = build(next.pending, 'pending');
    if (chunks.active) { chunks.active.level = 1; chunks.active.skip = next.active.skip ?? 0; if (next.active.animate) animateChunk(chunks.active, 1.2); }
    if (chunks.pending) { chunks.pending.level = 1; if (next.pending.animate) animateChunk(chunks.pending, 1.2); }
    $('.hud-time').textContent = next.time ?? '';
    const lane = $('.hud-lane'); lane.textContent = next.laneTitle ?? ''; lane.dataset.lane = next.lane ?? '';
    (BEATS[next.beat] ?? BEATS.establish)();
    if (next.beat !== 'capture-full' && next.beat !== 'capture-quick') { for (const k in frustumGlow) frustumGlow[k] = 0; }
  }

  // ---- Main loop ----
  let last = performance.now(), running = true, frame = 0;
  function tick(time) {
    frame = requestAnimationFrame(tick);
    if (!running) return;
    const dt = Math.min(.05, (time - last) / 1000); last = time; now = time;
    for (const step of sequence) if (!step.done && now - beatStart >= step.at) { step.done = true; step.fn(); }
    if (motion) { const k = clamp((now - motion.start) / motion.dur, 0, 1); T = motion.from + (motion.to - motion.from) * (k < 1 ? easeInOut(k) * .15 + k * .85 : 1); if (k >= 1) motion = null; }
    const state = stateAt(T); apply(state);
    // Servos glow faintly on the arm that is executing actions, only while it moves.
    for (const side of ['left', 'right']) {
      const arm = arms[side], target = motion && Math.abs(motion.to - motion.from) > 1e-6 && side === state.active ? 1 : 0;
      arm.glow = (arm.glow ?? 0) + (target - (arm.glow ?? 0)) * Math.min(1, dt * 4);
      arm.servoMaterial.emissive.copy(ACCENT).multiplyScalar(arm.glow * .16);
    }
    if (typing && now >= typing.start) { const shown = Math.min(typing.words.length, 1 + Math.floor((now - typing.start) / 260)); if (shown !== typing.shown) { typing.shown = shown; $('.hud-subtask').textContent = typing.words.slice(0, shown).join(' '); } if (shown >= typing.words.length) typing = null; }
    if (goalAnim) { const k = clamp((now - goalAnim.start) / goalAnim.dur, 0, 1); drawGoals(goalAnim.s, goalAnim.from + (goalAnim.to - goalAnim.from) * (reduced.matches ? 1 : k)); if (k >= 1) goalAnim = null; }
    for (const [name, f] of Object.entries(frusta)) { frustumGlow[name] = Math.max(0, frustumGlow[name] - dt * .45); f.set(Math.min(1, frustumGlow[name] * 1.4)); }
    jointPulse = Math.max(0, jointPulse - dt); updateJoints();
    const flow = props.stream.visible ? (time / 1000) : 0;
    props.stream.scale.set(1 + Math.sin(flow * 23) * .08, 1 + Math.cos(flow * 19) * .08, 1); props.ripple.scale.setScalar(1 + (flow * 2.2 % 1) * .9); props.ripple.material.opacity = .55 * (1 - flow * 2.2 % 1);
    drawChunks(now); apply(state);
    updateView(dt);
    renderer.render(scene, view);
  }
  document.addEventListener('visibilitychange', () => { running = !document.hidden; last = performance.now(); });
  apply(stateAt(0)); updateView(1); frame = requestAnimationFrame(tick);
  // Prime the camera-frame thumbnails with what each sensor sees at rest.
  click('overhead', false); click('left', false); click('right', false); shutter.classList.remove('is-firing');
  for (const t of frameThumbs) t.classList.remove('is-new');

  return { play, subtasks, debug: { look: (pos, target, fov) => cut(() => look(V(...pos), V(...target), fov), 0), shots, setT: t => { T = t; motion = null; } } };
}
