// The cell, in three dimensions, built from the cell's own numbers.
//
// Nothing here hardcodes a dimension. The tray, the table and the seam all come
// from /api/cell, which reads cell/geometry.py, so if a dimension changes in the
// simulation this view changes with it. A picture of a cell that is not driven by
// that cell is decoration.
//
// The scene is kept in the cell's own frame, metres and Z up, rather than being
// converted into the renderer's usual Y up. Converting would mean every number
// that arrives from the service has to be flipped on the way in, and one missed
// flip is a picture that looks plausible and is wrong.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const COLOUR = {
  sheet: 0x9aa7b4,
  table: 0x39434f,
  frame: 0x232b34,
  floor: 0x11161c,
    // One legend shared with the camera render: blue is where the robot was
  // taught to weld, green is the corrected path the cell welds along, amber is a
  // gap that needs attention, grey is a point the cell refused to weld. Red is
  // reserved for the true joint, which only the camera render can show apart.
  taught: 0x3987e5,
  tracked: 0x2ea043,
  ok: 0x3fb950,
  attention: 0xe3a008,
  refused: 0x6e7681,
  arc: 0xffd28a,
};

/* The viewport is a lit studio rather than a document, so it keeps its own
   environment: what changes with the page theme is the room the cell stands in,
   not the cell. The parts, the torch and the seam markers read the same on both,
   which is the point of leaving them alone. */
const STUDIO = {
  dark: {
    background: 0x070a0e,
    fogNear: 6, fogFar: 16,
    floor: 0x11161c,
    gridMajor: 0x1b2836, gridMinor: 0x141c26,
    skyLight: 0x8fa6c0, groundLight: 0x0a0d12, hemi: 1.05,
    keyLight: 0xffffff, key: 1.5,
    fillLight: 0x93b4ff, fill: 0.5,
  },
  light: {
    background: 0xeef1f4,
    fogNear: 7, fogFar: 19,
    floor: 0xdfe4e9,
    gridMajor: 0xc3ccd4, gridMinor: 0xd6dde3,
    skyLight: 0xffffff, groundLight: 0xb9c4cf, hemi: 1.35,
    keyLight: 0xffffff, key: 1.25,
    fillLight: 0xc8d8f0, fill: 0.45,
  },
};

function studio() {
  // Matches the stylesheet: light unless dark was explicitly chosen. The system
  // preference is deliberately not consulted, or a dark desktop would give a dark
  // viewport inside a light page.
  return document.documentElement.dataset.theme === "dark"
    ? STUDIO.dark : STUDIO.light;
}

export class Cell3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.exaggeration = 1;
    this.spinning = false;
    this.geometry = null;
    this.seamGroup = new THREE.Group();

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = false;

    this.scene = new THREE.Scene();
    const room = studio();
    this.scene.background = new THREE.Color(room.background);
    this.scene.fog = new THREE.Fog(room.background, room.fogNear, room.fogFar);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 80);
    this.camera.up.set(0, 0, 1);              // the cell is Z up
    this.camera.position.set(6.6, -3.2, 2.5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 14;
    this.controls.maxPolarAngle = Math.PI * 0.49;   // stay above the floor

    // Metal with nothing to reflect renders black. Sheet steel is the subject of
    // this view, so it needs an environment to be steel rather than a silhouette.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.#light();
    this.scene.add(this.seamGroup);

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this.#loop();
  }

  #light() {
    const room = studio();
    this.hemi = new THREE.HemisphereLight(room.skyLight, room.groundLight, room.hemi);
    this.scene.add(this.hemi);
    this.key = new THREE.DirectionalLight(room.keyLight, room.key);
    this.key.position.set(6, -5, 6);
    this.scene.add(this.key);
    this.fill = new THREE.DirectionalLight(room.fillLight, room.fill);
    this.fill.position.set(-4, 4, 3);
    this.scene.add(this.fill);
  }

  /** Repaint the room when the page theme changes. The cell itself is untouched. */
  retheme() {
    const room = studio();
    this.scene.background = new THREE.Color(room.background);
    this.scene.fog = new THREE.Fog(room.background, room.fogNear, room.fogFar);
    if (this.hemi) {
      this.hemi.color.setHex(room.skyLight);
      this.hemi.groundColor.setHex(room.groundLight);
      this.hemi.intensity = room.hemi;
    }
    if (this.key) { this.key.color.setHex(room.keyLight); this.key.intensity = room.key; }
    if (this.fill) { this.fill.color.setHex(room.fillLight); this.fill.intensity = room.fill; }
    if (this.grid) {
      this.grid.material.color.setHex(room.gridMajor);
      this.grid.material.needsUpdate = true;
    }
    if (this.floorMesh) this.floorMesh.material.color.setHex(room.floor);
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  #loop() {
    const tick = () => {
      requestAnimationFrame(tick);
      if (this.spinning) {
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.7;
      } else {
        this.controls.autoRotate = false;
      }
      this.controls.update();
      if (this.arc) {
        const t = performance.now() * 0.006;
        this.arc.intensity = 1.6 + Math.sin(t) * 0.5;
      }
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  // ------------------------------------------------------------ the static cell

  build(geometry) {
    this.geometry = geometry;
    const g = geometry;
    const seam = g.seam;
    this.controls.target.set(seam.x_m, seam.y_m, (seam.bottom_z_m + seam.top_z_m) / 2);

    this.#floor(g);
    this.#table(g);
    this.#tray(g);
    this.#sensor(g);
    this.#robot(g);
  }

  #box(w, l, h, x, y, z, colour, opts = {}, parent = this.scene) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, l, h),
      new THREE.MeshStandardMaterial({
        color: colour,
        metalness: opts.metalness ?? 0.7,
        roughness: opts.roughness ?? 0.45,
      })
    );
    mesh.position.set(x, y, z);
    parent.add(mesh);
    return mesh;
  }

  #floor(g) {
    const room = studio();
    const grid = new THREE.GridHelper(24, 48, room.gridMajor, room.gridMinor);
    grid.rotation.x = Math.PI / 2;            // GridHelper is XZ; the cell is XY
    grid.position.set(g.fixture.x_m, 0, 0);
    this.grid = grid;
    this.scene.add(grid);
    this.floorMesh = this.#box(24, 24, 0.02, g.fixture.x_m, 0, -0.011, room.floor,
                               { metalness: 0.0, roughness: 1.0 });
  }

  #table(g) {
    const top = g.fixture.table_top_m;
    this.#box(1.6, 1.7, 0.05, g.fixture.x_m, 0, top - 0.025, COLOUR.table,
              { metalness: 0.8, roughness: 0.4 });
    for (const dx of [-0.7, 0.7]) {
      for (const dy of [-0.72, 0.72]) {
        this.#box(0.08, 0.08, top - 0.05, g.fixture.x_m + dx, dy, (top - 0.05) / 2,
                  COLOUR.frame, { metalness: 0.5, roughness: 0.6 });
      }
    }
    // The two locating pads, set inboard of the welded corner so they do not
    // stand in the weld or in the camera's line to it.
    for (const side of [-1, 1]) {
      this.#box(0.05, 0.28, 0.12, g.fixture.x_m + 0.485, side * 0.34, top + 0.06,
                0x4a2a1c, { metalness: 0.6, roughness: 0.5 });
    }
  }

  #tray(g) {
    const p = g.part, top = g.fixture.table_top_m;
    const hw = p.width_m / 2, hl = p.length_m / 2, t = p.thickness_m;
    const wall = t + p.side_height_m;
    const x0 = g.fixture.x_m, y0 = 0;
    const mat = { metalness: 1.0, roughness: 0.32 };

    // The tray is one group, drawn with its welded corner at the nominal joint,
    // because that corner is where the program was taught to weld. When a
    // measured seam is shown the whole group is moved to where the part actually
    // sat, so the picture shows the part off its mark rather than a seam floating
    // off a tray that is drawn in the wrong place.
    const tray = new THREE.Group();
    this.trayGroup = tray;
    this.scene.add(tray);

    this.#box(p.width_m, p.length_m, t, x0, y0, top + t / 2, COLOUR.sheet, mat, tray);
    // Each side stands on its fold line and leans outward by the sheet thickness,
    // which is what leaves the notch at each corner.
    this.#box(t, p.length_m, wall, x0 + hw + t / 2, y0, top + wall / 2, COLOUR.sheet, mat, tray);
    this.#box(t, p.length_m, wall, x0 - hw - t / 2, y0, top + wall / 2, COLOUR.sheet, mat, tray);
    this.#box(p.width_m, t, wall, x0, y0 + hl + t / 2, top + wall / 2, COLOUR.sheet, mat, tray);
    this.#box(p.width_m, t, wall, x0, y0 - hl - t / 2, top + wall / 2, COLOUR.sheet, mat, tray);
  }

  /** The depth camera on its post.
   *
   *  Built in its own frame with +Z pointing down the sight line, so the housing,
   *  the two imagers and the projector between them all face the work without any
   *  of them having to be positioned by hand in world coordinates. It is a
   *  stereo head with an active projector, which is the class of sensor this
   *  method needs: it has to return usable depth on a bare folded sheet, and a
   *  bare folded sheet has nothing on it to match.
   */
  #sensor(g) {
    const eye = new THREE.Vector3(...g.camera.eye_m);
    const aim = new THREE.Vector3(...g.camera.aim_m);
    const look = aim.clone().sub(eye).normalize();

    const head = new THREE.Group();
    head.position.copy(eye).addScaledVector(look, -0.055);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), look);
    this.scene.add(head);

    const shell = (w, h, d, x, y, z, colour, opts = {}) => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({
          color: colour,
          metalness: opts.metalness ?? 0.55,
          roughness: opts.roughness ?? 0.42,
        })
      );
      mesh.position.set(x, y, z);
      head.add(mesh);
      return mesh;
    };
    const barrel = (r, len, x, colour, opts = {}) => {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, len, 24),
        new THREE.MeshStandardMaterial({
          color: colour,
          metalness: opts.metalness ?? 0.8,
          roughness: opts.roughness ?? 0.3,
          emissive: opts.emissive ?? 0x000000,
          emissiveIntensity: opts.emissiveIntensity ?? 0,
        })
      );
      mesh.rotation.x = Math.PI / 2;          // lie the cylinder along the sight line
      mesh.position.set(x, 0, len / 2);
      head.add(mesh);
      return mesh;
    };

    // Extruded aluminium housing, wide and shallow, the way a stereo bar is built.
    shell(0.175, 0.046, 0.052, 0, 0, 0, 0x6e7885, { metalness: 0.55, roughness: 0.36 });
    // A darker face plate the optics sit behind.
    shell(0.168, 0.038, 0.004, 0, 0, 0.028, 0x14181d, { metalness: 0.2, roughness: 0.55 });
    // Finned back, which is what these run hot enough to need.
    for (let i = -2; i <= 2; i++) {
      shell(0.006, 0.040, 0.020, i * 0.018, 0, -0.034, 0x4e5764, { roughness: 0.55 });
    }

    // Two imagers on a baseline, and the projector between them.
    [-0.062, 0.062].forEach((x) => {
      barrel(0.0175, 0.010, x, 0x8e99a6, { metalness: 0.9, roughness: 0.22 });  // turned ring
      barrel(0.0145, 0.016, x, 0x1a1f25, { metalness: 0.5, roughness: 0.5 });
      barrel(0.0105, 0.020, x, 0x0a1622, { metalness: 0.95, roughness: 0.08 });  // glass
    });
    barrel(0.014, 0.009, 0, 0x8e99a6, { metalness: 0.9, roughness: 0.22 });
    barrel(0.011, 0.014, 0, 0x1a1f25, { metalness: 0.5, roughness: 0.5 });
    barrel(0.0075, 0.018, 0, 0x2a1f0e, {
      metalness: 0.4, roughness: 0.25, emissive: 0x6b4a10, emissiveIntensity: 0.55,
    });
    // Status lamp, so the head reads as powered rather than as a block.
    const lamp = new THREE.Mesh(
      new THREE.SphereGeometry(0.004, 12, 12),
      new THREE.MeshStandardMaterial({
        color: 0x76b900, emissive: 0x76b900, emissiveIntensity: 1.4,
      })
    );
    lamp.position.set(0.078, 0.014, 0.027);
    head.add(lamp);

    // A short throw off the head itself. Without it the housing is a silhouette
    // from every angle that is not directly under the key light.
    const near = new THREE.PointLight(0xbcd0e6, 0.30, 1.1, 2);
    near.position.set(0.10, 0.22, 0.30);
    head.add(near);

    // Tilt bracket under the housing, then the post it hangs off.
    shell(0.05, 0.028, 0.044, 0, -0.037, -0.004, 0x515c69, { metalness: 0.75, roughness: 0.35 });

    const lateral = new THREE.Vector3(-look.y, look.x, 0).normalize().multiplyScalar(0.3);
    const mount = eye.clone().addScaledVector(look, -0.055);
    const foot = new THREE.Vector3(mount.x + lateral.x, mount.y + lateral.y, 0);
    const postTop = mount.z + 0.06;
    this.#box(0.07, 0.07, postTop, foot.x, foot.y, postTop / 2,
              0x3a4552, { metalness: 0.7, roughness: 0.4 });
    this.#box(0.16, 0.16, 0.02, foot.x, foot.y, 0.01, 0x2b333d,
              { metalness: 0.6, roughness: 0.5 });
    // A cross arm from the post to the bracket, short enough not to stand in the view.
    this.#cyl(
      new THREE.Vector3(foot.x, foot.y, postTop - 0.03),
      new THREE.Vector3(mount.x, mount.y, mount.z - 0.05),
      0.016, 0x3a4552, { metalness: 0.75, roughness: 0.35 }
    );

    // What it is looking at, drawn faintly so the standoff is legible.
    const sight = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([eye, aim]),
      new THREE.LineDashedMaterial({ color: 0x3d6ea8, dashSize: 0.03, gapSize: 0.03 })
    );
    sight.computeLineDistances();
    this.scene.add(sight);
  }

  #cyl(a, b, radius, colour, opts = {}) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = Math.max(1e-4, dir.length());
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, len, 16),
      new THREE.MeshStandardMaterial({
        color: colour,
        metalness: opts.metalness ?? 0.6,
        roughness: opts.roughness ?? 0.4,
        emissive: opts.emissive ?? 0x000000,
        emissiveIntensity: opts.emissiveIntensity ?? 0,
      })
    );
    mesh.position.copy(a).addScaledVector(dir, 0.5);
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), dir.clone().normalize()
    );
    this.scene.add(mesh);
    return mesh;
  }

  /** The torch, hung off the flange so it belongs to the arm rather than
   *  floating beside it.
   *
   *  Only the wire crosses the last few millimetres to the work. A nozzle taken
   *  all the way down to the joint would be sitting in the weld, which is what
   *  the first version of the Blender cell did before it was measured.
   */
  #torch(g, link6) {
    const seam = g.seam;
    const mid = (seam.bottom_z_m + seam.top_z_m) / 2;
    const out = new THREE.Vector3(seam.x_m - g.fixture.x_m, seam.y_m, 0).normalize();
    const stickout = (g.torch && g.torch.stickout_m) || 0.008;

    const wireTip = new THREE.Vector3(seam.x_m, seam.y_m, mid).addScaledVector(out, stickout);
    const nozzleFace = wireTip.clone().addScaledVector(out, 0.016);
    nozzleFace.z += 0.006;

    const flange = new THREE.Vector3();
    link6.getWorldPosition(flange);
    const axis = new THREE.Vector3().subVectors(nozzleFace, flange).normalize();

    const mount = flange.clone().addScaledVector(axis, 0.05);
    const neck = flange.clone().addScaledVector(axis, 0.16);
    const cupBack = nozzleFace.clone().addScaledVector(axis, -0.045);

    const parts = [
      this.#cyl(flange, mount, 0.034, 0x14161a, { metalness: 0.5, roughness: 0.45 }),
      this.#cyl(mount, neck, 0.026, 0x1b1e23, { metalness: 0.4, roughness: 0.4 }),
      this.#cyl(neck, cupBack, 0.019, 0x1b1e23, { metalness: 0.4, roughness: 0.4 }),
      this.#cyl(cupBack, nozzleFace, 0.013, 0xb06a2c, { metalness: 0.9, roughness: 0.3 }),
      this.#cyl(nozzleFace, wireTip, 0.0012, 0xffcf8a, {
        metalness: 0.2, roughness: 0.5, emissive: 0xffcf8a, emissiveIntensity: 1.2,
      }),
    ];
    // Reparent onto the flange, keeping the world placement just computed, so the
    // torch travels with the arm instead of staying where it was drawn.
    for (const part of parts) link6.attach(part);
  }

  #robot(g) {
    const base = { x: 3.84, y: -2.22 };
    const seam = g.seam;
    // Face the corner it welds.
    const yaw = Math.atan2(seam.y_m - base.y, seam.x_m - base.x);

    this.#box(0.72, 0.72, 0.08, base.x, base.y, 0.04, 0x1e252d,
              { metalness: 0.5, roughness: 0.6 });

    new GLTFLoader().load("/assets/robot/kr16_2.glb", (gltf) => {
      const root = gltf.scene;

      // The file is Y up and this scene is Z up, so the arm arrives lying on its
      // side. Measured rather than assumed: the model's bounding box is 0.85 m
      // tall in Y and 2.17 m long in X, which is a robot on the floor.
      //
      // Two parents, because the order matters. The inner one stands the model
      // up, the outer one turns it to face the corner it welds. Folding both into
      // one rotation applies the yaw about the model's own sideways axis and
      // tips it over again.
      const upright = new THREE.Group();
      upright.rotation.x = Math.PI / 2;
      upright.add(root);

      const pivot = new THREE.Group();
      pivot.position.set(base.x, base.y, 0.08);
      pivot.rotation.z = yaw;
      pivot.add(upright);

      // Shoulder up, elbow bent, so the arm reaches the joint instead of lying
      // straight out in its home pose, which is what makes a robot look like it
      // has fallen over.
      //
      // Axis and angles were both measured on this model rather than assumed.
      // Turning each joint in turn and watching the flange shows the pitch axis
      // is Y: a rotation about Z moves the flange sideways and does not lift it
      // at all, which is why an earlier version set Z and the arm never came up.
      // The angles then come from searching for the pair that puts the flange at
      // the joint, which lands within 3 mm of it.
      const pose = { link_2: -0.70, link_3: 0.94 };
      root.traverse((node) => {
        if (pose[node.name] !== undefined) node.rotation.y = pose[node.name];
        if (node.isMesh) {
          node.material = new THREE.MeshStandardMaterial({
            color: /RAL9005/i.test(node.material?.name || "") ? 0x14161a : 0xdc6b20,
            metalness: 0.25,
            roughness: 0.45,
          });
        }
      });
      this.scene.add(pivot);
      this.robot = pivot;

      let link6 = null;
      root.traverse((n) => { if (n.name === "link_6") link6 = n; });
      if (link6) this.#torch(g, link6);
    }, undefined, () => {
      // No model is not a broken dashboard. Stand a marker where it would be.
      this.#box(0.3, 0.3, 0.9, base.x, base.y, 0.53, 0xdc6b20,
                { metalness: 0.3, roughness: 0.5 });
    });
  }

  // ------------------------------------------------------------- the seam itself

  setExaggeration(factor) {
    this.exaggeration = factor;
    if (this.lastTrack) this.showSeam(this.lastTrack, this.lastSchedule);
  }

  showSeam(track, schedule) {
    this.lastTrack = track;
    this.lastSchedule = schedule;
    this.seamGroup.clear();
    if (this.arc) { this.scene.remove(this.arc); this.arc = null; }
    if (!track || !track.path || !track.path.length) return;

    const path = track.path;
    const pts = path.map(p => new THREE.Vector3(p.x_mm / 1000, p.y_mm / 1000, 0));
    const zBottom = this.geometry.seam.bottom_z_m;
    const zTop = this.geometry.seam.top_z_m;
    const along = path.map(p => p.along_mm / 1000);
    pts.forEach((v, i) => { v.z = zBottom + along[i]; });

    // The taught path is where the program was taught to weld: the nominal joint
    // from the cell's own geometry, straight up. An earlier version drew this line
    // from the MEASURED seam's own ends, which threw away the whole offset before
    // anything was drawn and left the picture showing a 0.4 mm bow beside a chip
    // that said 14.9 mm. The line has to stand where the robot would have gone.
    const nom = new THREE.Vector3(this.geometry.seam.x_m, this.geometry.seam.y_m, 0);
    const taught = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(nom.x, nom.y, zBottom),
        new THREE.Vector3(nom.x, nom.y, zTop),
      ]),
      new THREE.LineBasicMaterial({ color: COLOUR.taught, linewidth: 2 })
    );
    this.seamGroup.add(taught);

    // The measured path, with the whole gap from the taught line multiplied: the
    // offset of the part and the bow along it together, so what is exaggerated is
    // exactly the miss the chip quotes. Tens of millimetres on a metre of tray are
    // hard to read at a workcell's scale, so the factor is always shown on the
    // control, because an exaggerated picture that does not say so is a lie.
    const k = this.exaggeration;
    const shown = pts.map((v) => new THREE.Vector3(
      nom.x + (v.x - nom.x) * k,
      nom.y + (v.y - nom.y) * k,
      v.z
    ));

    // The part itself sits where its seam was found, not where the fixture
    // expected it. Move the tray with the seam so the picture is of a part off
    // its mark, and remember where to point the camera so both lines stay in
    // frame.
    const a = pts[0];
    if (this.trayGroup) {
      this.trayGroup.position.set((a.x - nom.x) * k, (a.y - nom.y) * k, 0);
    }
    this.seamFocus = new THREE.Vector3(
      (nom.x + shown[0].x) / 2, (nom.y + shown[0].y) / 2, (zBottom + zTop) / 2);
    this.seamGap = Math.hypot(shown[0].x - nom.x, shown[0].y - nom.y);

    const curve = new THREE.CatmullRomCurve3(shown);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 96, 0.006, 8, false),
      new THREE.MeshStandardMaterial({
        color: COLOUR.tracked, emissive: COLOUR.tracked, emissiveIntensity: 1.4,
        metalness: 0.2, roughness: 0.5,
      })
    );
    this.seamGroup.add(tube);

    const stations = (schedule && schedule.stations) || [];
    shown.forEach((v, i) => {
      const st = stations[i];
      const colour = !st ? COLOUR.tracked
        : !st.weldable ? COLOUR.refused
        : st.needs_attention ? COLOUR.attention : COLOUR.ok;
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.0105, 12, 10),
        new THREE.MeshStandardMaterial({
          color: colour, emissive: colour, emissiveIntensity: 1.1,
          metalness: 0.1, roughness: 0.6,
        })
      );
      dot.position.copy(v);
      this.seamGroup.add(dot);
    });

    // A light at the arc, so the eye goes to the joint rather than the furniture.
    this.arc = new THREE.PointLight(COLOUR.arc, 1.8, 1.4, 2);
    this.arc.position.copy(shown[Math.floor(shown.length * 0.45)]);
    this.scene.add(this.arc);
  }

  // ------------------------------------------------------------------- framing

  frameCell() {
    const g = this.geometry;
    if (!g) return;
    this.controls.target.set(g.fixture.x_m, 0, g.fixture.table_top_m + 0.2);
    this.camera.position.set(g.fixture.x_m + 2.9, -3.4, 2.4);
  }

  /** Stand where the sensor stands, close enough to read the head. */
  frameSensor() {
    const g = this.geometry;
    if (!g) return;
    const eye = new THREE.Vector3(...g.camera.eye_m);
    const aim = new THREE.Vector3(...g.camera.aim_m);
    const look = aim.clone().sub(eye).normalize();
    const side = new THREE.Vector3(-look.y, look.x, 0).normalize();
    const head = eye.clone().addScaledVector(look, -0.055);
    // Three quarters on, from above and outboard: the lens face is turned toward
    // the viewer and the cell falls away behind it, which is the view that shows
    // both what the sensor is and how far off the work it stands.
    this.controls.target.copy(head);
    this.camera.position.copy(head)
      .addScaledVector(look, 0.26)
      .addScaledVector(side, -0.30)
      .add(new THREE.Vector3(0, 0, 0.11));
  }

  frameSeam() {
    const g = this.geometry;
    if (!g) return;
    const s = g.seam, mid = (s.bottom_z_m + s.top_z_m) / 2;
    // Aim between the taught line and the seam as shown, and stand back a little
    // further the wider that gap is, so an exaggerated miss does not leave one of
    // the two lines out of the picture.
    const focus = this.seamFocus ?? new THREE.Vector3(s.x_m, s.y_m, mid);
    const back = 1 + Math.min(1.6, (this.seamGap ?? 0) * 2.2);
    this.controls.target.copy(focus);
    this.camera.position.set(focus.x + 0.62 * back, focus.y - 0.74 * back, mid + 0.34 * back);
  }
}
