/**
 * One-link First Light browser alpha. Physics: official MuJoCo 3.8.0 WASM.
 * The model and rules are selected source exports; this browser runtime is a
 * separate profile from the retained native MuJoCo 3.3.5 research campaigns.
 * Human input supplies cart force only. The paddle is passive and ball motion
 * is never edited after a declared serve.
 */
import loadMujoco from './vendor/mujoco.js';

export const BROWSER_PROFILE = 'floppy-first-light-human-mujoco-3.8.0-v1';
export const RUNTIME_VERSION = '3.8.0';
const COMMAND_PERIOD = .02;
const BALANCE_GAIN = [-29.999999999999503, -165.93010568290268, -33.01307364817709, -45.30136008546452];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const copy = value => structuredClone(value);

/** Same finite/F32/clipped delivery contract as the original rules revision 2. */
export function deliveredForce(force, limit = 80) {
  if (typeof force !== 'number' || !Number.isFinite(force)) throw new TypeError('Cart force must be finite.');
  return Math.fround(clamp(force, -limit, limit));
}

async function asset(name, kind = 'json') {
  const response = await fetch(new URL(name, import.meta.url));
  if (!response.ok) throw new Error(`Unable to load ${name}: HTTP ${response.status}`);
  return kind === 'text' ? response.text() : response.json();
}

/**
 * Assets may be supplied by the bounded Node validation harness. Normal browser
 * callers need no arguments. No external network or native service is required.
 */
export async function createGame(options = {}) {
  const [configuration, serves, xml, mj] = await Promise.all([
    options.configuration ?? asset('profile.json'),
    options.serves ?? asset('serves.json'),
    options.modelXML ?? asset('model.xml', 'text'),
    options.module ?? loadMujoco({ locateFile: name => new URL(`vendor/${name}`, import.meta.url).href }),
  ]);
  if (mj.mj_versionString() !== RUNTIME_VERSION) throw new Error('Unexpected browser physics runtime.');
  const duration = options.duration ?? 120;
  if (typeof duration !== 'number' || !Number.isFinite(duration) || duration <= 0 || duration > 120) {
    throw new RangeError('Duration must be in (0, 120] seconds.');
  }
  if (serves.seed !== 20261019 || serves.serves.length !== 3) throw new Error('Unexpected fixed serve schedule.');
  const p = configuration.source_profile;
  if (p.command_period !== COMMAND_PERIOD || p.timestep !== .0000625) throw new Error('Unexpected command clock.');
  // XML is read only during compilation; the temporary virtual file is removed.
  const path = `/floppy-first-light-${Math.random().toString(36).slice(2)}.xml`;
  let model;
  try {
    mj.FS.writeFile(path, xml);
    model = mj.MjModel.mj_loadXML(path);
  } finally {
    mj.FS.unlink(path);
  }
  if (!model) throw new Error('Unable to compile First Light physics model.');
  let data;
  try {
    data = new mj.MjData(model);
    return new FirstLightGame(mj, model, data, configuration, serves, duration);
  } catch (error) {
    data?.delete();
    model.delete();
    throw error;
  }
}

class FirstLightGame {
  constructor(mj, model, data, configuration, serves, duration) {
    this.mj = mj;
    this.model = model;
    this.data = data;
    this.p = configuration.source_profile;
    this.board = configuration.board;
    this.serves = serves.serves;
    this.duration = duration;
    this.disposed = false;
    this.ballID = mj.mj_name2id(model, 5, 'ball');
    this.geometryNames = new Map();
    for (let id = 0; id < model.ngeom; id++) this.geometryNames.set(id, mj.mj_id2name(model, 5, id));
    this.pairs = new Map(this.board.bricks.map(brick => [brick.id, mj.mj_name2id(model, 15, `brick_${brick.id}`)]));
    this.reset();
  }

  assertLive() { if (this.disposed) throw new Error('Game has been disposed.'); }

  forward() { this.mj.mj_forward(this.model, this.data); }

  mask() {
    for (const [key, id] of this.pairs) this.model.pair_margin[id] = this.active[key] ? 0 : -1e6;
  }

  event(kind, values = {}) {
    const event = { kind, time: this.time, step: this.steps, life: this.life, ...values };
    if (kind === 'contact_begin' || kind === 'contact_end') {
      event.motion = { native_time: this.data.time, own: this.mechanism(),
        ball_world: [this.data.qpos[2], this.data.qpos[3]],
        ball_velocity_world: [this.data.qvel[2], this.data.qvel[3]] };
    }
    this.events.push(event);
  }

  reset() {
    this.assertLive();
    this.time = this.steps = this.tick = this.score = this.hits = this.life = 0;
    this.lives = 3;
    this.phase = 'new_life';
    this.outcome = null;
    this.force = 0;
    this.reference = 0;
    this.health = Object.fromEntries(this.board.bricks.map(brick => [brick.id, brick.hp]));
    this.active = Object.fromEntries(this.board.bricks.map(brick => [brick.id, true]));
    this.pending = [];
    this.episodes = new Map();
    this.events = [];
    this.warmupEnd = 0;
    this.prepare();
    return this.snapshot();
  }

  prepare() {
    if (this.phase !== 'new_life') return;
    this.mj.mj_resetData(this.model, this.data);
    this.pending = [];
    this.active = Object.fromEntries(Object.entries(this.health).map(([key, hp]) => [key, hp > 0]));
    this.mask();
    this.forward();
    this.life++;
    this.episodes.clear();
    this.phase = 'warmup';
    this.warmupEnd = this.time + this.p.warmup;
    this.reference = 0;
    this.event('life_reset', { lives: this.lives });
  }

  serve() {
    const serve = this.serves[this.life - 1];
    if (!serve) throw new Error('Fixed three-life serve schedule exhausted.');
    this.data.qpos.set(serve.position, 2);
    this.data.qvel.set(serve.velocity, 2);
    this.forward();
    this.phase = 'rally';
    this.event('serve', { ball: [...serve.position], velocity: [...serve.velocity] });
  }

  end(outcome, detail = null) {
    this.phase = 'finished';
    this.outcome = outcome;
    this.event('episode_end', { outcome, detail });
  }

  lose(reasons) {
    this.lives--;
    this.event('life_lost', { reasons: [...new Set(reasons)].sort(), lives: this.lives });
    if (!Object.values(this.health).some(Boolean)) this.end('ambiguous', 'final damage and life fault before separation');
    else if (!this.lives) this.end('lost');
    else this.phase = 'new_life';
  }

  mechanism() {
    const [x, theta] = this.data.qpos;
    const [vx, omega] = this.data.qvel;
    const sn = Math.sin(theta), cs = Math.cos(theta), p = this.p;
    return { x, theta, vx, omega, cart_world: [x, -p.rail_height],
      paddle_world: [x + p.length * sn, -p.rail_height + p.length * cs],
      tip_height: p.length * cs, normal_world: [sn, cs] };
  }

  paddleMotion() {
    const own = this.mechanism(), p = this.p;
    const [nx, nz] = own.normal_world, tx = nz, tz = -nx;
    const deltaX = this.data.qpos[2] - own.paddle_world[0];
    const deltaZ = this.data.qpos[3] - own.paddle_world[1];
    const along = clamp(deltaX * tx + deltaZ * tz, -p.paddle_width / 2, p.paddle_width / 2);
    const ox = tx * along + nx * p.paddle_thickness / 2;
    const oz = tz * along + nz * p.paddle_thickness / 2;
    return { normal: [nx, nz], velocity: [own.vx + p.length * nz * own.omega + own.omega * oz,
      -p.length * nx * own.omega - own.omega * ox] };
  }

  contacts() {
    const result = new Map();
    if (this.data.ncon === 0) return result;
    // Embind returns heap-owned copies. Delete the vector and each entry.
    const contacts = this.data.contact;
    try {
      for (let i = 0; i < contacts.size(); i++) {
        const contact = contacts.get(i);
        try {
          if (contact.dist > 0 || (contact.geom1 !== this.ballID && contact.geom2 !== this.ballID)) continue;
          const geom = contact.geom1 === this.ballID ? contact.geom2 : contact.geom1;
          const name = this.geometryNames.get(geom);
          result.set(name, Math.min(result.get(name) ?? 0, contact.dist));
        } finally { contact?.delete(); }
      }
    } finally { contacts.delete(); }
    return result;
  }

  check(beforeVelocity, previousPaddle) {
    const p = this.p, own = this.mechanism(), faults = [];
    if (Math.abs(own.x) > p.rail_limit) faults.push('rail');
    if (Math.abs(own.theta) > 1.4 || own.tip_height < .35 * p.length) faults.push('collapse');
    const extent = Math.abs(own.paddle_world[0]) + Math.abs(Math.cos(own.theta)) * p.paddle_width / 2 + Math.abs(Math.sin(own.theta)) * p.paddle_thickness / 2;
    if (Math.max(extent, Math.abs(own.x) + .12) >= p.wall_x - p.crossing_clearance) faults.push('court_edge');
    if (this.phase !== 'rally') { if (faults.length) this.lose(faults); return; }
    const contacts = this.contacts();
    const fresh = [...contacts.keys()].filter(name => !this.episodes.has(name)).sort();
    for (const [name, penetration] of contacts) {
      if (!this.episodes.has(name)) {
        this.episodes.set(name, { start: this.time, max_penetration: -penetration });
        this.event('contact_begin', { geom: name });
      }
      const episode = this.episodes.get(name);
      episode.max_penetration = Math.max(episode.max_penetration, -penetration);
    }
    // Simultaneous brick damage precedes terminal/fault decisions.
    for (const name of fresh) {
      if (name.startsWith('brick_')) {
        const key = name.slice(6);
        if (this.active[key] && this.health[key] > 0) {
          this.health[key]--;
          this.score += this.health[key] === 0 ? 100 : 25;
          const brick = this.board.bricks.find(value => value.id === key);
          this.event('brick_damage', { id: key, hp: this.health[key], position: [brick.x, brick.z] });
          if (!this.health[key]) this.pending.push(key);
        }
      } else if (name === 'bottom_paddle') {
        const [nx, nz] = own.normal_world;
        const dx = this.data.qpos[2] - own.paddle_world[0], dz = this.data.qpos[3] - own.paddle_world[1];
        const approach = (beforeVelocity[0] - previousPaddle.velocity[0]) * previousPaddle.normal[0] + (beforeVelocity[1] - previousPaddle.velocity[1]) * previousPaddle.normal[1];
        const front = dx * nx + dz * nz > p.paddle_thickness / 2 - .002;
        if (front && Math.abs(dx * nz - dz * nx) <= p.paddle_width / 2 && approach < 0) {
          this.hits++;
          this.event('paddle_hit', { normal: [nx, nz] });
        } else faults.push('illegal_face');
      } else if (name === 'bottom_stem' || name === 'bottom_cart') faults.push('body_hit');
      else this.event('wall_hit', { wall: name });
    }
    for (const [name, episode] of this.episodes) {
      if (contacts.has(name)) continue;
      const geom = this.mj.mj_name2id(this.model, 5, name);
      if (this.mj.mj_geomDistance(this.model, this.data, this.ballID, geom, 10, null) <= p.separation_clearance) continue;
      this.episodes.delete(name);
      this.event('contact_end', { geom: name, ...episode });
      if (name.startsWith('brick_') && this.pending.includes(name.slice(6))) {
        const key = name.slice(6);
        this.pending.splice(this.pending.indexOf(key), 1);
        this.active[key] = false;
        this.mask();
        this.event('brick_removed', { id: key });
      }
    }
    if (this.data.qpos[3] < -p.goal_z) faults.push('miss');
    if (faults.length) this.lose(faults);
    else if (!Object.values(this.health).some(Boolean) && !this.pending.length) this.end('cleared');
  }

  /** Advance one 20ms command. Optional retain(snapshot) is for finite validation. */
  step(force, retain = null) {
    this.assertLive();
    this.events = [];
    const delivered = deliveredForce(force, this.p.force_limit);
    if (this.phase === 'finished') return this.snapshot();
    this.prepare();
    this.force = delivered;
    this.data.ctrl[0] = delivered;
    for (let i = 0; i < Math.round(COMMAND_PERIOD / this.p.timestep); i++) {
      if (this.phase === 'finished' || this.phase === 'new_life') break;
      if (this.phase === 'warmup' && this.time + 1e-12 >= this.warmupEnd) this.serve();
      const beforeVelocity = [this.data.qvel[2], this.data.qvel[3]];
      const previousPaddle = this.paddleMotion();
      this.mj.mj_step(this.model, this.data);
      this.forward();
      this.steps++;
      this.time = this.steps * this.p.timestep;
      for (let j = 0; j < 4; j++) {
        if (!Number.isFinite(this.data.qpos[j]) || !Number.isFinite(this.data.qvel[j])) throw new Error('Physics produced a nonfinite state.');
      }
      this.check(beforeVelocity, previousPaddle);
      if (this.phase !== 'finished' && this.time + 1e-12 >= this.duration) this.end('timeout');
      if (retain) retain(this.snapshot());
    }
    this.tick++;
    return this.snapshot();
  }

  /** Human chooses a cart target; this fixed posture helper never reads the ball. */
  assistedForce(targetX) {
    this.assertLive();
    if (!Number.isFinite(targetX)) throw new TypeError('Target must be finite.');
    const target = clamp(targetX, -1.65, 1.65);
    this.reference += clamp(.35 * (target - this.reference), -.06, .06);
    const own = this.mechanism();
    const error = [own.x - this.reference, own.theta, own.vx, own.omega];
    return deliveredForce(-BALANCE_GAIN.reduce((sum, gain, i) => sum + gain * error[i], 0), this.p.force_limit);
  }

  snapshot() {
    this.assertLive();
    return { browser_profile: BROWSER_PROFILE, runtime: RUNTIME_VERSION, rules_revision: 2,
      seed: 20261019, time: this.time, native_time: this.data.time, steps: this.steps, tick: this.tick,
      phase: this.phase, life: this.life, score: this.score, lives: this.lives, hits: this.hits,
      outcome: this.outcome, force: this.force, own: this.mechanism(),
      qpos: [...this.data.qpos], qvel: [...this.data.qvel],
      ball_world: [this.data.qpos[2], this.data.qpos[3]],
      ball_velocity_world: [this.data.qvel[2], this.data.qvel[3]],
      health: { ...this.health }, active: { ...this.active }, pending: [...this.pending],
      events: copy(this.events), profile: this.p, board: this.board };
  }

  /** Public reproducibility hook; UI never calls it and no run is ranked. */
  restoreValidationState(state) {
    this.assertLive();
    for (const key of ['qpos', 'qvel']) {
      if (!Array.isArray(state[key]) || state[key].length !== 4 || !state[key].every(Number.isFinite)) throw new Error(`Invalid ${key}.`);
    }
    // This is a fresh synthetic fixture, not an incomplete warm-start restore.
    this.mj.mj_resetData(this.model, this.data);
    this.data.qpos.set(state.qpos);
    this.data.qvel.set(state.qvel);
    this.data.time = state.native_time ?? state.time ?? 0;
    this.time = state.time ?? 0;
    this.steps = state.steps ?? Math.round(this.time / this.p.timestep);
    this.tick = state.tick ?? 0;
    this.phase = state.phase ?? 'rally';
    this.life = state.life ?? 1;
    this.lives = state.lives ?? 3;
    this.score = state.score ?? 0;
    this.hits = state.hits ?? 0;
    this.outcome = state.outcome ?? null;
    this.force = state.force ?? 0;
    this.warmupEnd = state.warmup_end ?? state.warmupEnd ?? this.time + this.p.warmup;
    this.health = state.health ? { ...state.health } : Object.fromEntries(this.board.bricks.map(brick => [brick.id, brick.hp]));
    this.active = state.active ? { ...state.active } : Object.fromEntries(Object.entries(this.health).map(([key, hp]) => [key, hp > 0]));
    this.pending = [...(state.pending ?? [])];
    this.episodes = new Map(Object.entries(state.episodes ?? {}));
    this.events = [];
    this.mask();
    this.forward();
    return this.snapshot();
  }

  dispose() {
    if (this.disposed) return;
    this.data.delete();
    this.model.delete();
    this.disposed = true;
  }
}
