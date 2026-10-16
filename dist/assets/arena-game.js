/* A browser toy. This is deliberately separate from the ranked MuJoCo task. */

export const TOY_PARAMETERS = Object.freeze({
  cartMass: 1,
  poleMass: 0.18,
  halfLength: 0.8,
  gravity: 9.81,
  cartDrag: 0.08,
  angularDrag: 0.035,
  maxForce: 9,
  railLimit: 2.4,
  uprightAngle: 0.2,
  goalSeconds: 5,
});

const TAU = Math.PI * 2;
const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
const wrap = (angle) => Math.abs(angle) <= Math.PI ? angle : Math.atan2(Math.sin(angle), Math.cos(angle));

export function createToyState(mode = "balance") {
  return {
    x: 0,
    velocity: 0,
    angle: mode === "swingup" ? Math.PI - 0.04 : 0.055,
    angularVelocity: 0,
    time: 0,
  };
}

function derivative(vector, force) {
  const [, velocity, angle, angularVelocity] = vector;
  const p = TOY_PARAMETERS;
  const sin = Math.sin(angle);
  const cos = Math.cos(angle);
  const totalMass = p.cartMass + p.poleMass;
  const poleMassLength = p.poleMass * p.halfLength;
  const temporary = (force - p.cartDrag * velocity + poleMassLength * angularVelocity ** 2 * sin) / totalMass;
  const angularAcceleration = (p.gravity * sin - cos * temporary - p.angularDrag * angularVelocity) /
    (p.halfLength * (4 / 3 - p.poleMass * cos ** 2 / totalMass));
  const acceleration = temporary - poleMassLength * angularAcceleration * cos / totalMass;
  return [velocity, acceleration, angularVelocity, angularAcceleration];
}

/** Integrate a uniform rod on a cart with bounded force and RK4 substeps. */
export function advanceToy(state, input, seconds) {
  const values = [state.x, state.velocity, state.angle, state.angularVelocity, state.time, input, seconds];
  if (!values.every(Number.isFinite) || seconds < 0 || seconds > 0.1) {
    throw new RangeError("Toy state, input, and step must be finite; step must be between 0 and 0.1 seconds.");
  }
  const count = Math.max(1, Math.ceil(seconds / (1 / 240)));
  const step = seconds / count;
  const force = clamp(input, -1, 1) * TOY_PARAMETERS.maxForce;
  let vector = [state.x, state.velocity, state.angle, state.angularVelocity];
  for (let index = 0; index < count; index += 1) {
    const k1 = derivative(vector, force);
    const k2 = derivative(vector.map((value, j) => value + step * k1[j] / 2), force);
    const k3 = derivative(vector.map((value, j) => value + step * k2[j] / 2), force);
    const k4 = derivative(vector.map((value, j) => value + step * k3[j]), force);
    vector = vector.map((value, j) => value + step * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6);
  }
  return {
    x: vector[0],
    velocity: vector[1],
    angle: wrap(vector[2]),
    angularVelocity: vector[3],
    time: state.time + seconds,
  };
}

export function inspectToy(state, mode = "balance") {
  const finite = [state.x, state.velocity, state.angle, state.angularVelocity, state.time].every(Number.isFinite);
  const angle = wrap(state.angle);
  return {
    finite,
    upright: finite && Math.abs(angle) < TOY_PARAMETERS.uprightAngle,
    railHit: finite && Math.abs(state.x) > TOY_PARAMETERS.railLimit,
    fallen: finite && mode !== "swingup" && Math.abs(angle) > Math.PI / 2,
  };
}

/** A deterministic balance demonstration; its score is never saved. */
export function demonstrationInput(state) {
  const force = 2.5819888975 * state.x + 5.3409163286 * state.velocity +
    53.1007749768 * wrap(state.angle) + 16.9438136252 * state.angularVelocity;
  return clamp(force / TOY_PARAMETERS.maxForce, -1, 1);
}

const styles = `
.arena-game{--arena-ink:#101519;--arena-paper:#f2f1e8;--arena-lime:#d9ff43;--arena-orange:#ff895a;color:var(--arena-paper);font-family:inherit;background:var(--arena-ink);border:1px solid #ffffff29;border-radius:20px;overflow:hidden}
.arena-game *{box-sizing:border-box}.arena-game button{font:inherit;cursor:pointer;-webkit-tap-highlight-color:transparent}.arena-game button:focus-visible,.arena-game canvas:focus-visible{outline:3px solid var(--arena-orange);outline-offset:4px}.arena-game button:disabled{opacity:.45;cursor:default}
.arena-game-head{padding:24px 26px 15px;display:flex;justify-content:space-between;align-items:flex-start;gap:16px}.arena-game-eyebrow{font-size:11px;letter-spacing:.16em;font-weight:750;color:var(--arena-lime);margin:0 0 7px}.arena-game-title{font-size:clamp(21px,3vw,29px);line-height:1.15;font-weight:700;letter-spacing:-.04em;margin:0}.arena-game-badge{font:600 10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--arena-orange);border:1px solid #ff895a66;border-radius:7px;padding:7px 10px;max-width:174px;flex-shrink:0;text-align:right}
.arena-game-modes{display:flex;gap:5px;padding:0 26px 8px}.arena-game-mode{color:#b1b6b3;background:transparent;border:1px solid #ffffff26;padding:7px 12px;border-radius:30px;font-size:12px;font-weight:600}.arena-game-mode[aria-pressed=true]{background:var(--arena-paper);border-color:var(--arena-paper);color:var(--arena-ink)}.arena-game-mode:hover{border-color:var(--arena-lime)}
.arena-game-stage{position:relative}.arena-game-canvas{width:100%;height:clamp(240px,28vw,350px);display:block;touch-action:pan-y}.arena-game-stats{display:flex;position:absolute;top:12px;left:26px;right:26px;gap:25px;pointer-events:none}.arena-game-stat{display:flex;flex-direction:column;gap:3px}.arena-game-stat-label{font:10px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.09em;text-transform:uppercase;color:#939f9d}.arena-game-stat-value{font:500 21px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.05em}.arena-game-hold{color:var(--arena-lime)}.arena-game-stat:last-child{margin-left:auto;text-align:right}
.arena-game-bottom{padding:0 26px 23px}.arena-game-message{min-height:21px;font-size:13px;line-height:1.5;color:#c7ceca;margin:2px 0 16px}.arena-game-message strong{color:var(--arena-lime)}.arena-game-bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.arena-game-start{background:var(--arena-lime);color:var(--arena-ink);border:1px solid var(--arena-lime);font-weight:750;padding:11px 19px;border-radius:8px;min-height:44px}.arena-game-quiet{background:transparent;color:var(--arena-paper);border:1px solid #ffffff30;border-radius:8px;padding:11px 13px;font-size:12px;min-height:44px}.arena-game-quiet:hover,.arena-game-push:hover{border-color:var(--arena-lime)}.arena-game-pushes{display:flex;gap:7px;margin-left:auto}.arena-game-push{border:1px solid #ffffff45;border-radius:8px;color:var(--arena-paper);background:#ffffff07;padding:8px 18px;font-size:22px;line-height:1;min-width:57px;min-height:44px;touch-action:none;user-select:none}.arena-game-push[aria-pressed=true]{background:var(--arena-orange);border-color:var(--arena-orange);color:var(--arena-ink)}.arena-game-instructions{font-size:11px;line-height:1.65;color:#94a09d;margin:13px 0 0}.arena-game-instructions kbd{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;border:1px solid #ffffff25;border-radius:3px;padding:1px 4px;color:#d6ddd7}.arena-game-accessible{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:560px){.arena-game-head{padding:20px 17px 15px;gap:10px}.arena-game-title{max-width:220px}.arena-game-badge{font-size:9px;max-width:118px;padding:5px 7px}.arena-game-modes{padding-left:17px;padding-right:17px}.arena-game-stats{left:17px;right:17px;gap:17px}.arena-game-stat-label{font-size:9px}.arena-game-stat-value{font-size:19px}.arena-game-bottom{padding:0 17px 20px}.arena-game-canvas{height:270px}.arena-game-bar{gap:7px}.arena-game-quiet{padding:11px 10px}.arena-game-push{min-width:49px;padding:8px 13px}.arena-game-start{padding:11px 15px}.arena-game-demo{order:5;width:100%;margin-top:3px}.arena-game-message{min-height:39px;margin-bottom:12px}}
`;

let instanceCount = 0;

/** Mount the independent browser challenge. Call destroy() before replacing it. */
export function mountArena(root) {
  if (!root || typeof root.replaceChildren !== "function") {
    throw new TypeError("mountArena needs a DOM element.");
  }
  const document = root.ownerDocument;
  const window = document.defaultView;
  const id = `arena-game-${++instanceCount}`;
  const shell = document.createElement("section");
  shell.className = "arena-game";
  shell.setAttribute("aria-label", "Playable browser balance toy");
  shell.innerHTML = `
    <style>${styles}</style>
    <div class="arena-game-head">
      <div><p class="arena-game-eyebrow">YOUR TURN</p><h3 class="arena-game-title">Looks easy.<br>Keep it up for five.</h3></div>
      <span class="arena-game-badge">Browser toy<br>Not the ranked MuJoCo benchmark</span>
    </div>
    <div class="arena-game-modes" role="group" aria-label="Toy starting position">
      <button type="button" class="arena-game-mode" data-mode="balance" aria-pressed="true">Balance</button>
      <button type="button" class="arena-game-mode" data-mode="swingup" aria-pressed="false">Swing-up · hard</button>
    </div>
    <div class="arena-game-stage">
      <canvas class="arena-game-canvas" tabindex="0" role="img" aria-label="A movable cart and one freely swinging pole on a limited rail. Use the buttons or keyboard to push the cart." aria-describedby="${id}-instructions"></canvas>
      <div class="arena-game-stats" aria-label="Toy session statistics">
        <div class="arena-game-stat"><span class="arena-game-stat-label">Upright streak</span><span class="arena-game-stat-value arena-game-hold" data-stat="hold">0.0s</span></div>
        <div class="arena-game-stat"><span class="arena-game-stat-label">Goal</span><span class="arena-game-stat-value">5.0s</span></div>
        <div class="arena-game-stat"><span class="arena-game-stat-label">Your best</span><span class="arena-game-stat-value" data-stat="best">0.0s</span></div>
      </div>
    </div>
    <div class="arena-game-bottom">
      <p class="arena-game-message" data-message>Tap toward the lean. Small corrections beat a long shove.</p>
      <div class="arena-game-bar">
        <button type="button" class="arena-game-start" data-action="start">Give it a go ↗</button>
        <button type="button" class="arena-game-quiet" data-action="pause" disabled>Pause</button>
        <button type="button" class="arena-game-quiet arena-game-demo" data-action="demo">Watch a demo</button>
        <div class="arena-game-pushes" role="group" aria-label="Push the cart">
          <button type="button" class="arena-game-push" data-direction="-1" aria-label="Push cart left" aria-pressed="false">←</button>
          <button type="button" class="arena-game-push" data-direction="1" aria-label="Push cart right" aria-pressed="false">→</button>
        </div>
      </div>
      <p class="arena-game-instructions" id="${id}-instructions"><kbd>←</kbd> <kbd>→</kbd> or <kbd>A</kbd> <kbd>D</kbd> to push. Hold the touch buttons on mobile. <kbd>Space</kbd> pauses. Keep the pole within about 11.5° of upright and the cart on the rail. This simplified single-pole toy has no leaderboard submissions. Best times stay in this browser.</p>
      <p class="arena-game-accessible" role="status" aria-live="polite" data-announcement></p>
    </div>`;
  root.replaceChildren(shell);

  const canvas = shell.querySelector("canvas");
  const context = canvas.getContext("2d");
  const startButton = shell.querySelector('[data-action="start"]');
  const pauseButton = shell.querySelector('[data-action="pause"]');
  const demoButton = shell.querySelector('[data-action="demo"]');
  const message = shell.querySelector("[data-message]");
  const announcement = shell.querySelector("[data-announcement]");
  const holdOutput = shell.querySelector('[data-stat="hold"]');
  const bestOutput = shell.querySelector('[data-stat="best"]');
  const modes = [...shell.querySelectorAll("[data-mode]")];
  const pushes = [...shell.querySelectorAll("[data-direction]")];
  const removers = [];
  const on = (target, name, listener, options) => {
    target.addEventListener(name, listener, options);
    removers.push(() => target.removeEventListener(name, listener, options));
  };

  let mode = "balance";
  let state = createToyState(mode);
  let running = false;
  let paused = false;
  let demo = false;
  let won = false;
  let ended = "";
  let streak = 0;
  let frame = 0;
  let previousTime = null;
  let accumulator = 0;
  let smoothedInput = 0;
  let alive = true;
  let width = 700;
  let height = 300;
  const keys = new Set();
  const pointers = new Map();
  const best = { balance: 0, swingup: 0 };
  try {
    const stored = JSON.parse(window.localStorage.getItem("pendulum-arena-toy-v1") || "null");
    for (const name of ["balance", "swingup"]) {
      if (stored && Number.isFinite(stored[name]) && stored[name] >= 0 && stored[name] <= 60) best[name] = stored[name];
    }
  } catch { /* Storage can be unavailable in private or embedded browsers. */ }

  function say(text, urgent = false) {
    message.textContent = text;
    if (urgent) announcement.textContent = text;
  }

  function clearInput() {
    keys.clear();
    pointers.clear();
    smoothedInput = 0;
    updatePushes();
  }

  function rawInput() {
    const left = keys.has("ArrowLeft") || keys.has("a") || [...pointers.values()].includes(-1);
    const right = keys.has("ArrowRight") || keys.has("d") || [...pointers.values()].includes(1);
    return Number(right) - Number(left);
  }

  function updatePushes() {
    const value = rawInput();
    for (const button of pushes) button.setAttribute("aria-pressed", String(value === Number(button.dataset.direction)));
  }

  function updateStats() {
    holdOutput.textContent = `${Math.min(streak, 60).toFixed(1)}s`;
    bestOutput.textContent = `${best[mode].toFixed(1)}s`;
  }

  function saveBest() {
    if (demo) return;
    best[mode] = Math.max(best[mode], streak);
    try { window.localStorage.setItem("pendulum-arena-toy-v1", JSON.stringify(best)); } catch { /* Best times are optional. */ }
  }

  function updateControls() {
    startButton.textContent = running || ended ? "Try again ↗" : "Give it a go ↗";
    pauseButton.disabled = !running;
    pauseButton.textContent = paused ? "Resume" : "Pause";
    demoButton.textContent = demo && running ? "Take the controls" : "Watch a demo";
    for (const button of modes) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }

  function requestFrame() {
    if (alive && running && !paused && !frame) frame = window.requestAnimationFrame(tick);
  }

  function finish(reason) {
    saveBest();
    running = false;
    paused = false;
    ended = reason;
    clearInput();
    say(reason === "rail" ? "End of the rail. Reset and try smaller taps." : reason === "time" ? "A full minute. Your cart has earned a break." : "Gravity wins this round. Tap toward the lean and try again.", true);
    updateControls();
    updateStats();
    draw();
  }

  function start(asDemo = false) {
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    saveBest();
    if (asDemo) mode = "balance";
    state = createToyState(mode);
    streak = 0;
    demo = asDemo;
    running = true;
    paused = false;
    won = false;
    ended = "";
    previousTime = null;
    accumulator = 0;
    clearInput();
    say(asDemo ? "Demo: feedback makes little corrections. Demo times are never saved." : mode === "swingup" ? "Hard mode: pump the hanging pole, then catch it. The rail is your limit." : "Go! Tap toward the lean. Keep it upright for five seconds.", true);
    updateControls();
    updateStats();
    canvas.focus({ preventScroll: true });
    requestFrame();
  }

  function pause(value = !paused) {
    if (!running || paused === value) return;
    paused = value;
    previousTime = null;
    accumulator = 0;
    clearInput();
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    say(paused ? "Paused. Resume when you are ready." : demo ? "The demo is running. Its times are not saved." : "Back to it. Small taps, steady cart.", true);
    updateControls();
    draw();
    requestFrame();
  }

  function tick(timestamp) {
    frame = 0;
    if (!running || paused || !alive) return;
    if (previousTime === null) previousTime = timestamp;
    const delta = (timestamp - previousTime) / 1000;
    previousTime = timestamp;
    if (delta > 0.25) {
      pause(true); // A stalled or suspended tab must never award free time.
      return;
    }
    accumulator += Math.max(0, delta);
    const step = 1 / 120;
    while (accumulator >= step && running) {
      const desired = demo ? demonstrationInput(state) : rawInput();
      smoothedInput += clamp(desired - smoothedInput, -step * 9, step * 9);
      try { state = advanceToy(state, demo ? desired : smoothedInput, step); }
      catch { finish("fallen"); break; }
      const outcome = inspectToy(state, mode);
      if (!outcome.finite || outcome.railHit || outcome.fallen) {
        finish(outcome.railHit ? "rail" : "fallen");
        break;
      }
      if (outcome.upright) streak += step;
      else { saveBest(); streak = 0; }
      if (!demo && streak > best[mode]) best[mode] = streak;
      if (streak >= TOY_PARAMETERS.goalSeconds && !won) {
        won = true;
        say(demo ? "Five seconds, courtesy of a tiny feedback controller. Your turn?" : "Five seconds! One pole down. Imagine fifteen. Keep going for a personal best.", true);
      }
      if (state.time >= 60) { finish("time"); break; }
      accumulator -= step;
    }
    updateStats();
    draw();
    requestFrame();
  }

  function roundedRect(x, y, boxWidth, boxHeight, radius) {
    context.beginPath();
    if (typeof context.roundRect === "function") context.roundRect(x, y, boxWidth, boxHeight, radius);
    else context.rect(x, y, boxWidth, boxHeight);
  }

  function draw() {
    if (!context) return;
    const p = TOY_PARAMETERS;
    const scale = Math.min((width - 65) / 6.4, (height - (mode === "swingup" ? 118 : 100)) / (mode === "swingup" ? 3.5 : 2.5));
    const railY = height * (mode === "swingup" ? 0.57 : 0.73);
    const middle = width / 2;
    const cartX = middle + state.x * scale;
    const pivotY = railY - 13;
    const length = 2 * p.halfLength * scale;
    const tipX = cartX + Math.sin(state.angle) * length;
    const tipY = pivotY - Math.cos(state.angle) * length;
    const bounds = p.railLimit * scale;
    const good = inspectToy(state, mode).upright;
    context.clearRect(0, 0, width, height);

    // Quiet dotted graph paper and a real, finite rail.
    context.fillStyle = "#ffffff12";
    for (let x = 26; x < width; x += 25) for (let y = 87; y < height - 20; y += 25) context.fillRect(x, y, 1.2, 1.2);
    context.strokeStyle = "#f2f1e83b";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(middle - bounds, railY);
    context.lineTo(middle + bounds, railY);
    context.stroke();
    context.strokeStyle = "#ff895a";
    for (const side of [-1, 1]) {
      context.beginPath();
      context.moveTo(middle + side * bounds, railY - 7);
      context.lineTo(middle + side * bounds, railY + 8);
      context.stroke();
    }
    context.fillStyle = "#929f9b";
    context.font = "9px ui-monospace, monospace";
    context.textAlign = "center";
    context.fillText("−2.4", middle - bounds, railY + 22);
    context.fillText("0", middle, railY + 22);
    context.fillText("+2.4 m", middle + bounds, railY + 22);

    // The target cone follows the cart, not the screen centre.
    context.fillStyle = "#d9ff430a";
    context.beginPath();
    context.moveTo(cartX, pivotY);
    context.arc(cartX, pivotY, length, -Math.PI / 2 - p.uprightAngle, -Math.PI / 2 + p.uprightAngle);
    context.closePath();
    context.fill();
    context.strokeStyle = "#d9ff4333";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(cartX, pivotY);
    context.lineTo(cartX, pivotY - length - 10);
    context.setLineDash([3, 5]);
    context.stroke();
    context.setLineDash([]);

    // Pole, cap, cart, and independent wheel details.
    context.strokeStyle = good ? "#d9ff43" : "#ff895a";
    context.lineWidth = 5;
    context.lineCap = "round";
    context.beginPath(); context.moveTo(cartX, pivotY); context.lineTo(tipX, tipY); context.stroke();
    context.fillStyle = good ? "#d9ff43" : "#ff895a";
    context.beginPath(); context.arc(tipX, tipY, 5.5, 0, TAU); context.fill();
    context.fillStyle = "#f2f1e8";
    roundedRect(cartX - 21, railY - 21, 42, 16, 4); context.fill();
    context.fillStyle = "#101519";
    context.beginPath(); context.arc(cartX, pivotY, 3, 0, TAU); context.fill();
    context.fillStyle = "#f2f1e8";
    for (const side of [-1, 1]) {
      context.beginPath(); context.arc(cartX + side * 12, railY - 3, 3.5, 0, TAU); context.fill();
    }
    const input = demo ? demonstrationInput(state) : smoothedInput;
    if (running && Math.abs(input) > 0.02) {
      const side = Math.sign(input);
      const tail = cartX + side * 31;
      const head = tail + side * (12 + Math.abs(input) * 23);
      context.strokeStyle = "#ff895a";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(tail, railY - 13); context.lineTo(head, railY - 13);
      context.moveTo(head - side * 5, railY - 18); context.lineTo(head, railY - 13); context.lineTo(head - side * 5, railY - 8);
      context.stroke();
    }
    // Streak progress remains legible without colour perception.
    const barY = height - 18;
    const barWidth = width - 52;
    context.fillStyle = "#ffffff16";
    roundedRect(26, barY, barWidth, 3, 2); context.fill();
    context.fillStyle = "#d9ff43";
    roundedRect(26, barY, barWidth * Math.min(streak / p.goalSeconds, 1), 3, 2); context.fill();

    if (!running || paused) {
      const label = paused ? "PAUSED" : ended ? "ONE MORE?" : "FIVE SECONDS. YOU VS. GRAVITY.";
      const labelTop = height - 45;
      context.font = "600 10px ui-monospace, monospace";
      const labelWidth = context.measureText(label).width;
      context.fillStyle = "#101519e8";
      roundedRect(middle - labelWidth / 2 - 12, labelTop, labelWidth + 24, 24, 4); context.fill();
      context.fillStyle = paused ? "#ff895a" : "#b1bbb5";
      context.fillText(label, middle, labelTop + 16);
    }
    if (demo && running) {
      context.textAlign = "right";
      context.fillStyle = "#ff895a";
      context.font = "600 9px ui-monospace, monospace";
      context.fillText("DEMO · NO SCORE", width - 26, height - 33);
    }
  }

  function resize() {
    const rectangle = canvas.getBoundingClientRect();
    if (rectangle.width <= 0 || rectangle.height <= 0) return;
    width = rectangle.width;
    height = rectangle.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context?.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  on(startButton, "click", () => start(false));
  on(pauseButton, "click", () => pause());
  on(demoButton, "click", () => start(!(demo && running)));
  for (const button of modes) {
    on(button, "click", () => {
      saveBest();
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
      mode = button.dataset.mode;
      state = createToyState(mode);
      running = false;
      paused = false;
      demo = false;
      ended = "";
      streak = 0;
      clearInput();
      say(mode === "swingup" ? "Start beneath the rail. Pump it up, then catch it. This one takes practice." : "Tap toward the lean. Small corrections beat a long shove.");
      updateControls(); updateStats(); draw();
    });
  }
  for (const button of pushes) {
    on(button, "pointerdown", (event) => {
      if (event.button !== 0 || !running || paused || demo) return;
      event.preventDefault();
      try { button.setPointerCapture(event.pointerId); } catch { /* Capture is a convenience. */ }
      pointers.set(event.pointerId, Number(button.dataset.direction));
      updatePushes();
    });
    for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
      on(button, eventName, (event) => { pointers.delete(event.pointerId); updatePushes(); });
    }
    // Switch/keyboard users can activate a direction button for a short impulse.
    on(button, "click", (event) => {
      if (event.detail !== 0 || !running || paused || demo) return;
      const key = Number(button.dataset.direction) < 0 ? "ArrowLeft" : "ArrowRight";
      keys.add(key); updatePushes();
      window.setTimeout(() => { if (alive) { keys.delete(key); updatePushes(); } }, 100);
    });
  }
  on(window, "keydown", (event) => {
    if (!shell.contains(document.activeElement) || event.altKey || event.ctrlKey || event.metaKey) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    if (key === " " && document.activeElement === canvas && !event.repeat) {
      event.preventDefault(); pause(); return;
    }
    if (!["ArrowLeft", "ArrowRight", "a", "d"].includes(key) || !running || paused || demo) return;
    event.preventDefault(); keys.add(key); updatePushes();
  });
  on(window, "keyup", (event) => {
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    keys.delete(key); updatePushes();
  });
  on(window, "blur", () => pause(true));
  on(shell, "focusout", (event) => {
    if (!shell.contains(event.relatedTarget)) pause(true);
  });
  on(document, "visibilitychange", () => { if (document.hidden) pause(true); });
  on(window, "pagehide", () => { saveBest(); pause(true); });

  const observer = typeof window.ResizeObserver === "function" ? new window.ResizeObserver(resize) : null;
  observer?.observe(canvas);
  if (!observer) on(window, "resize", resize);
  if (!context) {
    startButton.disabled = true;
    demoButton.disabled = true;
    say("This browser cannot draw the toy. The benchmark records and submission tools still work.");
  }
  updateStats();
  resize();

  return {
    destroy() {
      if (!alive) return;
      saveBest();
      alive = false;
      if (frame) window.cancelAnimationFrame(frame);
      observer?.disconnect();
      for (const remove of removers) remove();
      shell.remove();
    },
  };
}
