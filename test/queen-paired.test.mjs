import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {PairedPlayback, queenEmbedURL, QUEEN_VIDEO_ID, youtubeErrorMessage} from '../dist/assets/queen-paired-controller.js';
import {initPairedView, loadYouTubeAPI} from '../dist/assets/queen-paired.js';

class Events {
  constructor() { this.listeners = new Map(); }
  addEventListener(name, callback) { if (!this.listeners.has(name)) this.listeners.set(name, new Set()); this.listeners.get(name).add(callback); }
  removeEventListener(name, callback) { this.listeners.get(name)?.delete(callback); }
  emit(name) { for (const callback of this.listeners.get(name) ?? []) callback(); }
}

class Replay extends Events {
  constructor() {
    super(); this.readyState = 2; this.duration = 30; this.paused = true; this.ended = false;
    this.currentTime = 0; this.playbackRate = 1; this.src = '/assets/canonical18-telemetry.mp4'; this.playCalls = 0;
    this.playResult = () => Promise.resolve();
  }
  play() { this.playCalls += 1; this.paused = false; this.emit('playing'); return this.playResult(); }
  pause() { if (!this.paused) { this.paused = true; this.emit('pause'); } }
}

function clock() {
  let id = 0;
  const waiting = new Map();
  return {waiting, schedule(callback) { waiting.set(++id, callback); return id; },
    cancel(key) { waiting.delete(key); }, flush() { const callbacks = [...waiting.values()]; waiting.clear(); callbacks.forEach(callback => callback()); }};
}

function fixture({ready = true} = {}) {
  const replay = new Replay();
  const calls = [];
  const youtube = {seekTo: (time, ahead) => calls.push(['seek', time, ahead]), playVideo: () => calls.push(['play']), pauseVideo: () => calls.push(['pause'])};
  const timer = clock();
  const messages = [];
  const controller = new PairedPlayback(replay, {...timer, onChange: state => messages.push(state)});
  if (ready) controller.setYouTubeReady(youtube);
  return {replay, calls, youtube, timer, messages, controller};
}

test('loading and readiness never start either player', () => {
  const f = fixture();
  assert.equal(f.replay.playCalls, 0);
  assert.deepEqual(f.calls, []);
  assert.equal(f.controller.snapshot().canPlayTogether, true);
  assert.equal(f.timer.waiting.size, 0);
});

test('both ready players are required for the paired click', () => {
  const f = fixture({ready: false});
  assert.equal(f.controller.playTogether(), false);
  assert.equal(f.replay.playCalls, 0);
  f.controller.setYouTubeReady(f.youtube);
  f.replay.duration = 60; f.controller.replayAvailable();
  assert.equal(f.controller.snapshot().canPlayTogether, false);
  assert.equal(f.controller.playTogether(), false);
  assert.deepEqual(f.calls, []);
});

test('a click lines up the official player with the current replay time, preserving source and rate', () => {
  const f = fixture(); f.replay.currentTime = 12.25;
  assert.equal(f.controller.playTogether(), true);
  assert.deepEqual(f.calls, [['seek', 12.25, true], ['play']]);
  assert.equal(f.replay.playCalls, 1);
  assert.equal(f.controller.snapshot().started, false);
  f.controller.youtubeChanged(1);
  assert.equal(f.controller.snapshot().started, true);
  assert.equal(f.timer.waiting.size, 0);
  assert.equal(f.replay.src, '/assets/canonical18-telemetry.mp4');
  assert.equal(f.replay.playbackRate, 1);
});

test('the explicit paired click restarts a completed replay from zero', () => {
  const f = fixture(); f.replay.currentTime = 30; f.replay.ended = true;
  f.controller.playTogether();
  assert.equal(f.replay.currentTime, 0);
  assert.deepEqual(f.calls[0], ['seek', 0, true]);
});

test('native replay pause pauses Queen and cancels the startup timer', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1);
  f.replay.pause();
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.calls.at(-1)[0], 'pause');
  assert.equal(f.timer.waiting.size, 0);
});

for (const event of ['ended', 'timeupdate']) {
  test(`${event} at thirty seconds pauses Queen and never loops automatically`, () => {
    const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1);
    f.replay.currentTime = 30; f.replay.ended = event === 'ended'; f.replay.emit(event);
    assert.equal(f.controller.snapshot().paired, false);
    assert.equal(f.replay.paused, true);
    assert.equal(f.calls.at(-1)[0], 'pause');
    assert.equal(f.replay.playCalls, 1);
    assert.match(f.controller.snapshot().message, /finished/);
  });
}

test('replay-only click never starts Queen, including after a paired session', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1);
  const previousPlays = f.calls.filter(call => call[0] === 'play').length;
  f.controller.playReplayOnly();
  assert.equal(f.calls.filter(call => call[0] === 'play').length, previousPlays);
  assert.equal(f.calls.at(-1)[0], 'pause');
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.replay.paused, false);
});

test('blocked native play stops Queen and reports the direct-player fallback', async () => {
  const f = fixture(); f.replay.playResult = () => Promise.reject(new Error('NotAllowedError'));
  f.controller.playTogether(); await Promise.resolve();
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.replay.paused, true);
  assert.equal(f.calls.at(-1)[0], 'pause');
  assert.match(f.controller.snapshot().message, /direct tap/);
});

test('an old rejected play promise cannot stop a newer user request', async () => {
  const f = fixture(); let rejectOld;
  f.replay.playResult = () => new Promise((resolve, reject) => { rejectOld = reject; });
  f.controller.playTogether();
  f.replay.playResult = () => Promise.resolve(); f.controller.playTogether(); f.controller.youtubeChanged(1);
  rejectOld(new Error('Interrupted previous request')); await Promise.resolve();
  assert.equal(f.controller.snapshot().paired, true);
  assert.equal(f.controller.snapshot().started, true);
});

test('YouTube buffering during startup is bounded without an automatic retry loop', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(3);
  assert.equal(f.controller.snapshot().paired, true);
  f.timer.flush();
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.calls.filter(call => call[0] === 'play').length, 1);
  assert.match(f.controller.snapshot().message, /still loading/);
});

test('replay startup buffering cannot falsely clear the bounded startup timer', () => {
  const f = fixture(); f.controller.playTogether(); f.replay.emit('waiting'); f.controller.youtubeChanged(1);
  assert.equal(f.controller.snapshot().started, false);
  assert.equal(f.timer.waiting.size, 1);
  f.replay.emit('playing');
  assert.equal(f.controller.snapshot().started, true);
  assert.equal(f.timer.waiting.size, 0);
});

for (const source of ['Queen', 'Replay']) {
  test(`${source} buffering after startup pauses both and requires another click`, () => {
    const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1);
    if (source === 'Queen') f.controller.youtubeChanged(3); else f.replay.emit('waiting');
    assert.equal(f.controller.snapshot().paired, false);
    assert.equal(f.replay.paused, true);
    assert.equal(f.calls.at(-1)[0], 'pause');
    assert.match(f.controller.snapshot().message, /buffering/);
  });
}

for (const state of [0, 2]) {
  test(`Queen's visible player state${state} stops an active paired replay`, () => {
    const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1); f.controller.youtubeChanged(state);
    assert.equal(f.controller.snapshot().paired, false);
    assert.equal(f.replay.paused, true);
  });
}

test('manual Queen playback never automatically starts the saved replay', () => {
  const f = fixture(); f.controller.youtubeChanged(1);
  assert.equal(f.replay.playCalls, 0);
  assert.equal(f.controller.snapshot().paired, false);
});

test('seeking an active replay pauses the pair instead of allowing silent drift', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1); f.replay.emit('seeking');
  assert.equal(f.controller.snapshot().paired, false);
  assert.match(f.controller.snapshot().message, /Position changed/);
});

test('official autoplay-blocked event stops both without hiding or retrying Queen', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.autoplayBlocked();
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.replay.paused, true);
  assert.equal(f.calls.filter(call => call[0] === 'play').length, 1);
  assert.match(f.controller.snapshot().message, /visible Play control/);
});

for (const code of [2, 5, 100, 101, 150, 153]) {
  test(`YouTube error${code} retains replay-only and official-source fallbacks`, () => {
    const f = fixture(); f.controller.playTogether(); f.controller.youtubeFailed(code);
    assert.equal(f.controller.snapshot().canPlayTogether, false);
    assert.equal(f.controller.snapshot().canPlayReplay, true);
    assert.equal(f.replay.paused, true);
    assert.match(youtubeErrorMessage(code), /official YouTube link/);
  });
}

test('a missing or corrupt replay pauses Queen and disables paired playback', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1); f.replay.emit('error');
  assert.equal(f.controller.snapshot().canPlayTogether, false);
  assert.equal(f.controller.snapshot().canPlayReplay, false);
  assert.equal(f.calls.at(-1)[0], 'pause');
  assert.match(f.controller.snapshot().message, /scientific recording/);
});

test('legacy undefined video.play return does not crash the paired request', () => {
  const f = fixture(); f.replay.playResult = () => undefined;
  assert.equal(f.controller.playTogether(), true);
  f.controller.pause();
});

test('embed URL always uses the official video, real origin, visible controls and no autoplay', () => {
  const url = new URL(queenEmbedURL('https://arena.neuralyogi.com'));
  assert.equal(url.origin, 'https://www.youtube.com');
  assert.equal(url.pathname, `/embed/${QUEEN_VIDEO_ID}`);
  assert.equal(url.searchParams.get('origin'), 'https://arena.neuralyogi.com');
  for (const [key, value] of Object.entries({enablejsapi: '1', autoplay: '0', controls: '1', playsinline: '1'})) assert.equal(url.searchParams.get(key), value);
  assert.throws(() => queenEmbedURL('file://'), /HTTP/);
  assert.throws(() => queenEmbedURL('https://arena.neuralyogi.com/some-path'), /origin/);
});

function documentStub() {
  return {scripts: [], createElement(tag) { assert.equal(tag, 'script'); return {}; },
    head: {append(script) { this.owner.scripts.push(script); }}};
}

function apiFixture() {
  const timer = clock(); const win = {}; const doc = documentStub(); doc.head.owner = doc;
  return {timer, win, doc};
}

test('official API loading is shared, preserves previous ready callback and never plays media', async () => {
  const f = apiFixture(); let previousCalls = 0; const previous = () => { previousCalls += 1; };
  f.win.onYouTubeIframeAPIReady = previous;
  const first = loadYouTubeAPI(f.win, f.doc, f.timer);
  assert.equal(loadYouTubeAPI(f.win, f.doc, f.timer), first);
  assert.equal(f.doc.scripts.length, 1);
  assert.equal(f.doc.scripts[0].src, 'https://www.youtube.com/iframe_api');
  f.win.YT = {Player() { assert.fail('API loading must not construct or play a player'); }};
  f.win.onYouTubeIframeAPIReady();
  assert.equal(await first, f.win.YT);
  assert.equal(previousCalls, 1);
  assert.equal(f.win.onYouTubeIframeAPIReady, previous);
  assert.equal(f.timer.waiting.size, 0);
});

for (const failure of ['network', 'timeout']) {
  test(`API ${failure} rejects once and clears its bounded timer`, async () => {
    const f = apiFixture(); const promise = loadYouTubeAPI(f.win, f.doc, f.timer);
    if (failure === 'network') f.doc.scripts[0].onerror(); else f.timer.flush();
    await assert.rejects(promise, /unavailable/);
    assert.equal(f.timer.waiting.size, 0);
  });
}

test('bootstrap wires ready events and buttons without autoplay', async () => {
  const replay = new Replay(); const frame = {src: ''}; const play = new Events(); const pause = new Events(); const silent = new Events(); const status = {};
  const elements = {'eighteen-replay': replay, 'queen-player': frame, 'play-together': play,
    'pause-together': pause, 'play-replay-only': silent, 'playback-status': status};
  let options;
  const win = new Events(); win.location = {origin: 'https://arena.neuralyogi.com'};
  win.YT = {Player: class {constructor(target, value) { assert.equal(target, frame); options = value; }}};
  const doc = {getElementById: id => elements[id]};
  const controller = await initPairedView(win, doc);
  const calls = []; const player = {playVideo: () => calls.push('play'), pauseVideo: () => calls.push('pause'), seekTo: time => calls.push(time)};
  assert.equal(replay.playCalls, 0);
  options.events.onReady({target: player});
  assert.deepEqual(calls, []);
  assert.equal(play.disabled, false);
  play.emit('click');
  assert.deepEqual(calls, [0, 'play']);
  options.events.onStateChange({data: 1}); pause.emit('click');
  assert.equal(calls.at(-1), 'pause');
  assert.equal(replay.paused, true);
  controller.destroy();
});

test('staged page has visible official player, direct fallbacks, no autoplay and mobile minimum viewport', async () => {
  const html = await readFile(new URL('../dist/queen-eighteen.html', import.meta.url), 'utf8');
  const css = await readFile(new URL('../dist/assets/queen-paired.css', import.meta.url), 'utf8');
  assert.match(html, /<video[^>]+controls[^>]+playsinline/);
  assert.match(html, /\/assets\/canonical18-telemetry\.mp4/);
  assert.match(html, /\/assets\/canonical18\.mp4/);
  assert.match(html, /youtube\.com\/watch\?v=HgzGwKwLmgM/);
  assert.match(html, /referrerpolicy="strict-origin-when-cross-origin"/);
  assert.match(html, /role="status" aria-live="polite"/);
  assert.doesNotMatch(html, /<audio|<iframe[^>]+hidden|\sautoplay(?:\s|>)/i);
  assert.match(css, /min-width:200px/);
  assert.match(css, /min-height:270px/);
  assert.doesNotMatch(css, /\.queen-frame[^}]+(?:display:none|visibility:hidden|opacity:0|position:absolute)/);
});

test('effective CSP permits the visible official YouTube API and frame', async () => {
  const headers = await readFile(new URL('../dist/_headers', import.meta.url), 'utf8');
  const csp = headers.split('\n').find(line => line.includes('Content-Security-Policy:'));
  assert.match(csp, /script-src 'self' https:\/\/www\.youtube\.com;/);
  assert.match(csp, /frame-src https:\/\/www\.youtube\.com;/);
  assert.match(csp, /media-src 'self'/);
  assert.match(csp, /connect-src 'self'/);
  assert.doesNotMatch(csp, /unsafe-eval|googlevideo|ytimg/);
  assert.match(headers, /Referrer-Policy: strict-origin-when-cross-origin/);
});

test('default browser timers keep the Window/global receiver on startup and pause', () => {
  const originalSet = globalThis.setTimeout, originalClear = globalThis.clearTimeout;
  const calls = [];
  try {
    globalThis.setTimeout = function (_callback, delay) { assert.equal(this, globalThis); calls.push(['set', delay]); return 17; };
    globalThis.clearTimeout = function (id) { assert.equal(this, globalThis); calls.push(['clear', id]); };
    const replay = new Replay();
    const controller = new PairedPlayback(replay);
    controller.setYouTubeReady({seekTo() {}, playVideo() {}, pauseVideo() {}});
    assert.equal(controller.playTogether(), true);
    controller.pause();
    assert.deepEqual(calls, [['set', 8000], ['clear', 17]]);
    controller.destroy();
  } finally { globalThis.setTimeout = originalSet; globalThis.clearTimeout = originalClear; }
});

test('a network stall with sufficient decoded replay data keeps both players running', () => {
  const f = fixture(); f.replay.readyState = 4;
  f.controller.playTogether(); f.controller.youtubeChanged(1);
  f.replay.emit('stalled');
  assert.equal(f.controller.snapshot().paired, true);
  assert.equal(f.controller.snapshot().started, true);
  assert.equal(f.replay.paused, false);
  assert.equal(f.calls.filter(call => call[0] === 'pause').length, 0);
});

test('a network stall without enough replay data pauses both players', () => {
  const f = fixture(); f.controller.playTogether(); f.controller.youtubeChanged(1);
  f.replay.readyState = 2; f.replay.emit('stalled');
  assert.equal(f.controller.snapshot().paired, false);
  assert.equal(f.replay.paused, true);
  assert.match(f.controller.snapshot().message, /buffering/);
});
