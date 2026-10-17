import {PairedPlayback, queenEmbedURL} from './queen-paired-controller.js';

const apiLoads = new WeakMap();

export function loadYouTubeAPI(win, doc, {timeoutMs = 12000, schedule = setTimeout, cancel = clearTimeout} = {}) {
  if (win.YT?.Player) return Promise.resolve(win.YT);
  if (apiLoads.has(win)) return apiLoads.get(win);
  const promise = new Promise((resolve, reject) => {
    const previous = win.onYouTubeIframeAPIReady;
    let timer;
    let settled = false;
    const cleanup = () => {
      if (timer !== undefined) cancel(timer);
      if (win.onYouTubeIframeAPIReady === ready) win.onYouTubeIframeAPIReady = previous;
    };
    const fail = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('The official YouTube player API is unavailable.'));
    };
    const ready = () => {
      if (settled) return;
      try { if (typeof previous === 'function') previous(); } catch { /* Preserve independent widgets. */ }
      if (!win.YT?.Player) { fail(); return; }
      settled = true;
      cleanup();
      resolve(win.YT);
    };
    win.onYouTubeIframeAPIReady = ready;
    const script = doc.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.async = true;
    script.onerror = fail;
    timer = schedule(fail, timeoutMs);
    doc.head.append(script);
  });
  apiLoads.set(win, promise);
  return promise;
}

export async function initPairedView(win, doc) {
  const replay = doc.getElementById('eighteen-replay');
  const frame = doc.getElementById('queen-player');
  const play = doc.getElementById('play-together');
  const pause = doc.getElementById('pause-together');
  const silent = doc.getElementById('play-replay-only');
  const status = doc.getElementById('playback-status');
  if (![replay, frame, play, pause, silent, status].every(Boolean)) return null;
  const controller = new PairedPlayback(replay, {onChange(state) {
    play.disabled = !state.canPlayTogether;
    silent.disabled = !state.canPlayReplay;
    status.textContent = state.message;
  }});
  play.addEventListener('click', () => controller.playTogether());
  pause.addEventListener('click', () => controller.pause());
  silent.addEventListener('click', () => controller.playReplayOnly());
  win.addEventListener('pagehide', () => controller.pause(), {once: true});
  // Set the real deployed/preview origin; never use an unrelated fixed origin.
  try {
    frame.src = queenEmbedURL(win.location.origin);
    const api = await loadYouTubeAPI(win, doc);
    new api.Player(frame, {events: {
      onReady: event => controller.setYouTubeReady(event.target),
      onStateChange: event => controller.youtubeChanged(event.data),
      onAutoplayBlocked: () => controller.autoplayBlocked(),
      onError: event => controller.youtubeFailed(event.data),
    }});
  } catch {
    controller.apiUnavailable();
  }
  return controller;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') initPairedView(window, document);
