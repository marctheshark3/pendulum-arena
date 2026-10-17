export const QUEEN_VIDEO_ID = 'HgzGwKwLmgM';
export const QUEEN_WATCH_URL = `https://www.youtube.com/watch?v=${QUEEN_VIDEO_ID}`;
export const REPLAY_SECONDS = 30;

export function queenEmbedURL(origin) {
  const host = new URL(origin);
  if (!['https:', 'http:'].includes(host.protocol) || host.origin !== origin) {
    throw new TypeError('An HTTP(S) page origin is required.');
  }
  const url = new URL(`https://www.youtube.com/embed/${QUEEN_VIDEO_ID}`);
  for (const [key, value] of Object.entries({enablejsapi: '1', origin, autoplay: '0', controls: '1', playsinline: '1'})) {
    url.searchParams.set(key, value);
  }
  return url.href;
}

export function youtubeErrorMessage(code) {
  if ([101, 150].includes(code)) return 'Queen’s video cannot play on this page. Open the official YouTube link, or play the replay only.';
  if (code === 153) return 'YouTube could not identify this page. Open the official YouTube link, or play the replay only.';
  if (code === 100) return 'Queen’s video is unavailable here. Open the official YouTube link, or play the replay only.';
  return 'Queen’s player is unavailable. Open the official YouTube link, or play the replay only.';
}

/** Coordinates two existing players. It never changes a media source or rate. */
export class PairedPlayback {
  constructor(replay, {onChange = () => {}, schedule = globalThis.setTimeout.bind(globalThis), cancel = globalThis.clearTimeout.bind(globalThis), startTimeoutMs = 8000} = {}) {
    this.replay = replay;
    this.onChange = onChange;
    this.schedule = schedule;
    this.cancel = cancel;
    this.startTimeoutMs = startTimeoutMs;
    this.youtube = null;
    this.youtubeReady = false;
    this.replayReady = false;
    this.paired = false;
    this.started = false;
    this.finished = false;
    this.attempt = 0;
    this.startTimer = null;
    this.youtubeState = -1;
    this.replayPlaying = false;
    this.message = 'Loading the players. The replay and official YouTube link remain available.';
    this.listeners = new Map([
      ['loadeddata', () => this.replayAvailable()], ['canplay', () => this.replayAvailable()],
      ['playing', () => this.replayStarted()], ['pause', () => this.replayPaused()],
      ['waiting', () => this.buffering('Replay')],
      // A network stall may fire after this small recording is fully buffered.
      ['stalled', () => { if (this.replay.readyState < 3) this.buffering('Replay'); }],
      ['seeking', () => { if (this.paired && this.started) this.pause('Position changed. Press Play together to line up the players.'); }],
      ['ended', () => this.finish()], ['timeupdate', () => { if (this.replay.currentTime >= REPLAY_SECONDS) this.finish(); }],
      ['error', () => this.replayFailed()],
    ]);
    for (const [event, listener] of this.listeners) replay.addEventListener(event, listener);
    this.replayAvailable(false);
    this.report();
  }

  snapshot() {
    return {paired: this.paired, started: this.started, canPlayTogether: this.youtubeReady && this.replayReady,
      canPlayReplay: this.replayReady, message: this.message};
  }

  report(message = this.message) {
    this.message = message;
    this.onChange(this.snapshot());
  }

  replayAvailable(announce = true) {
    this.replayReady = this.replay.readyState >= 2 && Number.isFinite(this.replay.duration)
      && Math.abs(this.replay.duration - REPLAY_SECONDS) < .1;
    if (announce && this.replayReady) this.report(this.youtubeReady ? 'Ready. Press Play together, or play the replay only.' : this.message);
  }

  setYouTubeReady(player) {
    this.youtube = player;
    this.youtubeReady = true;
    this.report(this.replayReady ? 'Ready. Press Play together, or play the replay only.' : 'Queen is ready. Waiting for the thirty-second replay.');
  }

  clearStartTimer() {
    if (this.startTimer !== null) this.cancel(this.startTimer);
    this.startTimer = null;
  }

  pauseQueen() {
    try {
      if (this.youtubeReady && this.youtubeState !== 2) {
        this.youtubeState = 2;
        this.youtube.pauseVideo();
      }
    } catch { /* Visible official-source fallback remains. */ }
  }

  pause(message = 'Paused. Press Play together to resume both players.') {
    this.attempt += 1;
    this.paired = this.started = this.replayPlaying = false;
    this.clearStartTimer();
    this.pauseQueen();
    if (!this.replay.paused) this.replay.pause();
    this.report(message);
  }

  /** Called only from the Play together button's user gesture. */
  playTogether() {
    if (!this.youtubeReady || !this.replayReady) return false;
    this.clearStartTimer();
    const attempt = ++this.attempt;
    this.finished = false;
    this.paired = true;
    this.started = this.replayPlaying = false;
    this.youtubeState = -1;
    if (this.replay.ended || this.replay.currentTime >= REPLAY_SECONDS) this.replay.currentTime = 0;
    this.report('Starting both players…');
    this.startTimer = this.schedule(() => {
      if (this.attempt === attempt && this.paired && !this.started) {
        this.pause('A player is still loading. Press Play together again, or use the replay and official YouTube link.');
      }
    }, this.startTimeoutMs);
    try {
      // Both play calls remain in the original click stack for browser policy.
      this.youtube.seekTo(Math.max(0, this.replay.currentTime), true);
      this.youtube.playVideo();
      const playing = this.replay.play();
      if (playing?.catch) playing.catch(() => {
        if (this.attempt === attempt) this.pause('Playback needs a direct tap. Use either visible player, or try Play together again.');
      });
    } catch {
      if (this.attempt === attempt) this.pause('Playback could not start. Use either visible player or the official YouTube link.');
      return false;
    }
    return true;
  }

  /** A separate user action; this never starts Queen. */
  playReplayOnly() {
    if (!this.replayReady) return false;
    this.pause('Playing the replay only. Queen is paused.');
    this.finished = false;
    const attempt = this.attempt;
    if (this.replay.ended || this.replay.currentTime >= REPLAY_SECONDS) this.replay.currentTime = 0;
    try {
      const playing = this.replay.play();
      if (playing?.catch) playing.catch(() => {
        if (this.attempt === attempt) this.pause('Tap the replay’s own Play control, or download its MP4.');
      });
    } catch {
      this.pause('The replay could not start. Its MP4 and scientific recording remain available below.');
      return false;
    }
    return true;
  }

  replayStarted() {
    this.replayPlaying = true;
    this.bothStarted();
  }

  bothStarted() {
    if (this.paired && this.replayPlaying && this.youtubeState === 1) {
      this.started = true;
      this.clearStartTimer();
      this.report('Playing together. Pause both at any time.');
    }
  }

  replayPaused() {
    this.replayPlaying = false;
    if (this.finished) return;
    if (this.replay.ended || this.replay.currentTime >= REPLAY_SECONDS) this.finish();
    else if (this.paired) this.pause();
    else this.pauseQueen();
  }

  buffering(name) {
    if (!this.paired) return;
    if (name === 'Replay') this.replayPlaying = false;
    if (this.started) this.pause(`${name} is buffering. Press Play together when it is ready, or play the replay only.`);
    else this.report('Starting both players… waiting for the connection.');
  }

  youtubeChanged(state) {
    this.youtubeState = state;
    if (!this.paired) return;
    if (state === 1) this.bothStarted();
    else if (state === 0) this.pause('Queen’s video ended. The replay is paused.');
    else if (state === 2 && this.started) this.pause('Queen is paused. Press Play together to resume both players.');
    else if (state === 3) this.buffering('Queen');
  }

  youtubeFailed(code) {
    this.pause(youtubeErrorMessage(code));
    this.youtubeReady = false;
    this.report();
  }

  autoplayBlocked() {
    this.pause('YouTube needs a direct tap. Use its visible Play control, then try Play together; the replay also works on its own.');
  }

  apiUnavailable() {
    this.pause('Paired controls could not reach YouTube. Use the visible Queen player or official link, and play the replay on its own.');
    this.youtubeReady = false;
    this.report();
  }

  replayFailed() {
    this.replayReady = false;
    this.pause('The replay could not load. Use its download or scientific recording below; Queen’s official link remains available.');
  }

  finish() {
    if (this.finished) return;
    this.finished = true;
    this.pause('The thirty-second replay is finished. Press Play together to start it again.');
  }

  destroy() {
    this.pause();
    for (const [event, listener] of this.listeners) this.replay.removeEventListener(event, listener);
    this.listeners.clear();
  }
}
