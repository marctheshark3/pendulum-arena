import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const media='/assets/canonical18-telemetry-dont-stop-me-now.mp4';
for(const page of ['index.html','queen-eighteen.html']){
  test(`${page} offers one native benchmark video with its soundtrack`,async()=>{
    const html=await readFile(new URL(`../dist/${page}`,import.meta.url),'utf8');
    assert.equal((html.match(/<video\b/gi)??[]).length,1);
    assert.match(html,/<video[^>]+controls[^>]+playsinline/i);
    assert.doesNotMatch(html,/<(?:iframe|audio)\b|queen-paired\.js|play-together|\s(?:autoplay|muted)(?:\s|>)/i);
    assert.ok(html.includes(`<source src="${media}" type="video/mp4">`));
    assert.ok(html.includes(`href="${media}"`));
    for(const original of ['canonical18.mp4','canonical18-telemetry.mp4'])assert.ok(html.includes(`href="/assets/${original}"`));
    assert.match(html,/Don’t Stop Me Now/);
  });
}
