import { mountArena } from './arena-game.js';

const $ = (id) => document.getElementById(id);
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safeUrl = (value) => { try { const u = new URL(value); return ['https:','http:'].includes(u.protocol) && !u.username && !u.password ? u.href : '#'; } catch { return '#'; } };
const anchor = (url, label) => `<a href="${escape(safeUrl(url))}" target="_blank" rel="noopener noreferrer">${escape(label)} ↗</a>`;
const date = (value, full=false) => value ? new Intl.DateTimeFormat('en-GB',{day:'2-digit',month:'short',year:'numeric',...(full?{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'short'}:{}),timeZone:'UTC'}).format(new Date(value)) : 'Not established';
const labels = {'author-attested':'Author-attested','evidence-reviewed':'Evidence reviewed','third-party-reproduced':'Third-party reproduced'};
let registry;

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status}).`);
  return response.json();
}

function renderRegistry() {
  const links = $('links-filter').value;
  const verification = $('verification-filter').value;
  const disposition = new Map(registry.records.filter(r => r.type !== 'result').map(r => [r.targetId,r]));
  const results = registry.records.filter(r => r.type === 'result' && (links === 'all' || r.links === Number(links)) && (verification === 'all' || r.verification.level === verification));
  results.sort((a,b) => b.links-a.links || Number(disposition.has(a.id))-Number(disposition.has(b.id)) || a.timestamps.registeredAt.localeCompare(b.timestamps.registeredAt));
  $('result-count').textContent = results.filter(r=>r.qualified&&!disposition.has(r.id)).length;
  if (!results.length) { $('registry').innerHTML = '<p class="empty-state">No entries at this evidence level yet. The next reproducible result can be yours.</p>'; return; }
  const groups = new Map();
  for (const r of results) { if (!groups.has(r.protocolId)) groups.set(r.protocolId,[]); groups.get(r.protocolId).push(r); }
  $('registry').innerHTML = [...groups].map(([protocolId,records]) => {
    const protocol = registry.protocols.find(p => p.id === protocolId);
    return `<div class="result-group"><p class="protocol-group-label">${escape(protocol.title)} / ${escape(protocol.id)} / fingerprint ${escape(protocol.fingerprint.slice(0,12))}…</p><table class="result-table"><caption class="sr-only">${escape(protocol.title)} results</caption><thead><tr><th scope="col">Links</th><th scope="col">Entrant / method</th><th scope="col">Reserved cohort</th><th scope="col">Evidence status</th><th scope="col">First public evidence (UTC)</th></tr></thead><tbody>${records.map(r=>renderRow(r,disposition.get(r.id))).join('')}</tbody></table></div>`;
  }).join('');
  for (const button of $('registry').querySelectorAll('[data-receipt]')) button.addEventListener('click',()=>{
    const detail = document.getElementById(button.getAttribute('aria-controls'));
    const opening = detail.hidden;
    detail.hidden = !opening;
    button.setAttribute('aria-expanded',String(opening));
    button.textContent = opening ? 'Close receipt ↑' : 'Open receipt ↗';
  });
  for (const link of $('registry').querySelectorAll('[data-open-result]')) link.addEventListener('click',()=>{
    const id = link.dataset.openResult;
    const detail = document.getElementById(`receipt-${id}`);
    const button = [...$('registry').querySelectorAll('[data-receipt]')].find(b=>b.dataset.receipt===id);
    if(detail && button){ detail.hidden=false; button.setAttribute('aria-expanded','true'); button.textContent='Close receipt ↑'; }
  });
}

function renderRow(r, disposition) {
  const numbers = [r.links, ...Object.values(r.metrics), ...r.cohorts.flatMap(c=>[c.required,c.attempted,c.passed,...(c.seedStart==null?[]:[c.seedStart,c.seedEnd])])];
  if (!numbers.every(v=>typeof v==='number' && Number.isFinite(v))) throw new Error('Registry contains invalid numeric fields.');
  const held = r.cohorts.filter(c => ['reserved','video'].includes(c.stage));
  const passed = held.reduce((sum,c)=>sum+c.passed,0);
  const attempted = held.reduce((sum,c)=>sum+c.attempted,0);
  const level = r.verification.level;
  const detailId = `receipt-${r.id}`;
  const status = disposition ? disposition.type === 'retraction' ? 'Retracted' : 'Superseded' : r.qualified ? labels[level] : 'Partial campaign';
  const badgeClass = disposition ? 'withdrawn' : level === 'third-party-reproduced' ? 'reproduced' : level === 'evidence-reviewed' ? 'reviewed' : '';
  return `<tr class="result-row"><td class="links-cell" data-label="Links">${r.links}<small>links</small></td><td class="result-identity">${anchor(r.entrant.url,r.entrant.name).replace('<a ','<a class="entrant-name" ')}<span class="method-name">${escape(r.method.name)}</span><span class="table-sub">${escape(r.runtime.engine)} ${escape(r.runtime.engineVersion)} · ${escape(r.runtime.architecture)}</span></td><td class="result-hold"><span class="pass-stat">${passed} / ${attempted}</span><span class="table-sub">${r.metrics.minUprightSeconds.toFixed(2)}s min across all actual runs<br>${r.qualified?'Complete qualification claim':'Not qualified'}</span></td><td class="result-verification"><span class="badge ${badgeClass}">${escape(status)}</span><span class="table-sub">${level==='author-attested'?'Published by the entrant':escape(r.verification.reviewer?.name || 'Reviewer evidence linked')}<br>${r.method.wallClockRealtime?'Realtime claim disclosed':'Realtime not established'}</span></td><td class="result-publication"><time class="result-date" ${r.timestamps.firstPublicAt?`datetime="${escape(r.timestamps.firstPublicAt)}"`:''}>${date(r.timestamps.firstPublicAt)}</time><span class="table-sub">Freeze: ${date(r.timestamps.frozenAt)}</span><button class="receipt-toggle" data-receipt="${escape(r.id)}" aria-expanded="false" aria-controls="${escape(detailId)}">Open receipt ↗</button></td></tr>
  <tr class="result-detail-row" id="${escape(detailId)}" hidden><td colspan="5"><div class="result-detail"><div><h3>${escape(r.entrant.name)} / ${r.links} links</h3><p>${escape(r.method.summary)}</p>${renderPresentationNote(r)}<ul>${[...(r.method.disclosures||[]),...(r.notes||[])].map(n=>`<li>${escape(n)}</li>`).join('')}</ul><div class="evidence-links">${anchor(r.source.url,'Frozen source')}${anchor(r.model.url,'Model XML')}${r.evidence.map(e=>anchor(e.url,e.id.replaceAll('-',' '))).join('')}</div><p class="hash">Source commit: ${escape(r.source.commit)}<br>Model SHA256: ${escape(r.model.sha256)}<br>Protocol SHA256: ${escape(r.protocolFingerprint)}</p>${renderDisposition(disposition)}</div><div><h3>Clocks & checks</h3><dl><dt>Final run (declared)</dt><dd>${date(r.timestamps.runAt,true)}</dd><dt>Controller freeze</dt><dd>${date(r.timestamps.frozenAt,true)}</dd><dt>Public evidence</dt><dd>${date(r.timestamps.firstPublicAt,true)}</dd><dt>Registry insertion</dt><dd>${date(r.timestamps.registeredAt,true)}</dd><dt>Actual native audit</dt><dd>${r.metrics.auditedEpisodes} episodes · ${r.metrics.auditedCommands.toLocaleString('en')} commands</dd><dt>Audited defect / gap</dt><dd>${r.metrics.maxNativeDefect} / ${r.metrics.maxInitialGap}</dd><dt>Max |cart|</dt><dd>${r.metrics.maxCartMeters.toFixed(6)} m</dd></dl><p>These clocks describe different events. Registry insertion is not proof of an earlier publication. Same-owner audit code does not count as independent reproduction.</p><ul>${r.cohorts.map(c=>`<li>${escape(c.id)}: ${c.passed}/${c.attempted}${c.seedStart!=null?` · seeds ${c.seedStart}–${c.seedEnd}`:''}</li>`).join('')}</ul></div></div></td></tr>`;
}

// Current presentation context is separate from the append-only registry record.
function renderPresentationNote(record) {
  if (record.id !== 'rage-mill-canonical18-20261016-published') return '';
  return `<div class="presentation-update"><p><strong>Presentation update · 18 October 2026 UTC</strong></p><p>The hero and eighteen-link video page now offer a timed Queen “Don’t Stop Me Now” soundtrack MP4. The paired-player and no-rehosting wording in the published note below describes the earlier presentation. The scientific recording and benchmark qualification are unchanged.</p><div class="evidence-links"><a href="/assets/canonical18-telemetry-dont-stop-me-now-timed.provenance.json">Current soundtrack provenance ↗</a><a href="/assets/canonical18-timed-soundtrack-CREDITS.md">Current credits &amp; rights ↗</a></div><p><strong>Published disclosures and notes (preserved)</strong></p></div>`;
}

function renderDisposition(event) {
  if (!event) return '';
  return `<div class="disposition-event"><p><strong>${escape(event.type)}:</strong> ${escape(event.reason)}</p><p>Registered ${date(event.timestamps.registeredAt,true)} by ${anchor(event.authority.url,event.authority.name)}.${event.replacementId?` Replaced by record <a href="#receipt-${escape(event.replacementId)}" data-open-result="${escape(event.replacementId)}">${escape(event.replacementId)}</a>.`:''}</p><div class="evidence-links">${event.evidence.map(e=>anchor(e.url,e.id)).join('')}</div></div>`;
}

async function initRegistry() {
  try {
    registry = await loadJson('/data/registry.json');
    for (const links of [...new Set(registry.records.filter(r=>r.type==='result').map(r=>r.links))].sort((a,b)=>b-a)) {
      const option = document.createElement('option'); option.value=String(links); option.textContent=`${links} links`; $('links-filter').append(option);
    }
    $('links-filter').addEventListener('change',renderRegistry);
    $('verification-filter').addEventListener('change',renderRegistry);
    renderRegistry();
  } catch (error) {
    console.error(error);
    $('registry').innerHTML = '<p class="error-message">The registry could not load. You can <a href="https://github.com/marctheshark3/pendulum-arena/tree/main/dist/data">inspect the public data on GitHub</a>.</p>';
  }
}

async function initLineage() {
  try {
    const lineage = await loadJson('/data/lineage.json');
    $('lineage-people').innerHTML = lineage.people.map((p,i)=>`<article class="person"><span class="person-index">0${i+1} / IN THE LINEAGE</span><h3>${anchor(p.url,p.name)}</h3><span class="person-handle">${escape(p.handle)}</span><p class="person-role">${escape(p.role)}</p><p class="person-summary">${escape(p.summary)}</p><div class="person-links">${p.sources.map(s=>anchor(s.url,s.title)).join('')}</div></article>`).join('');
    $('source-timeline').innerHTML = lineage.timeline.map(t=>`<li><time ${t.date?`datetime="${escape(t.date)}"`:''}>${date(t.date,true)}<br>Platform publication</time><div><h3>${anchor(t.url,t.title)}</h3><p>${escape(t.description)}</p></div></li>`).join('');
  } catch(error) { console.error(error); $('lineage-people').innerHTML='<p>Primary-source credits are available in <a href="https://github.com/marctheshark3/pendulum-arena/blob/main/docs/attribution.md">the attribution record</a>.</p>'; }
}

$('submission-form').addEventListener('submit',(event)=>{
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  if (!['source','evidence'].every(k=>safeUrl(form.get(k))!=='#')) { $('submission-status').textContent='Please use a full HTTP or HTTPS source and evidence URL.'; return; }
  const url = new URL('https://github.com/marctheshark3/pendulum-arena/issues/new');
  url.searchParams.set('template','benchmark.yml');
  url.searchParams.set('title',`[Result] ${form.get('entrant')} — ${form.get('links')} links`);
  for (const key of ['entrant','links','source','evidence']) url.searchParams.set(key,form.get(key));
  $('submission-status').textContent='Opening your public GitHub submission…';
  window.location.assign(url.href);
});

try { mountArena($('arena-game')); } catch(error) { console.error(error); $('arena-game').textContent='The browser toy could not start. The native benchmark video and registry remain available above.'; }
initRegistry();
initLineage();
