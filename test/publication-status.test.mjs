import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import {canonicalJson,validateRegistry} from '../scripts/validate-registry.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const read=path=>readFileSync(new URL(`../${path}`,import.meta.url),'utf8');
const originalCommit='45326b078c54fff1265f5dfa4d48408a74ca4a6c';
const originalId='rage-mill-canonical17-20261016';
const replacementId=`${originalId}-published`;
const eventId='publish-canonical17-20261016';
const releasePublishedAt='2026-10-16T23:25:50Z';

function publicationSnapshot(){
  const registry=JSON.parse(read('dist/data/registry.json'));
  const base=JSON.parse(execFileSync('git',['show',`${originalCommit}:dist/data/registry.json`],
    {cwd:root,encoding:'utf8',maxBuffer:1024*1024}));
  const original=registry.records.find(record=>record.id===originalId);
  const replacement=registry.records.find(record=>record.id===replacementId);
  const event=registry.records.find(record=>record.id===eventId);
  assert.ok(original,'original pre-publication snapshot must remain visible');
  assert.ok(replacement,'published replacement snapshot is missing');
  assert.ok(event,'publication supersession event is missing');
  return {registry,base,original,replacement,event};
}

test('seventeen publication clocks append a replacement without rewriting qualification or history',()=>{
  const {registry,base,original,replacement,event}=publicationSnapshot();
  assert.equal(base.records.length,3);
  assert.equal(canonicalJson(registry.records.slice(0,base.records.length)),canonicalJson(base.records));
  assert.equal(canonicalJson(registry.protocols.slice(0,base.protocols.length)),canonicalJson(base.protocols));
  assert.deepEqual(validateRegistry(registry,{base}),[]);
  assert.equal(registry.records[base.records.length].id,replacementId);
  assert.equal(registry.records[base.records.length+1].id,eventId);
  for(const field of Object.keys(original).filter(field=>!['id','timestamps','evidence','notes'].includes(field)))
    assert.equal(canonicalJson(replacement[field]),canonicalJson(original[field]),`${field} changed in a publication-only correction`);
  assert.equal(replacement.timestamps.runAt,original.timestamps.runAt);
  assert.equal(replacement.timestamps.frozenAt,original.timestamps.frozenAt);
  assert.equal(original.timestamps.firstPublicAt,null);
  assert.equal(replacement.timestamps.firstPublicAt,releasePublishedAt);
  assert.ok(Date.parse(replacement.timestamps.registeredAt)>Date.parse(releasePublishedAt));
  assert.ok(Date.parse(replacement.timestamps.registeredAt)>Date.parse(original.timestamps.registeredAt));
  assert.equal(original.evidence.length,9);
  for(const evidence of original.evidence){
    assert.equal(evidence.publishedAt,null);
    const published=replacement.evidence.find(entry=>entry.id===evidence.id);
    assert.ok(published,`missing original evidence ${evidence.id}`);
    assert.deepEqual(published,{...evidence,publishedAt:releasePublishedAt});
  }
  assert.equal(event.type,'supersession');
  assert.equal(event.targetId,originalId);
  assert.equal(event.replacementId,replacementId);
  assert.equal(event.authority.id,original.entrant.id);
  assert.ok(Date.parse(event.timestamps.registeredAt)>=Date.parse(replacement.timestamps.registeredAt));
  const disposed=new Set(registry.records.filter(record=>record.type!=='result').map(record=>record.targetId));
  const active=registry.records.filter(record=>record.type==='result'&&record.links===17&&!disposed.has(record.id));
  assert.deepEqual(active.map(record=>record.id),[replacementId]);
});

test('publication supersession binds public proof bytes without inventing an earlier proof-publication clock',()=>{
  const {event,original,replacement}=publicationSnapshot();
  const proofBytes=readFileSync(new URL('../dist/evidence/canonical17/publication-status.json',import.meta.url));
  const proof=JSON.parse(proofBytes);
  const evidence=event.evidence.find(entry=>entry.url==='https://arena.neuralyogi.com/evidence/canonical17/publication-status.json');
  assert.ok(evidence,'supersession must link publicly hosted publication proof');
  assert.equal(evidence.sha256,createHash('sha256').update(proofBytes).digest('hex'));
  assert.equal(evidence.kind,'receipt');
  assert.equal(evidence.publishedAt,null,'the newly deployed proof must not inherit the earlier release publication time');
  assert.deepEqual(replacement.evidence.find(entry=>entry.id===evidence.id),evidence);
  assert.equal(proof.schema,'pendulum_arena.canonical17_publication_status.v1');
  assert.equal(proof.release.id,415452725);
  assert.equal(proof.release.tag_name,'canonical17-20261016');
  assert.equal(proof.release.html_url,'https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical17-20261016');
  assert.equal(proof.release.published_at,releasePublishedAt);
  assert.equal(proof.release.target_commitish,originalCommit);
  assert.equal(proof.release.draft,false);
  assert.equal(proof.asset_count,27);
  assert.equal(proof.assets.length,27);
  assert.equal(new Set(proof.assets.map(asset=>asset.name)).size,27);
  assert.equal(proof.all_asset_hashes_sizes_and_states_match_original_manifest,true);
  assert.ok(Date.parse(proof.verified_UTC)>=Date.parse(releasePublishedAt));
  assert.equal(proof.new_physical_episodes,0);
  assert.equal(proof.native_commands,0);
  for(const asset of proof.assets){
    assert.equal(asset.state,'uploaded');
    assert.ok(Number.isSafeInteger(asset.size)&&asset.size>0);
    assert.match(asset.digest,/^sha256:[0-9a-f]{64}$/);
    assert.ok(Date.parse(asset.updated_at)<=Date.parse(releasePublishedAt),'release asset was changed after its claimed publication');
  }
  for(const entry of [...original.evidence,original.source,original.model]){
    const asset=proof.assets.find(asset=>asset.browser_download_url===entry.url);
    assert.ok(asset,`release API proof omits ${entry.url}`);
    assert.equal(asset.digest,`sha256:${entry.sha256}`);
  }
});

test('registry shows active published seventeen before retained superseded history and counts it once',()=>{
  const {registry,base}=publicationSnapshot();
  // Exercise the real correction snapshot while allowing later counts and entrants to append.
  const snapshot={...registry,records:registry.records.slice(0,base.records.length+2)};
  const elements=new Map();
  for(const id of ['links-filter','verification-filter','registry','result-count'])
    elements.set(id,{value:'all',innerHTML:'',textContent:'',querySelectorAll:()=>[]});
  const source=read('dist/assets/app.js');
  const start=source.indexOf('const $ =');
  const end=source.indexOf("$('submission-form').addEventListener");
  assert.ok(start>=0&&end>start,'could not locate registry renderer without browser startup');
  runInNewContext(`${source.slice(start,end)}\nregistry=JSON.parse(snapshotJson);renderRegistry();`,{
    document:{getElementById:id=>elements.get(id)},
    snapshotJson:JSON.stringify(snapshot),URL,Intl
  },{timeout:1000});
  assert.equal(String(elements.get('result-count').textContent),'3');
  const html=elements.get('registry').innerHTML;
  const rows=[...html.matchAll(/<tr class="result-row">([\s\S]*?)<\/tr>/g)].map(match=>match[1]);
  assert.equal(rows.length,4,'all original results and the replacement remain visible');
  assert.match(rows[0],new RegExp(`data-receipt="${replacementId}"`));
  assert.match(rows[0],/datetime="2026-10-16T23:25:50Z">16 Oct 2026<\/time>/);
  assert.match(rows[0],/>141 \/ 141<\/span>/);
  assert.match(rows[0],/>Author-attested<\/span>/);
  assert.match(rows[1],new RegExp(`data-receipt="${originalId}"`));
  assert.match(rows[1],/>Superseded<\/span>/);
  assert.match(rows[1],/>Not established<\/time>/);
  assert.match(html,new RegExp(`data-open-result="${replacementId}"`));
});
