import {readFileSync,existsSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join,relative,basename,isAbsolute} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {canonicalJson} from './validate-registry.mjs';

export function checkSite(projectRoot=process.cwd()) {
  const root=resolve(projectRoot,'dist'), problems=[];
  const hash=file=>createHash('sha256').update(readFileSync(file)).digest('hex');
  const readJson=file=>{
    try{return JSON.parse(readFileSync(file,'utf8'));}
    catch{problems.push(`Invalid or missing JSON: ${relative(root,file)}`);return null;}
  };
  const localFile=(directory,path)=>{
    const file=resolve(directory,path), rel=relative(root,file);
    if(rel==='..'||rel.startsWith('../')||isAbsolute(rel)){
      problems.push(`Local path escapes site: ${path}`);return null;
    }
    return file;
  };
  const html=readFileSync(join(root,'index.html'),'utf8');
  const paths=[...html.matchAll(/(?:src|href|poster)\s*=\s*["'](\/[^"']*)["']/gi)]
    .map(m=>new URL(m[1],'https://site.invalid').pathname);
  const plainLinks=new Set([...html.matchAll(/<a\b[^>]*\bhref\s*=\s*["'](\/[^"']*)["']/gi)]
    .map(m=>new URL(m[1],'https://site.invalid').pathname));
  for(const path of paths){
    const file=localFile(root,`.${path}`);
    if(file&&!existsSync(file))problems.push(`Missing asset: ${path}`);
  }
  for(const file of readdirSync(join(root,'assets')).filter(f=>f.endsWith('.js'))){
    try{execFileSync(process.execPath,['--check',join(root,'assets',file)],{stdio:'pipe'});}
    catch{problems.push(`JavaScript syntax: ${file}`);}
  }
  const data=new Map(readdirSync(join(root,'data')).filter(f=>f.endsWith('.json'))
    .map(file=>[file,readJson(join(root,'data',file))]));
  for(const file of ['registry.json','lineage.json'])if(!data.has(file))problems.push(`Missing JSON: data/${file}`);
  const registry=data.get('registry.json');
  const protocolCopies=new Map();
  for(const [file,protocol] of data){
    if(!file.startsWith('protocol-')||!protocol)continue;
    const registered=registry?.protocols?.find(p=>p.id===protocol.id);
    if(protocolCopies.has(protocol.id))problems.push(`Duplicate protocol JSON: ${protocol.id}`);
    protocolCopies.set(protocol.id,file);
    if(!registered||canonicalJson(protocol)!==canonicalJson(registered))
      problems.push(`Protocol JSON differs from registry: data/${file}`);
  }
  for(const protocol of registry?.protocols??[])
    if(!protocolCopies.has(protocol.id))problems.push(`Missing protocol JSON: ${protocol.id}`);
  const registeredVideos=new Map(), hostedVideoNames=new Set(), registeredEvidence=new Map();
  const isHosted=url=>url.hostname==='github.com'&&url.pathname.startsWith('/marctheshark3/pendulum-arena/releases/download/')
    ||url.hostname==='arena.neuralyogi.com'&&/^\/(assets|evidence)\//.test(url.pathname);
  for(const record of registry?.records?.filter(r=>r.type==='result')??[]){
    const stem=`canonical${record.links}`;
    for(const evidence of record.evidence){
      const url=new URL(evidence.url);
      if(!isHosted(url))continue;
      const name=basename(decodeURIComponent(url.pathname));
      const file=url.hostname==='arena.neuralyogi.com'
        ?localFile(root,`.${decodeURIComponent(url.pathname)}`)
        :join(root,evidence.kind==='video'?'assets':`evidence/${stem}`,name);
      if(!file)continue;
      if(evidence.kind==='video')hostedVideoNames.add(name);
      if(!existsSync(file)){
        if(evidence.kind==='video')problems.push(`Missing registered held-out recording: ${relative(root,file)}`);
        continue;
      }
      if(hash(file)!==evidence.sha256)
        problems.push(`Evidence bytes do not match registry: ${relative(root,file)}`);
      else registeredEvidence.set(file,evidence.sha256);
      if(evidence.kind==='video'&&name===`${stem}.mp4`){
        if(hash(file)!==evidence.sha256)
          problems.push(`Video is not the registered held-out recording: ${stem}.mp4`);
        else registeredVideos.set(name,evidence.sha256);
      }
    }
  }
  for(const file of readdirSync(join(root,'assets')).filter(f=>/^canonical\d+\.mp4$/.test(f)))
    if(!registeredVideos.has(file))problems.push(`Missing registered scientific video: ${file}`);
  let derivatives=0;
  const mediaFiles=readdirSync(join(root,'assets'));
  const derivativeFiles=mediaFiles.filter(f=>/^canonical\d+-.+\.mp4$/.test(f)
    &&(!hostedVideoNames.has(f)||mediaFiles.includes(f.replace(/\.mp4$/,'.provenance.json'))));
  for(const file of derivativeFiles)
    if(!mediaFiles.includes(file.replace(/\.mp4$/,'.provenance.json')))
      problems.push(`Missing soundtrack or telemetry provenance: ${file}`);
  const sidecars=new Map(mediaFiles.filter(f=>f.endsWith('.provenance.json'))
    .map(file=>[file,readJson(join(root,'assets',file))]));
  const telemetryVideos=new Map();
  const sameDimensions=(value,expected)=>Array.isArray(value)&&value.length===2&&
    value.every((dimension,index)=>dimension===expected[index]);
  const checkVideoBytes=(name,digest,label)=>{
    const target=typeof name==='string'?localFile(join(root,'assets'),name):null;
    if(!target||!existsSync(target)||hash(target)!==digest){
      problems.push(`${label} provenance bytes differ: ${name}`);return false;
    }
    return true;
  };
  // Presentation renders may change pixels and dimensions. Their authority comes
  // from the registered recording/receipt pins, never pixel equality with a prior render.
  for(const [file,provenance] of sidecars){
    if(provenance?.schema!=='pendulum_arena.telemetry_video.v1')continue;
    const before=problems.length, count=provenance.n_links;
    const pairedVideo=file.replace(/\.provenance\.json$/,'.mp4');
    if(!Number.isInteger(count)||count<1||provenance.video!==pairedVideo||
       !pairedVideo.startsWith(`canonical${count}-`)||provenance.scientific_video!==`canonical${count}.mp4`)
      problems.push(`Telemetry filenames/count differ from provenance: ${file}`);
    const scientific=registeredVideos.get(provenance.scientific_video);
    if(!scientific||provenance.scientific_video_sha256!==scientific)
      problems.push(`Telemetry source is not the registered scientific video: ${file}`);
    checkVideoBytes(provenance.video,provenance.video_sha256,'Telemetry');
    const evidenceRoot=join(root,'evidence',`canonical${count}`);
    const videoSummary=join(evidenceRoot,'video.json'), manifest=join(evidenceRoot,'files.json');
    if(!registeredEvidence.has(videoSummary)||!registeredEvidence.has(manifest))
      problems.push(`Telemetry source summaries are not registered evidence: ${file}`);
    const source=readJson(videoSummary), files=readJson(manifest)?.files;
    const seed=provenance.seed, prefix=`reserved/video/episode-${seed}`;
    for(const [path,digest] of [[`${prefix}/tape.npz`,provenance.source_recording_sha256],
                               [`${prefix}/receipt.json`,provenance.source_receipt_sha256]]){
      const rows=Array.isArray(files)?files.filter(row=>row?.path===path):[];
      if(rows.length!==1||!digest||rows[0].sha256!==digest)
        problems.push(`Telemetry recording/receipt is not in the registered manifest: ${file} (${path})`);
    }
    if(source?.video_sha256!==scientific||!Number.isInteger(seed)||source?.seed!==seed||
       source?.frames!==1500||source?.fps!==50||source?.seconds!==30)
      problems.push(`Telemetry seed/timing differs from scientific evidence: ${file}`);
    if(provenance.frames!==1500||provenance.fps!==50||provenance.duration_seconds!==30||
       !sameDimensions(provenance.dimensions,[1280,1080])||
       provenance.state_samples!=='post-command 1..1500'||
       !Number.isInteger(provenance.park_ticks)||provenance.park_ticks<0||provenance.park_ticks>1500||
       provenance.new_simulation_or_controller_inference!==false||provenance.complete!==true)
      problems.push(`Telemetry full-episode recorded-state render is incomplete: ${file}`);
    if(!plainLinks.has(`/assets/${provenance.scientific_video}`))
      problems.push(`Missing plain scientific video link: ${provenance.scientific_video}`);
    if(problems.length===before)telemetryVideos.set(pairedVideo,{provenance,file});
  }
  for(const [file,provenance] of sidecars){
    const pairedVideo=file.replace(/\.provenance\.json$/,'.mp4');
    const schema=typeof provenance?.schema==='string'
      ?provenance.schema.match(/^rage_mill\.canonical(\d+)_music_share\.v1$/):null;
    if(!schema){
      if(provenance?.schema!=='pendulum_arena.telemetry_video.v1'&&derivativeFiles.includes(pairedVideo))
        problems.push(`Invalid soundtrack or telemetry provenance schema: ${file}`);
      continue;
    }
    derivatives++;
    const scientificVideo=`canonical${schema[1]}.mp4`;
    const telemetry=telemetryVideos.get(provenance.original_video);
    if(provenance.video!==pairedVideo||
       !(provenance.original_video===scientificVideo||telemetry?.provenance.n_links===Number(schema[1]))||
       !pairedVideo.startsWith(`canonical${schema[1]}-`))
      problems.push(`Soundtrack filenames/count differ from provenance: ${file}`);
    const original=telemetry?.provenance.video_sha256??registeredVideos.get(provenance.original_video);
    if(!original||provenance.original_video_sha256!==original)
      problems.push(`Soundtrack original is not the registered scientific video or a validated telemetry render: ${file}`);
    if(provenance.video===provenance.original_video)
      problems.push(`Soundtrack must be a separate derivative: ${file}`);
    checkVideoBytes(provenance.video,provenance.video_sha256,'Soundtrack');
    checkVideoBytes(provenance.original_video,provenance.original_video_sha256,'Soundtrack');
    const source=typeof provenance.original_provenance==='string'
      ?localFile(join(root,'assets'),provenance.original_provenance):null;
    if(!source||!existsSync(source))problems.push(`Missing scientific video provenance: ${file}`);
    const expectedSource=telemetry?join(root,'assets',telemetry.file):join(root,'evidence',`canonical${schema[1]}`,'video.json');
    if(source!==expectedSource)
      problems.push(`Soundtrack original provenance points to a different video: ${file}`);
    if(telemetry&&(provenance.scientific_video!==scientificVideo||
       provenance.scientific_video_sha256!==registeredVideos.get(scientificVideo)||
       provenance.frames!==1500||provenance.fps!==50||provenance.duration_seconds!==30||
       !sameDimensions(provenance.resolution??provenance.dimensions,telemetry.provenance.dimensions)))
      problems.push(`Soundtrack telemetry/scientific lineage or timing differs: ${file}`);
    if(provenance.complete!==true||provenance.video_stream_copied_without_reencoding!==true||
       provenance.independent_review?.all_1500_decoded_video_frames_and_timestamps_identical!==true)
      problems.push(`Soundtrack frame-preservation review is incomplete: ${file}`);
    if(!plainLinks.has(`/assets/${scientificVideo}`))
      problems.push(`Missing plain scientific video link: ${scientificVideo}`);
  }
  for(const required of ['benchmark.yml','correction.yml'])
    if(!existsSync(join(projectRoot,'.github/ISSUE_TEMPLATE',required)))problems.push(`Missing public issue form ${required}`);
  if(/codex-preview|lorem ipsum|example\.org/i.test(html))problems.push('Placeholder content in public HTML.');
  return {problems,localAssets:new Set(paths).size,protocols:protocolCopies.size,
          scientificVideos:registeredVideos.size,telemetryRenders:telemetryVideos.size,soundtrackDerivatives:derivatives};
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const result=checkSite();
  if(result.problems.length){console.error(result.problems.join('\n'));process.exitCode=1;}
  else console.log(`Site checks passed: ${result.localAssets} local assets, JavaScript syntax, JSON, ${result.protocols} protocols, evidence hashes, ${result.scientificVideos} exact held-out videos, ${result.telemetryRenders} recorded-state telemetry renders and ${result.soundtrackDerivatives} separate soundtrack derivatives.`);
}
