import {readFileSync,existsSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join,relative,basename,dirname,isAbsolute} from 'node:path';
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
  const registeredVideos=new Map(), hostedVideoNames=new Set();
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
  const soundtrackFiles=mediaFiles.filter(f=>/^canonical\d+-.+\.mp4$/.test(f)
    &&(!hostedVideoNames.has(f)||mediaFiles.includes(f.replace(/\.mp4$/,'.provenance.json'))));
  for(const file of soundtrackFiles)
    if(!mediaFiles.includes(file.replace(/\.mp4$/,'.provenance.json')))
      problems.push(`Missing soundtrack provenance: ${file}`);
  for(const file of mediaFiles.filter(f=>f.endsWith('.provenance.json'))){
    const provenance=readJson(join(root,'assets',file));
    const pairedVideo=file.replace(/\.provenance\.json$/,'.mp4');
    const schema=provenance?.schema?.match(/^rage_mill\.canonical(\d+)_music_share\.v1$/);
    if(!schema){
      if(soundtrackFiles.includes(pairedVideo))problems.push(`Invalid soundtrack provenance schema: ${file}`);
      continue;
    }
    derivatives++;
    if(provenance.video!==pairedVideo||provenance.original_video!==`canonical${schema[1]}.mp4`||
       !pairedVideo.startsWith(`canonical${schema[1]}-`))
      problems.push(`Soundtrack filenames/count differ from provenance: ${file}`);
    const original=registeredVideos.get(provenance.original_video);
    if(!original||provenance.original_video_sha256!==original)
      problems.push(`Soundtrack original is not the registered scientific video: ${file}`);
    if(provenance.video===provenance.original_video)
      problems.push(`Soundtrack must be a separate derivative: ${file}`);
    for(const [name,digest] of [[provenance.video,provenance.video_sha256],
                               [provenance.original_video,provenance.original_video_sha256]]){
      const target=typeof name==='string'?localFile(join(root,'assets'),name):null;
      if(!target||!existsSync(target)||hash(target)!==digest)
        problems.push(`Soundtrack provenance bytes differ: ${file} (${name})`);
    }
    const source=typeof provenance.original_provenance==='string'
      ?localFile(join(root,'assets'),provenance.original_provenance):null;
    if(!source||!existsSync(source))problems.push(`Missing scientific video provenance: ${file}`);
    if(provenance.complete!==true||provenance.video_stream_copied_without_reencoding!==true||
       provenance.independent_review?.all_1500_decoded_video_frames_and_timestamps_identical!==true)
      problems.push(`Soundtrack frame-preservation review is incomplete: ${file}`);
    if(!plainLinks.has(`/assets/${provenance.original_video}`))
      problems.push(`Missing plain scientific video link: ${provenance.original_video}`);
  }
  for(const required of ['benchmark.yml','correction.yml'])
    if(!existsSync(join(projectRoot,'.github/ISSUE_TEMPLATE',required)))problems.push(`Missing public issue form ${required}`);
  if(/codex-preview|lorem ipsum|example\.org/i.test(html))problems.push('Placeholder content in public HTML.');
  return {problems,localAssets:new Set(paths).size,protocols:protocolCopies.size,
          scientificVideos:registeredVideos.size,soundtrackDerivatives:derivatives};
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const result=checkSite();
  if(result.problems.length){console.error(result.problems.join('\n'));process.exitCode=1;}
  else console.log(`Site checks passed: ${result.localAssets} local assets, JavaScript syntax, JSON, ${result.protocols} protocols, evidence hashes, ${result.scientificVideos} exact held-out videos and ${result.soundtrackDerivatives} separate soundtrack derivatives.`);
}
