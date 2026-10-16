import {readFileSync,existsSync,readdirSync,statSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import {execFileSync} from 'node:child_process';

const root=resolve('dist');
const html=readFileSync(join(root,'index.html'),'utf8');
const problems=[];
const paths=[...html.matchAll(/(?:src|href|poster)="(\/[^"#?]*)/g)].map(m=>m[1]);
for(const path of paths) if(!existsSync(join(root,path)))problems.push(`Missing asset: ${path}`);
for(const file of readdirSync(join(root,'assets')).filter(f=>f.endsWith('.js')))try{execFileSync(process.execPath,['--check',join(root,'assets',file)],{stdio:'pipe'});}catch{problems.push(`JavaScript syntax: ${file}`);}
for(const file of ['data/registry.json','data/lineage.json','data/protocol-canonical15.json'])try{JSON.parse(readFileSync(join(root,file),'utf8'));}catch{problems.push(`Invalid JSON: ${file}`);}
const registry=JSON.parse(readFileSync(join(root,'data/registry.json'),'utf8'));
for(const record of registry.records.filter(r=>r.type==='result')){
  for(const evidence of record.evidence){
    const file=join(root,'evidence/canonical15',new URL(evidence.url).pathname.split('/').at(-1));
    if(existsSync(file)){
      const hash=createHash('sha256').update(readFileSync(file)).digest('hex');
      if(hash!==evidence.sha256)problems.push(`Evidence bytes do not match registry: ${file}`);
    }
  }
}
const video=join(root,'assets/canonical15.mp4');
const expected=registry.records.find(r=>r.type==='result'&&r.links===15)?.evidence.find(e=>e.kind==='video')?.sha256;
if(expected&&createHash('sha256').update(readFileSync(video)).digest('hex')!==expected)problems.push('Video is not the registered held-out recording.');
for(const required of ['benchmark.yml','correction.yml'])if(!existsSync(join('.github/ISSUE_TEMPLATE',required)))problems.push(`Missing public issue form ${required}`);
if(/codex-preview|lorem ipsum|example\.org/i.test(html))problems.push('Placeholder content in public HTML.');
if(problems.length){console.error(problems.join('\n'));process.exit(1);}
console.log(`Site checks passed: ${new Set(paths).size} local assets, JavaScript syntax, JSON, evidence hashes and exact held-out video (${statSync(video).size} bytes).`);
