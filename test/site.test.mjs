import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {checkSite} from '../scripts/check-site.mjs';

const sha=value=>createHash('sha256').update(value).digest('hex');
function fixture(t,counts=[15,16]){
  const root=mkdtempSync(join(tmpdir(),'arena-site-'));
  t.after(()=>rmSync(root,{recursive:true,force:true}));
  const write=(path,value)=>{
    mkdirSync(join(root,path,'..'),{recursive:true});
    writeFileSync(join(root,path),typeof value==='string'?value:JSON.stringify(value));
  };
  write('.github/ISSUE_TEMPLATE/benchmark.yml','name: Benchmark');
  write('.github/ISSUE_TEMPLATE/correction.yml','name: Correction');
  write('dist/data/lineage.json',[]);
  const registry={protocols:[],records:[]};
  for(const count of counts){
    const protocol={id:`canonical${count}-v1`,linksMin:count,linksMax:count,requirements:{commands:1500}};
    registry.protocols.push(protocol);
    write(`dist/data/protocol-canonical${count}.json`,protocol);
    const video=`scientific held-out ${count}`;
    write(`dist/assets/canonical${count}.mp4`,video);
    const receipt={links:count,qualified:true};
    write(`dist/evidence/canonical${count}/audit.json`,receipt);
    write(`dist/evidence/canonical${count}/video.json`,{video_sha256:sha(video)});
    registry.records.push({type:'result',links:count,evidence:[
      {kind:'receipt',url:`https://github.com/marctheshark3/pendulum-arena/releases/download/canonical${count}/audit.json`,sha256:sha(JSON.stringify(receipt))},
      {kind:'video',url:`https://github.com/marctheshark3/pendulum-arena/releases/download/canonical${count}/canonical${count}.mp4`,sha256:sha(video)}
    ]});
  }
  write('dist/data/registry.json',registry);
  write('dist/index.html',counts.map(n=>`<a href="/assets/canonical${n}.mp4">Plain ${n}</a>`).join('\n'));
  const mutate=(path,fn)=>{const x=JSON.parse(readFileSync(join(root,path),'utf8'));fn(x);write(path,x);};
  const music=()=>{
    const video='soundtrack mux with same frames',original='scientific held-out 16';
    write('dist/assets/canonical16-im-still-standing.mp4',video);
    const p={schema:'rage_mill.canonical16_music_share.v1',video:'canonical16-im-still-standing.mp4',
      original_video:'canonical16.mp4',video_sha256:sha(video),original_video_sha256:sha(original),
      original_provenance:'../evidence/canonical16/video.json',complete:true,
      video_stream_copied_without_reencoding:true,
      independent_review:{all_1500_decoded_video_frames_and_timestamps_identical:true}};
    write('dist/assets/canonical16-im-still-standing.provenance.json',p);
    write('dist/index.html',readFileSync(join(root,'dist/index.html'),'utf8')+'\n<video src="/assets/canonical16-im-still-standing.mp4"></video>');
  };
  const telemetry=()=>{
    const summary={video_sha256:sha('scientific held-out 16'),seed:216500,frames:1500,fps:50,seconds:30};
    const manifest={files:[
      {path:'reserved/video/episode-216500/tape.npz',sha256:sha('saved tape 16')},
      {path:'reserved/video/episode-216500/receipt.json',sha256:sha('saved episode receipt 16')}
    ]};
    for(const [file,contents] of [['video.json',summary],['files.json',manifest]]){
      write(`dist/evidence/canonical16/${file}`,contents);
      mutate('dist/data/registry.json',r=>r.records[1].evidence.push({kind:'receipt',
        url:`https://github.com/marctheshark3/pendulum-arena/releases/download/canonical16/${file}`,
        sha256:sha(JSON.stringify(contents))}));
    }
    const p={schema:'pendulum_arena.telemetry_video.v1',video:'canonical16-telemetry.mp4',
      video_sha256:sha('new 1080p branded pixels'),scientific_video:'canonical16.mp4',
      scientific_video_sha256:sha('scientific held-out 16'),
      source_recording_sha256:sha('saved tape 16'),source_receipt_sha256:sha('saved episode receipt 16'),
      n_links:16,seed:216500,park_ticks:850,frames:1500,fps:50,duration_seconds:30,
      dimensions:[1280,1080],state_samples:'post-command 1..1500',
      new_simulation_or_controller_inference:false,complete:true};
    write('dist/assets/canonical16-telemetry.mp4','new 1080p branded pixels');
    write('dist/assets/canonical16-telemetry.provenance.json',p);
    return p;
  };
  const telemetryMusic=()=>{
    const p={schema:'rage_mill.canonical16_music_share.v1',video:'canonical16-telemetry-im-still-standing.mp4',
      video_sha256:sha('music mux of new 1080p pixels'),original_video:'canonical16-telemetry.mp4',
      original_video_sha256:sha('new 1080p branded pixels'),original_provenance:'canonical16-telemetry.provenance.json',
      scientific_video:'canonical16.mp4',scientific_video_sha256:sha('scientific held-out 16'),
      frames:1500,fps:50,duration_seconds:30,resolution:[1280,1080],complete:true,
      video_stream_copied_without_reencoding:true,
      independent_review:{all_1500_decoded_video_frames_and_timestamps_identical:true}};
    write(`dist/assets/${p.video}`,'music mux of new 1080p pixels');
    write('dist/assets/canonical16-telemetry-im-still-standing.provenance.json',p);
    write('dist/index.html',readFileSync(join(root,'dist/index.html'),'utf8')+`\n<video src="/assets/${p.video}"></video>`);
  };
  return {root,write,mutate,music,telemetry,telemetryMusic};
}
const errors=f=>checkSite(f.root).problems.join('\n');

test('all result counts and standalone protocols pass, including future counts',t=>{
  const f=fixture(t,[15,16,17]);
  assert.deepEqual(checkSite(f.root),{problems:[],localAssets:3,protocols:3,scientificVideos:3,telemetryRenders:0,soundtrackDerivatives:0});
});
test('receipt lookup uses each result count rather than the fifteen directory',t=>{
  const f=fixture(t);f.write('dist/evidence/canonical16/audit.json',{links:16,qualified:false});
  assert.match(errors(f),/Evidence bytes do not match registry: evidence\/canonical16\/audit.json/);
});
test('fifteen scientific video hash protection is retained',t=>{
  const f=fixture(t);f.write('dist/assets/canonical15.mp4','changed');
  assert.match(errors(f),/registered held-out recording: canonical15.mp4/);
});
test('a separately bound soundtrack can be featured while canonical sixteen stays registered',t=>{
  const f=fixture(t);f.music();
  f.mutate('dist/data/registry.json',r=>r.records[1].evidence.unshift({kind:'video',
    url:'https://github.com/marctheshark3/pendulum-arena/releases/download/canonical16/canonical16-im-still-standing.mp4',sha256:sha('soundtrack mux with same frames')}));
  assert.equal(errors(f),'');assert.equal(checkSite(f.root).soundtrackDerivatives,1);
});
test('soundtrack evidence cannot substitute for the registered scientific video',t=>{
  const f=fixture(t);f.music();
  f.mutate('dist/data/registry.json',r=>r.records[1].evidence=r.records[1].evidence.filter(e=>e.kind!=='video').concat({kind:'video',url:'https://github.com/marctheshark3/pendulum-arena/releases/download/canonical16/canonical16-im-still-standing.mp4',sha256:sha('soundtrack mux with same frames')}));
  assert.match(errors(f),/Missing registered scientific video: canonical16.mp4/);
});
test('a scored clip cannot hide a missing scientific asset or plain download link',t=>{
  const f=fixture(t);f.music();
  f.write('dist/index.html','<video src="/assets/canonical16-im-still-standing.mp4"></video>');
  assert.match(errors(f),/Missing plain scientific video link: canonical16.mp4/);
  rmSync(join(f.root,'dist/assets/canonical16.mp4'));
  assert.match(errors(f),/Missing registered held-out recording: assets\/canonical16.mp4/);
});
test('soundtrack original and derivative bytes must independently match provenance',t=>{
  const f=fixture(t);f.music();
  f.write('dist/assets/canonical16-im-still-standing.mp4','different soundtrack');
  assert.match(errors(f),/Soundtrack provenance bytes differ/);
  f.mutate('dist/assets/canonical16-im-still-standing.provenance.json',p=>p.original_video_sha256='0'.repeat(64));
  assert.match(errors(f),/Soundtrack original is not the registered scientific video/);
});
test('missing or relabeled soundtrack provenance cannot bypass its checks',t=>{
  const f=fixture(t);f.music();
  f.mutate('dist/assets/canonical16-im-still-standing.provenance.json',p=>p.schema='unrelated');
  assert.match(errors(f),/Invalid soundtrack or telemetry provenance schema/);
  rmSync(join(f.root,'dist/assets/canonical16-im-still-standing.provenance.json'));
  assert.match(errors(f),/Missing soundtrack or telemetry provenance/);
});
test('incomplete frame-preservation review and count aliases are rejected',t=>{
  const f=fixture(t);f.music();
  f.mutate('dist/assets/canonical16-im-still-standing.provenance.json',p=>{
    p.independent_review.all_1500_decoded_video_frames_and_timestamps_identical=false;
    p.original_video='canonical15.mp4';
  });
  assert.match(errors(f),/frame-preservation review is incomplete/);
  assert.match(errors(f),/filenames\/count differ/);
});
test('standalone protocol bytes must represent the embedded registry contract',t=>{
  const f=fixture(t);f.mutate('dist/data/protocol-canonical16.json',p=>p.requirements.commands=1499);
  assert.match(errors(f),/Protocol JSON differs from registry: data\/protocol-canonical16.json/);
});
test('every protocol JSON is parsed and every embedded protocol has one local copy',t=>{
  const f=fixture(t);f.write('dist/data/protocol-canonical17.json','{');
  assert.match(errors(f),/Invalid or missing JSON: data\/protocol-canonical17.json/);
  rmSync(join(f.root,'dist/data/protocol-canonical16.json'));
  assert.match(errors(f),/Missing protocol JSON: canonical16-v1/);
});
test('existing issue form, missing asset and placeholder checks remain active',t=>{
  const f=fixture(t);rmSync(join(f.root,'.github/ISSUE_TEMPLATE/correction.yml'));
  f.write('dist/index.html','lorem ipsum<img src="/assets/missing.png">');
  assert.match(errors(f),/Missing public issue form correction.yml/);
  assert.match(errors(f),/Missing asset: \/assets\/missing.png/);
  assert.match(errors(f),/Placeholder content/);
});


test('external community evidence does not require local copies or collide with a hosted same-count result',t=>{
  const f=fixture(t);
  f.mutate('dist/data/registry.json',r=>{
    r.records.push({type:'result',links:16,evidence:[
      {kind:'receipt',url:'https://community.example/releases/audit.json',sha256:'a'.repeat(64)},
      {kind:'video',url:'https://community.example/releases/canonical16.mp4',sha256:'b'.repeat(64)}
    ]});
    r.records.push({type:'result',links:18,evidence:[
      {kind:'video',url:'https://another.example/canonical18.mp4',sha256:'c'.repeat(64)}
    ]});
  });
  assert.equal(errors(f),'');
});
test('a second repository-hosted controller at the same count can use its own scientific video',t=>{
  const f=fixture(t);const second='distinct controller16 recording';
  f.write('dist/assets/canonical16-controller2.mp4',second);
  f.mutate('dist/data/registry.json',r=>r.records.push({type:'result',links:16,evidence:[
    {kind:'video',url:'https://arena.neuralyogi.com/assets/canonical16-controller2.mp4',sha256:sha(second)}
  ]}));
  assert.equal(errors(f),'');
  f.write('dist/assets/canonical16-controller2.mp4','tampered');
  assert.match(errors(f),/Evidence bytes do not match registry: assets\/canonical16-controller2.mp4/);
});

test('telemetry and its music derivative preserve the scientific root while permitting new pixels',t=>{
  const f=fixture(t);f.music();f.telemetry();f.telemetryMusic();
  assert.equal(errors(f),'');
  assert.equal(checkSite(f.root).telemetryRenders,1);
  assert.equal(checkSite(f.root).soundtrackDerivatives,2);
  assert.notEqual(sha('new 1080p branded pixels'),sha('scientific held-out 16'));
});
test('telemetry tape and episode receipt must match the exact registered seed manifest rows',async t=>{
  for(const [field,value] of [['source_recording_sha256','a'.repeat(64)],
                            ['source_receipt_sha256','b'.repeat(64)],['seed',216501]]){
    await t.test(field,t=>{
      const f=fixture(t);f.telemetry();
      f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>p[field]=value);
      assert.match(errors(f),/recording\/receipt is not in the registered manifest/);
    });
  }
});
test('telemetry cannot trust a forged or unregistered source manifest',t=>{
  const f=fixture(t);f.telemetry();
  f.mutate('dist/evidence/canonical16/files.json',p=>p.files[0].sha256='a'.repeat(64));
  f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>p.source_recording_sha256='a'.repeat(64));
  assert.match(errors(f),/Telemetry source summaries are not registered evidence/);
  f.mutate('dist/data/registry.json',r=>r.records[1].evidence=r.records[1].evidence.filter(e=>!e.url.endsWith('/files.json')));
  assert.match(errors(f),/Telemetry source summaries are not registered evidence/);
});
test('telemetry must include every post-command sample without new dynamics or altered timing',async t=>{
  for(const [field,value] of [['frames',1499],['fps',25],['duration_seconds',29.98],
      ['dimensions',[1280,720]],['dimensions',null],['state_samples','initial and selected states'],
      ['new_simulation_or_controller_inference',true],['complete',false],['park_ticks',1501]]){
    await t.test(`${field}=${JSON.stringify(value)}`,t=>{
      const f=fixture(t);f.telemetry();
      f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>p[field]=value);
      assert.match(errors(f),/full-episode recorded-state render is incomplete/);
    });
  }
});
test('telemetry count aliases, scientific root substitution and changed render bytes are refused',t=>{
  const f=fixture(t);f.telemetry();
  f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>{
    p.n_links=15;p.scientific_video_sha256='a'.repeat(64);
  });
  f.write('dist/assets/canonical16-telemetry.mp4','tampered render');
  assert.match(errors(f),/Telemetry filenames\/count differ/);
  assert.match(errors(f),/Telemetry source is not the registered scientific video/);
  assert.match(errors(f),/Telemetry provenance bytes differ/);
});
test('music provenance follows its immediate telemetry parent, not the old pixel render',t=>{
  const f=fixture(t);f.telemetry();f.telemetryMusic();
  f.mutate('dist/assets/canonical16-telemetry-im-still-standing.provenance.json',p=>{
    p.original_video_sha256=sha('scientific held-out 16');
    p.original_provenance='../evidence/canonical16/video.json';
    p.resolution=[1280,720];
  });
  assert.match(errors(f),/Soundtrack original is not the registered scientific video or a validated telemetry render/);
  assert.match(errors(f),/original provenance points to a different video/);
  assert.match(errors(f),/telemetry\/scientific lineage or timing differs/);
});
test('music cannot bypass invalid telemetry lineage or omit immediate-parent frame preservation',t=>{
  const f=fixture(t);f.telemetry();f.telemetryMusic();
  f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>p.source_recording_sha256='c'.repeat(64));
  f.mutate('dist/assets/canonical16-telemetry-im-still-standing.provenance.json',p=>{
    p.independent_review.all_1500_decoded_video_frames_and_timestamps_identical=false;
  });
  assert.match(errors(f),/Soundtrack original is not the registered scientific video or a validated telemetry render/);
  assert.match(errors(f),/frame-preservation review is incomplete/);
});
test('missing or relabeled telemetry sidecars cannot turn presentation renders into scientific evidence',t=>{
  const f=fixture(t);f.telemetry();
  f.mutate('dist/assets/canonical16-telemetry.provenance.json',p=>p.schema='unrelated');
  assert.match(errors(f),/Invalid soundtrack or telemetry provenance schema/);
  rmSync(join(f.root,'dist/assets/canonical16-telemetry.provenance.json'));
  assert.match(errors(f),/Missing soundtrack or telemetry provenance/);
});
