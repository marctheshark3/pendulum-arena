import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const root=new URL('../dist/floppy/engine/',import.meta.url).pathname;
const native=JSON.parse(readFileSync(new URL('../test/data/floppy/native-fixtures.json',import.meta.url),'utf8'));
const {default:load}=await import(pathToFileURL(root+'vendor/mujoco.js'));
const {createGame}=await import(pathToFileURL(root+'game.js'));
const module=await load({wasmBinary:readFileSync(root+'vendor/mujoco.wasm')});
const options={module,configuration:JSON.parse(readFileSync(root+'profile.json','utf8')),serves:JSON.parse(readFileSync(root+'serves.json','utf8')),modelXML:readFileSync(root+'model.xml','utf8')};
const digest=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const checks=[];let qposMax=0,qvelMax=0,commands=0;
const eventKey=e=>JSON.stringify({kind:e.kind,step:e.step,life:e.life,geom:e.geom,id:e.id,reasons:e.reasons,wall:e.wall,outcome:e.outcome});
for(const fixture of native.fixtures){
 const game=await createGame(options);game.restoreValidationState(fixture.initial);
 let qpos=0,qvel=0;const faults=[],events=[];
 for(const expected of fixture.commands){
  const actual=game.step(fixture.force);commands++;
  for(let j=0;j<4;j++){qpos=Math.max(qpos,Math.abs(actual.qpos[j]-expected.qpos[j]));qvel=Math.max(qvel,Math.abs(actual.qvel[j]-expected.qvel[j]));}
  for(const key of ['phase','score','hits','health','active','pending','lives','outcome']) if(JSON.stringify(actual[key])!==JSON.stringify(expected.frame[key]))faults.push(key);
  events.push(...actual.events.map(eventKey));
 }
 const expectedEvents=fixture.commands.at(-1).events.map(eventKey);
 if(JSON.stringify(events)!==JSON.stringify(expectedEvents))faults.push('event_sequence');
 qposMax=Math.max(qposMax,qpos);qvelMax=Math.max(qvelMax,qvel);
 checks.push({id:fixture.id,qpos_max:qpos,qvel_max:qvel,state_faults:[...new Set(faults)],pass:qpos<=1e-6&&qvel<=1e-5&&!faults.length});
 game.dispose();
}
const report={schema:'floppy-browser-port-check-v1',native_fixture_sha256:digest(new URL('../test/data/floppy/native-fixtures.json',import.meta.url)),engine_sha256:digest(root+'game.js'),xml_sha256:digest(root+'model.xml'),runtime:module.mj_versionString(),commands,qpos_max:qposMax,qvel_max:qvelMax,checks,pass:checks.every(x=>x.pass),claim:'Selected fixed integration cases only. Different engine versions remain separate profiles.'};
console.log(JSON.stringify(report));
if(!report.pass)process.exitCode=1;
