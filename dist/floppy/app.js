import {createRenderer} from './render.js';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['court','court-wrap','court-mode','court-status','court-overlay','overlay-kicker','overlay-title','overlay-copy','overlay-action','score','bricks','lives','time','time-label','target-label','target-value','steer-left','steer-right','steer-center','mobile-pause','go','reset','feedback','play-tab','watch-tab','play-panel','watch-panel','replay-go','replay-reset','replay-status','replay-time','replay-position','runtime-version','clock-note'].map(id=>[id,$(id)]));
const renderer = createRenderer(ui.court);
const SEED = 20261019;
const DT = .02;
let game = null, state = null, mode = 'play', running = false, started = false, finished = false;
let target = 0, controlMode = 'assisted', lastFrame = null, accumulator = 0;
let replay = null, replayTime = 0, replayPlaying = false, replayIndex = 0;
let simulationError = null;
let animationPending = false, keyboardPulse = 0, pulseTicks = 0;
const heldKeys = new Set(), heldPointers = new Map();

const clamp = (v,lo,hi) => Math.min(hi,Math.max(lo,v));
const clock = value => `${Math.floor(Math.max(0,value)/60)}:${Math.floor(Math.max(0,value)%60).toString().padStart(2,'0')}`;
function direction(){
  let left=heldKeys.has('ArrowLeft')||heldKeys.has('KeyA');
  let right=heldKeys.has('ArrowRight')||heldKeys.has('KeyD');
  for(const d of heldPointers.values()){if(d<0)left=true;else right=true;}
  return left||right?Number(right)-Number(left):pulseTicks>0?keyboardPulse:0;
}
function releaseInput(){heldKeys.clear();heldPointers.clear();keyboardPulse=0;pulseTicks=0;ui['steer-left'].classList.remove('pressed');ui['steer-right'].classList.remove('pressed');}
function setStatus(text){ui['court-status'].replaceChildren();const dot=document.createElement('i');dot.className='status-light';ui['court-status'].append(dot,document.createTextNode(text));}
function overlay({kicker,title,copy,action,disabled=false}){
  ui['court-overlay'].hidden=false;ui['overlay-kicker'].textContent=kicker;ui['overlay-title'].replaceChildren(...title.split('\n').flatMap((s,i)=>i?[document.createElement('br'),document.createTextNode(s)]:[document.createTextNode(s)]));
  ui['overlay-copy'].textContent=copy;ui['overlay-action'].textContent=action;ui['overlay-action'].disabled=disabled;
}
function drawStats(s){
  if(!s)return;
  ui.score.textContent=String(s.score||0).padStart(4,'0');
  const count=Object.values(s.health||{}).filter(hp=>hp<=0).length;
  ui.bricks.textContent=count;ui.lives.textContent=s.lives ?? 3;ui.time.textContent=clock(mode==='watch'?(s.time||0):120-(s.time||0));
  ui['time-label'].textContent=mode==='watch'?'ELAPSED':'TIME LEFT';
  ui.court.setAttribute('aria-label',`First Light ${mode==='watch'?'recording':'human game'}: ${count} of 15 bricks, score ${s.score||0}, ${s.lives??3} lives, ${clock(mode==='watch'?(s.time||0):120-(s.time||0))} ${mode==='watch'?'elapsed':'remaining'}. ${s.phase||'ready'}.`);
}
function showState(s,clearTrail=false){
  drawStats(s);renderer.render(s,{target,showTarget:mode==='play'&&controlMode==='assisted',replay:mode==='watch',clearTrail});
}
function syncControlReadout(){
  ui['target-label'].textContent=controlMode==='assisted'?'TARGET POSITION':'CART FORCE';
  const value=controlMode==='assisted'?target:direction()*20;
  ui['target-value'].replaceChildren(document.createTextNode(`${value>=0?'+':''}${controlMode==='assisted'?value.toFixed(2):value.toFixed(0)} `));
  const unit=document.createElement('small');unit.textContent=controlMode==='assisted'?'m':'N';ui['target-value'].append(unit);
  ui['steer-center'].setAttribute('aria-label',controlMode==='assisted'?'Center target':'Release cart force');
  ui['court-mode'].textContent=mode==='watch'?'AUTONOMOUS / RETAINED RECORDING':controlMode==='assisted'?'HUMAN / ASSISTED DIRECTOR':'HUMAN / DIRECT CART FORCE';
}
function pause(reason='Paused. Press Resume to continue.'){
  if(mode==='watch'){replayPlaying=false;syncReplay();return;}
  if(!running)return;
  running=false;document.body.classList.remove('is-playing');accumulator=0;lastFrame=null;releaseInput();ui.go.textContent='Resume ↗';setStatus('PAUSED');ui.feedback.textContent=reason;
  overlay({kicker:'TAKE YOUR TIME',title:'The court can wait.',copy:reason,action:'Resume ↗'});
}
function start(){
  if(mode==='watch'){toggleReplay();return;}
  if(!game||simulationError)return;
  if(finished){reset();return;}
  started=true;running=true;document.body.classList.add('is-playing');lastFrame=null;accumulator=0;releaseInput();ui['court-overlay'].hidden=true;ui.go.textContent='Pause Ⅱ';setStatus('LIVE');
  ui.feedback.textContent=controlMode==='assisted'?'Steer the target. Balance assist handles the force; it does not chase the ball.':'Direct force: hold left or right for ±20 N. The passive hinge needs your balance.';
  ui.court.focus({preventScroll:true});
  requestFrame();
}
function reset(){
  if(!game)return;
  running=false;document.body.classList.remove('is-playing');started=false;finished=false;target=0;accumulator=0;lastFrame=null;releaseInput();
  state=game.reset();if(!state)state=game.snapshot();
  if(mode==='watch'){ui.go.textContent='Go. Keep it together. ↗';syncReplay();return;}
  ui.go.textContent='Go. Keep it together. ↗';setStatus('READY');ui.feedback.textContent='Board reset. Same board and serve seed. Press Go when you are ready.';syncControlReadout();showState(state,true);
  overlay({kicker:'YOUR TURN, HUMAN',title:'A little balance.\nA lot of nerve.',copy:'15 bricks. Three lives. Steer the cart target and keep the paddle ready for the return.',action:'Go. Keep it together. ↗'});
}
function endBoard(){
  running=false;document.body.classList.remove('is-playing');finished=true;releaseInput();setStatus(state.outcome==='cleared'?'BOARD CLEAR':'FINISHED');ui.go.textContent='Try again ↺';
  const clear=state.outcome==='cleared';
  const title=clear?'Skyline cleared.':state.outcome==='timeout'?'Time is up.':'Gravity gets a turn.';
  const copy=clear?`All 15 bricks. ${state.score} points. A First Light clear in this human browser alpha.`:state.outcome==='timeout'?`${Object.values(state.health).filter(hp=>hp<=0).length} of 15 bricks removed. A timeout ends the board without a win.`:`${Object.values(state.health).filter(hp=>hp<=0).length} of 15 bricks removed. Reset for the same board and serve.`;
  ui.feedback.textContent=copy;overlay({kicker:clear?'FIRST LIGHT / COMPLETE':'FIRST LIGHT / RUN ENDED',title,copy,action:'Try again ↺'});
}
function gameEvents(events=[]){
  for(const event of events){
    if(event.kind==='life_lost')ui.feedback.textContent=`Life lost: ${(event.reasons||[]).map(x=>x.replaceAll('_',' ')).join(', ') || 'fault'}. Bricks stay removed; a new warmup follows.`;
    if(event.kind==='brick_removed')ui.feedback.textContent='Brick removed. +100 points. Keep the paddle ready for the return.';
    if(event.kind==='serve')ui.feedback.textContent='Ball in play. Read the return and steer ahead of it.';
    if(event.kind==='paddle_hit')ui.feedback.textContent='Legal paddle return. Stay balanced for the next one.';
  }
}
function stepGame(){
  const d=direction();
  if(controlMode==='assisted')target=clamp(target+d*.025,-1.65,1.65);
  const force=controlMode==='assisted'?game.assistedForce(target):d*20;
  state=game.step(force);if(!state)state=game.snapshot();
  if(pulseTicks>0)pulseTicks--;
  gameEvents(state.events);syncControlReadout();showState(state);
  if(state.phase==='finished')endBoard();
}

function replayState(){
  if(!replay)return null;
  const frames=replay.frames;
  while(replayIndex<frames.length-1&&frames[replayIndex+1].time<=replayTime)replayIndex++;
  while(replayIndex>0&&frames[replayIndex].time>replayTime)replayIndex--;
  return {...frames[replayIndex],board:replay.board,profile:replay.profile};
}
function syncReplay(){
  const duration=replay?.frames.at(-1)?.time || 45.0281875;
  ui['replay-time'].textContent=`${clock(replayTime)} / ${clock(duration)}`;ui['replay-position'].value=String(replayTime/duration);
  ui['replay-go'].textContent=replayPlaying?'Pause Ⅱ':replayTime>=duration?'Replay ↺':'Play recording ▶';
  ui['replay-go'].disabled=!replay;setStatus(replayPlaying?'RECORDING':'REPLAY PAUSED');
  if(replay)showState(replayState());
}
function toggleReplay(){
  if(!replay)return;
  if(replayTime>=replay.frames.at(-1).time){replayTime=0;replayIndex=0;}
  replayPlaying=!replayPlaying;lastFrame=null;ui['court-overlay'].hidden=true;
  ui['replay-status'].textContent=replayPlaying?'Playing the retained First Light recording. No live simulation or human input.':'Recording paused. Scrub to inspect any part of this board.';syncReplay();
  if(replayPlaying)requestFrame();
}
function switchMode(next){
  if(mode===next)return;
  pause('Paused while you watch the retained recording.');replayPlaying=false;mode=next;lastFrame=null;releaseInput();
  for(const [id,selected]of[['play-tab',next==='play'],['watch-tab',next==='watch']]){ui[id].setAttribute('aria-selected',String(selected));ui[id].tabIndex=selected?0:-1;ui[id].classList.toggle('active',selected);}
  ui['play-panel'].hidden=next!=='play';ui['watch-panel'].hidden=next!=='watch';syncControlReadout();
  if(next==='watch'){
    ui['clock-note'].textContent='RETAINED 10 Hz RECORDING';ui['court-overlay'].hidden=true;syncReplay();
    if(!replay)overlay({kicker:'THE CONTROLLER GOES FIRST',title:'Loading the recording.',copy:'A full retained First Light clear. Human play stays paused.',action:'Loading…',disabled:true});
  }else{
    ui['clock-note'].textContent='SIMULATION CLOCK';
    if(state)showState(state,true);
    if(finished)endBoard();
    else if(started){setStatus('PAUSED');overlay({kicker:'YOUR GAME IS PAUSED',title:'Ready for the return?',copy:'The simulation has stayed still while you watched.',action:'Resume ↗'});}
    else if(game){setStatus('READY');overlay({kicker:'YOUR TURN, HUMAN',title:'A little balance.\nA lot of nerve.',copy:'15 bricks. Three lives. Steer the cart target and keep the paddle ready for the return.',action:'Go. Keep it together. ↗'});}
  }
}

ui.go.addEventListener('click',()=>running?pause():start());ui['overlay-action'].addEventListener('click',()=>start());ui.reset.addEventListener('click',reset);
ui['play-tab'].addEventListener('click',()=>switchMode('play'));ui['watch-tab'].addEventListener('click',()=>switchMode('watch'));
for(const [id,d]of[['steer-left',-1],['steer-right',1]]){
  const button=ui[id];
  button.addEventListener('pointerdown',e=>{if(mode!=='play'||!running)return;e.preventDefault();button.setPointerCapture(e.pointerId);heldPointers.set(e.pointerId,d);button.classList.add('pressed');syncControlReadout();});
  const up=e=>{heldPointers.delete(e.pointerId);button.classList.remove('pressed');syncControlReadout();};
  button.addEventListener('pointerup',up);button.addEventListener('pointercancel',up);button.addEventListener('lostpointercapture',up);
  button.addEventListener('click',e=>{
    if(e.detail!==0||!running||mode!=='play')return;
    if(controlMode==='assisted'){target=clamp(target+d*.12,-1.65,1.65);syncControlReadout();showState(state);}
    else{keyboardPulse=d;pulseTicks=5;syncControlReadout();}
  });
}
ui['mobile-pause'].addEventListener('click',()=>pause());
ui['steer-center'].addEventListener('click',()=>{releaseInput();target=0;syncControlReadout();if(state&&mode==='play')showState(state);});
for(const input of document.querySelectorAll('input[name="control-mode"]'))input.addEventListener('change',()=>{
  if(!input.checked)return;pause('Control style changed. Resume when you are ready.');controlMode=input.value;releaseInput();target=state?.own?.x||0;syncControlReadout();
  if(state&&mode==='play')showState(state);
});
function keyScope(e){return mode==='play'&&(e.target===ui.court||e.target.closest?.('#play-panel'));}
document.addEventListener('keydown',e=>{
  if(!keyScope(e)||e.target.matches('input,select,textarea'))return;
  if(['ArrowLeft','ArrowRight','KeyA','KeyD'].includes(e.code)){e.preventDefault();if(running){heldKeys.add(e.code);ui['steer-left'].classList.toggle('pressed',direction()<0);ui['steer-right'].classList.toggle('pressed',direction()>0);syncControlReadout();}}
  if(e.code==='Space'&&e.target===ui.court&&!e.repeat){e.preventDefault();running?pause():start();}
});
document.addEventListener('keyup',e=>{if(!heldKeys.has(e.code))return;heldKeys.delete(e.code);ui['steer-left'].classList.toggle('pressed',direction()<0);ui['steer-right'].classList.toggle('pressed',direction()>0);syncControlReadout();});
window.addEventListener('blur',()=>{if(mode==='play')pause('Window left. Human play paused; press Resume to continue.');else{replayPlaying=false;syncReplay();}releaseInput();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(mode==='play')pause('Tab hidden. Human play paused; press Resume to continue.');else{replayPlaying=false;syncReplay();}releaseInput();}});
window.addEventListener('pagehide',()=>{pause('Page left. Human play paused; press Resume when you return.');replayPlaying=false;releaseInput();});
ui['replay-go'].addEventListener('click',toggleReplay);ui['replay-reset'].addEventListener('click',()=>{replayPlaying=false;replayTime=0;replayIndex=0;syncReplay();});
ui['replay-position'].addEventListener('input',()=>{if(!replay)return;replayPlaying=false;replayTime=Number(ui['replay-position'].value)*replay.frames.at(-1).time;syncReplay();});
for(const tab of [ui['play-tab'],ui['watch-tab']])tab.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?'play':e.key==='End'?'watch':mode==='play'?'watch':'play';switchMode(next);ui[next==='play'?'play-tab':'watch-tab'].focus();}});

function requestFrame(){if(!animationPending){animationPending=true;requestAnimationFrame(animate);}}
function animate(now){
  animationPending=false;
  const elapsed=lastFrame===null?0:now-lastFrame;lastFrame=now;
  if(mode==='play'&&running){
    if(elapsed>140){pause('Slow frame detected. Play paused instead of advancing unseen time. Press Resume to continue.');}
    else{
      accumulator+=elapsed/1000;
      let count=0;
      try{
        while(accumulator>=DT&&running&&count<4){stepGame();accumulator-=DT;count++;}
        if(accumulator>=DT&&running)pause('Your browser fell behind. Play paused instead of skipping physics. Press Resume to continue.');
      }catch(error){
        simulationError=error;running=false;document.body.classList.remove('is-playing');releaseInput();setStatus('STOPPED');ui.go.disabled=true;
        ui.feedback.textContent='The physics stopped unexpectedly. Reload this page to start a clean session.';
        overlay({kicker:'SIMULATION STOPPED',title:'Let’s reset the court.',copy:ui.feedback.textContent,action:'Reload to restart',disabled:true});console.error('Floppy physics stopped',error);
      }
    }
  }else if(mode==='watch'&&replayPlaying&&replay){
    if(elapsed<250){replayTime=Math.min(replay.frames.at(-1).time,replayTime+elapsed/1000);if(replayTime>=replay.frames.at(-1).time){replayPlaying=false;ui['replay-status'].textContent='First Light cleared: 15 bricks, 1,500 points. This segment ends here; the later campaign was unfinished.';}syncReplay();}
  }
  if(running||replayPlaying)requestFrame();
}

async function loadGame(){
  try{
    const engine=await import('./engine/game.js');game=await engine.createGame({seed:SEED,duration:120});state=game.snapshot();
    if(typeof game.assistedForce!=='function'){
      controlMode='direct';document.querySelector('input[value="direct"]').checked=true;document.querySelector('input[value="assisted"]').disabled=true;
      document.querySelector('input[value="assisted"]').closest('label').title='Balance assistance is unavailable in this browser build.';
    }
    const version=(typeof state.runtime==='string'?state.runtime:state.runtime?.version)||state.profile?.runtime_version||engine.RUNTIME_VERSION||engine.RUNTIME?.version||game.runtime?.version;
    if(version)ui['runtime-version'].textContent=`MuJoCo ${version} WebAssembly`;
    ui.go.disabled=false;ui.reset.disabled=false;reset();
  }catch(error){
    simulationError=error;setStatus('UNAVAILABLE');ui.go.textContent='Physics unavailable';ui.feedback.textContent='This browser could not start the physics. Try a current browser, or watch the retained recording.';
    overlay({kicker:'BROWSER PHYSICS UNAVAILABLE',title:'The recording is ready.',copy:ui.feedback.textContent,action:'Play unavailable',disabled:true});console.error('Floppy initialization failed',error);
  }
}
async function loadReplay(){
  try{
    const response=await fetch('./replay.json');if(!response.ok)throw new Error(`Recording ${response.status}`);replay=await response.json();
    if(!Array.isArray(replay.frames)||!replay.frames.length)throw new Error('Empty recording');
    $('replay-description').textContent=`${replay.controller?.name||'Retained autonomous controller'}. A complete First Light clear, ending with the campaign's fourth-life clear bonus; the later campaign was unfinished.`;
    ui['replay-go'].disabled=false;ui['replay-position'].disabled=false;ui['replay-go'].textContent='Play recording ▶';
    ui['replay-status'].textContent='Ready: complete retained First Light segment, 45.03 seconds. Later campaign unfinished.';
    if(mode==='watch'){ui['court-overlay'].hidden=true;syncReplay();}
  }catch(error){ui['replay-status'].textContent='The recording could not load. Human play is available separately.';ui['replay-go'].textContent='Recording unavailable';console.error('Floppy recording failed',error);}
}

// A small read-only status surface supports release QA without controlling play.
window.floppyStatus=()=>({mode,running,started,finished,controlMode,target,replayLoaded:!!replay,replayPlaying,replayTime,state:state?{time:state.time,phase:state.phase,score:state.score,lives:state.lives,outcome:state.outcome}:null});
syncControlReadout();loadGame();loadReplay();
