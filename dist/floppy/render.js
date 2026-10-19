const PALETTE = ['#70e8df', '#b1ed8d', '#d9ff43'];
const INK = '#0c130e';

export function createRenderer(canvas) {
  const ctx = canvas.getContext('2d', {alpha: false});
  let previousBall = null;
  const trail = [];
  let previousTime = -1;

  function render(state, {target = 0, showTarget = false, replay = false, clearTrail = false} = {}) {
    if (!ctx || !state) return;
    const W = canvas.width, H = canvas.height;
    const p = state.profile || {};
    const wall = p.wall_x || 2.41875;
    const top = (p.goal_z || 4.3) + .26;
    const bottom = -(p.goal_z || 4.3) - .26;
    const scale = Math.min((W - 64) / (wall * 2), (H - 48) / (top - bottom));
    const X = x => W / 2 + x * scale;
    const Z = z => H / 2 - z * scale;
    ctx.fillStyle = INK;
    ctx.fillRect(0, 0, W, H);
    if (clearTrail || state.time < previousTime || state.phase === 'warmup' || state.phase === 'new_life') {
      trail.length = 0;
      previousBall = null;
    }
    previousTime = state.time;

    // The grid is presentation only; contact and scoring come from the engine.
    ctx.strokeStyle = '#18251a';
    ctx.lineWidth = 1;
    for (let x = -2; x <= 2; x += .5) {
      ctx.beginPath(); ctx.moveTo(X(x), Z(top)); ctx.lineTo(X(x), Z(bottom)); ctx.stroke();
    }
    for (let z = -4; z <= 4; z += .5) {
      ctx.beginPath(); ctx.moveTo(X(-wall), Z(z)); ctx.lineTo(X(wall), Z(z)); ctx.stroke();
    }
    ctx.strokeStyle = '#607b4b';
    ctx.lineWidth = 3;
    ctx.beginPath();ctx.moveTo(X(-wall),Z(-4.3));ctx.lineTo(X(-wall),Z(4.3));ctx.lineTo(X(wall),Z(4.3));ctx.lineTo(X(wall),Z(-4.3));ctx.stroke();
    ctx.strokeStyle = '#63764a';ctx.lineWidth = 1;
    ctx.setLineDash([5,7]);ctx.beginPath();ctx.moveTo(X(-2),Z(-4));ctx.lineTo(X(2),Z(-4));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#759561';ctx.font='10px ui-monospace, monospace';ctx.textAlign='left';
    ctx.fillText('−2 m',X(-2),Z(-4)-13);ctx.textAlign='right';ctx.fillText('+2 m',X(2),Z(-4)-13);

    const bricks = state.board?.bricks || [];
    for (const b of bricks) {
      const hp = state.health?.[b.id] ?? b.hp ?? 1;
      const pending = state.pending?.includes(b.id);
      if (hp <= 0 && !pending) continue;
      const bx = X(b.x - b.width / 2), by = Z(b.z + b.height / 2);
      const bw = b.width * scale, bh = b.height * scale;
      ctx.fillStyle = pending ? '#ff704a' : PALETTE[b.row % PALETTE.length];
      ctx.globalAlpha = pending ? .4 : 1;
      ctx.fillRect(bx, by, bw, bh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#0c180c66';ctx.fillRect(bx+4,by+bh-4,bw-8,2);
      ctx.strokeStyle = pending ? '#ffbd72' : '#dfffc533';ctx.lineWidth=1;ctx.strokeRect(bx+.5,by+.5,bw-1,bh-1);
    }

    const own = state.own || {};
    const cart = own.cart_world || [own.x || 0, -4];
    const theta = own.theta || 0;
    const paddle = own.paddle_world || [cart[0] + Math.sin(theta),cart[1] + Math.cos(theta)];
    if (showTarget) {
      ctx.strokeStyle='#d9ff4366';ctx.lineWidth=2;ctx.setLineDash([4,5]);
      ctx.beginPath();ctx.moveTo(X(target),Z(-4.18));ctx.lineTo(X(target),Z(-2.65));ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='#d9ff43';ctx.beginPath();ctx.moveTo(X(target),Z(-4.27));ctx.lineTo(X(target)-6,Z(-4.36));ctx.lineTo(X(target)+6,Z(-4.36));ctx.closePath();ctx.fill();
    }

    ctx.strokeStyle='#a8b791';ctx.lineWidth=4;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(X(cart[0]),Z(cart[1]));ctx.lineTo(X(paddle[0]),Z(paddle[1]));ctx.stroke();ctx.lineCap='butt';
    ctx.fillStyle='#445b35';ctx.fillRect(X(cart[0]-.12),Z(cart[1]),.24*scale,.14*scale);
    ctx.fillStyle='#d9ff43';ctx.fillRect(X(cart[0]-.12),Z(cart[1]),.24*scale,3);
    ctx.fillStyle='#0c130e';ctx.strokeStyle='#a5bd7b';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(X(cart[0]),Z(cart[1]),6,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.save();ctx.translate(X(paddle[0]),Z(paddle[1]));ctx.rotate(theta);
    const pw = (p.paddle_width || .64) * scale;
    const ph = Math.max((p.paddle_thickness || .03) * scale, 5);
    ctx.fillStyle='#d9ff43';ctx.fillRect(-pw/2,-ph/2,pw,ph);ctx.fillStyle='#efffc6';ctx.fillRect(-pw/2,-ph/2,pw,1.5);ctx.restore();

    const ball = state.ball_world || [0, -2.85];
    const visible = state.phase === 'rally' || state.phase === 'finished' || state.phase === 'level_clear';
    if (visible) {
      if (!previousBall || Math.hypot(ball[0]-previousBall[0],ball[1]-previousBall[1]) > .025) {
        trail.push([...ball]);if(trail.length>10)trail.shift();previousBall=[...ball];
      }
      for(let i=0;i<trail.length;i++){
        ctx.globalAlpha = .045 + (i / trail.length) * .17;
        ctx.fillStyle='#70e8df';ctx.beginPath();ctx.arc(X(trail[i][0]),Z(trail[i][1]),(p.ball_radius||.06)*scale*.65,0,Math.PI*2);ctx.fill();
      }
      ctx.globalAlpha=1;ctx.fillStyle='#f2f1e8';ctx.beginPath();ctx.arc(X(ball[0]),Z(ball[1]),Math.max(5,(p.ball_radius||.06)*scale),0,Math.PI*2);ctx.fill();
      ctx.strokeStyle='#70e8df77';ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(ball[0]),Z(ball[1]),Math.max(7,(p.ball_radius||.06)*scale+4),0,Math.PI*2);ctx.stroke();
    }
    if (state.phase === 'warmup' || state.phase === 'new_life') {
      ctx.fillStyle='#829770';ctx.font='10px ui-monospace, monospace';ctx.textAlign='center';
      ctx.fillText(state.phase==='new_life'?'A NEW LIFE / BRICKS RETAINED':'BALANCE / SERVE PREPARING',W/2,Z(-1.4));
    }
    ctx.fillStyle='#617954';ctx.font='9px ui-monospace, monospace';ctx.textAlign='center';
    ctx.fillText(replay ? 'RETAINED RECORDING / NOT LIVE' : 'HUMAN PLAY / BROWSER ALPHA',W/2,Z(.1));
  }
  return {render};
}
