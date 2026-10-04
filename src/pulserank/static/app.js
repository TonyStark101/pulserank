const state={user:'maya',data:null};
const $=s=>document.querySelector(s);
const users=['maya','leo','nora','sam'];

function userButtons(){
  $('#users').innerHTML=users.map(u=>`<button class="${u===state.user?'active':''}" data-user="${u}">${u}</button>`).join('');
  document.querySelectorAll('[data-user]').forEach(b=>b.onclick=()=>{state.user=b.dataset.user;userButtons();load()});
}
async function load(){
  const start=performance.now();
  const [data,overview]=await Promise.all([
    fetch(`/api/recommendations?user_id=${state.user}`).then(r=>r.json()),
    fetch('/api/overview').then(r=>r.json())
  ]);
  state.data=data; render(data,overview,performance.now()-start);
}
function render(d,o,latency){
  $('#viewer').textContent=state.user.toUpperCase();
  $('#variant').textContent=d.assignment.variant;
  $('#bucket').textContent=`bucket ${d.assignment.bucket}`;
  $('#serving-mode').textContent=d.serving.mode.replaceAll('_',' ');
  $('#model-version').textContent=d.serving.model_version||'heuristic';
  const learned=d.serving.mode==='learned_challenger';
  $('#retrieval-stage').textContent=learned?'Two-tower embedding recall':'Taste + popularity recall';
  $('#ranking-stage').textContent=learned?'Calibrated learned ranker':'Heuristic baseline';
  $('#metrics').innerHTML=[['CATALOG ITEMS',o.stats.items],['BEHAVIOR EVENTS',o.stats.events.toLocaleString()],['ACTIVE VIEWERS',o.stats.users],['API ROUND TRIP',`${latency.toFixed(1)} ms`]].map(x=>`<div class="metric"><span>${x[0]}</span><strong>${x[1]}</strong></div>`).join('');
  const fs=Object.entries(d.features.genre_affinity).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,6);
  $('#features').innerHTML=fs.length?fs.map(([g,v])=>`<div class="feature"><div><span>${g}</span><span>${v>0?'+':''}${v.toFixed(3)}</span></div><div class="feature-line"><i style="width:${Math.min(100,Math.abs(v)*280)}%;background:${v<0?'#ef5d78':''}"></i></div></div>`).join(''):'<p class="muted">Cold-start mode: ranking by quality and freshness.</p>';
  $('#feed').innerHTML=d.recommendations.slice(0,9).map((r,i)=>`<article class="card"><div class="art" style="--c:${r.item.color}"><span class="rank">#${i+1}</span><span class="score">${(r.score*100).toFixed(1)}</span><h3>${r.item.title}</h3></div><div class="card-body"><div class="tags">${r.item.year} · ${r.item.genres.join(' / ')}</div><div class="reason">${r.reasons[0]}</div><div class="scorebar"><i style="width:${Math.max(4,Math.min(100,r.score*100))}%"></i></div><div class="actions"><button data-item="${r.item.item_id}" data-action="like">♥ Like</button><button data-item="${r.item.item_id}" data-action="complete">✓ Watch</button><button data-item="${r.item.item_id}" data-action="skip">Skip</button></div></div></article>`).join('');
  document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>signal(b.dataset.item,b.dataset.action));
}
async function signal(item,action){
  const event={event_id:crypto.randomUUID(),user_id:state.user,item_id:item,action,timestamp:Date.now()/1000,watch_pct:action==='complete'?1:action==='like'?.8:.04};
  await fetch('/api/events',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({events:[event]})});
  toast(`${action} signal ingested — recomputing ${state.user}'s features`); await load();
}
function toast(t){const el=$('#toast');el.textContent=t;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2300)}
$('#refresh').onclick=load;setInterval(()=>$('#clock').textContent=new Date().toLocaleTimeString(),1000);userButtons();load();
