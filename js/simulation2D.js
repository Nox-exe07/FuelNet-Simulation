// simulation2D.js — Canvas road, car, particles, live chart
class Sim2D {
  constructor(){
    this.road = document.getElementById("road");
    this.particlesCanvas = document.getElementById("particles");
    this.rctx = this.road.getContext("2d");
    this.pctx = this.particlesCanvas.getContext("2d");
    this.liveCanvas = document.getElementById("chartLive");
    this.speedEl = document.getElementById("speed");
    this.speedVal = document.getElementById("speedVal");
    this.distEl = document.getElementById("distVal");
    this.fuelUsedEl = document.getElementById("fuelUsed");
    this.co2TotalEl = document.getElementById("co2Total");
    this.timeEl = document.getElementById("timeVal");
    this.btnDrive = document.getElementById("btnDrive");
    this.btnPause = document.getElementById("btnPause");
    this.btnClear = document.getElementById("btnClear");

    this.width = 900; this.height=260;
    this.car = {x: 140, y: 150, w: 64, h: 30, wheel:0};
    this.particles=[];
    this.distance=0; // km
    this.fuelUsed=0; // L
    this.co2Total=0; // kg
    this.time=0;
    this.running=false;
    this.lastTs=0;
    this.speed=60; // km/h
    this.co2Current=250; // g/km
    this.fcCurrent=10.5;
    this.roadOffset=0;

    this.liveChart=null;
    this.tick=0;
    this.maxPoints=60;

    this.initChart();
    this.bind();
    this.resize();
    window.addEventListener("resize",()=>this.resize());
    requestAnimationFrame((t)=>this.loop(t));
  }
  initChart(){
    const ctx=this.liveCanvas.getContext("2d");
    this.liveChart=new Chart(ctx,{
      type:"line",
      data:{
        labels: Array.from({length:this.maxPoints},(_,i)=>i),
        datasets:[
          {label:"CO₂ g/km", data:Array(this.maxPoints).fill(null), borderColor:"#ef4444", backgroundColor:"rgba(239,68,68,.15)", tension:.35, fill:true, pointRadius:0, borderWidth:2},
          {label:"Fuel L/100km", data:Array(this.maxPoints).fill(null), borderColor:"#38bdf8", backgroundColor:"rgba(56,189,248,.12)", tension:.35, fill:true, pointRadius:0, borderWidth:2, yAxisID:"y1"}
        ]
      },
      options:{
        responsive:true,
        animation:false,
        plugins:{legend:{labels:{color:"#e5e7eb"}}},
        scales:{
          x:{ticks:{color:"#94a3b8", maxTicksLimit:8}, grid:{color:"rgba(148,163,184,.15)"}},
          y:{title:{display:true,text:"CO₂ g/km",color:"#94a3b8"}, ticks:{color:"#94a3b8"}, grid:{color:"rgba(148,163,184,.15)"}},
          y1:{position:"right", title:{display:true,text:"Fuel",color:"#94a3b8"}, ticks:{color:"#94a3b8"}, grid:{drawOnChartArea:false}}
        }
      }
    });
  }
  bind(){
    this.speedEl.addEventListener("input",()=>{
      this.speed=parseInt(this.speedEl.value);
      this.speedVal.textContent=this.speed;
    });
    this.btnDrive.addEventListener("click",()=>{ this.running=true; this.lastTs=performance.now(); });
    this.btnPause.addEventListener("click",()=>{ this.running=false; });
    this.btnClear.addEventListener("click",()=>{ this.reset(); });
  }
  reset(){
    this.running=false;
    this.distance=0; this.fuelUsed=0; this.co2Total=0; this.time=0;
    this.particles=[]; this.tick=0;
    this.distEl.textContent="0.00"; this.fuelUsedEl.textContent="0.00"; this.co2TotalEl.textContent="0.00"; this.timeEl.textContent="0";
    if(this.liveChart){
      this.liveChart.data.datasets.forEach(ds=>ds.data=Array(this.maxPoints).fill(null));
      this.liveChart.update();
    }
    this.car.x=140;
  }
  updatePrediction(res){
    this.co2Current=res.co2;
    this.fcCurrent=res.fuel_consumption;
  }
  resize(){
    // keep fixed internal res but scale via CSS responsive; canvas already 900x260
    // ensure overlay matches
    this.particlesCanvas.width=this.road.width;
    this.particlesCanvas.height=this.road.height;
    this.drawRoad(0);
  }
  spawnParticles(){
    const intensity = Math.min(1, this.co2Current/400);
    const count = this.running ? Math.ceil(intensity*3)+ (this.speed>80?2:0) : 0;
    for(let i=0;i<count;i++){
      const tier = this.co2Current<200?"low":this.co2Current<300?"mid":"high";
      const color = tier=="low"?"#22c55e":tier=="mid"?"#f59e0b":"#ef4444";
      this.particles.push({
        x: this.car.x - 6,
        y: this.car.y + 16 + (Math.random()*8-4),
        vx: - (1 + Math.random()*3 + this.speed*0.02),
        vy: (Math.random()*2-1)*0.6,
        r: 2 + Math.random()*3 + intensity*3,
        alpha: 0.85,
        color,
        life: 0,
        maxLife: 60 + Math.random()*40
      });
    }
  }
  updateParticles(){
    for(let p of this.particles){
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.02; // gravity slight
      p.vx *= 0.998;
      p.r *= 0.998;
      p.life++;
      p.alpha = Math.max(0, 0.85 * (1 - p.life/p.maxLife));
    }
    this.particles = this.particles.filter(p=> p.alpha>0.05 && p.x>-20 && p.life<p.maxLife);
  }
  drawRoad(dt){
    const ctx=this.rctx;
    ctx.clearRect(0,0,this.width,this.height);
    // sky
    const grad=ctx.createLinearGradient(0,0,0,this.height);
    grad.addColorStop(0,"#0b1222");
    grad.addColorStop(1,"#1e293b");
    ctx.fillStyle=grad;
    ctx.fillRect(0,0,this.width,this.height);
    // hills
    ctx.fillStyle="#0f172a";
    ctx.beginPath();
    ctx.moveTo(0,120);
    for(let x=0;x<=this.width;x+=40){
      ctx.lineTo(x, 110 + Math.sin(x*0.01 + this.roadOffset*0.005)*12 + Math.cos(x*0.02)*6);
    }
    ctx.lineTo(this.width,0); ctx.lineTo(0,0); ctx.closePath(); ctx.fill();
    // road
    const roadY= 170;
    const roadH= 70;
    ctx.fillStyle="#1f2937";
    ctx.fillRect(0,roadY,this.width,roadH);
    // road border
    ctx.fillStyle="#0f172a";
    ctx.fillRect(0,roadY-3,this.width,3);
    ctx.fillRect(0,roadY+roadH,this.width,8);
    // lane markers moving
    ctx.fillStyle="#e5e7eb";
    this.roadOffset += this.running ? this.speed*0.22 : 0.2;
    // wrap offset
    if(this.roadOffset>60) this.roadOffset-=60;
    for(let x=-60; x<this.width+60; x+=60){
      const off = (x - (this.roadOffset%60));
      ctx.fillRect(off, roadY+roadH/2 -1.5, 28, 3);
    }
    // distance markers
    ctx.fillStyle="#94a3b8"; ctx.font="11px Inter, Arial"; ctx.textAlign="center";
    for(let x=-200; x<this.width+200; x+=200){
      const worldX = this.distance*1000 + (x - this.car.x);
      const marker = Math.floor(worldX/100)*100;
      if(marker<0) continue;
      const sx = x - (this.roadOffset%200)*0.3;
      if(sx< -10 || sx> this.width+10) continue;
      // post
      ctx.fillStyle="#cbd5e1";
      ctx.fillRect(sx, roadY-28, 2, 28);
      ctx.fillStyle="#1f2937";
      ctx.fillRect(sx-10, roadY-38, 22,14);
      ctx.fillStyle="#e5e7eb";
      ctx.fillText(marker+"m", sx+1, roadY-28);
    }
    // sidewalk
    ctx.fillStyle="#334155";
    ctx.fillRect(0, roadY+roadH, this.width, 6);
    // clouds
    ctx.fillStyle="rgba(255,255,255,.08)";
    for(let i=0;i<3;i++){
      const cx = (i*300 + (this.roadOffset*0.05))%this.width;
      ctx.beginPath(); ctx.ellipse(cx, 40, 48, 18, 0,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(cx+18, 36, 30,12,0,0,Math.PI*2); ctx.fill();
    }
    // car shadow
    ctx.fillStyle="rgba(0,0,0,.35)";
    ctx.beginPath(); ctx.ellipse(this.car.x+12, roadY+roadH-4, 36,8,0,0,Math.PI*2); ctx.fill();
    // car body
    const cx=this.car.x, cy= roadY+18;
    // body
    ctx.fillStyle="#e5e7eb";
    roundRect(ctx,cx-28,cy-12,56,22,6,true);
    // cabin
    ctx.fillStyle="#38bdf8";
    roundRect(ctx,cx-6,cy-18,22,14,3,true);
    ctx.fillStyle="rgba(255,255,255,.4)";
    ctx.fillRect(cx-4,cy-16,18,4);
    // wheels
    this.car.wheel += this.running? this.speed*0.18 : 0.1;
    for(let wx of [cx-18,cx+18]){
      ctx.fillStyle="#0f172a"; ctx.beginPath(); ctx.arc(wx,cy+12,9,0,Math.PI*2); ctx.fill();
      ctx.fillStyle="#64748b"; ctx.beginPath(); ctx.arc(wx,cy+12,5,0,Math.PI*2); ctx.fill();
      // spoke
      ctx.strokeStyle="#e5e7eb"; ctx.lineWidth=1.2;
      ctx.beginPath();
      const a=this.car.wheel*0.12 + (wx==cx-18?0:Math.PI/3);
      ctx.moveTo(wx+Math.cos(a)*5, cy+12+Math.sin(a)*5);
      ctx.lineTo(wx+Math.cos(a+Math.PI)*5, cy+12+Math.sin(a+Math.PI)*5);
      ctx.stroke();
      // second spoke
      ctx.beginPath();
      ctx.moveTo(wx+Math.cos(a+Math.PI/2)*5, cy+12+Math.sin(a+Math.PI/2)*5);
      ctx.lineTo(wx+Math.cos(a+Math.PI/2+Math.PI)*5, cy+12+Math.sin(a+Math.PI/2+Math.PI)*5);
      ctx.stroke();
    }
    // headlight
    ctx.fillStyle="#fef9c3";
    ctx.beginPath(); ctx.arc(cx+28,cy-2,3,0,Math.PI*2); ctx.fill();
    // taillight
    ctx.fillStyle="#ef4444";
    ctx.beginPath(); ctx.arc(cx-28,cy-3,2.2,0,Math.PI*2); ctx.fill();
    // exhaust pipe
    ctx.fillStyle="#475569";
    ctx.fillRect(cx-29,cy+2,4,4);
    // speedo small
    ctx.fillStyle="rgba(0,0,0,.5)";
    roundRect(ctx, this.width-78, 12, 66,36,8,true);
    ctx.fillStyle="#e5e7eb"; ctx.font="700 14px Inter, Arial"; ctx.textAlign="center";
    ctx.fillText(this.speed+" km/h", this.width-45, 30);
    ctx.fillStyle="#94a3b8"; ctx.font="11px Inter"; ctx.fillText(this.co2Current.toFixed(0)+" g/km", this.width-45, 42);

    // co2 tier badge on car
    const tier=this.co2Current<200?"ECO":this.co2Current<300?"MID":"HIGH";
    const tierColor=tier=="ECO"?"#22c55e":tier=="MID"?"#f59e0b":"#ef4444";
    ctx.fillStyle=tierColor; ctx.font="700 9px Inter"; ctx.textAlign="center";
    ctx.fillText(tier, cx+2, cy-24);
  }
  drawParticles(){
    const ctx=this.pctx;
    ctx.clearRect(0,0,this.width,this.height);
    for(let p of this.particles){
      ctx.globalAlpha=p.alpha;
      // radial gradient puff
      const g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r);
      // color with alpha
      const col=p.color;
      // convert hex to rgba approx
      let r=239,g2=68,b=68;
      if(col=="#22c55e"){r=34;g2=197;b=94}
      else if(col=="#f59e0b"){r=245;g2=158;b=11}
      g.addColorStop(0, `rgba(${r},${g2},${b},${p.alpha})`);
      g.addColorStop(0.6, `rgba(${r},${g2},${b},${p.alpha*0.35})`);
      g.addColorStop(1, `rgba(${r},${g2},${b},0)`);
      ctx.fillStyle=g;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  updateLiveChart(){
    this.tick++;
    if(this.tick%6!==0) return; // update 10Hz
    const labels=this.liveChart.data.labels;
    labels.push(labels.length);
    labels.shift();
    const co2Arr=this.liveChart.data.datasets[0].data;
    const fcArr=this.liveChart.data.datasets[1].data;
    co2Arr.push(this.running? this.co2Current : null);
    fcArr.push(this.running? this.fcCurrent : null);
    co2Arr.shift(); fcArr.shift();
    this.liveChart.update("none");
  }
  loop(ts){
    if(!this.lastTs) this.lastTs=ts;
    const dt = Math.min(0.05, (ts-this.lastTs)/1000);
    this.lastTs=ts;
    if(this.running){
      const dkm = this.speed * dt / 3600; // km, speed km/h * dt seconds /3600
      this.distance += dkm;
      this.time += dt;
      const speedFactor = 1 + Math.max(0,(this.speed-60))*0.004 + Math.pow(this.speed/120,2)*0.08;
      const fcEff = this.fcCurrent * speedFactor;
      const co2Eff = this.co2Current * speedFactor;
      const fuelDelta = fcEff * dkm / 100; // L, because fcEff is L/100km
      const co2DeltaKg = co2Eff * dkm / 1000; // kg, g/km * km = g -> /1000
      this.fuelUsed += fuelDelta;
      this.co2Total += co2DeltaKg;
      this.distEl.textContent=this.distance.toFixed(3);
      this.fuelUsedEl.textContent=this.fuelUsed.toFixed(3);
      this.co2TotalEl.textContent=this.co2Total.toFixed(3);
      this.timeEl.textContent=Math.floor(this.time).toString();
    }
    this.spawnParticles();
    this.updateParticles();
    this.drawRoad(dt);
    this.drawParticles();
    this.updateLiveChart();
    requestAnimationFrame((t)=>this.loop(t));
  }
}
function roundRect(ctx,x,y,w,h,r,fill){
  ctx.beginPath();
  ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);
  ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);
  ctx.arcTo(x,y,x+w,y,r);
  ctx.closePath();
  if(fill) ctx.fill();
}

document.addEventListener("DOMContentLoaded",()=>{
  window.sim = new Sim2D();
});
