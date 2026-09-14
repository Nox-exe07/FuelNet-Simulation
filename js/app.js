// app.js — static what-if + API bindings
const $ = id=>document.getElementById(id);
const apiBase = ""; // same origin localhost:8000

let currentCo2 = null, currentFc = null;
let staticChart = null, sensChart=null;

async function fetchHealth(){
  try{
    const r = await fetch(apiBase+"/health");
    const j = await r.json();
    $("apiStatus").textContent = j.model_loaded ? "API: ready (FuelNet)" : "API: fallback";
    $("apiStatus").style.borderColor = j.model_loaded ? "#22c55e" : "#f59e0b";
  }catch(e){ $("apiStatus").textContent="API: offline (fallback)"; $("apiStatus").style.borderColor="#ef4444"; }
}
async function fetchStats(){
  try{
    const r = await fetch(apiBase+"/stats");
    const j = await r.json();
    $("datasetInfo").textContent = `Rows: ${j.rows}\nClasses: ${j.vehicle_classes?.join(", ")}\nFuel: ${j.fuel_types?.join(", ")}\nFC ${j.fuel_consumption?.min?.toFixed(1)}–${j.fuel_consumption?.max?.toFixed(1)} mean ${j.fuel_consumption?.mean?.toFixed(1)}\nCO2 ${j.co2?.min?.toFixed(0)}–${j.co2?.max?.toFixed(0)} mean ${j.co2?.mean?.toFixed(0)}`;
    if(j.per_class_avg){
      const list = $("classList"); list.innerHTML="";
      Object.entries(j.per_class_avg).forEach(([cls,v])=>{
        const div=document.createElement("div"); div.className="class-item";
        div.innerHTML=`<span>${cls}</span><span>${v.FUEL_CONSUMPTION?.toFixed(1)} L · ${v.CO2?.toFixed(0)}g</span>`;
        list.appendChild(div);
      });
    }
  }catch(e){ $("datasetInfo").textContent="stats unavailable offline"; }
}
async function fetchMetrics(){
  try{
    const r=await fetch(apiBase+"/metrics"); const j=await r.json();
    const row=$("metricsRow"); row.innerHTML="";
    const defs=[["MSE", j.mse?.toFixed(1)],["RMSE", j.rmse?.toFixed(2)],["MAE", j.mae?.toFixed(2)],["R²", (j.r2??0).toFixed(4)],["MAPE", (j.mape??0).toFixed(2)+"%"]];
    defs.forEach(([k,v])=>{
      const d=document.createElement("div"); d.className="metric"; d.innerHTML=`<span>${k}</span><br><b>${v??"—"}</b>`;
      row.appendChild(d);
    });
  }catch(e){}
}

function collectInput(){
  return {
    Year: parseInt($("year").value),
    MAKE: $("make").value,
    MODEL: $("model").value,
    VEHICLE_CLASS: $("vclass").value,
    "VEHICLE CLASS": $("vclass").value,
    ENGINE_SIZE: parseFloat($("engine").value),
    "ENGINE SIZE": parseFloat($("engine").value),
    CYLINDERS: parseInt($("cyl").value),
    TRANSMISSION: $("trans").value,
    FUEL: $("fuel").value,
    FUEL_CONSUMPTION: parseFloat($("fc").value),
    "FUEL CONSUMPTION": parseFloat($("fc").value),
    FC: parseFloat($("fc").value)
  };
}
function co2Tier(v){
  if(v<200) return "low";
  if(v<300) return "mid";
  return "high";
}
function showStatus(msg, type){
  const el=$("predictStatus");
  if(!el) return;
  el.textContent=msg;
  el.className="muted small "+(type||"");
  if(type==="ok"){
    setTimeout(()=>{ if(el.textContent===msg) el.textContent=""; }, 4000);
  }
}
function pulseGauges(){
  document.querySelectorAll(".gauge").forEach(g=>{
    g.classList.remove("pulse");
    void g.offsetWidth;
    g.classList.add("pulse");
    setTimeout(()=> g.classList.remove("pulse"), 800);
  });
}
function updateGauges(res){
  currentCo2 = res.co2; currentFc=res.fuel_consumption;
  $("co2Val").textContent = res.co2.toFixed(1);
  $("fcGauge").textContent = res.fuel_consumption.toFixed(1);
  $("physicsVal").textContent = res.physics_base.toFixed(1);
  $("residVal").textContent = res.co2_residual.toFixed(1);
  const pct = Math.min(100, Math.max(0, (res.co2-100)/400*100));
  $("co2Bar").style.width = pct+"%";
  const tier = co2Tier(res.co2);
  const colors={low:"#22c55e",mid:"#f59e0b",high:"#ef4444"};
  $("co2Bar").style.background = colors[tier];
  pulseGauges();
  if(window.sim) window.sim.updatePrediction(res);
  if(typeof window._refreshStaticCurrent === "function") window._refreshStaticCurrent();
}

// API predict with fallback — click feedback ensures output appears
async function doPredict(){
  const btn=$("btnPredict");
  const inp = collectInput();
  if(btn){ btn.classList.add("loading"); btn.textContent="⏳ Predicting…"; }
  showStatus("Predicting…", "");
  try{
    const r=await fetch(apiBase+"/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(inp)});
    if(!r.ok) throw new Error("bad "+r.status);
    const j=await r.json();
    updateGauges(j);
    const t=new Date().toLocaleTimeString();
    showStatus(`✓ Predicted via ${j.model} at ${t} — CO₂ ${j.co2.toFixed(1)} g/km (click again to re-predict)`, "ok");
  }catch(e){
    // fallback local physics + empirical residual
    const fc=inp.FUEL_CONSUMPTION;
    const physics=fc*23.7;
    const resid = inp.ENGINE_SIZE*1.2 + inp.CYLINDERS*0.8 -4;
    const co2=physics+resid;
    updateGauges({co2, fuel_consumption:fc, physics_base:physics, co2_residual:resid, model:"fallback-local"});
    showStatus(`⚠ API offline — fallback: CO₂ ${co2.toFixed(1)} g/km (physics ${physics.toFixed(1)} + resid ${resid.toFixed(1)})`, "err");
  } finally {
    if(btn){ btn.classList.remove("loading"); btn.textContent="🔮 Predict (Static)"; }
  }
  if(staticChart) addStaticPoint();
  try{ await updateSensitivity(); }catch(e){}
}

function addStaticPoint(){
  if(!staticChart) return;
  // we keep last 1 point highlight
  const tier = co2Tier(currentCo2);
  const colors={low:"#22c55e",mid:"#f59e0b",high:"#ef4444"};
  // find dataset index?
}

// sensitivity sweep — tries real API batch, falls back to physics+residual
async function updateSensitivity(){
  const base = collectInput();
  const xs=[], ys=[];
  const engineVals = [];
  for(let e=1.0; e<=6.0; e+=0.5){ engineVals.push(e); xs.push(e.toFixed(1)); }
  // try API batch for true model sensitivity
  try{
    const batch = engineVals.map(e=> ({...base, ENGINE_SIZE:e, "ENGINE SIZE":e, CYLINDERS: base.CYLINDERS, FUEL_CONSUMPTION: base.FUEL_CONSUMPTION, "FUEL CONSUMPTION": base.FUEL_CONSUMPTION, FC: base.FUEL_CONSUMPTION, VEHICLE_CLASS: base.VEHICLE_CLASS, TRANSMISSION: base.TRANSMISSION, FUEL: base.FUEL, MAKE: base.MAKE, MODEL: base.MODEL, Year: base.Year }));
    const r = await fetch(apiBase+"/predict_batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(batch)});
    if(r.ok){
      const arr = await r.json();
      arr.forEach(j=> ys.push(j.co2));
    } else throw new Error("batch failed");
  }catch(e){
    // fallback: physics + calibrated residual (matches predict.py fallback)
    engineVals.forEach(ev=>{
      const physics=base.FUEL_CONSUMPTION*23.7;
      const resid = ev*1.2 + base.CYLINDERS*0.8 - 4 + (ev-2.0)*4.5;
      ys.push(physics+resid);
    });
  }
  if(sensChart){
    sensChart.data.labels=xs;
    sensChart.data.datasets[0].data=ys;
    sensChart.update();
  }
}

// static chart with sample data — robust to Chart.js offline
async function initStaticChart(){
  const base=collectInput();
  const fcs=[], co2s=[];
  for(let fc=5; fc<=25; fc+=0.5){
    const physics=fc*23.7;
    const resid= base.ENGINE_SIZE*2 + base.CYLINDERS*0.5 -2;
    fcs.push(fc);
    co2s.push(physics+resid);
  }
  let fuelSweepOk = false;
  try{
    const sweepBatch = fcs.map(fc=> ({...base, FUEL_CONSUMPTION:fc, "FUEL CONSUMPTION":fc, FC:fc }));
    const rr = await fetch(apiBase+"/predict_batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(sweepBatch)});
    if(rr.ok){
      const arr = await rr.json();
      for(let i=0;i<arr.length;i++) co2s[i]=arr[i].co2;
      fuelSweepOk = true;
    }
  }catch(e){ /* fallback keeps physics estimate */ }

  // Wrap Chart creation so failure does not break Predict button
  try{
    if(typeof Chart === "undefined") throw new Error("Chart.js not loaded");
    const ctx=$("chartStatic").getContext("2d");
    staticChart=new Chart(ctx,{
      type:"scatter",
      data:{
        datasets:[{
          label: fuelSweepOk ? "Fuel→CO2 (FuelNet)" : "Fuel→CO2 (physics est.)",
          data: fcs.map((fc,i)=>({x:fc,y:co2s[i]})),
          backgroundColor: co2s.map(v=> co2Tier(v)=="low"?"#22c55e":co2Tier(v)=="mid"?"#f59e0b":"#ef4444"),
          pointRadius:4,
          showLine:true,
          borderColor:"#38bdf8",
          borderWidth:1.2,
          fill:false
        },
        {
          label:"Current",
          data: [{x: base.FUEL_CONSUMPTION, y: (currentCo2|| base.FUEL_CONSUMPTION*23.7)}],
          backgroundColor: "#fff",
          borderColor:"#000",
          pointRadius:7,
          pointStyle:"rectRot"
        }]
      },
      options:{
        responsive:true,
        plugins:{legend:{labels:{color:"#e5e7eb"}}},
        scales:{
          x:{title:{display:true,text:"Fuel Consumption L/100km",color:"#94a3b8"}, grid:{color:"rgba(148,163,184,.2)"}, ticks:{color:"#94a3b8"}},
          y:{title:{display:true,text:"CO₂ g/km",color:"#94a3b8"}, grid:{color:"rgba(148,163,184,.2)"}, ticks:{color:"#94a3b8"}}
        }
      }
    });
    const sctx=$("chartSensitivity").getContext("2d");
    sensChart=new Chart(sctx,{
      type:"line",
      data:{labels:[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], datasets:[{label:"CO₂ vs Engine Size (FC fixed)", data:[0,0,0,0,0,0,0,0,0,0,0], borderColor:"#22c55e", backgroundColor:"rgba(34,197,94,.2)", tension:.3, fill:true, pointRadius:4}]},
      options:{responsive:true, plugins:{legend:{labels:{color:"#e5e7eb"}}}, scales:{x:{ticks:{color:"#94a3b8"}, grid:{color:"rgba(148,163,184,.2)"}}, y:{ticks:{color:"#94a3b8"}, grid:{color:"rgba(148,163,184,.2)"}}}}
    });
  }catch(e){
    console.warn("Chart init failed (offline CDN?) — gauges still work:", e);
    showStatus("Charts offline (CDN blocked) — Predict still works via gauges", "err");
  }
  // Ensure refresh hook exists even if chart failed
  window._refreshStaticCurrent = ()=>{
    try{
      if(staticChart && currentCo2!=null){
        staticChart.data.datasets[1].data=[{x: parseFloat($("fc").value), y: currentCo2}];
        staticChart.update("none");
      }
    }catch(e){}
  };
  try{ await updateSensitivity(); }catch(e){}
}

// Tabs
function initTabs(){
  document.querySelectorAll(".tab").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
      document.querySelectorAll(".tabpane").forEach(p=>p.classList.remove("active"));
      btn.classList.add("active");
      $("tab-"+btn.dataset.tab).classList.add("active");
      if(btn.dataset.tab=="realtime" && window.sim){ window.sim.resize(); }
    });
  });
}

// Inputs
function initInputs(){
  ["year","engine","cyl","fc"].forEach(id=>{
    $(id).addEventListener("input",()=>{
      $("yearVal").textContent=$("year").value;
      $("engVal").textContent=parseFloat($("engine").value).toFixed(1);
      $("cylVal").textContent=$("cyl").value;
      $("fcVal").textContent=parseFloat($("fc").value).toFixed(1);
      // live update static immediately
      if(document.querySelector('.tab[data-tab="static"].active')) doPredict();
      else {
        // just update gauge via local compute for responsiveness, real predict on drive
        const fc=parseFloat($("fc").value);
        $("fcGauge").textContent=fc.toFixed(1);
        // schedule api
        clearTimeout(window._predDeb);
        window._predDeb=setTimeout(doPredict,300);
      }
    });
  });
  ["make","model","vclass","trans","fuel"].forEach(id=>{
    $(id).addEventListener("change", doPredict);
  });
  $("btnPredict").addEventListener("click", doPredict);
  $("btnReset").addEventListener("click",()=>{
    $("engine").value=2.0; $("cyl").value=4; $("fc").value=10.5; $("vclass").value="COMPACT";
    $("year").value=2000; $("trans").value="A4"; $("fuel").value="X";
    $("yearVal").textContent="2000"; $("engVal").textContent="2.0"; $("cylVal").textContent="4"; $("fcVal").textContent="10.5";
    doPredict();
  });
}

window.addEventListener("DOMContentLoaded",()=>{
  fetchHealth(); fetchStats(); fetchMetrics();
  initTabs(); initInputs(); initStaticChart();
  // initial predict
  setTimeout(doPredict,400);
});
