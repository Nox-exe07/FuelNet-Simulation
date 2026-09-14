// charts.js — utility helpers for Chart.js (shared palette, export, theme)
(function(){
  const tierColor = (v)=> v<200 ? "#22c55e" : v<300 ? "#f59e0b" : "#ef4444";
  const tierLabel = (v)=> v<200 ? "Eco" : v<300 ? "Medium" : "High";

  // Apply dark theme defaults to all Chart.js charts
  if(window.Chart){
    try{
      Chart.defaults.color = "#e5e7eb";
      Chart.defaults.borderColor = "rgba(148,163,184,.2)";
    }catch(e){}
  }

  // Utility: build fuel→CO₂ trace locally (fallback) matching predict.py physics
  function localFuelTrace(base, fcs){
    const eng = base.ENGINE_SIZE || base["ENGINE SIZE"] || 2.0;
    const cyl = base.CYLINDERS || 4;
    return fcs.map(fc=>{
      const physics = fc*23.7;
      // residual calibrated to dataset: negative at moderate fc, positive for large engines
      const resid = eng*1.2 + cyl*0.8 -4 + (fc-10)*0.2;
      return physics + resid;
    });
  }

  // Export canvas as PNG download
  function downloadCanvas(canvas, filename){
    if(!canvas) return;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href=url; a.download=filename; a.click();
  }

  window.ChartsUtil = { tierColor, tierLabel, localFuelTrace, downloadCanvas };
  console.log("charts.js loaded — utils ready");
})();
