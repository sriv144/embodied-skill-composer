import ReactECharts from "echarts-for-react";
import { Ban, Check, Cpu, Play, RefreshCw, X } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../api";
import { FidelityBadge } from "../components/WorkbenchControls";
import type { CoppeliaHealth, LabMode, LabPolicy, LabRun } from "../types";

export function ExperimentsView({
  mode,
  runs,
  policies,
  coppelia,
  onRefresh
}: {
  mode: LabMode;
  runs: LabRun[];
  policies: LabPolicy[];
  coppelia: CoppeliaHealth | null;
  onRefresh: () => void;
}) {
  const [algorithm, setAlgorithm] = useState<"mappo" | "ippo">("mappo");
  const [profile, setProfile] = useState<"unit" | "smoke" | "research">("smoke");
  const [seed, setSeed] = useState(7);
  const [confirmed, setConfirmed] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  const launch = async () => {
    setLaunching(true);
    setNotice(null);
    try {
      const response = await api.launchTraining({ algorithm, profile, seed, confirmed, device: "auto" });
      setNotice(`Run ${response.run_id} queued.`);
      setConfirmed(false);
      onRefresh();
    } catch (reason) {
      setNotice(String(reason));
    } finally {
      setLaunching(false);
    }
  };

  const chart = useMemo(() => ({
    animationDuration: 500,
    grid: { left: 44, right: 16, top: 24, bottom: 34 },
    xAxis: { type: "category", data: runs.slice(0, 12).reverse().map((run) => run.id.slice(9, 19)), axisLabel: { color: "#738079", fontSize: 9 } },
    yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%", color: "#738079" }, splitLine: { lineStyle: { color: "#e3e8e5" } } },
    series: [{ type: "line", smooth: true, symbolSize: 7, data: runs.slice(0, 12).reverse().map((run) => Math.round(run.progress * 100)), lineStyle: { color: "#2d8376", width: 2 }, itemStyle: { color: "#f2672e" }, areaStyle: { color: "rgba(45,131,118,.1)" } }]
  }), [runs]);

  return (
    <div className="experiments-layout">
      <section className="experiment-main">
        <header className="operational-toolbar">
          <div><p className="eyebrow">Research runs</p><h2>Training and evaluation registry</h2></div>
          <button className="icon-button light" onClick={onRefresh} title="Refresh runs"><RefreshCw size={16} /></button>
        </header>
        <div className="experiment-body">
          <section className="run-chart-band">
            <div><p className="eyebrow">Run progress</p><h3>Recent local jobs</h3></div>
            <ReactECharts option={chart} style={{ height: 220 }} />
          </section>
          <section className="run-history">
            <div className="table-header"><span>Run</span><span>Kind</span><span>Status</span><span>Progress</span><span /></div>
            {runs.length === 0 && <div className="empty-row">No registered runs</div>}
            {runs.slice(0, 12).map((run) => (
              <div className="run-row" key={run.id}>
                <strong>{run.id}</strong><span>{run.kind}</span><Status value={run.status} /><div className="progress-track"><i style={{ width: `${run.progress * 100}%` }} /></div>
                <button disabled={!['queued', 'running', 'cancel_requested'].includes(run.status)} title="Cancel run" onClick={async () => { await api.cancelRun(run.id); onRefresh(); }}><X size={14} /></button>
              </div>
            ))}
          </section>
          <section className="policy-table">
            <div><p className="eyebrow">Policy registry</p><h3>Exported actors</h3></div>
            {policies.length === 0 ? <p className="muted-line">No MAPPO or IPPO checkpoint has been registered.</p> : policies.map((policy) => (
              <div className="policy-row" key={policy.id}><strong>{policy.controller.toUpperCase()}</strong><span>{policy.id}</span><small>{policy.manifest.transition_count?.toLocaleString() ?? 0} transitions</small><Check size={14} /></div>
            ))}
          </section>
        </div>
      </section>
      <aside className="experiment-launcher">
        <p className="eyebrow">New training run</p><h2>Swarm policy</h2>
        <label className="control-label">Algorithm</label>
        <div className="segmented full"><button className={algorithm === "mappo" ? "selected" : ""} onClick={() => setAlgorithm("mappo")}>MAPPO</button><button className={algorithm === "ippo" ? "selected" : ""} onClick={() => setAlgorithm("ippo")}>IPPO</button></div>
        <label className="control-label">Profile</label>
        <div className="segmented full">{(["unit", "smoke", "research"] as const).map((item) => <button key={item} className={profile === item ? "selected" : ""} onClick={() => setProfile(item)}>{item}</button>)}</div>
        <label className="control-label" htmlFor="training-seed">Seed</label>
        <input id="training-seed" className="number-field" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        <label className="confirm-control"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>Approve compute launch</span></label>
        <button className="primary-button launch-button" disabled={mode === "static" || !confirmed || launching} onClick={launch}><Play size={16} />{launching ? "Launching" : "Start training"}</button>
        {notice && <p className="notice-line">{notice}</p>}
        <div className="fidelity-matrix">
          <h3>Simulator fidelity</h3>
          <div><FidelityBadge level="Event Sim" active /><span>temporal MARL</span></div>
          <div><FidelityBadge level="MuJoCo" active /><span>skill profile</span></div>
          <div><FidelityBadge level="CoppeliaSim" active={Boolean(coppelia?.reachable)} /><span>{coppelia?.reachable ? "wheel control online" : "offline"}</span></div>
        </div>
        <div className="runtime-note">{mode === "local" ? <><Cpu size={15} />Local lab controls enabled</> : <><Ban size={15} />Read-only public trace</>}</div>
      </aside>
    </div>
  );
}

function Status({ value }: { value: LabRun["status"] }) {
  return <span className={`run-status ${value}`}>{value.replaceAll("_", " ")}</span>;
}
