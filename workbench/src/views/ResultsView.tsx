import ReactECharts from "echarts-for-react";
import { FileDown, Gauge, Route, Sparkles } from "lucide-react";
import { api } from "../api";
import { downloadText } from "../components/WorkbenchControls";
import type { LabPolicy, Project } from "../types";

export function ResultsView({ project, policies }: { project: Project; policies: LabPolicy[] }) {
  const names = ["sequential", "greedy", "optimized"];
  const option = {
    animationDuration: 700,
    grid: { left: 42, right: 20, top: 30, bottom: 34 },
    xAxis: { type: "category", data: ["Sequential", "Greedy", "CP-SAT"], axisLine: { lineStyle: { color: "#9ba6a1" } }, axisLabel: { color: "#53605b" } },
    yAxis: { type: "value", name: "seconds", nameTextStyle: { color: "#718078" }, axisLabel: { color: "#718078" }, splitLine: { lineStyle: { color: "#e3e8e5" } } },
    series: [{ type: "bar", data: names.map((name) => ({ value: project.controllers[name].makespan_s, itemStyle: { color: name === "optimized" ? "#f05b2a" : name === "greedy" ? "#41958d" : "#798781", borderRadius: [3, 3, 0, 0] } })), barWidth: "42%" }]
  };
  return (
    <div className="results-layout">
      <section className="results-lead">
        <p className="eyebrow">Fixture benchmark</p>
        <h2><span>{project.optimized_improvement_percent}%</span> shorter planned makespan</h2>
        <p>Independent jobs overlap while two-robot teams remain reserved for foundation and roof modules.</p>
        <div className="result-kpis">
          <Kpi icon={Gauge} label="CP-SAT plan" value={`${project.controllers.optimized.makespan_s}s`} />
          <Kpi icon={Route} label="Fleet travel" value={`${project.controllers.optimized.total_travel_m.toFixed(1)}m`} />
          <Kpi icon={Sparkles} label="Completion" value={`${Math.round(project.controllers.optimized.structure_completion_rate * 100)}%`} />
        </div>
        <button className="primary-button" onClick={() => api.report().then(({ markdown }) => downloadText("construction-report.md", markdown))}><FileDown size={17} /> Export report</button>
        <div className="result-boundary"><strong>Evidence boundary</strong><span>Planner replay is deterministic. Learned-policy acceptance remains open until five research seeds are evaluated.</span></div>
      </section>
      <section className="chart-section">
        <div><p className="eyebrow">Controller benchmark</p><h2>Makespan comparison</h2></div>
        <ReactECharts option={option} style={{ height: 310 }} />
        <div className="controller-table">
          {names.map((name) => <div key={name}><strong>{name === "optimized" ? "CP-SAT" : name}</strong><span>{project.controllers[name].makespan_s}s</span><small>{project.controllers[name].idle_robot_seconds}s idle</small></div>)}
        </div>
        <div className="acceptance-grid">
          <Acceptance label="Sequential" value="reference" ready />
          <Acceptance label="Greedy" value="reference" ready />
          <Acceptance label="Auction" value="event baseline" ready />
          <Acceptance label="CP-SAT" value="planning bound" ready />
          <Acceptance label="IPPO" value={policies.some((item) => item.controller === "ippo") ? "checkpoint" : "not trained"} ready={policies.some((item) => item.controller === "ippo")} />
          <Acceptance label="MAPPO" value={policies.some((item) => item.controller === "mappo") ? "checkpoint" : "not trained"} ready={policies.some((item) => item.controller === "mappo")} />
        </div>
      </section>
    </div>
  );
}

function Kpi({ icon: Icon, label, value }: { icon: typeof Gauge; label: string; value: string }) {
  return <div className="kpi"><Icon size={19} /><span>{label}</span><strong>{value}</strong></div>;
}

function Acceptance({ label, value, ready }: { label: string; value: string; ready: boolean }) {
  return <div><strong>{label}</strong><span>{value}</span><i className={ready ? "ready" : "pending"} /></div>;
}
