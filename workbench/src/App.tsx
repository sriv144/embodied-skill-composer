import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BarChart3, BrainCircuit, Building2, Check, FlaskConical, Layers3, LoaderCircle, PlaySquare } from "lucide-react";
import { api } from "./api";
import type { CoppeliaHealth, LabMode, LabPolicy, LabRun, Project, Trace } from "./types";

const BrainView = lazy(() => import("./views/BrainView").then((module) => ({ default: module.BrainView })));
const DesignView = lazy(() => import("./views/DesignView").then((module) => ({ default: module.DesignView })));
const ExperimentsView = lazy(() => import("./views/ExperimentsView").then((module) => ({ default: module.ExperimentsView })));
const ResultsView = lazy(() => import("./views/ResultsView").then((module) => ({ default: module.ResultsView })));
const SimulateView = lazy(() => import("./views/SimulateView").then((module) => ({ default: module.SimulateView })));

type View = "design" | "brain" | "simulate" | "experiments" | "results";
const views: Array<{ id: View; label: string; icon: typeof Building2 }> = [
  { id: "design", label: "Design", icon: Building2 },
  { id: "brain", label: "Brain", icon: BrainCircuit },
  { id: "simulate", label: "Simulate", icon: PlaySquare },
  { id: "experiments", label: "Experiments", icon: FlaskConical },
  { id: "results", label: "Results", icon: BarChart3 }
];

export default function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [view, setView] = useState<View>(readView());
  const [controller, setController] = useState("optimized");
  const [mode, setMode] = useState<LabMode>(api.mode());
  const [runs, setRuns] = useState<LabRun[]>([]);
  const [policies, setPolicies] = useState<LabPolicy[]>([]);
  const [coppelia, setCoppelia] = useState<CoppeliaHealth | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshLab = useCallback(async () => {
    const [nextRuns, nextPolicies, health] = await Promise.all([
      api.runs(),
      api.policies(),
      api.coppeliaHealth()
    ]);
    setRuns(nextRuns);
    setPolicies(nextPolicies);
    setCoppelia(health);
    setMode(api.mode());
  }, []);

  useEffect(() => {
    Promise.all([api.project(), api.trace(controller)])
      .then(([nextProject, nextTrace]) => {
        setProject(nextProject);
        setTrace(nextTrace);
        setMode(api.mode());
      })
      .then(refreshLab)
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    api.trace(controller).then(setTrace).catch((reason) => setError(String(reason)));
  }, [controller]);

  useEffect(() => {
    const update = () => setView(readView());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  const navigate = (next: View) => {
    window.location.hash = `/${next}`;
    setView(next);
  };

  if (error) return <StartupState error={error} />;
  if (!project || !trace) return <StartupState />;

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand-mark"><Layers3 size={21} strokeWidth={2.2} /></div>
        <nav aria-label="Workbench views">
          {views.map((item) => (
            <button key={item.id} className={view === item.id ? "rail-button active" : "rail-button"} onClick={() => navigate(item.id)} title={item.label}>
              <item.icon size={20} /><span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-status" title={mode === "local" ? "Local research lab" : "Read-only public preview"}><span className={mode === "local" ? "status-dot" : "status-dot static"} />{mode === "local" ? "local" : "preview"}</div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <div><p className="eyebrow">Construction Intelligence v1</p><h1>{project.design.title}</h1></div>
          <div className="topbar-meta">
            <span>{project.plan.modules.length} modules</span><span>{project.plan.robots.length} robots</span>
            <span className="solver-ready"><Check size={14} /> {mode === "local" ? "Lab connected" : "Preview replay"}</span>
          </div>
        </header>
        <motion.section key={view} className="view-stage" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
          <Suspense fallback={<div className="view-loading"><LoaderCircle className="spin" size={24} /></div>}>
            {view === "design" && <DesignView project={project} mode={mode} onProject={(updated) => { setProject(updated); api.trace(controller).then(setTrace); }} />}
            {view === "brain" && <BrainView project={project} trace={trace} controller={controller} policies={policies} onController={setController} />}
            {view === "simulate" && <SimulateView project={project} trace={trace} controller={controller} coppelia={coppelia} onController={setController} onTrace={setTrace} />}
            {view === "experiments" && <ExperimentsView mode={mode} runs={runs} policies={policies} coppelia={coppelia} onRefresh={() => refreshLab().catch((reason) => setError(String(reason)))} />}
            {view === "results" && <ResultsView project={project} policies={policies} />}
          </Suspense>
        </motion.section>
      </main>
    </div>
  );
}

function readView(): View {
  const candidate = window.location.hash.replace(/^#\/?/, "") as View;
  return views.some((item) => item.id === candidate) ? candidate : "simulate";
}

function StartupState({ error }: { error?: string }) {
  return <div className="startup-state">{error ? <Layers3 size={32} /> : <LoaderCircle className="spin" size={32} />}<h1>{error ? "Workbench unavailable" : "Loading construction trace"}</h1><p>{error ?? ""}</p></div>;
}
