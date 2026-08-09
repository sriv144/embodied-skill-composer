import {
  FileDown,
  Gauge,
  Radio,
  Route,
  ShieldCheck,
  Sparkles
} from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";
import { api, formatApiError } from "../api";
import { EvidenceReferences } from "../components/EvidenceReference";
import { downloadText } from "../components/WorkbenchControls";
import "./results-evidence.css";
import type {
  AblationDecision,
  CoppeliaEvidenceRun,
  CoppeliaEvidenceSummary,
  CoppeliaHealth,
  ControllerEvaluation,
  LearningCurveSeries,
  LabPolicy,
  PrimaryAcceptanceAudit,
  Project,
  ResearchSummary
} from "../types";

type ResearchEvidence = {
  research: ResearchSummary;
  coppeliaHealth: CoppeliaHealth;
  coppelia: CoppeliaEvidenceSummary;
};

export function ResultsView({
  project,
  policies
}: {
  project: Project;
  policies: LabPolicy[];
}) {
  const runtime = api.runtime();
  const [evidence, setEvidence] = useState<ResearchEvidence | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setRefreshing(true);
      const [research, coppeliaHealth, coppelia] = await Promise.all([
        api.researchSummary(),
        api.coppeliaHealth(),
        api.coppeliaEvidence()
      ]);
      if (!cancelled) {
        setEvidence({
          research,
          coppeliaHealth,
          coppelia
        });
        setEvidenceError(null);
        setRefreshing(false);
      }
    };
    load().catch((reason) => {
      if (!cancelled) {
        setEvidenceError(formatApiError(reason));
        setRefreshing(false);
      }
    });
    const refresh =
      runtime.mode === "local"
        ? window.setInterval(() => {
            void load().catch((reason) => {
              if (!cancelled) {
                setEvidenceError(formatApiError(reason));
                setRefreshing(false);
              }
            });
          }, 5_000)
        : null;
    return () => {
      cancelled = true;
      if (refresh !== null) window.clearInterval(refresh);
    };
  }, [runtime.mode]);

  const learnedSummaries = useMemo(
    () =>
      (evidence?.research.confidence_intervals ?? []).filter(
        (summary) =>
          ["mappo", "ippo"].includes(summary.controller) &&
          summary.experiment_variant?.endsWith("_full")
      ),
    [evidence]
  );
  const acceptance = evidence?.research.acceptance ?? null;
  const ablations = evidence?.research.ablations ?? [];
  const names = ["sequential", "greedy", "optimized"] as const;

  const exportReport = async () => {
    setExportError(null);
    try {
      const { markdown } = await api.report();
      downloadText("construction-report.md", markdown);
    } catch (reason) {
      setExportError(formatApiError(reason));
    }
  };

  return (
    <div className="results-layout">
      <section className="results-lead" aria-labelledby="results-title">
        <p className="eyebrow">Fixture benchmark</p>
        <h2 id="results-title">
          <span>{project.optimized_improvement_percent}%</span> shorter planned
          makespan
        </h2>
        <p>
          Independent jobs overlap while two-robot teams remain reserved for
          foundation and roof modules.
        </p>
        <div className="result-kpis">
          <Kpi
            icon={Gauge}
            label="CP-SAT plan"
            value={`${project.controllers.optimized.makespan_s}s`}
          />
          <Kpi
            icon={Route}
            label="Fleet travel"
            value={`${project.controllers.optimized.total_travel_m.toFixed(1)}m`}
          />
          <Kpi
            icon={Sparkles}
            label="Completion"
            value={`${Math.round(
              project.controllers.optimized.structure_completion_rate * 100
            )}%`}
          />
        </div>
        <button className="primary-button" onClick={exportReport}>
          <FileDown aria-hidden="true" size={17} /> Export report
        </button>
        {exportError && (
          <p className="notice-line" role="alert">
            {exportError}
          </p>
        )}
        <div className="result-boundary">
          <strong>Evidence boundary</strong>
          <span>
            Planner replay is deterministic. Learned claims below appear only
            when a frozen held-out evaluation is registered.
          </span>
        </div>
        <div className="result-boundary">
          <strong>{runtime.label}</strong>
          <span>{runtime.detail}</span>
        </div>
      </section>

      <section className="chart-section" aria-labelledby="benchmark-title">
        {evidenceError && (
          <p className="notice-line" role="alert">
            Research evidence could not be loaded: {evidenceError}
          </p>
        )}
        <p className="sr-only" role="status" aria-live="polite">
          {refreshing
            ? "Refreshing research and simulator evidence."
            : "Research and simulator evidence is up to date."}
        </p>
        <div>
          <p className="eyebrow">Controller benchmark</p>
          <h2 id="benchmark-title">Makespan comparison</h2>
        </div>
        <ControllerBarChart project={project} />
        <div className="controller-table">
          {names.map((name) => (
            <div key={name}>
              <strong>{name === "optimized" ? "CP-SAT" : name}</strong>
              <span>{project.controllers[name].makespan_s}s</span>
              <small>
                {project.controllers[name].idle_robot_seconds}s idle
              </small>
            </div>
          ))}
        </div>

        <section className="policy-table" aria-labelledby="learned-title">
          <div>
            <p className="eyebrow">Held-out test split</p>
            <h3 id="learned-title">Learned coordination</h3>
          </div>
          <EvidenceStatus summary={evidence?.research ?? null} />
          {learnedSummaries.length === 0 ? (
            <p className="muted-line">
              No frozen MAPPO/IPPO held-out result is included in this data
              source.
            </p>
          ) : (
            learnedSummaries.map((summary) => (
              <LearnedSummary
                key={`${summary.controller}-${summary.experiment_variant}-${summary.failure_enabled}`}
                summary={summary}
              />
            ))
          )}
          <EvidenceReferences
            references={evidence?.research.artifact_references ?? []}
          />
        </section>

        <LearningCurves
          series={evidence?.research.learning_curves ?? []}
        />

        <PerSeedEvidence
          trainingRows={evidence?.research.per_training_seed ?? []}
          scenarioRows={evidence?.research.per_scenario_seed ?? []}
        />

        <AcceptanceEvidence
          acceptance={acceptance}
          hasMappo={policies.some((item) => item.controller === "mappo")}
          hasIppo={policies.some((item) => item.controller === "ippo")}
        />

        <AblationEvidence decisions={ablations} />

        <section className="policy-table" aria-labelledby="coppelia-title">
          <div>
            <p className="eyebrow">Simulator evidence</p>
            <h3 id="coppelia-title">CoppeliaSim boundary</h3>
          </div>
          <div className="acceptance-grid">
            <EvidenceFact
              label="Remote API"
              value={
                evidence?.coppeliaHealth.reachable
                  ? `${evidence.coppeliaHealth.host}:${evidence.coppeliaHealth.port}`
                  : "offline"
              }
              ready={Boolean(evidence?.coppeliaHealth.reachable)}
            />
            <EvidenceFact
              label="Nominal gate"
              value={
                passingCoppeliaRun(evidence?.coppelia.nominal)
                  ? "validated live"
                  : "not validated"
              }
              ready={passingCoppeliaRun(evidence?.coppelia.nominal)}
            />
            <EvidenceFact
              label="Recovery gate"
              value={
                passingCoppeliaRun(evidence?.coppelia.recovery)
                  ? "validated live"
                  : "not validated"
              }
              ready={passingCoppeliaRun(evidence?.coppelia.recovery)}
            />
            <EvidenceFact
              label="Payload"
              value="logical transport"
              ready
            />
          </div>
          <p
            className="notice-line"
            role="status"
          >
            <Radio aria-hidden="true" size={12} />{" "}
            {coppeliaReady(evidence?.coppelia)
              ? "Nominal and unavailable-robot recovery manifests independently pass the live gate."
              : evidence?.coppelia.reason ??
                "Both validated live-gate manifests are required before simulator evidence is ready."}
          </p>
          <p className="muted-line">
            Wheel commands and measured base telemetry are the dynamic fidelity
            boundary. This workbench does not claim arm, gripper, payload
            contact, or transport dynamics.
          </p>
          <CoppeliaRunEvidence
            label="Nominal evidence"
            run={evidence?.coppelia.nominal ?? null}
          />
          <CoppeliaRunEvidence
            label="Unavailable-robot recovery"
            run={evidence?.coppelia.recovery ?? null}
          />
        </section>
      </section>
    </div>
  );
}

function EvidenceStatus({ summary }: { summary: ResearchSummary | null }) {
  if (!summary) {
    return <p className="muted-line">Loading the evidence summary…</p>;
  }
  const canonical = summary.status === "validated" && summary.claim_allowed;
  return (
    <p className="notice-line" role="status">
      <Radio aria-hidden="true" size={12} />{" "}
      {canonical
        ? "Canonical, hash-verified release evidence."
        : summary.status === "local_complete"
          ? "Complete local evidence; canonical publication is still required."
          : summary.status === "in_progress"
            ? "Local experiment evidence is still in progress."
            : summary.reason ?? "Canonical learned-policy evidence is absent."}
    </p>
  );
}

function LearningCurves({ series }: { series: LearningCurveSeries[] }) {
  const titleId = useId();
  const visible = series.filter(
    (item) =>
      item.experiment_variant?.endsWith("_full") &&
      item.points.some((point) =>
        Number.isFinite(point.rollout_terminal_fraction)
      )
  );
  if (visible.length === 0) {
    return (
      <section className="policy-table" aria-labelledby="learning-curves-title">
        <div>
          <p className="eyebrow">Training evidence</p>
          <h3 id="learning-curves-title">Learning curves</h3>
        </div>
        <p className="muted-line">
          No canonical or active primary-run learning curves are available in
          this data source.
        </p>
      </section>
    );
  }
  const width = 620;
  const height = 270;
  const left = 50;
  const right = 16;
  const top = 20;
  const bottom = 42;
  const maxTransitions = Math.max(
    1,
    ...visible.flatMap((item) =>
      item.points.map((point) => point.transitions)
    )
  );
  const colors = [
    "#f05b2a",
    "#21756f",
    "#9b4dca",
    "#1f6aa5",
    "#b16b08",
    "#8b3d46",
    "#2f855a",
    "#6b5b95",
    "#0f766e",
    "#a33b20"
  ];

  return (
    <section className="policy-table" aria-labelledby="learning-curves-title">
      <div>
        <p className="eyebrow">Training evidence</p>
        <h3 id="learning-curves-title">Primary learning curves</h3>
      </div>
      <svg
        className="evidence-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={titleId}
      >
        <title id={titleId}>
          Rollout completion over transitions for {visible.length} MAPPO and
          IPPO primary seed runs
        </title>
        {[0, 0.5, 1].map((fraction) => {
          const y = top + (1 - fraction) * (height - top - bottom);
          return (
            <g key={fraction}>
              <line
                x1={left}
                x2={width - right}
                y1={y}
                y2={y}
                stroke="#dce3df"
              />
              <text x="4" y={y + 4} fontSize="10" fill="#63716a">
                {Math.round(fraction * 100)}%
              </text>
            </g>
          );
        })}
        {visible.map((item, index) => {
          const points = item.points.filter((point) =>
            Number.isFinite(point.rollout_terminal_fraction)
          );
          const path = points
            .map((point, pointIndex) => {
              const x =
                left +
                (point.transitions / maxTransitions) *
                  (width - left - right);
              const completion = Math.max(
                0,
                Math.min(1, point.rollout_terminal_fraction ?? 0)
              );
              const y =
                top + (1 - completion) * (height - top - bottom);
              return `${pointIndex === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(" ");
          return (
            <path
              key={item.run_key}
              d={path}
              fill="none"
              stroke={colors[index % colors.length]}
              strokeWidth="2"
              opacity="0.82"
            >
              <title>
                {item.controller.toUpperCase()} seed {item.training_seed}
              </title>
            </path>
          );
        })}
        <text
          x={(left + width - right) / 2}
          y={height - 9}
          textAnchor="middle"
          fontSize="10"
          fill="#63716a"
        >
          transitions (max {maxTransitions.toLocaleString()})
        </text>
      </svg>
      <div className="curve-legend" aria-label="Learning curve legend">
        {visible.map((item, index) => (
          <span key={item.run_key}>
            <i
              aria-hidden="true"
              style={{ backgroundColor: colors[index % colors.length] }}
            />
            {item.controller.toUpperCase()} s{item.training_seed}
          </span>
        ))}
      </div>
    </section>
  );
}

function PerSeedEvidence({
  trainingRows,
  scenarioRows
}: {
  trainingRows: Array<Record<string, unknown>>;
  scenarioRows: Array<Record<string, unknown>>;
}) {
  const learnedTraining = learnedSeedRows(trainingRows, "training_seed");
  const learnedScenarios = learnedSeedRows(scenarioRows, "scenario_seed");
  return (
    <section className="policy-table" aria-labelledby="per-seed-title">
      <div>
        <p className="eyebrow">Hierarchical evaluation</p>
        <h3 id="per-seed-title">Per-seed evidence</h3>
      </div>
      {learnedTraining.length === 0 && learnedScenarios.length === 0 ? (
        <p className="muted-line">
          Per-training-seed and per-scenario-seed tables appear with the
          canonical held-out evaluation.
        </p>
      ) : (
        <div className="seed-table-grid">
          <SeedTable
            caption="Training seeds"
            rows={learnedTraining}
            seedLabel="Training seed"
          />
          <SeedTable
            caption="Scenario seeds"
            rows={learnedScenarios}
            seedLabel="Scenario seed"
          />
        </div>
      )}
    </section>
  );
}

type DisplaySeedRow = {
  id: string;
  controller: string;
  failures: boolean;
  seed: string;
  completion: number;
  makespan: number;
};

function SeedTable({
  caption,
  rows,
  seedLabel
}: {
  caption: string;
  rows: DisplaySeedRow[];
  seedLabel: string;
}) {
  return (
    <div
      className="seed-table-scroll"
      role="region"
      aria-label={`${caption} table, horizontally scrollable`}
    >
      <table className="seed-table">
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Policy</th>
            <th scope="col">Failures</th>
            <th scope="col">{seedLabel}</th>
            <th scope="col">Completion</th>
            <th scope="col">Makespan</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <th scope="row">{row.controller}</th>
              <td>{row.failures ? "yes" : "no"}</td>
              <td>{row.seed}</td>
              <td>{formatPercent(row.completion)}</td>
              <td>{row.makespan.toFixed(1)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function learnedSeedRows(
  rows: Array<Record<string, unknown>>,
  seedKey: "training_seed" | "scenario_seed"
): DisplaySeedRow[] {
  return rows.flatMap((row, index) => {
    const controller = String(row.controller ?? "");
    const variant = String(row.experiment_variant ?? "");
    if (
      !["mappo", "ippo"].includes(controller) ||
      (variant && !variant.endsWith("_full"))
    ) {
      return [];
    }
    const metrics =
      typeof row.metrics === "object" && row.metrics !== null
        ? (row.metrics as Record<string, unknown>)
        : {};
    const completion = metricMean(metrics.structure_completion_rate);
    const makespan = metricMean(metrics.makespan_s);
    if (completion === null || makespan === null) return [];
    const seed = row[seedKey];
    return [
      {
        id: `${controller}-${variant}-${String(seed)}-${String(row.failure_enabled)}-${index}`,
        controller: controller.toUpperCase(),
        failures: row.failure_enabled === true,
        seed: seed === null || seed === undefined ? "n/a" : String(seed),
        completion,
        makespan
      }
    ];
  });
}

function metricMean(value: unknown): number | null {
  if (
    typeof value !== "object" ||
    value === null ||
    !("mean" in value) ||
    typeof value.mean !== "number"
  ) {
    return null;
  }
  return value.mean;
}

function CoppeliaRunEvidence({
  label,
  run
}: {
  label: string;
  run: CoppeliaEvidenceRun | null;
}) {
  if (!run) return null;
  const installed = Array.isArray(run.metrics.installed_module_ids)
    ? run.metrics.installed_module_ids.length
    : 0;
  return (
    <div className="coppelia-evidence-run">
      <strong>{label}</strong>
      <span>
        {installed}/{run.metrics.expected_module_count ?? "?"} modules · seed{" "}
        {run.manifest.scenario_seed}
      </span>
      <EvidenceReferences references={run.artifact_references ?? []} />
    </div>
  );
}

function passingCoppeliaRun(
  run: CoppeliaEvidenceRun | null | undefined
): boolean {
  return Boolean(
    run &&
      run.manifest.evidence_kind === "live_coppelia" &&
      run.manifest.live_evidence &&
      run.manifest.run_status === "completed" &&
      run.manifest.live_gate_passed &&
      run.manifest.approval_gate_confirmed &&
      run.metrics.status === "completed" &&
      run.metrics.live_gate_passed &&
      run.metrics.acceptance &&
      Object.values(run.metrics.acceptance).length > 0 &&
      Object.values(run.metrics.acceptance).every((value) => value === true)
  );
}

function coppeliaReady(
  summary: CoppeliaEvidenceSummary | null | undefined
): boolean {
  return Boolean(
    summary &&
      summary.status === "validated" &&
      (summary.ready === true || summary.claim_allowed) &&
      passingCoppeliaRun(summary.nominal) &&
      passingCoppeliaRun(summary.recovery) &&
      summary.nominal?.manifest.source_commit ===
        summary.recovery?.manifest.source_commit
  );
}

function ControllerBarChart({ project }: { project: Project }) {
  const titleId = useId();
  const controllers = [
    { key: "sequential", label: "Sequential", color: "#798781" },
    { key: "greedy", label: "Greedy", color: "#41958d" },
    { key: "optimized", label: "CP-SAT", color: "#f05b2a" }
  ] as const;
  const width = 620;
  const height = 240;
  const top = 24;
  const bottom = 34;
  const max = Math.max(
    ...controllers.map((item) => project.controllers[item.key].makespan_s)
  );
  const slot = width / controllers.length;
  const barWidth = Math.min(90, slot * 0.48);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby={titleId}
      style={{ width: "100%", height: 280 }}
    >
      <title id={titleId}>
        Makespan comparison: {controllers
          .map(
            (item) =>
              `${item.label} ${project.controllers[item.key].makespan_s} seconds`
          )
          .join(", ")}
      </title>
      {[0, 0.5, 1].map((fraction) => {
        const y = height - bottom - fraction * (height - top - bottom);
        return (
          <g key={fraction}>
            <line
              x1="0"
              x2={width}
              y1={y}
              y2={y}
              stroke="#e3e8e5"
              strokeWidth="1"
            />
            <text x="4" y={y - 4} fontSize="9" fill="#718078">
              {Math.round(max * fraction)}s
            </text>
          </g>
        );
      })}
      {controllers.map((item, index) => {
        const value = project.controllers[item.key].makespan_s;
        const barHeight = (value / max) * (height - top - bottom);
        const x = index * slot + (slot - barWidth) / 2;
        const y = height - bottom - barHeight;
        return (
          <g key={item.key}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill={item.color}
              rx="3"
            >
              <title>
                {item.label}: {value} seconds
              </title>
            </rect>
            <text
              x={x + barWidth / 2}
              y={height - 12}
              textAnchor="middle"
              fontSize="11"
              fill="#53605b"
            >
              {item.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LearnedSummary({ summary }: { summary: ControllerEvaluation }) {
  const completion = summary.metrics.structure_completion_rate;
  const makespan = summary.metrics.makespan_s;
  return (
    <div className="policy-row">
      <strong>{summary.controller.toUpperCase()}</strong>
      <span>
        {summary.failure_enabled ? "failures enabled" : "nominal"} ·{" "}
        {summary.episode_count} episodes
      </span>
      <small>
        {completion
          ? `${formatPercent(completion.mean)} completion · 95% CI ${formatPercent(
              completion.bootstrap_ci95_low
            )}–${formatPercent(completion.bootstrap_ci95_high)}`
          : makespan
            ? `${makespan.median.toFixed(1)}s median`
            : "metrics unavailable"}
      </small>
      <ShieldCheck aria-hidden="true" size={14} />
    </div>
  );
}

function AcceptanceEvidence({
  acceptance,
  hasMappo,
  hasIppo
}: {
  acceptance: PrimaryAcceptanceAudit | null;
  hasMappo: boolean;
  hasIppo: boolean;
}) {
  return (
    <section className="policy-table" aria-labelledby="acceptance-title">
      <div>
        <p className="eyebrow">Release gates</p>
        <h3 id="acceptance-title">Primary acceptance</h3>
      </div>
      {acceptance ? (
        <>
          <p className="notice-line" role="status">
            <Radio aria-hidden="true" size={12} /> Frozen held-out acceptance{" "}
            {acceptance.passed ? "passed" : "did not pass"}.
          </p>
          <div className="acceptance-grid">
            {acceptance.results.map((result) => (
              <EvidenceFact
                key={result.name}
                label={friendlyAcceptanceName(result.name)}
                value={`${formatAcceptanceValue(result.name, result.observed)} ${
                  result.comparison === "min" ? "≥" : "≤"
                } ${formatAcceptanceValue(result.name, result.threshold)}`}
                ready={result.passed}
              />
            ))}
          </div>
        </>
      ) : (
        <div className="acceptance-grid">
          <EvidenceFact label="Sequential" value="reference" ready />
          <EvidenceFact label="Greedy" value="reference" ready />
          <EvidenceFact label="Auction" value="event baseline" ready />
          <EvidenceFact label="CP-SAT" value="planning bound" ready />
          <EvidenceFact
            label="IPPO"
            value={hasIppo ? "checkpoint only" : "not trained"}
            ready={false}
          />
          <EvidenceFact
            label="MAPPO"
            value={hasMappo ? "checkpoint only" : "not trained"}
            ready={false}
          />
        </div>
      )}
    </section>
  );
}

function AblationEvidence({ decisions }: { decisions: AblationDecision[] }) {
  return (
    <section className="policy-table" aria-labelledby="ablation-title">
      <div>
        <p className="eyebrow">Pre-registered hypotheses</p>
        <h3 id="ablation-title">Ablations</h3>
      </div>
      {decisions.length === 0 ? (
        <p className="muted-line">
          Ablation interpretations appear after the complete held-out matrix.
        </p>
      ) : (
        decisions.map((decision) => (
          <div className="policy-row" key={decision.hypothesis}>
            <strong>{decision.supported ? "SUPPORTED" : "UNSUPPORTED"}</strong>
            <span>{decision.hypothesis.replaceAll("_", " ")}</span>
            <small title={decision.interpretation}>
              {decision.interpretation}
            </small>
            <ShieldCheck aria-hidden="true" size={14} />
          </div>
        ))
      )}
    </section>
  );
}

function Kpi({
  icon: Icon,
  label,
  value
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
}) {
  return (
    <div className="kpi">
      <Icon aria-hidden="true" size={19} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvidenceFact({
  label,
  value,
  ready
}: {
  label: string;
  value: string;
  ready: boolean;
}) {
  return (
    <div>
      <strong>{label}</strong>
      <span>{value}</span>
      <i className={ready ? "ready" : "pending"} aria-hidden="true" />
    </div>
  );
}

function friendlyAcceptanceName(name: string): string {
  const labels: Record<string, string> = {
    mappo_no_failure_mean_completion: "MAPPO nominal",
    ippo_no_failure_mean_completion: "IPPO nominal",
    mappo_median_makespan_cp_sat_ratio: "MAPPO / CP-SAT",
    mappo_failure_mean_completion: "MAPPO failures"
  };
  return labels[name] ?? name.replaceAll("_", " ");
}

function formatAcceptanceValue(name: string, value: number): string {
  return name.includes("completion")
    ? formatPercent(value)
    : `${value.toFixed(3)}×`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
