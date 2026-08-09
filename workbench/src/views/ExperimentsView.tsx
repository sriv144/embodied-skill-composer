import {
  Ban,
  Check,
  Cpu,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  X
} from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { api, formatApiError } from "../api";
import { EvidenceReferences } from "../components/EvidenceReference";
import { FidelityBadge } from "../components/WorkbenchControls";
import { useRunEventStream } from "../hooks/useRunEventStream";
import type {
  ArtifactReference,
  CoppeliaHealth,
  ExperimentMatrix,
  ExperimentProfile,
  LabEvaluation,
  LabMode,
  LabPolicy,
  LabRun,
  PolicySelectionRecord,
  RunEventEnvelope
} from "../types";

/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Axe requires the named scrolling experiment region to be keyboard-focusable. */

export type LauncherMode = "single" | "matrix";

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
  const runtime = api.runtime();
  const [launcherMode, setLauncherMode] = useState<LauncherMode>("matrix");
  const [algorithm, setAlgorithm] = useState<"mappo" | "ippo">("mappo");
  const [profile, setProfile] = useState<ExperimentProfile>("smoke");
  const [seed, setSeed] = useState(7);
  const [confirmed, setConfirmed] = useState(false);
  const [confirmedHeldout, setConfirmedHeldout] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [matrices, setMatrices] = useState<ExperimentMatrix[]>([]);
  const [selectedMatrixId, setSelectedMatrixId] = useState<string>("");
  const [selections, setSelections] = useState<PolicySelectionRecord[]>([]);
  const [evaluations, setEvaluations] = useState<LabEvaluation[]>([]);
  const liveRefreshAt = useRef(0);

  const loadResearch = useCallback(async () => {
    const [nextMatrices, nextEvaluations] = await Promise.all([
      api.experimentMatrices(),
      api.evaluations()
    ]);
    setMatrices(nextMatrices);
    setEvaluations(nextEvaluations);
    setSelectedMatrixId((current) =>
      nextMatrices.some((item) => item.id === current)
        ? current
        : (nextMatrices[0]?.id ?? "")
    );
  }, []);

  const loadSelections = useCallback(async (matrixId: string) => {
    if (!matrixId) {
      setSelections([]);
      return;
    }
    setSelections(await api.matrixSelections(matrixId));
  }, []);

  useEffect(() => {
    loadResearch().catch((reason) => setError(formatApiError(reason)));
  }, [loadResearch]);

  useEffect(() => {
    loadSelections(selectedMatrixId).catch((reason) =>
      setError(formatApiError(reason))
    );
  }, [loadSelections, selectedMatrixId]);

  const selectedMatrix = useMemo(
    () => matrices.find((item) => item.id === selectedMatrixId) ?? null,
    [matrices, selectedMatrixId]
  );
  const visibleRuns = useMemo(
    () => runsForLauncher(launcherMode, runs, matrices, selectedMatrix),
    [launcherMode, matrices, runs, selectedMatrix]
  );
  const displayedMatrix = launcherMode === "matrix" ? selectedMatrix : null;
  const activeRun = runForLiveEvents(visibleRuns);

  const handleLiveEvent = useCallback(
    (event: RunEventEnvelope) => {
      const now = Date.now();
      const terminalEvent = isTerminalRunEvent(event.payload.event);
      if (!terminalEvent && now - liveRefreshAt.current < 1_250) return;
      liveRefreshAt.current = now;
      onRefresh();
      loadResearch().catch((reason) => setError(formatApiError(reason)));
    },
    [loadResearch, onRefresh]
  );

  const stream = useRunEventStream(
    activeRun?.id ?? null,
    mode === "local",
    handleLiveEvent
  );

  const refreshAll = async () => {
    setBusyAction("refresh");
    setError(null);
    try {
      onRefresh();
      await loadResearch();
      await loadSelections(selectedMatrixId);
      setNotice("Registry refreshed.");
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setBusyAction(null);
    }
  };

  const launch = async () => {
    setBusyAction("launch");
    setNotice(null);
    setError(null);
    try {
      if (launcherMode === "matrix") {
        const response = await api.launchExperimentMatrix(profile);
        setSelectedMatrixId(response.matrix_id);
        setNotice(
          `${response.run_count}-run ${response.profile} matrix queued.`
        );
      } else {
        const response = await api.launchTraining({
          algorithm,
          profile,
          seed,
          confirmed: true,
          device: "auto"
        });
        setNotice(`Run ${response.run_id} queued.`);
      }
      setConfirmed(false);
      onRefresh();
      await loadResearch();
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setBusyAction(null);
    }
  };

  const selectedRunKeys = useMemo(
    () => new Set(selections.map((item) => item.run_key)),
    [selections]
  );
  const nextValidationRun = selectedMatrix?.runs.find(
    (run) => run.status === "completed" && !selectedRunKeys.has(run.run_key)
  );

  const freezeNextSelection = async () => {
    if (!selectedMatrix || !nextValidationRun) return;
    setBusyAction("selection");
    setError(null);
    try {
      const response = await api.freezeValidationSelection(
        selectedMatrix.id,
        nextValidationRun.run_key
      );
      setNotice(
        `Validation selected the ${Math.round(
          response.selection.checkpoint_fraction * 100
        )}% checkpoint for seed ${response.selection.training_seed}.`
      );
      await Promise.all([
        loadResearch(),
        loadSelections(selectedMatrix.id)
      ]);
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setBusyAction(null);
    }
  };

  const launchHeldout = async () => {
    if (!selectedMatrix) return;
    setBusyAction("heldout");
    setError(null);
    try {
      const response = await api.launchHeldoutEvaluation(selectedMatrix.id);
      setConfirmedHeldout(false);
      setNotice(
        `Held-out evaluation queued as ${response.run_ids.join(", ")}.`
      );
      onRefresh();
      await loadResearch();
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setBusyAction(null);
    }
  };

  const actOnRun = async (run: LabRun) => {
    const resumable = isResumable(run);
    setBusyAction(run.id);
    setError(null);
    try {
      if (resumable) {
        await api.resumeRun(run.id);
        setNotice(`Run ${run.id} is resuming.`);
      } else {
        await api.cancelRun(run.id);
        setNotice(`Cancellation requested for ${run.id}.`);
      }
      onRefresh();
      await loadResearch();
    } catch (reason) {
      setError(formatApiError(reason));
    } finally {
      setBusyAction(null);
    }
  };

  const matrixEvaluationCount = selectedMatrix
    ? evaluations.filter((item) => item.matrix_id === selectedMatrix.id).length
    : 0;
  const matrixProgress = selectedMatrix
    ? selectedMatrix.runs.reduce((sum, run) => sum + run.progress, 0) /
      Math.max(selectedMatrix.expected_run_count, 1)
    : 0;

  return (
    <div className="experiments-layout">
      <section className="experiment-main" aria-labelledby="experiments-title">
        <header className="operational-toolbar">
          <div>
            <p className="eyebrow">Research runs</p>
            <h2 id="experiments-title">Durable experiment registry</h2>
          </div>
          <button
            className="icon-button light"
            onClick={refreshAll}
            disabled={busyAction === "refresh"}
            aria-label="Refresh experiment registry"
            title="Refresh experiment registry"
          >
            <RefreshCw aria-hidden="true" size={16} />
          </button>
        </header>

        <div
          className="experiment-body"
          role="region"
          aria-label="Experiment registry details"
          tabIndex={0}
        >
          {error && (
            <p className="notice-line" role="alert">
              {error}
            </p>
          )}
          <p className="notice-line" role="status" aria-live="polite">
            {notice ?? runtime.detail}
          </p>

          <section className="run-chart-band" aria-labelledby="progress-title">
            <div>
              <p className="eyebrow">Run progress</p>
              <h3 id="progress-title">
                {displayedMatrix ? "Selected matrix" : "Recent single jobs"}
              </h3>
              {displayedMatrix && (
                <p className="muted-line">
                  {Math.round(matrixProgress * 100)}% aggregate ·{" "}
                  {displayedMatrix.selection_count}/{displayedMatrix.expected_run_count}{" "}
                  selected
                </p>
              )}
            </div>
            <ProgressPlot runs={visibleRuns.slice(0, 20)} />
          </section>

          <section className="policy-table" aria-labelledby="matrix-title">
            <div>
              <p className="eyebrow">Frozen protocol</p>
              <h3 id="matrix-title">Experiment matrix</h3>
            </div>
            {matrices.length === 0 ? (
              <p className="muted-line">
                {mode === "static"
                  ? "This public bundle contains deterministic fixture evidence only."
                  : "No experiment matrix has been launched."}
              </p>
            ) : (
              <>
                <label className="control-label" htmlFor="experiment-matrix">
                  Registered matrix
                </label>
                <select
                  id="experiment-matrix"
                  className="number-field"
                  value={selectedMatrixId}
                  onChange={(event) => setSelectedMatrixId(event.target.value)}
                >
                  {matrices.map((matrix) => (
                    <option value={matrix.id} key={matrix.id}>
                      {matrix.execution_profile} · {matrix.status} ·{" "}
                      {matrix.id.slice(-12)}
                    </option>
                  ))}
                </select>
                {selectedMatrix && (
                  <>
                    <div className="acceptance-grid">
                      <MatrixFact
                        label="Training"
                        value={`${selectedMatrix.status_counts.completed ?? 0}/${selectedMatrix.expected_run_count} complete`}
                        ready={
                          (selectedMatrix.status_counts.completed ?? 0) ===
                          selectedMatrix.expected_run_count
                        }
                      />
                      <MatrixFact
                        label="Validation"
                        value={`${selectedMatrix.selection_count}/${selectedMatrix.expected_run_count} frozen`}
                        ready={
                          selectedMatrix.selection_count ===
                          selectedMatrix.expected_run_count
                        }
                      />
                      <MatrixFact
                        label="Held-out"
                        value={
                          matrixEvaluationCount
                            ? `${matrixEvaluationCount} evaluation`
                            : "not launched"
                        }
                        ready={matrixEvaluationCount > 0}
                      />
                    </div>
                    <div
                      className="toolbar-actions"
                      style={{ marginTop: 14, flexWrap: "wrap" }}
                    >
                      <button
                        className="primary-button"
                        onClick={freezeNextSelection}
                        disabled={
                          runtime.readOnly ||
                          !nextValidationRun ||
                          busyAction !== null
                        }
                      >
                        <Check aria-hidden="true" size={15} />
                        {nextValidationRun
                          ? `Freeze next validation (${selectedMatrix.selection_count + 1}/${selectedMatrix.expected_run_count})`
                          : "Validation selections complete"}
                      </button>
                    </div>
                    <label className="confirm-control">
                      <input
                        type="checkbox"
                        checked={confirmedHeldout}
                        disabled={runtime.readOnly}
                        onChange={(event) =>
                          setConfirmedHeldout(event.target.checked)
                        }
                      />
                      <span>Approve held-out evaluation after all selections</span>
                    </label>
                    <button
                      className="primary-button"
                      onClick={launchHeldout}
                      disabled={
                        runtime.readOnly ||
                        !confirmedHeldout ||
                        selectedMatrix.selection_count !==
                          selectedMatrix.expected_run_count ||
                        busyAction !== null
                      }
                    >
                      <Play aria-hidden="true" size={15} />
                      Launch held-out matrix
                    </button>
                  </>
                )}
              </>
            )}
          </section>

          <section className="run-history" aria-labelledby="run-history-title">
            <div>
              <p className="eyebrow">Durable queue</p>
              <h3 id="run-history-title">Run history</h3>
            </div>
            <div className="table-header" aria-hidden="true">
              <span>Run</span>
              <span>Kind</span>
              <span>Status</span>
              <span>Progress</span>
              <span />
            </div>
            {visibleRuns.length === 0 && (
              <div className="empty-row">No registered runs</div>
            )}
            {visibleRuns.slice(0, 24).map((run) => {
              const resumable = isResumable(run);
              const cancellable = isCancellable(run);
              return (
                <div className="run-row" key={run.id} title={run.error ?? run.id}>
                  <strong>{run.id}</strong>
                  <span>{run.kind.replaceAll("_", " ")}</span>
                  <Status value={run.status} />
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-label={`${run.id} progress`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(run.progress * 100)}
                  >
                    <i style={{ width: `${run.progress * 100}%` }} />
                  </div>
                  {resumable || cancellable ? (
                    <button
                      disabled={runtime.readOnly || busyAction !== null}
                      aria-label={
                        resumable ? `Resume ${run.id}` : `Cancel ${run.id}`
                      }
                      title={resumable ? "Resume run" : "Cancel run"}
                      onClick={() => actOnRun(run)}
                    >
                      {resumable ? (
                        <RotateCcw aria-hidden="true" size={14} />
                      ) : (
                        <X aria-hidden="true" size={14} />
                      )}
                    </button>
                  ) : (
                    <span />
                  )}
                </div>
              );
            })}
            {visibleRuns.some((run) => run.error) && (
              <div role="alert">
                {visibleRuns
                  .filter((run) => run.error)
                  .map((run) => (
                    <p className="notice-line" key={`${run.id}-error`}>
                      <strong>{run.id}:</strong> {run.error}
                    </p>
                  ))}
              </div>
            )}
          </section>

          <section className="policy-table" aria-labelledby="event-stream-title">
            <div>
              <p className="eyebrow">Typed events</p>
              <h3 id="event-stream-title">Live worker activity</h3>
            </div>
            <p className="notice-line" aria-live="polite">
              {activeRun ? (
                <>
                  <Radio aria-hidden="true" size={12} /> {activeRun.id} ·{" "}
                  {stream.connection}
                </>
              ) : (
                "No active local run."
              )}
            </p>
            {stream.error && (
              <p className="notice-line" role="alert">
                {stream.error}
              </p>
            )}
            {stream.events
              .slice(-6)
              .reverse()
              .map((event) => (
                <div className="policy-row" key={event.sequence}>
                  <strong>#{event.sequence}</strong>
                  <span>{describeEvent(event)}</span>
                  <small>{formatTimestamp(event.created_at)}</small>
                  <Check aria-hidden="true" size={14} />
                </div>
              ))}
          </section>

          <section className="policy-table" aria-labelledby="policy-title">
            <div>
              <p className="eyebrow">Policy registry</p>
              <h3 id="policy-title">Exported actors</h3>
            </div>
            {policies.length === 0 ? (
              <p className="muted-line">
                No MAPPO or IPPO checkpoint has been registered.
              </p>
            ) : (
              policies.map((policy) => (
                <div className="policy-row" key={policy.id}>
                  <strong>{policy.controller.toUpperCase()}</strong>
                  <span>{policy.id}</span>
                  <small>
                    {policy.manifest.transition_count?.toLocaleString() ?? 0}{" "}
                    transitions
                  </small>
                  <Check aria-hidden="true" size={14} />
                </div>
              ))
            )}
            {selectedMatrix &&
              evaluations
                .filter((item) => item.matrix_id === selectedMatrix.id)
                .slice(0, 1)
                .map((evaluation) => (
                  <EvaluationEvidenceLinks
                    key={evaluation.id}
                    evaluationId={evaluation.id}
                  />
                ))}
          </section>
        </div>
      </section>

      <aside className="experiment-launcher" aria-labelledby="launcher-title">
        <p className="eyebrow">Compute launch</p>
        <h2 id="launcher-title">Research queue</h2>
        <div
          className="segmented full"
          style={{ gridTemplateColumns: "1fr 1fr" }}
          aria-label="Launch type"
        >
          <button
            className={launcherMode === "matrix" ? "selected" : ""}
            aria-pressed={launcherMode === "matrix"}
            onClick={() => setLauncherMode("matrix")}
          >
            Matrix
          </button>
          <button
            className={launcherMode === "single" ? "selected" : ""}
            aria-pressed={launcherMode === "single"}
            onClick={() => setLauncherMode("single")}
          >
            Single
          </button>
        </div>

        {launcherMode === "single" && (
          <>
            <span className="control-label">Algorithm</span>
            <div
              className="segmented full"
              style={{ gridTemplateColumns: "1fr 1fr" }}
              aria-label="Training algorithm"
            >
              <button
                className={algorithm === "mappo" ? "selected" : ""}
                aria-pressed={algorithm === "mappo"}
                onClick={() => setAlgorithm("mappo")}
              >
                MAPPO
              </button>
              <button
                className={algorithm === "ippo" ? "selected" : ""}
                aria-pressed={algorithm === "ippo"}
                onClick={() => setAlgorithm("ippo")}
              >
                IPPO
              </button>
            </div>
          </>
        )}

        <span className="control-label">Profile</span>
        <div className="segmented full" aria-label="Experiment profile">
          {(["unit", "smoke", "research"] as const).map((item) => (
            <button
              key={item}
              className={profile === item ? "selected" : ""}
              aria-pressed={profile === item}
              onClick={() => setProfile(item)}
            >
              {item}
            </button>
          ))}
        </div>

        {launcherMode === "single" && (
          <>
            <label className="control-label" htmlFor="training-seed">
              Training seed
            </label>
            <input
              id="training-seed"
              className="number-field"
              type="number"
              min={0}
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
          </>
        )}

        <label className="confirm-control">
          <input
            type="checkbox"
            checked={confirmed}
            disabled={runtime.readOnly}
            onChange={(event) => setConfirmed(event.target.checked)}
          />
          <span>
            Approve {launcherMode === "matrix" ? "serial matrix" : "training"}{" "}
            compute
          </span>
        </label>
        <button
          className="primary-button launch-button"
          disabled={runtime.readOnly || !confirmed || busyAction !== null}
          onClick={launch}
        >
          <Play aria-hidden="true" size={16} />
          {busyAction === "launch"
            ? "Launching"
            : launcherMode === "matrix"
              ? "Queue matrix"
              : "Start training"}
        </button>

        <div className="fidelity-matrix">
          <h3>Simulator fidelity</h3>
          <div>
            <FidelityBadge level="Event Sim" active />
            <span>temporal MARL</span>
          </div>
          <div>
            <FidelityBadge level="MuJoCo" active />
            <span>skill profile</span>
          </div>
          <div>
            <FidelityBadge
              level="CoppeliaSim"
              active={Boolean(coppelia?.reachable)}
            />
            <span>
              {coppelia?.reachable ? "wheel control online" : "offline"}
            </span>
          </div>
        </div>
        <div className="runtime-note" role="status">
          {mode === "local" ? (
            <>
              <Cpu aria-hidden="true" size={15} />
              Local lab controls enabled
            </>
          ) : (
            <>
              <Ban aria-hidden="true" size={15} />
              Read-only public preview
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

function EvaluationEvidenceLinks({
  evaluationId
}: {
  evaluationId: string;
}) {
  const [references, setReferences] = useState<ArtifactReference[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    api
      .evaluationArtifacts(evaluationId)
      .then((items) => {
        if (!cancelled) setReferences(items);
      })
      .catch((reason) => {
        if (!cancelled) setError(formatApiError(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [evaluationId]);
  if (error) {
    return (
      <span className="muted-line" role="alert">
        Artifact links unavailable: {error}
      </span>
    );
  }
  return <EvidenceReferences references={references} />;
}

function ProgressPlot({ runs }: { runs: LabRun[] }) {
  const titleId = useId();
  if (runs.length === 0) {
    return <p className="muted-line">No run progress to chart.</p>;
  }
  const width = 640;
  const height = 170;
  const padding = 18;
  const step = runs.length > 1 ? (width - padding * 2) / (runs.length - 1) : 0;
  const points = runs
    .map((run, index) => {
      const x = padding + index * step;
      const y = height - padding - run.progress * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const average =
    runs.reduce((sum, run) => sum + run.progress, 0) / runs.length;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby={titleId}
      style={{ width: "100%", height: 190 }}
    >
      <title id={titleId}>
        Progress for {runs.length} runs; average {Math.round(average * 100)} percent
      </title>
      {[0, 0.5, 1].map((fraction) => {
        const y = height - padding - fraction * (height - padding * 2);
        return (
          <g key={fraction}>
            <line
              x1={padding}
              x2={width - padding}
              y1={y}
              y2={y}
              stroke="#dbe1dd"
              strokeWidth="1"
            />
            <text x={padding} y={y - 4} fill="#738079" fontSize="9">
              {Math.round(fraction * 100)}%
            </text>
          </g>
        );
      })}
      <polyline
        points={points}
        fill="none"
        stroke="#2d8376"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      {runs.map((run, index) => {
        const x = padding + index * step;
        const y = height - padding - run.progress * (height - padding * 2);
        return (
          <circle
            key={run.id}
            cx={x}
            cy={y}
            r="4"
            fill={run.status === "failed" ? "#a2453a" : "#f2672e"}
          >
            <title>
              {run.id}: {Math.round(run.progress * 100)}%
            </title>
          </circle>
        );
      })}
    </svg>
  );
}

function Status({ value }: { value: LabRun["status"] }) {
  return (
    <span className={`run-status ${value}`}>
      {value.replaceAll("_", " ")}
    </span>
  );
}

function MatrixFact({
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

export function isResumable(run: LabRun): boolean {
  if (typeof run.can_resume === "boolean") return run.can_resume;
  if (!["interrupted", "failed", "cancelled"].includes(run.status)) return false;
  if (!["training", "matrix_evaluation"].includes(run.kind)) return false;
  return run.latest_checkpoint === null || run.latest_checkpoint === undefined
    ? true
    : Boolean(run.latest_checkpoint);
}

export function isCancellable(run: LabRun): boolean {
  if (typeof run.can_cancel === "boolean") return run.can_cancel;
  return ["queued", "running", "resuming"].includes(run.status);
}

export function runsForLauncher(
  launcherMode: LauncherMode,
  runs: LabRun[],
  matrices: ExperimentMatrix[],
  selectedMatrix: ExperimentMatrix | null
): LabRun[] {
  if (launcherMode === "matrix") return selectedMatrix?.runs ?? [];
  const matrixRunIds = new Set(
    matrices.flatMap((matrix) => matrix.runs.map((run) => run.id))
  );
  return runs.filter(
    (run) =>
      run.kind !== "matrix_evaluation" &&
      !matrixRunIds.has(run.id) &&
      typeof run.config.matrix_id !== "string"
  );
}

export function runForLiveEvents(runs: LabRun[]): LabRun | null {
  for (const status of [
    "running",
    "resuming",
    "cancel_requested",
    "queued"
  ] as const) {
    const run = runs.find((candidate) => candidate.status === status);
    if (run) return run;
  }
  return null;
}

export function isTerminalRunEvent(
  event: RunEventEnvelope["payload"]["event"]
): boolean {
  return [
    "training_completed",
    "evaluation_completed",
    "matrix_evaluation_completed",
    "run_interrupted",
    "worker_exited",
    "failed",
    "cancelled",
    "worker_launch_failed"
  ].includes(event);
}

function describeEvent(event: RunEventEnvelope): string {
  const payload = event.payload;
  switch (payload.event) {
    case "ppo_update":
      return `${payload.transitions.toLocaleString()} transitions · update ${payload.update}`;
    case "checkpoint_saved":
    case "behavior_cloning_checkpoint":
      return `${payload.event.replaceAll("_", " ")} · ${payload.transitions.toLocaleString()} transitions`;
    case "training_resumed":
      return `resumed at ${payload.transitions.toLocaleString()} transitions`;
    case "run_claimed":
      return `worker claimed attempt ${payload.attempt}`;
    case "matrix_evaluation_completed":
      return `held-out evaluation ${payload.acceptance.passed ? "passed" : "did not pass"}`;
    case "failed":
    case "worker_launch_failed":
      return `${payload.event.replaceAll("_", " ")} · ${payload.error}`;
    default:
      return payload.event.replaceAll("_", " ");
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
