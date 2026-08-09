import type {
  ArtifactReference,
  CoppeliaEvidenceSummary,
  CoppeliaHealth,
  DesignValidationResult,
  ExperimentMatrix,
  ExperimentProfile,
  HeldoutLaunchResponse,
  LabEvaluation,
  LabMode,
  LabPolicy,
  LabRun,
  LabScenario,
  MatrixLaunchResponse,
  PolicySelectionRecord,
  Project,
  RunEventEnvelope,
  ResearchSummary,
  RuntimeStatus,
  Trace,
  ValidationSelectionResponse
} from "./types";

const base = import.meta.env.BASE_URL;
const configuredMode: LabMode =
  import.meta.env.VITE_STATIC_DEMO === "true" || import.meta.env.MODE === "static"
    ? "static"
    : "local";

export function runtimeStatusFor(mode: LabMode): RuntimeStatus {
  return Object.freeze(
    mode === "static"
      ? {
          mode,
          source: "versioned-static-demo",
          readOnly: true,
          interactive: false,
          label: "Read-only public preview",
          detail:
            "Versioned demo artifacts are loaded from this deployment. Compute and simulator controls are disabled."
        }
      : {
          mode,
          source: "loopback-api",
          readOnly: false,
          interactive: true,
          label: "Local research lab",
          detail:
            "The workbench is connected to the loopback API. API failures are reported and never replaced with demo data."
        }
  );
}

const runtimeStatus = runtimeStatusFor(configuredMode);
const passiveStaticArtifactSuffixes = new Set([
  ".csv",
  ".glb",
  ".json",
  ".jsonl",
  ".md",
  ".mp4",
  ".onnx",
  ".png",
  ".pt",
  ".ttt"
]);

export class ApiRequestError extends Error {
  readonly status: number | null;
  readonly path: string;
  readonly details: unknown;

  constructor(
    message: string,
    path: string,
    status: number | null = null,
    details: unknown = null
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.path = path;
    this.status = status;
    this.details = details;
  }
}

function demoUrl(path: string) {
  return `${base}demo/${path}`;
}

function errorDetail(payload: unknown): string | null {
  if (typeof payload === "string" && payload.trim()) return payload.trim();
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return null;
  }
  if (typeof payload.detail === "string") return payload.detail;
  return validationDetail(payload.detail);
}

function validationDetail(value: unknown): string | null {
  if (Array.isArray(value)) {
    const messages = value
      .map((issue) => {
        if (
          typeof issue !== "object" ||
          issue === null ||
          !("loc" in issue) ||
          !("msg" in issue) ||
          !Array.isArray(issue.loc) ||
          typeof issue.msg !== "string"
        ) {
          return null;
        }
        return `${issue.loc.map(String).join(".")}: ${issue.msg}`;
      })
      .filter((item): item is string => Boolean(item));
    return messages.length
      ? `Request validation failed — ${messages.join("; ")}`
      : null;
  }
  if (
    typeof value !== "object" ||
    value === null ||
    !("issues" in value) ||
    !Array.isArray(value.issues)
  ) {
    return null;
  }
  const messages = value.issues
    .map((issue) => {
      if (
        typeof issue !== "object" ||
        issue === null ||
        !("path" in issue) ||
        !("message" in issue) ||
        typeof issue.path !== "string" ||
        typeof issue.message !== "string"
      ) {
        return null;
      }
      return `${issue.path}: ${issue.message}`;
    })
    .filter((item): item is string => Boolean(item));
  return messages.length
    ? `Design validation failed — ${messages.join("; ")}`
    : null;
}

async function readJson<T>(response: Response, path: string): Promise<T> {
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = errorDetail(payload) ?? response.statusText ?? "Request failed";
    throw new ApiRequestError(
      `${response.status} ${detail}`.trim(),
      path,
      response.status,
      payload
    );
  }
  return payload as T;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (reason) {
    const detail = reason instanceof Error ? reason.message : String(reason);
    throw new ApiRequestError(`Could not reach ${path}: ${detail}`, path);
  }
  return readJson<T>(response, path);
}

async function fetchText(path: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch (reason) {
    const detail = reason instanceof Error ? reason.message : String(reason);
    throw new ApiRequestError(`Could not reach ${path}: ${detail}`, path);
  }
  if (!response.ok) {
    throw new ApiRequestError(
      `${response.status} ${response.statusText}`.trim(),
      path,
      response.status
    );
  }
  return response.text();
}

async function optionalStaticJson<T>(path: string, fallback: T): Promise<T> {
  const url = demoUrl(path);
  let response: Response;
  try {
    response = await fetch(url);
  } catch (reason) {
    const detail = reason instanceof Error ? reason.message : String(reason);
    throw new ApiRequestError(`Could not reach ${url}: ${detail}`, url);
  }
  if (response.status === 404) return fallback;
  if (!response.headers.get("Content-Type")?.includes("application/json")) {
    return fallback;
  }
  return readJson<T>(response, url);
}

async function runtimeJson<T>(localPath: string, staticPath: string): Promise<T> {
  return runtimeStatus.mode === "static"
    ? fetchJson<T>(demoUrl(staticPath))
    : fetchJson<T>(localPath);
}

async function requireLocal<T>(path: string, init?: RequestInit): Promise<T> {
  if (runtimeStatus.readOnly) {
    throw new ApiRequestError(
      "This control is disabled in the read-only public preview.",
      path
    );
  }
  return fetchJson<T>(path, init);
}

function normalizeProject(project: Project): Project {
  if (runtimeStatus.mode === "local") return project;
  return {
    ...project,
    geometry_asset_url: demoUrl("house.glb"),
    robot_asset_url: demoUrl("construction_robot.glb")
  };
}

function jsonRequest(body: unknown, method = "POST"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
}

function encoded(value: string): string {
  return encodeURIComponent(value);
}

export function formatApiError(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

export function staticRecoveryTracePath(failureType: string): string {
  if (failureType !== "obstacle") {
    throw new ApiRequestError(
      "The public preview ships only the obstacle recovery trace.",
      demoUrl("traces/recovery.json")
    );
  }
  return demoUrl("traces/recovery.json");
}

export const api = {
  mode: () => runtimeStatus.mode,
  runtime: () => runtimeStatus,

  project: () =>
    runtimeJson<Project>("/api/project", "project.json").then(normalizeProject),
  trace: (controller: string) =>
    runtimeJson<Trace>(
      `/api/traces/${encoded(controller)}`,
      `traces/${encoded(controller)}.json`
    ),
  disrupt: (controller: string, failureType: string, timestamp: number) =>
    runtimeStatus.mode === "static"
      ? fetchJson<Trace>(staticRecoveryTracePath(failureType))
      : requireLocal<Trace>(
          `/api/traces/${encoded(controller)}/disrupt`,
          jsonRequest({ failure_type: failureType, timestamp_s: timestamp })
        ),
  report: async () =>
    runtimeStatus.mode === "static"
      ? { markdown: await fetchText(demoUrl("report.md")) }
      : requireLocal<{ markdown: string }>("/api/report"),
  rebuild: (design: Project["design"]) =>
    requireLocal<Project>(
      "/api/design/rebuild",
      jsonRequest({ design })
    ),
  validateDesign: (design: Project["design"]) =>
    requireLocal<DesignValidationResult>(
      "/api/design/validate",
      jsonRequest({ design })
    ),
  parseFloorPlan: (file: File, width: number) => {
    const data = new FormData();
    data.append("file", file);
    return requireLocal<Project["design"]["floor_plan"]>(
      `/api/intent/parse?known_width_m=${width}`,
      { method: "POST", body: data }
    );
  },

  scenarios: () =>
    runtimeJson<LabScenario[]>("/api/lab/scenarios", "scenarios.json"),
  generateScenario: (seed: number) =>
    requireLocal<Record<string, unknown>>(
      "/api/lab/scenarios",
      jsonRequest({ seed })
    ),
  policies: () =>
    runtimeJson<LabPolicy[]>("/api/lab/policies", "policies.json"),
  runs: () => runtimeJson<LabRun[]>("/api/lab/runs", "runs.json"),
  run: async (runId: string) => {
    if (runtimeStatus.mode === "local") {
      return requireLocal<LabRun>(`/api/lab/runs/${encoded(runId)}`);
    }
    const runs = await api.runs();
    const run = runs.find((item) => item.id === runId);
    if (!run) {
      throw new ApiRequestError(`Static run not found: ${runId}`, demoUrl("runs.json"), 404);
    }
    return run;
  },
  runEvents: (runId: string, after = 0) =>
    runtimeStatus.mode === "local"
      ? requireLocal<RunEventEnvelope[]>(
          `/api/lab/runs/${encoded(runId)}/events?after=${Math.max(0, after)}`
        )
      : optionalStaticJson<RunEventEnvelope[]>(
          `events/${encoded(runId)}.json`,
          []
        ).then((events) => events.filter((event) => event.sequence > after)),
  runEventsSocketUrl: (runId: string) => {
    if (runtimeStatus.mode !== "local") return null;
    const url = new URL(
      `/api/lab/runs/${encoded(runId)}/events/ws`,
      window.location.href
    );
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  },

  launchTraining: (payload: {
    algorithm: "mappo" | "ippo";
    profile: ExperimentProfile;
    seed: number;
    transitions?: number;
    confirmed: boolean;
    device: "auto" | "cpu" | "cuda";
  }) =>
    requireLocal<{ run_id: string }>(
      "/api/lab/training",
      jsonRequest(payload)
    ),
  launchExperimentMatrix: (profile: ExperimentProfile) =>
    requireLocal<MatrixLaunchResponse>(
      "/api/lab/experiment-matrices",
      jsonRequest({ profile, confirmed: true })
    ),
  experimentMatrices: () =>
    runtimeStatus.mode === "local"
      ? requireLocal<ExperimentMatrix[]>("/api/lab/experiment-matrices")
      : optionalStaticJson<ExperimentMatrix[]>("experiment-matrices.json", []),
  experimentMatrix: async (matrixId: string) => {
    if (runtimeStatus.mode === "local") {
      return requireLocal<ExperimentMatrix>(
        `/api/lab/experiment-matrices/${encoded(matrixId)}`
      );
    }
    const matrices = await api.experimentMatrices();
    const matrix = matrices.find((item) => item.id === matrixId);
    if (!matrix) {
      throw new ApiRequestError(
        `Static experiment matrix not found: ${matrixId}`,
        demoUrl("experiment-matrices.json"),
        404
      );
    }
    return matrix;
  },
  matrixSelections: (matrixId: string) =>
    runtimeStatus.mode === "local"
      ? requireLocal<PolicySelectionRecord[]>(
          `/api/lab/experiment-matrices/${encoded(matrixId)}/selections`
        )
      : optionalStaticJson<PolicySelectionRecord[]>(
          `experiment-matrices/${encoded(matrixId)}/selections.json`,
          []
        ),
  freezeValidationSelection: (matrixId: string, runKey: string) =>
    requireLocal<ValidationSelectionResponse>(
      `/api/lab/experiment-matrices/${encoded(matrixId)}/selections/${encoded(runKey)}`,
      jsonRequest({ confirmed: true }, "PUT")
    ),
  launchHeldoutEvaluation: (matrixId: string) =>
    requireLocal<HeldoutLaunchResponse>(
      `/api/lab/experiment-matrices/${encoded(matrixId)}/heldout`,
      jsonRequest({ confirmed: true })
    ),
  evaluations: async (matrixId?: string) => {
    if (runtimeStatus.mode === "local") {
      const query = matrixId ? `?matrix_id=${encoded(matrixId)}` : "";
      return requireLocal<LabEvaluation[]>(`/api/lab/evaluations${query}`);
    }
    const evaluations = await optionalStaticJson<LabEvaluation[]>(
      "evaluations.json",
      []
    );
    return matrixId
      ? evaluations.filter((item) => item.matrix_id === matrixId)
      : evaluations;
  },
  researchSummary: () =>
    runtimeJson<ResearchSummary>(
      "/api/lab/evidence/research-summary",
      "research-summary.json"
    ),
  coppeliaEvidence: () =>
    runtimeJson<CoppeliaEvidenceSummary>(
      "/api/lab/evidence/coppelia",
      "coppelia-evidence.json"
    ),
  runArtifacts: (runId: string) =>
    runtimeStatus.mode === "local"
      ? requireLocal<ArtifactReference[]>(
          `/api/lab/runs/${encoded(runId)}/artifacts`
        )
      : Promise.resolve<ArtifactReference[]>([]),
  evaluationArtifacts: (evaluationId: string) =>
    runtimeStatus.mode === "local"
      ? requireLocal<ArtifactReference[]>(
          `/api/lab/evaluations/${encoded(evaluationId)}/artifacts`
        )
      : api
          .researchSummary()
          .then((summary) => summary.artifact_references ?? []),
  cancelRun: (runId: string) =>
    requireLocal<{ run_id: string; cancel_requested: boolean }>(
      `/api/lab/runs/${encoded(runId)}/cancel`,
      { method: "POST" }
    ),
  resumeRun: (runId: string) =>
    requireLocal<{ run_id: string; status: "resuming" }>(
      `/api/lab/runs/${encoded(runId)}/resume`,
      { method: "POST" }
    ),

  coppeliaHealth: () =>
    runtimeStatus.mode === "static"
      ? Promise.resolve<CoppeliaHealth>({
          reachable: false,
          host: "public demo",
          port: 23000,
          detail: "CoppeliaSim control is available in local mode.",
          controller: "dynamic_base_logical_payload"
        })
      : requireLocal<CoppeliaHealth>("/api/lab/coppelia/health"),

  artifactHref: (path: string | null | undefined) => {
    if (!path) return null;
    if (runtimeStatus.mode === "static") {
      const relative = safeStaticArtifactPath(path.replace(/^demo\//, ""));
      return relative ? demoUrl(relative) : null;
    }
    return safeLocalArtifactPath(path);
  }
};

export function safeStaticArtifactPath(path: string): string | null {
  if (
    !path ||
    path.includes("\\") ||
    path.startsWith("/") ||
    /^[a-z][a-z0-9+.-]*:/i.test(path)
  ) {
    return null;
  }
  const segments = path.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    return null;
  }
  try {
    const decoded = segments.map((segment) => decodeURIComponent(segment));
    if (
      decoded.some(
        (segment) =>
          !segment ||
          segment === "." ||
          segment === ".." ||
          segment.includes("/") ||
          segment.includes("\\")
      )
    ) {
      return null;
    }
    const filename = decoded.at(-1)?.toLowerCase() ?? "";
    const suffixOffset = filename.lastIndexOf(".");
    if (
      suffixOffset <= 0 ||
      !passiveStaticArtifactSuffixes.has(filename.slice(suffixOffset))
    ) {
      return null;
    }
    return decoded.map((segment) => encodeURIComponent(segment)).join("/");
  } catch {
    return null;
  }
}

function safeLocalArtifactPath(path: string): string | null {
  if (
    !path.startsWith("/api/lab/") ||
    !path.includes("/artifacts/") ||
    path.includes("\\") ||
    path.includes("?") ||
    path.includes("#")
  ) {
    return null;
  }
  let decoded: string;
  try {
    decoded = decodeURIComponent(path);
  } catch {
    return null;
  }
  if (
    decoded.split("/").some((segment) => segment === "." || segment === "..") ||
    !/^\/api\/lab\/(?:runs|evaluations|coppelia\/evidence)\/[^/]+\/artifacts\/[^/].*/.test(
      decoded
    )
  ) {
    return null;
  }
  return path;
}
