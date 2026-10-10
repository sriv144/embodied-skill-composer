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

export function runtimeStatusFor(
  mode: LabMode,
  channel: "preview" | "release" | null = null
): RuntimeStatus {
  return Object.freeze(
    mode === "static"
      ? {
          mode,
          channel,
          source: "versioned-static-demo",
          readOnly: true,
          interactive: false,
          label: channel === "release"
            ? "Read-only public release v1.0.0"
            : channel === "preview"
              ? "Read-only public preview"
              : "Read-only static workbench",
          detail:
            "Versioned demo artifacts are loaded from this deployment. Compute and simulator controls are disabled."
        }
      : {
          mode,
          channel: null,
          source: "loopback-api",
          readOnly: false,
          interactive: true,
          label: "Local research lab",
          detail:
            "The workbench is connected to the loopback API. API failures are reported and never replaced with demo data."
        }
  );
}

let runtimeStatus = runtimeStatusFor(configuredMode);
let runtimeInitialization: Promise<RuntimeStatus> | null = null;
let verifiedReleaseAssets: ReadonlySet<string> = new Set();
const releaseAssetBaseUrl =
  "https://github.com/sriv144/embodied-skill-composer/releases/download/v1.0.0/";
const coppeliaArchivePath = "evidence/coppelia/archive-index.json";
const coppeliaArchiveAsset = "construction-intelligence-v1-coppelia-evidence.json";
const hashPattern = /^[a-f0-9]{64}$/;
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

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

async function sha256Text(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

export async function verifiedStaticRuntime(
  provenanceValue: unknown,
  statusText: string
): Promise<RuntimeStatus> {
  const provenance = record(provenanceValue);
  const status = record(JSON.parse(statusText));
  const channel = status?.channel;
  if (
    !provenance || !status ||
    provenance.schema_version !== "construction-intelligence-public-demo-provenance-v1" ||
    status.schema_version !== "construction-intelligence-public-status-v1" ||
    (channel !== "preview" && channel !== "release") ||
    provenance.channel !== channel ||
    status.release_ready !== (channel === "release") ||
    status.release_version !== provenance.release_version ||
    status.release_tag !== provenance.release_tag
  ) {
    throw new Error("Static release channel metadata is missing or inconsistent.");
  }
  const integrityPayload = { ...provenance, manifest_payload_sha256: null };
  const canonical = JSON.stringify(integrityPayload, (_key, value: unknown) => {
    const object = record(value);
    return object
      ? Object.fromEntries(Object.keys(object).sort().map((key) => [key, object[key]]))
      : value;
  }, 2) + "\n";
  if (provenance.manifest_payload_sha256 !== await sha256Text(canonical)) {
    throw new Error("Static provenance manifest failed its integrity check.");
  }
  const artifacts = Array.isArray(provenance.artifacts) ? provenance.artifacts : [];
  const matching = artifacts.map(record).filter((item) => item?.path === "release-status.json");
  const bytes = new TextEncoder().encode(statusText);
  if (matching.length !== 1 || matching[0]?.bytes !== bytes.byteLength) {
    throw new Error("Static release status is absent from the provenance inventory.");
  }
  const observed = await sha256Text(statusText);
  if (matching[0]?.sha256 !== observed) {
    throw new Error("Static release status failed its provenance hash check.");
  }
  if (channel === "release") {
    const evidence = record(status.evidence);
    const source = record(provenance.source);
    if (
      status.release_version !== "1.0.0" || status.release_tag !== "v1.0.0" ||
      status.claim_status !== "release_evidence" ||
      !Array.isArray(status.limitations) || status.limitations.length !== 0 ||
      evidence?.deterministic !== "canonical" || evidence.research !== "validated" ||
      evidence.coppelia !== "validated" || source?.dirty !== false ||
      typeof source.commit !== "string" || !/^[a-f0-9]{40}$/.test(source.commit) ||
      typeof source.tree_digest !== "string" || !/^[a-f0-9]{64}$/.test(source.tree_digest)
    ) {
      throw new Error("Static release metadata does not attest complete canonical v1 evidence.");
    }
  } else if (status.release_version !== null || status.release_tag !== null) {
    throw new Error("Static preview metadata cannot declare a release identity.");
  }
  return runtimeStatusFor("static", channel);
}

function initializeRuntime(): Promise<RuntimeStatus> {
  if (runtimeStatus.mode === "local") return Promise.resolve(runtimeStatus);
  runtimeInitialization ??= Promise.all([
    fetchJson<unknown>(demoUrl("provenance.json")),
    fetchText(demoUrl("release-status.json"))
  ]).then(async ([provenance, statusText]) => {
    const verified = await verifiedStaticRuntime(provenance, statusText);
    if (verified.channel === "release") {
      const indexText = await fetchText(demoUrl(coppeliaArchivePath));
      verifiedReleaseAssets = await verifiedReleaseAssetUrls(provenance, indexText);
    }
    runtimeStatus = verified;
    return runtimeStatus;
  });
  return runtimeInitialization;
}

export async function verifiedReleaseAssetUrls(
  provenanceValue: unknown,
  indexText: string
): Promise<ReadonlySet<string>> {
  const provenance = record(provenanceValue);
  const binding = record(provenance?.coppelia_archive);
  const index = record(JSON.parse(indexText));
  const inputs = Array.isArray(provenance?.inputs) ? provenance.inputs.map(record) : [];
  const simulator = inputs.filter((input) => input?.kind === "simulator");
  const inventory = Array.isArray(provenance?.artifacts) ? provenance.artifacts.map(record) : [];
  const indexInventory = inventory.filter((item) => item?.path === coppeliaArchivePath);
  const digest = await sha256Text(indexText);
  const bytes = new TextEncoder().encode(indexText).byteLength;
  if (
    provenance?.channel !== "release" || provenance.release_tag !== "v1.0.0" ||
    binding?.path !== coppeliaArchivePath || binding.asset_file !== coppeliaArchiveAsset ||
    binding.asset_base_url !== releaseAssetBaseUrl || binding.sha256 !== digest ||
    indexInventory.length !== 1 || indexInventory[0]?.sha256 !== digest ||
    indexInventory[0]?.bytes !== bytes || bytes > 4 * 1024 ** 2 ||
    index?.schema_version !== "construction-intelligence-coppelia-archive-v1" ||
    index.generated_at !== "1970-01-01T00:00:00Z" || simulator.length !== 1 ||
    index.descriptor_sha256 !== simulator[0]?.manifest_sha256
  ) {
    throw new Error("Static release archive index is missing or not bound to provenance.");
  }
  const source = record(index.source);
  const inputSource = record(simulator[0]?.source);
  if (
    !source || !inputSource || source.dirty !== false || inputSource.dirty !== false ||
    source.commit !== inputSource.commit || source.tree_digest !== inputSource.tree_digest ||
    typeof source.commit !== "string" || !/^[a-f0-9]{40}$/.test(source.commit) ||
    typeof source.tree_digest !== "string" || !hashPattern.test(source.tree_digest)
  ) {
    throw new Error("Static release archive source identity is inconsistent.");
  }
  const parts = Array.isArray(index.parts) ? index.parts.map(record) : [];
  const files = Array.isArray(index.files) ? index.files.map(record) : [];
  const nativeNames = [
    "manifest.json", "scenario.json", "planned_jobs.json", "planned_vs_measured_replay.json",
    "wheel_commands.jsonl", "measured_telemetry.jsonl", "trace.json", "metrics.json",
    "report.md", "construction_intelligence.ttt", "runtime_attestation.json"
  ];
  const paths = ["nominal", "recovery"].flatMap((scenario) => [
    ...nativeNames.map((name) => `evidence/coppelia/${scenario}/${name}`),
    `evidence/coppelia/visualizations/${scenario}.mp4`
  ]).sort();
  const positiveInteger = (value: unknown, maximum: number): value is number =>
    typeof value === "number" && Number.isSafeInteger(value) && value > 0 && value <= maximum;
  const validHash = (value: unknown): value is string =>
    typeof value === "string" && hashPattern.test(value);
  if (
    !parts.length || parts.length > 256 || files.length !== paths.length ||
    !positiveInteger(index.chunk_bytes, 128 * 1024 ** 2) ||
    parts.some((part, position) => (
      part?.file !== `construction-intelligence-v1-coppelia-evidence-part-${String(position + 1).padStart(6, "0")}.zip` ||
      !positiveInteger(part.bytes, 384 * 1024 ** 2) || !validHash(part.sha256) ||
      !positiveInteger(part.entry_count, 2)
    )) ||
    files.some((file, position) => (
      file?.path !== paths[position] || !positiveInteger(file.bytes, 8 * 1024 ** 3) ||
      !validHash(file.sha256) || !Array.isArray(file.chunks) || !file.chunks.length
    ))
  ) {
    throw new Error("Static release archive inventory is invalid.");
  }
  let ordinal = 0;
  let totalBytes = 0;
  const scenarioBytes = { nominal: 0, recovery: 0 };
  const counts = new Map<string, number>();
  for (const file of files) {
    const size = file?.bytes as number;
    const chunks = (file?.chunks as unknown[]).map(record);
    const chunkBytes = index.chunk_bytes;
    totalBytes += size;
    const scenario = String(file?.path).startsWith("evidence/coppelia/nominal/") ||
      file?.path === "evidence/coppelia/visualizations/nominal.mp4" ? "nominal" : "recovery";
    scenarioBytes[scenario] += size;
    if (chunks.length !== Math.ceil(size / chunkBytes)) {
      throw new Error("Static release archive chunks are invalid.");
    }
    for (const [offset, chunk] of chunks.entries()) {
      const part = parts[Math.floor(ordinal / 2)];
      if (
        !part || !chunk || chunk.part !== part.file || !validHash(chunk.sha256) ||
        chunk.bytes !== Math.min(chunkBytes, size - offset * chunkBytes) ||
        chunk.member !== `chunks/${String(file?.path)}/${String(offset + 1).padStart(6, "0")}.bin`
      ) throw new Error("Static release archive chunks are invalid.");
      counts.set(String(part.file), (counts.get(String(part.file)) ?? 0) + 1);
      ordinal += 1;
    }
  }
  if (
    totalBytes > 16 * 1024 ** 3 ||
    scenarioBytes.nominal > 8 * 1024 ** 3 || scenarioBytes.recovery > 8 * 1024 ** 3
  ) throw new Error("Static release archive exceeds its native byte budget.");
  if (parts.some((part) => counts.get(String(part?.file)) !== part?.entry_count)) {
    throw new Error("Static release archive part coverage is invalid.");
  }
  // Part names come only from the verified contiguous inventory, never from href fields.
  return new Set([
    releaseAssetBaseUrl + coppeliaArchiveAsset,
    ...parts.map((part) => releaseAssetBaseUrl + String(part?.file))
  ]);
}

export function safeReleaseArtifactUrl(path: string, allowed: ReadonlySet<string>): string | null {
  if (
    !path.startsWith(releaseAssetBaseUrl) || !allowed.has(path) ||
    !/^construction-intelligence-v1-coppelia-evidence(?:\.json|-part-\d{6}\.zip)$/.test(
      path.slice(releaseAssetBaseUrl.length)
    )
  ) return null;
  return path;
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
      "This control is disabled in the read-only static workbench.",
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
      "The public workbench ships only the obstacle recovery trace.",
      demoUrl("traces/recovery.json")
    );
  }
  return demoUrl("traces/recovery.json");
}

export const api = {
  mode: () => runtimeStatus.mode,
  runtime: () => runtimeStatus,
  initializeRuntime,

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
      const releaseUrl = safeReleaseArtifactUrl(path, verifiedReleaseAssets);
      if (releaseUrl) return releaseUrl;
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
