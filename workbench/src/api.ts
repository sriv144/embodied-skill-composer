import type {
  CoppeliaHealth,
  LabMode,
  LabPolicy,
  LabRun,
  LabScenario,
  Project,
  Trace
} from "./types";

const base = import.meta.env.BASE_URL;
let mode: LabMode =
  import.meta.env.VITE_STATIC_DEMO === "true" || import.meta.env.MODE === "static"
    ? "static"
    : "local";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  return response.json() as Promise<T>;
}

function demoUrl(path: string) {
  return `${base}demo/${path}`;
}

async function localOrDemo<T>(localPath: string, demoPath: string): Promise<T> {
  if (mode === "static") return fetch(demoUrl(demoPath)).then(readJson<T>);
  try {
    const response = await fetch(localPath);
    if (!response.ok) throw new Error(response.statusText);
    return (await response.json()) as T;
  } catch {
    mode = "static";
    return fetch(demoUrl(demoPath)).then(readJson<T>);
  }
}

function normalizeProject(project: Project): Project {
  if (mode === "local") return project;
  return {
    ...project,
    geometry_asset_url: demoUrl("house.glb"),
    robot_asset_url: demoUrl("construction_robot.glb")
  };
}

async function requireLocal<T>(path: string, init?: RequestInit): Promise<T> {
  if (mode === "static") throw new Error("This control is available in the local research lab.");
  return fetch(path, init).then(readJson<T>);
}

export const api = {
  mode: () => mode,
  project: () => localOrDemo<Project>("/api/project", "project.json").then(normalizeProject),
  trace: (controller: string) =>
    localOrDemo<Trace>(`/api/traces/${controller}`, `traces/${controller}.json`),
  disrupt: (controller: string, failureType: string, timestamp: number) =>
    mode === "static"
      ? fetch(demoUrl("traces/recovery.json")).then(readJson<Trace>)
      : requireLocal<Trace>(`/api/traces/${controller}/disrupt`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ failure_type: failureType, timestamp_s: timestamp })
        }),
  report: async () => {
    if (mode === "static") {
      const response = await fetch(demoUrl("report.md"));
      return { markdown: await response.text() };
    }
    return requireLocal<{ markdown: string }>("/api/report");
  },
  rebuild: (design: Project["design"]) =>
    requireLocal<Project>("/api/design/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design })
    }),
  parseFloorPlan: (file: File, width: number) => {
    const data = new FormData();
    data.append("file", file);
    return requireLocal<Project["design"]["floor_plan"]>(
      `/api/intent/parse?known_width_m=${width}`,
      { method: "POST", body: data }
    );
  },
  scenarios: () => localOrDemo<LabScenario[]>("/api/lab/scenarios", "scenarios.json"),
  generateScenario: (seed: number) =>
    requireLocal<Record<string, unknown>>("/api/lab/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed })
    }),
  policies: () => localOrDemo<LabPolicy[]>("/api/lab/policies", "policies.json"),
  runs: () => localOrDemo<LabRun[]>("/api/lab/runs", "runs.json"),
  launchTraining: (payload: {
    algorithm: "mappo" | "ippo";
    profile: "unit" | "smoke" | "research";
    seed: number;
    confirmed: boolean;
    device: "auto" | "cpu" | "cuda";
  }) =>
    requireLocal<{ run_id: string }>("/api/lab/training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }),
  cancelRun: (runId: string) =>
    requireLocal<{ run_id: string; cancel_requested: boolean }>(
      `/api/lab/runs/${runId}/cancel`,
      { method: "POST" }
    ),
  coppeliaHealth: () =>
    mode === "static"
      ? Promise.resolve<CoppeliaHealth>({
          reachable: false,
          host: "public demo",
          port: 23000,
          detail: "CoppeliaSim control is available in local mode.",
          controller: "dynamic_base_logical_payload"
        })
      : requireLocal<CoppeliaHealth>("/api/lab/coppelia/health")
};
