import type { Project, Trace } from "./types";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  return response.json() as Promise<T>;
}

export const api = {
  project: () => fetch("/api/project").then(readJson<Project>),
  trace: (controller: string) => fetch(`/api/traces/${controller}`).then(readJson<Trace>),
  disrupt: (controller: string, failureType: string, timestamp: number) =>
    fetch(`/api/traces/${controller}/disrupt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ failure_type: failureType, timestamp_s: timestamp })
    }).then(readJson<Trace>),
  report: () => fetch("/api/report").then(readJson<{ markdown: string }>),
  rebuild: (design: Project["design"]) =>
    fetch("/api/design/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design })
    }).then(readJson<Project>),
  parseFloorPlan: (file: File, width: number) => {
    const data = new FormData();
    data.append("file", file);
    return fetch(`/api/intent/parse?known_width_m=${width}`, { method: "POST", body: data }).then(
      readJson<Project["design"]["floor_plan"]>
    );
  }
};
