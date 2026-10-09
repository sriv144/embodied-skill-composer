import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash, webcrypto } from "node:crypto";
import { TextEncoder } from "node:util";
import {
  api,
  ApiRequestError,
  runtimeStatusFor,
  verifiedStaticRuntime,
  safeStaticArtifactPath,
  staticRecoveryTracePath
} from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("runtime contract", () => {
  it("requires hash-verified metadata before assigning a static channel", async () => {
    vi.stubGlobal("crypto", webcrypto);
    vi.stubGlobal("TextEncoder", TextEncoder);
    expect(runtimeStatusFor("static").channel).toBeNull();
    for (const channel of ["preview", "release"] as const) {
      const { provenance, statusText } = staticMetadata(channel);
      const status = await verifiedStaticRuntime(provenance, statusText);
      expect(status.channel).toBe(channel);
      expect(status.label).toContain(channel);
      expect(status.readOnly).toBe(true);
      expect(Object.isFrozen(status)).toBe(true);
    }
  });

  it("rejects tampered metadata and incomplete canonical claims", async () => {
    vi.stubGlobal("crypto", webcrypto);
    vi.stubGlobal("TextEncoder", TextEncoder);
    const { provenance, statusText } = staticMetadata("release");
    await expect(verifiedStaticRuntime(
      provenance, statusText.replace("release_evidence", "release_evidencX")
    )).rejects.toThrow("provenance hash check");
    const incomplete = staticMetadata("release", { evidence: { research: "absent" } });
    await expect(verifiedStaticRuntime(
      incomplete.provenance, incomplete.statusText
    )).rejects.toThrow("complete canonical v1 evidence");
    await expect(verifiedStaticRuntime({}, statusText)).rejects.toThrow(
      "missing or inconsistent"
    );
  });

  it("loads static identity once and never falls back when its fetch fails", async () => {
    vi.stubEnv("VITE_STATIC_DEMO", "true");
    vi.resetModules();
    const { api: staticApi } = await import("./api");
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("metadata unavailable"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(staticApi.initializeRuntime()).rejects.toThrow("metadata unavailable");
    await expect(staticApi.initializeRuntime()).rejects.toThrow("metadata unavailable");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.every(([url]) => String(url).includes("demo/"))).toBe(true);
    expect(staticApi.mode()).toBe("static");
    expect(staticApi.runtime().channel).toBeNull();
  });

  it("is explicit, immutable, and honest about static controls", () => {
    const status = runtimeStatusFor("static");

    expect(status).toEqual(
      expect.objectContaining({
        mode: "static",
        source: "versioned-static-demo",
        readOnly: true,
        interactive: false
      })
    );
    expect(Object.isFrozen(status)).toBe(true);
    expect(staticRecoveryTracePath("obstacle")).toContain(
      "demo/traces/recovery.json"
    );
    expect(() => staticRecoveryTracePath("robot_unavailable")).toThrow(
      "only the obstacle recovery trace"
    );
    expect(() => staticRecoveryTracePath("dropped_resource")).toThrow(
      "only the obstacle recovery trace"
    );
  });

  it("does not switch a failed local request to demo data", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("connection refused"));
    vi.stubGlobal("fetch", fetchMock);

    const reason = await api.project().catch((error: unknown) => error);
    expect(reason).toBeInstanceOf(ApiRequestError);
    expect(reason).toEqual(
      expect.objectContaining({ name: "ApiRequestError", path: "/api/project" })
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("/api/project", undefined);
    expect(api.mode()).toBe("local");
  });

  it("surfaces the backend detail for a failed local request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "registry is unavailable" }), {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "application/json" }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.runs()).rejects.toThrow("503 registry is unavailable");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("/api/lab/runs", undefined);
  });

  it("renders structured backend design validation details", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: {
            valid: false,
            issues: [
              {
                code: "opening_out_of_bounds",
                path: "floor_plan.openings[0]",
                message: "opening span must fit wall length"
              },
              {
                code: "wall_overlap",
                path: "floor_plan.walls[3]",
                message: "wall overlaps floor_plan.walls[1]"
              }
            ]
          }
        }),
        {
          status: 422,
          statusText: "Unprocessable Entity",
          headers: { "Content-Type": "application/json" }
        }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const reason = await api
      .rebuild({} as Parameters<typeof api.rebuild>[0])
      .catch((error: unknown) => error);
    expect(reason).toBeInstanceOf(ApiRequestError);
    expect(reason).toEqual(
      expect.objectContaining({
        status: 422,
        path: "/api/design/rebuild"
      })
    );
    expect((reason as Error).message).toContain(
      "floor_plan.openings[0]: opening span must fit wall length"
    );
    expect((reason as Error).message).toContain(
      "floor_plan.walls[3]: wall overlaps floor_plan.walls[1]"
    );
  });

  it("renders FastAPI request-field validation details", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: [
            {
              type: "greater_than",
              loc: ["body", "design", "footprint_width_m"],
              msg: "Input should be greater than 0"
            }
          ]
        }),
        {
          status: 422,
          statusText: "Unprocessable Entity",
          headers: { "Content-Type": "application/json" }
        }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      api.rebuild({} as Parameters<typeof api.rebuild>[0])
    ).rejects.toThrow(
      "body.design.footprint_width_m: Input should be greater than 0"
    );
  });

  it("accepts only concrete loopback artifact download routes", () => {
    expect(
      api.artifactHref(
        "/api/lab/evaluations/eval-1/artifacts/report.md"
      )
    ).toBe("/api/lab/evaluations/eval-1/artifacts/report.md");
    expect(
      api.artifactHref(
        "/api/lab/coppelia/evidence/live-1/artifacts/metrics.json"
      )
    ).toBe("/api/lab/coppelia/evidence/live-1/artifacts/metrics.json");
    expect(api.artifactHref("C:\\private\\run\\report.md")).toBeNull();
    expect(api.artifactHref("https://evil.example/report.md")).toBeNull();
    expect(
      api.artifactHref(
        "/api/lab/evaluations/eval-1/artifacts/%2e%2e/secret.json"
      )
    ).toBeNull();
    expect(api.artifactHref("/api/lab/runs/run-1/artifacts")).toBeNull();
    expect(safeStaticArtifactPath("evidence/research/report.md")).toBe(
      "evidence/research/report.md"
    );
    expect(
      safeStaticArtifactPath("evidence/%2e%2e/secret.json")
    ).toBeNull();
    expect(
      safeStaticArtifactPath("evidence/%2Fetc/passwd")
    ).toBeNull();
    for (const suffix of [
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
    ]) {
      expect(safeStaticArtifactPath(`evidence/result${suffix}`)).toBe(
        `evidence/result${suffix}`
      );
    }
    for (const path of [
      "evidence/report.html",
      "evidence/icon.svg",
      "evidence/worker.js",
      "evidence/archive.zip",
      "evidence/README",
      "evidence/report%2Ehtml"
    ]) {
      expect(safeStaticArtifactPath(path)).toBeNull();
    }
  });

  it("loads evidence summaries only from the local evidence API", async () => {
    const research = {
      schema_version: "construction-intelligence-research-summary-v1",
      status: "absent",
      claim_allowed: false,
      confidence_intervals: [],
      per_training_seed: [],
      per_scenario_seed: [],
      acceptance: null,
      ablations: []
    };
    const coppelia = {
      schema_version: "construction-intelligence-coppelia-public-v1",
      status: "absent",
      claim_allowed: false,
      payload_transport: "logical",
      nominal: null,
      recovery: null,
      limitation: "logical transport"
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(research), {
          headers: { "Content-Type": "application/json" }
        })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(coppelia), {
          headers: { "Content-Type": "application/json" }
        })
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.researchSummary()).resolves.toEqual(research);
    await expect(api.coppeliaEvidence()).resolves.toEqual(coppelia);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/lab/evidence/research-summary",
      undefined
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/lab/evidence/coppelia",
      undefined
    );
  });
});

function staticMetadata(
  channel: "preview" | "release",
  overrides: Record<string, unknown> = {}
) {
  const released = channel === "release";
  const status = {
    schema_version: "construction-intelligence-public-status-v1",
    channel,
    release_ready: released,
    release_version: released ? "1.0.0" : null,
    release_tag: released ? "v1.0.0" : null,
    claim_status: released ? "release_evidence" : "fixture_preview",
    limitations: released ? [] : ["Preview fixture"],
    evidence: released
      ? { deterministic: "canonical", research: "validated", coppelia: "validated" }
      : { deterministic: "fixture", research: "absent", coppelia: "absent" },
    ...overrides
  };
  const statusText = JSON.stringify(status);
  const provenance = {
    artifact_count: 1,
    artifacts: [{
      bytes: new TextEncoder().encode(statusText).byteLength,
      path: "release-status.json",
      sha256: createHash("sha256").update(statusText, "utf8").digest("hex")
    }],
    channel,
    coverage: { complete: true, excluded_self: "provenance.json" },
    manifest_payload_sha256: null as string | null,
    release_tag: status.release_tag,
    release_version: status.release_version,
    schema_version: "construction-intelligence-public-demo-provenance-v1",
    source: { commit: "a".repeat(40), dirty: false, tree_digest: "b".repeat(64) }
  };
  provenance.manifest_payload_sha256 = createHash("sha256")
    .update(JSON.stringify(provenance, null, 2) + "\n", "utf8").digest("hex");
  return { provenance, statusText };
}
