import { afterEach, describe, expect, it, vi } from "vitest";
import { createHash, webcrypto } from "node:crypto";
import { TextEncoder } from "node:util";
import {
  api,
  ApiRequestError,
  runtimeStatusFor,
  verifiedStaticRuntime,
  verifiedReleaseAssetUrls,
  safeReleaseArtifactUrl,
  safeStaticArtifactPath,
  staticRecoveryTracePath
} from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("runtime contract", () => {
  it("permits only exact versioned asset URLs from the hash-verified native index", async () => {
    vi.stubGlobal("crypto", webcrypto);
    vi.stubGlobal("TextEncoder", TextEncoder);
    const { provenance, indexText } = archiveMetadata();
    const allowed = await verifiedReleaseAssetUrls(provenance, indexText);
    const base = "https://github.com/sriv144/embodied-skill-composer/releases/download/v1.0.0/";
    const filename = "construction-intelligence-v1-coppelia-evidence-part-000001.zip";
    expect(safeReleaseArtifactUrl(base + filename, allowed)).toBe(base + filename);
    for (const path of [
      base + filename + "?download=1", base + filename + "#redirect",
      base + "construction-intelligence-v1-coppelia-evidence-part-000013.zip",
      base + "../v2.0.0/" + filename, base + "%2e%2e/" + filename,
      base + filename.replace("-part-", "%2fpart-"),
      (base + filename).replace("github.com", "github.com.evil.example"),
      (base + filename).replace("https:", "http:"),
      (base + filename).replace("v1.0.0", "latest"),
      (base + filename).replace("sriv144", "untrusted")
    ]) expect(safeReleaseArtifactUrl(path, allowed)).toBeNull();
    expect(safeReleaseArtifactUrl(base + filename, new Set())).toBeNull();
    await expect(verifiedReleaseAssetUrls(provenance, indexText + " ")).rejects.toThrow(
      "not bound to provenance"
    );
    const unsafe = archiveMetadata((index) => {
      index.parts[0].file = "../../untrusted.zip";
    });
    await expect(verifiedReleaseAssetUrls(unsafe.provenance, unsafe.indexText)).rejects.toThrow(
      "inventory is invalid"
    );
    const mismatch = archiveMetadata((index) => { index.source.commit = "c".repeat(40); });
    await expect(verifiedReleaseAssetUrls(mismatch.provenance, mismatch.indexText)).rejects.toThrow(
      "source identity is inconsistent"
    );
    const brokenChunks = archiveMetadata((index) => { index.files[0].chunks[0].member = "../x"; });
    await expect(verifiedReleaseAssetUrls(
      brokenChunks.provenance, brokenChunks.indexText
    )).rejects.toThrow("chunks are invalid");
    const dirtyInput = archiveMetadata();
    dirtyInput.provenance.inputs[0].source.dirty = true;
    await expect(verifiedReleaseAssetUrls(
      dirtyInput.provenance, dirtyInput.indexText
    )).rejects.toThrow("source identity is inconsistent");
  });

  it("rejects a scenario exceeding its native reconstruction budget", async () => {
    vi.stubGlobal("crypto", webcrypto);
    vi.stubGlobal("TextEncoder", TextEncoder);
    const oversized = archiveMetadata((index) => {
      // A legal 8 GiB file plus the remaining nominal members exceeds the
      // per-scenario bound even though the whole archive remains below 16 GiB.
      index.files[0].bytes = 8 * 1024 ** 3;
      let ordinal = 0;
      const partName = (position: number) =>
        `construction-intelligence-v1-coppelia-evidence-part-${String(position + 1).padStart(6, "0")}.zip`;
      for (const file of index.files) {
        file.chunks = Array.from({ length: Math.ceil(file.bytes / index.chunk_bytes) }, (_, offset) => {
          const chunk = {
            part: partName(Math.floor(ordinal / 2)),
            member: `chunks/${file.path}/${String(offset + 1).padStart(6, "0")}.bin`,
            bytes: Math.min(index.chunk_bytes, file.bytes - offset * index.chunk_bytes),
            sha256: "e".repeat(64)
          };
          ordinal += 1;
          return chunk;
        });
      }
      index.parts = Array.from({ length: Math.ceil(ordinal / 2) }, (_, position) => ({
        file: partName(position), bytes: 1, sha256: "f".repeat(64),
        entry_count: Math.min(2, ordinal - position * 2)
      }));
    });
    await expect(verifiedReleaseAssetUrls(
      oversized.provenance, oversized.indexText
    )).rejects.toThrow("native byte budget");
  });

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

  it("loads the release index once before enabling versioned download links", async () => {
    vi.stubGlobal("crypto", webcrypto);
    vi.stubGlobal("TextEncoder", TextEncoder);
    vi.stubEnv("VITE_STATIC_DEMO", "true");
    vi.resetModules();
    const { api: staticApi } = await import("./api");
    const metadata = staticMetadata("release");
    const archive = archiveMetadata();
    const provenance = {
      ...metadata.provenance,
      coppelia_archive: archive.provenance.coppelia_archive,
      inputs: archive.provenance.inputs,
      artifacts: [...metadata.provenance.artifacts, ...archive.provenance.artifacts],
      artifact_count: 2,
      manifest_payload_sha256: null as string | null
    };
    const canonical = JSON.stringify(provenance, (_key, value: unknown) => {
      if (typeof value !== "object" || value === null || Array.isArray(value)) return value;
      const object = value as Record<string, unknown>;
      return Object.fromEntries(Object.keys(object).sort().map((key) => [key, object[key]]));
    }, 2) + "\n";
    provenance.manifest_payload_sha256 = createHash("sha256").update(canonical).digest("hex");
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("provenance.json")) return Promise.resolve(new Response(JSON.stringify(provenance)));
      if (url.endsWith("release-status.json")) return Promise.resolve(new Response(metadata.statusText));
      if (url.endsWith("archive-index.json")) return Promise.resolve(new Response(archive.indexText));
      throw new Error("Unexpected request");
    });
    vi.stubGlobal("fetch", fetchMock);
    const href = archive.provenance.coppelia_archive.asset_base_url +
      "construction-intelligence-v1-coppelia-evidence-part-000001.zip";
    expect(staticApi.artifactHref(href)).toBeNull();
    await expect(staticApi.initializeRuntime()).resolves.toMatchObject({ channel: "release" });
    await staticApi.initializeRuntime();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(staticApi.artifactHref(href)).toBe(href);
    expect(staticApi.artifactHref(href + "?redirect=1")).toBeNull();
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

function archiveMetadata(change?: (index: ReturnType<typeof archiveIndex>) => void) {
  const index = archiveIndex();
  change?.(index);
  const indexText = JSON.stringify(index);
  const sha256 = createHash("sha256").update(indexText).digest("hex");
  const provenance = {
    channel: "release", release_tag: "v1.0.0",
    coppelia_archive: {
      path: "evidence/coppelia/archive-index.json",
      asset_file: "construction-intelligence-v1-coppelia-evidence.json",
      asset_base_url: "https://github.com/sriv144/embodied-skill-composer/releases/download/v1.0.0/",
      sha256
    },
    inputs: [{
      kind: "simulator", manifest_sha256: "d".repeat(64),
      source: { commit: "a".repeat(40), dirty: false, tree_digest: "b".repeat(64) }
    }],
    artifacts: [{
      path: "evidence/coppelia/archive-index.json", sha256,
      bytes: new TextEncoder().encode(indexText).byteLength
    }]
  };
  return { provenance, indexText };
}

function archiveIndex() {
  const paths = ["nominal", "recovery"].flatMap((scenario) => [
    ...[
      "manifest.json", "scenario.json", "planned_jobs.json", "planned_vs_measured_replay.json",
      "wheel_commands.jsonl", "measured_telemetry.jsonl", "trace.json", "metrics.json",
      "report.md", "construction_intelligence.ttt", "runtime_attestation.json"
    ].map((name) => `evidence/coppelia/${scenario}/${name}`),
    `evidence/coppelia/visualizations/${scenario}.mp4`
  ]).sort();
  const partName = (position: number) =>
    `construction-intelligence-v1-coppelia-evidence-part-${String(position + 1).padStart(6, "0")}.zip`;
  return {
    schema_version: "construction-intelligence-coppelia-archive-v1",
    generated_at: "1970-01-01T00:00:00Z", chunk_bytes: 128 * 1024 ** 2,
    source: { commit: "a".repeat(40), dirty: false, tree_digest: "b".repeat(64) },
    descriptor_sha256: "d".repeat(64),
    files: paths.map((path, position) => ({
      path, bytes: 1, sha256: "e".repeat(64), chunks: [{
        part: partName(Math.floor(position / 2)), member: `chunks/${path}/000001.bin`,
        bytes: 1, sha256: "e".repeat(64)
      }]
    })),
    parts: Array.from({ length: 12 }, (_, position) => ({
      file: partName(position), bytes: 1, sha256: "f".repeat(64), entry_count: 2
    }))
  };
}

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
