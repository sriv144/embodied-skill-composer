import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { fileURLToPath } from "node:url";

const workspaceRoot = fileURLToPath(new URL("../../..", import.meta.url));
const workbenchRoot = fileURLToPath(new URL("../..", import.meta.url));
const artifactDir = fileURLToPath(
  new URL("../../../output/playwright", import.meta.url)
);
const databasePath = `${artifactDir}/workbench-e2e.sqlite`;
const processStatePath = `${artifactDir}/api-process.json`;
const processLogPath = `${artifactDir}/api-server.log`;
const localPython = `${workspaceRoot}/.venv/Scripts/python.exe`;
const python = existsSync(localPython) ? localPython : "python";
const viteBin = `${workbenchRoot}/node_modules/vite/bin/vite.js`;
const tscBin = `${workbenchRoot}/node_modules/typescript/bin/tsc`;

export default async function globalSetup(): Promise<void> {
  mkdirSync(artifactDir, { recursive: true });
  for (const path of [
    databasePath,
    `${databasePath}-shm`,
    `${databasePath}-wal`,
    processStatePath,
    processLogPath
  ]) {
    rmSync(path, { force: true });
  }

  const occupied = await Promise.all([
      endpointIsReady("http://127.0.0.1:8008/api/health"),
      endpointIsReady("http://127.0.0.1:4173"),
      endpointIsReady("http://127.0.0.1:4174")
    ]);
  if (occupied.some(Boolean)) {
    throw new Error(
      "An acceptance port (8008, 4173, or 4174) is already in use. Stop that " +
        "service so the single-user browser flow cannot mutate another session."
    );
  }

  runNode(
    [tscBin, "--noEmit", "-p", "tsconfig.app.json"],
    "TypeScript acceptance build"
  );
  runNode([viteBin, "build", "--outDir", "dist-local"], "local Vite build");
  runNode(
    [viteBin, "build", "--mode", "static", "--outDir", "dist-static"],
    "static Vite build"
  );
  seedAcceptanceRegistry();

  const processes = [
    startManaged(
      python,
      [
        `${workspaceRoot}/scripts/run_construction_api.py`,
        "--port",
        "8008",
        "--registry-path",
        databasePath
      ],
      workspaceRoot,
      processLogPath
    ),
    startManaged(
      process.execPath,
      [
        viteBin,
        "preview",
        "--host",
        "127.0.0.1",
        "--port",
        "4173",
        "--outDir",
        "dist-local"
      ],
      workbenchRoot,
      `${artifactDir}/preview-local.log`
    ),
    startManaged(
      process.execPath,
      [
        viteBin,
        "preview",
        "--host",
        "127.0.0.1",
        "--port",
        "4174",
        "--outDir",
        "dist-static"
      ],
      workbenchRoot,
      `${artifactDir}/preview-static.log`
    )
  ];
  const pids = processes
    .map((child) => child.pid)
    .filter((pid): pid is number => pid !== undefined);
  writeFileSync(
    processStatePath,
    JSON.stringify({ pids, startedAt: new Date().toISOString() })
  );

  try {
    await waitUntilReady(
      [
        "http://127.0.0.1:8008/api/health",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:4174"
      ],
      processes
    );
  } catch (error) {
    stopManaged(pids);
    rmSync(processStatePath, { force: true });
    throw error;
  }
}

function seedAcceptanceRegistry(): void {
  const script = [
    "from pathlib import Path",
    "from embodied_skill_composer.construction.lab_registry import LabRegistry",
    `registry = LabRegistry(Path(${JSON.stringify(databasePath)}))`,
    "registry.create_run(",
    "    'training',",
    "    {'seed': 7},",
    "    status='interrupted',",
    "    input_payload={},",
    "    run_id='e2e-resumable-run',",
    ")"
  ].join("\n");
  const result = spawnSync(python, ["-c", script], {
    cwd: workspaceRoot,
    env: {
      ...process.env,
      PYTHONPATH: `${workspaceRoot}/src`
    },
    stdio: "inherit",
    windowsHide: true
  });
  if (result.status !== 0) {
    throw new Error(
      `Acceptance registry fixture failed with status ${result.status}.`
    );
  }
}

function runNode(arguments_: string[], label: string): void {
  const result = spawnSync(process.execPath, arguments_, {
    cwd: workbenchRoot,
    stdio: "inherit",
    windowsHide: true
  });
  if (result.status !== 0) {
    throw new Error(`${label} failed with status ${result.status}.`);
  }
}

function startManaged(
  command: string,
  arguments_: string[],
  cwd: string,
  logPath: string
): ChildProcess {
  const log = openSync(logPath, "w");
  const child = spawn(command, arguments_, {
    cwd,
    detached: process.platform !== "win32",
    windowsHide: true,
    stdio: ["ignore", log, log]
  });
  closeSync(log);
  child.unref();
  return child;
}

async function waitUntilReady(
  urls: string[],
  processes: ChildProcess[]
): Promise<void> {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (processes.some((child) => child.exitCode !== null)) {
      throw new Error(
        `An acceptance server exited early; inspect logs in ${artifactDir}.`
      );
    }
    if ((await Promise.all(urls.map(endpointIsReady))).every(Boolean)) return;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Acceptance servers did not become ready; see ${artifactDir}.`);
}

async function endpointIsReady(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(500)
    });
    return response.ok;
  } catch {
    return false;
  }
}

function stopManaged(pids: number[]): void {
  for (const pid of pids.reverse()) {
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true
      });
    } else {
      try {
        process.kill(-pid, "SIGKILL");
      } catch {
        // The process already exited.
      }
    }
  }
}
