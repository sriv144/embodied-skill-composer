import { spawnSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

const processStatePath = fileURLToPath(
  new URL("../../../output/playwright/api-process.json", import.meta.url)
);

export default async function globalTeardown(): Promise<void> {
  let pids: number[] = [];
  try {
    const state = JSON.parse(readFileSync(processStatePath, "utf8")) as {
      pids?: number[];
    };
    pids = state.pids ?? [];
  } catch {
    return;
  }

  try {
    for (const pid of pids.reverse()) {
      if (process.platform === "win32") {
        spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], {
          stdio: "ignore",
          windowsHide: true
        });
      } else {
        try {
          process.kill(-pid, "SIGTERM");
          await waitForExit(pid, 3_000);
          if (isRunning(pid)) process.kill(-pid, "SIGKILL");
        } catch {
          // The process already exited.
        }
      }
    }
  } finally {
    rmSync(processStatePath, { force: true });
  }
}

async function waitForExit(pid: number, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline && isRunning(pid)) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

function isRunning(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
