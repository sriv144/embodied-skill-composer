import { expect, test, type Page } from "@playwright/test";
import type { ArtifactReference, LabEvaluation, LabPolicy, LabRun, MatrixLaunchResponse, RunEventEnvelope } from "../../src/types";
import {
  expectDocumentScrollsToBottom,
  expectNoHorizontalOverflow,
  expectNoSeriousAccessibilityViolations,
  expectNoVisibleControlOcclusion,
  expectPageAtTop,
  expectVisibleFocusIndicator
} from "./accessibility";
import {
  expectHardenedDocumentHeaders,
  watchForCspViolations
} from "./security";

test.describe.configure({ mode: "serial" });

async function auditPrimaryView(
  page: Page,
  options: { projectHeader?: boolean } = {}
): Promise<void> {
  if (options.projectHeader !== false) {
    await expect(
      page.getByRole("heading", { name: "Cedar Ridge Modular Cottage" })
    ).toBeVisible();
  }
  await expectNoHorizontalOverflow(page);
  await expectNoVisibleControlOcclusion(page);
  await expectNoSeriousAccessibilityViolations(page);
}

async function pressRepeated(
  page: Page,
  key: string,
  count: number
): Promise<void> {
  for (let index = 0; index < count; index += 1) {
    await page.keyboard.press(key);
  }
}

test("local design-to-results flow proves editing, durable controls, and evidence", async ({
  page
}, testInfo) => {
  test.setTimeout(240_000);
  const cspViolations = watchForCspViolations(page);
  const documentResponse = await page.goto("/#/design");
  expectHardenedDocumentHeaders(documentResponse);
  await expect(
    page.getByRole("heading", { name: "Orthogonal floor-plan editor" })
  ).toBeVisible();
  await expect(page.getByText("Lab connected")).toBeVisible();

  const canvas = page.getByRole("application", {
    name: /Editable floor plan/
  });
  await expectVisibleFocusIndicator(canvas);

  const northWall = page.getByRole("button", { name: /^Wall north,/ });
  await northWall.focus();
  await page.keyboard.press("Enter");
  await expect(northWall).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("button", {
      name: /Resize end endpoint of wall north/
    })
  ).toBeVisible();

  await canvas.focus();
  await page.keyboard.press("Escape");
  await page.keyboard.press("w");
  await page.keyboard.press("Enter");
  await pressRepeated(page, "ArrowRight", 12);
  await page.keyboard.press("Enter");

  let createdWall = page.getByRole("button", { name: /^Wall wall_1,/ });
  await expect(createdWall).toHaveAccessibleName(/3\.00 metres/);
  await expect(page.getByText("Approval required", { exact: true })).toBeVisible();
  await createdWall.focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowUp");

  let wallEnd = page.getByRole("button", {
    name: /Resize end endpoint of wall wall_1/
  });
  await wallEnd.focus();
  await page.keyboard.press("ArrowRight");
  createdWall = page.getByRole("button", { name: /^Wall wall_1,/ });
  await expect(createdWall).toHaveAccessibleName(/3\.25 metres/);

  await createdWall.focus();
  await page.keyboard.press("d");
  await page.keyboard.press("Enter");
  let createdDoor = page.getByRole("button", { name: /^door door_1,/i });
  await expect(createdDoor).toHaveCount(1);
  await createdDoor.focus();
  await page.keyboard.press("ArrowRight");
  const doorEnd = page.getByRole("button", {
    name: /Resize end edge of door door_1/
  });
  await doorEnd.focus();
  await page.keyboard.press("ArrowRight");
  createdDoor = page.getByRole("button", { name: /^door door_1,/i });
  await expect(createdDoor).toHaveAccessibleName(/1\.50 metres wide/);
  await createdDoor.focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("Delete");
  await expect(
    page.getByRole("button", { name: /^door door_1,/i })
  ).toHaveCount(0);

  createdWall = page.getByRole("button", { name: /^Wall wall_1,/ });
  await createdWall.focus();
  await page.keyboard.press("Space");
  wallEnd = page.getByRole("button", {
    name: /Resize end endpoint of wall wall_1/
  });
  await wallEnd.focus();
  await page.keyboard.press("Shift+ArrowRight");
  await page.keyboard.press("Shift+ArrowRight");
  await expect(page.getByText(/extends outside the footprint/i)).toBeVisible();
  const approve = page.getByRole("button", { name: "Approve design" });
  const compile = page.getByRole("button", { name: "Compile build plan" });
  await expect(approve).toBeDisabled();
  await expect(compile).toBeDisabled();

  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("Shift+ArrowLeft");
  await expect(page.getByText("Ready", { exact: true })).toBeVisible();
  createdWall = page.getByRole("button", { name: /^Wall wall_1,/ });
  await createdWall.focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("Delete");
  await expect(
    page.getByRole("button", { name: /^Wall wall_1,/ })
  ).toHaveCount(0);

  await expect(approve).toBeEnabled();
  await approve.click();
  await expect(
    page.getByRole("button", { name: "Design approved" })
  ).toBeDisabled();

  await expect(compile).toBeEnabled();
  await compile.click();
  await expect(
    page.getByText("Approved design compiled into a new build plan.")
  ).toBeVisible({ timeout: 30_000 });
  await auditPrimaryView(page);

  await page.getByRole("button", { name: "Brain" }).click();
  await expect(
    page.getByRole("heading", { name: "Task graph and fleet allocation" })
  ).toBeVisible();
  await auditPrimaryView(page);

  await page.getByRole("button", { name: "Simulate" }).click();
  await expect(
    page.getByRole("heading", { name: "Decision epoch" })
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "CP-SAT" })
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Obstacle" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Robot offline" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Drop payload" })).toBeVisible();
  await auditPrimaryView(page);

  await page.getByRole("button", { name: "Experiments" }).click();
  await expect(
    page.getByRole("heading", { name: "Durable experiment registry" })
  ).toBeVisible();
  await expect(page.getByText("Local lab controls enabled")).toBeVisible();
  const resumeId = `e2e-resumable-${testInfo.project.name}`;
  await page.getByRole("button", { name: "Single", exact: true }).click();
  const beforeResume = await (await page.request.get(`/api/lab/runs/${resumeId}`)).json() as LabRun;
  expect(beforeResume.config.device).toBe("cpu");
  expect(beforeResume.config.profile).toBe("unit");
  expect(beforeResume.latest_checkpoint).toBeTruthy();
  // A retry can reuse a genuinely completed job, but never an unexecuted fixture.
  if (beforeResume.status !== "completed") {
    expect(beforeResume.status).toBe("interrupted");
    expect(beforeResume.progress).toBe(0.5);
    const resume = page.getByRole("button", { name: `Resume ${resumeId}` });
    await expect(resume).toBeVisible();
    await resume.click();
    await expect(
      page.getByText(`Run ${resumeId} is resuming.`)
    ).toBeVisible();
  }
  await expect.poll(async () => {
    const run = await (await page.request.get(`/api/lab/runs/${resumeId}`)).json() as LabRun;
    expect(run.status, run.error ?? "CPU resume must not fail").not.toBe("failed");
    return run.status;
  }, { timeout: 90_000, intervals: [500, 1_000, 2_000] }).toBe("completed");
  await expect(page.getByRole("progressbar", { name: `${resumeId} progress` })).toHaveAttribute("aria-valuenow", "100");
  const events = await (await page.request.get(`/api/lab/runs/${resumeId}/events`)).json() as RunEventEnvelope[];
  expect(events.some((event) => event.payload.event === "training_resumed" && event.payload.transitions === 32)).toBe(true);
  expect(events.some((event) => event.payload.event === "training_completed")).toBe(true);
  const artifacts = await (await page.request.get(`/api/lab/runs/${resumeId}/artifacts`)).json() as ArtifactReference[];
  const manifestReference = artifacts.find((item) => item.path === "policy_manifest.json");
  expect(manifestReference).toBeTruthy();
  expect(artifacts.some((item) => item.path === "actor.onnx")).toBe(true);
  const manifestResponse = await page.request.get(manifestReference!.href);
  expect(manifestResponse.ok()).toBe(true);
  const manifest = await manifestResponse.json();
  expect(manifest.transition_count).toBe(64);
  expect(manifest.resume_provenance.run_id).toBe(resumeId);
  expect(manifest.configuration_digest).toMatch(/^[a-f0-9]{64}$/);
  const policies = await (await page.request.get("/api/lab/policies")).json() as LabPolicy[];
  expect(policies.some((item) => item.id === manifest.policy_id)).toBe(true);
  await expect(page.getByText(manifest.policy_id, { exact: true })).toBeVisible();

  // Diagnostic validation seed only: this is not held-out model selection or a release gate.
  const launchEvaluation = await page.request.post("/api/lab/evaluations", { data: {
    seeds: [800], controllers: [manifest.controller], policy_ids: { [manifest.controller]: manifest.policy_id }, include_failures: false
  } });
  expect(launchEvaluation.status()).toBe(202);
  const { run_id: evaluationRunId } = await launchEvaluation.json();
  await expect.poll(async () => {
    const run = await (await page.request.get(`/api/lab/runs/${evaluationRunId}`)).json() as LabRun;
    expect(run.status, run.error ?? "CPU diagnostic evaluation must not fail").not.toBe("failed");
    return run.status;
  }, { timeout: 90_000, intervals: [500, 1_000, 2_000] }).toBe("completed");
  const evaluationRunArtifacts = await (await page.request.get(`/api/lab/runs/${evaluationRunId}/artifacts`)).json() as ArtifactReference[];
  const suiteReference = evaluationRunArtifacts.find((item) => item.path === "evaluation.json");
  expect(suiteReference).toBeTruthy();
  const suite = await (await page.request.get(suiteReference!.href)).json();
  expect(suite.seeds).toEqual([800]);
  expect(suite.controllers).toEqual([manifest.controller]);
  expect(suite.episodes).toHaveLength(1);
  expect(suite.episodes[0].split).toBe("validation");
  const records = await (await page.request.get("/api/lab/evaluations")).json() as LabEvaluation[];
  const record = records.find((item) => item.id === suite.evaluation_id);
  expect(record?.matrix_id).toBeNull();
  expect(record?.split).toBe("diagnostic");
  await page.getByRole("button", { name: "Matrix", exact: true }).click();

  let observedRunSockets = 0;
  let receivedRunFrames = 0;
  page.on("websocket", (socket) => {
    if (!socket.url().includes("/events/ws")) return;
    observedRunSockets += 1;
    socket.on("framereceived", () => {
      receivedRunFrames += 1;
    });
  });
  const matrixProfile =
    testInfo.project.name === "local-desktop" ? "unit" : "smoke";
  await page.getByRole("button", {
    name: matrixProfile,
    exact: true
  }).click();
  await page
    .getByRole("checkbox", { name: "Approve serial matrix compute" })
    .check();
  const matrixResponse = page.waitForResponse((response) => response.url().endsWith("/api/lab/experiment-matrices") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Queue matrix" }).click();
  const matrix = await (await matrixResponse).json() as MatrixLaunchResponse;
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: new RegExp(`20-run ${matrixProfile} matrix queued\\.`) })
  ).toBeVisible();
  if (testInfo.project.name === "local-desktop") {
    await expect
      .poll(() => observedRunSockets, { timeout: 30_000 })
      .toBeGreaterThan(0);
    await expect
      .poll(() => receivedRunFrames, { timeout: 30_000 })
      .toBeGreaterThan(0);
  }
  await expect(
    page.getByRole("heading", { name: "Live worker activity" })
  ).toBeVisible();

  const cancelButtons = page.getByRole("button", { name: /^Cancel / });
  await expect(cancelButtons.first()).toBeVisible();
  const cancel = cancelButtons.last();
  const cancelLabel = await cancel.getAttribute("aria-label");
  await cancel.click();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: `Cancellation requested for ${cancelLabel?.slice(7)}.` })
  ).toBeVisible();
  await auditPrimaryView(page);

  // Bound CI compute: controls/socket proof does not require training all 20 jobs.
  await Promise.all(matrix.run_ids.map((runId) => page.request.post(`/api/lab/runs/${runId}/cancel`)));
  await expect.poll(async () => {
    const runs = await Promise.all(matrix.run_ids.map(async (runId) => (await (await page.request.get(`/api/lab/runs/${runId}`)).json()) as LabRun));
    return runs.every((run) => ["completed", "cancelled", "failed", "interrupted"].includes(run.status));
  }, { timeout: 30_000 }).toBe(true);

  await page.getByRole("button", { name: "Results" }).click();
  await expect(
    page.getByRole("heading", { name: "Makespan comparison" })
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "CoppeliaSim boundary" })
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /learning curves/i })
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Per-seed evidence" })
  ).toBeVisible();
  await expect(page.getByText("Nominal gate", { exact: true })).toBeVisible();
  await expect(page.getByText("Recovery gate", { exact: true })).toBeVisible();
  await expect(page.getByText("logical transport")).toBeVisible();
  const localRecord = page.getByRole("article", { name: `Local evaluation ${suite.evaluation_id}` });
  await expect(localRecord).toBeVisible();
  await expect(localRecord).toContainText("Scenario seeds: 800 · 1 episodes · diagnostic");
  await expect(page.getByText(/unit evaluations do not satisfy the frozen research matrix/)).toBeVisible();
  const reportLink = localRecord.getByRole("link", { name: /report/i });
  await expect(reportLink).toBeVisible();
  const downloadedReport = await page.request.get((await reportLink.getAttribute("href"))!);
  expect(downloadedReport.ok()).toBe(true);
  expect(await downloadedReport.text()).toContain(suite.evaluation_id);

  if ((page.viewportSize()?.width ?? 0) <= 720) {
    await expectDocumentScrollsToBottom(page);
    await page.getByRole("button", { name: "Design" }).click();
    await expect(
      page.getByRole("heading", { name: "Orthogonal floor-plan editor" })
    ).toBeVisible();
    await expectPageAtTop(page);
  }
  await auditPrimaryView(page);
  expect(cspViolations).toEqual([]);
});

test("a failed post-compile trace refresh is visible and cannot leave the previous replay", async ({ page }) => {
  await page.goto("/#/design");
  const compile = page.getByRole("button", { name: "Compile build plan" });
  await expect(compile).toBeEnabled();
  await page.route("**/api/traces/*", (route) => route.abort("failed"));
  const rebuilt = page.waitForResponse((response) => response.url().endsWith("/api/design/rebuild") && response.request().method() === "POST");
  await compile.click();
  expect((await rebuilt).ok()).toBe(true);
  const alert = page.getByRole("alert");
  await expect(alert.getByRole("heading", { name: "Workbench unavailable" })).toBeVisible();
  await expect(alert).toContainText(/Could not reach|Failed to fetch/i);
  await expect(page.getByRole("heading", { name: "Decision epoch" })).toHaveCount(0);
  await expect(page.getByText("Read-only public preview")).toHaveCount(0);
});

test("local API failure is shown and never replaced by preview data", async ({
  page
}) => {
  await page.route("**/api/**", (route) => route.abort("failed"));
  await page.goto("/#/simulate");

  const alert = page.getByRole("alert");
  await expect(
    alert.getByRole("heading", { name: "Workbench unavailable" })
  ).toBeVisible();
  await expect(alert).toContainText(/Could not reach|Failed to fetch/i);
  await expect(page.getByText("Read-only public preview")).toHaveCount(0);
  await auditPrimaryView(page, { projectHeader: false });
});
