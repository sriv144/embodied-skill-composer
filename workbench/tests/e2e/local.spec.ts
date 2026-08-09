import { expect, test, type Page } from "@playwright/test";
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
  test.setTimeout(120_000);
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
  if (testInfo.project.name === "local-desktop") {
    await page.getByRole("button", { name: "Single", exact: true }).click();
    const resume = page.getByRole("button", {
      name: "Resume e2e-resumable-run"
    });
    await expect(resume).toBeVisible();
    await resume.click();
    await expect(
      page.getByText("Run e2e-resumable-run is resuming.")
    ).toBeVisible();
    await page.getByRole("button", { name: "Matrix", exact: true }).click();
  }

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
  await page.getByRole("button", { name: "Queue matrix" }).click();
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
