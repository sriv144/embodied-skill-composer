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

async function auditPrimaryView(page: Page): Promise<void> {
  await expect(
    page.getByRole("heading", { name: "Cedar Ridge Modular Cottage" })
  ).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await expectNoVisibleControlOcclusion(page);
  await expectNoSeriousAccessibilityViolations(page);
}

test("static workbench is visibly read-only across the complete journey", async ({
  page
}) => {
  test.setTimeout(90_000);
  const apiRequests: string[] = [];
  const cspViolations = watchForCspViolations(page);
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/")) apiRequests.push(url.pathname);
  });

  const documentResponse = await page.goto("/#/design");
  expectHardenedDocumentHeaders(documentResponse);
  await expect(
    page.getByRole("heading", { name: "Orthogonal floor-plan editor" })
  ).toBeVisible();
  await expect(
    page.getByLabel("Design inspector").getByText("Read-only preview", {
      exact: true
    })
  ).toBeVisible();
  await expect(
    page.getByRole("spinbutton", { name: "Footprint width in metres" })
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Compile build plan" })
  ).toBeDisabled();
  const canvas = page.getByRole("application", {
    name: /Read-only floor plan/
  });
  await expectVisibleFocusIndicator(canvas);
  const northWall = page.getByRole("button", { name: /^Wall north,/ });
  await northWall.focus();
  await page.keyboard.press("Space");
  const readOnlyHandle = page.getByRole("button", {
    name: /Resize end endpoint of wall north/
  });
  await expect(readOnlyHandle).toHaveAttribute("tabindex", "-1");
  const wallCount = await page.getByRole("button", { name: /^Wall / }).count();
  await page.keyboard.press("Delete");
  await expect(page.getByRole("button", { name: /^Wall / })).toHaveCount(
    wallCount
  );
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
  await expect(page.getByRole("button", { name: "Robot offline" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Drop payload" })).toHaveCount(0);
  await expect(
    page.getByText("This preview ships only the obstacle recovery trace.")
  ).toBeVisible();
  await page.getByRole("button", { name: "Obstacle" }).click();
  await expect(
    page.getByText("Obstacle inserted; schedule recovered.")
  ).toBeVisible();
  await auditPrimaryView(page);

  await page.getByRole("button", { name: "Experiments" }).click();
  await expect(
    page.getByRole("heading", { name: "Durable experiment registry" })
  ).toBeVisible();
  await expect(page.getByText("Read-only public preview", { exact: true }))
    .toBeVisible();
  await expect(page.getByRole("button", { name: "Queue matrix" })).toBeDisabled();
  await auditPrimaryView(page);

  await page.getByRole("button", { name: "Results" }).click();
  await expect(
    page.getByRole("heading", { name: "Makespan comparison" })
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Learned coordination" })
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
  await expect(
    page.getByText(/no canonical 20-run research bundle was supplied/i)
  ).toBeVisible();
  await expect(page.getByText(/does not claim arm, gripper, payload contact/i))
    .toBeVisible();
  if ((page.viewportSize()?.width ?? 0) <= 720) {
    await expectDocumentScrollsToBottom(page);
    await page.getByRole("button", { name: "Design" }).click();
    await expect(
      page.getByRole("heading", { name: "Orthogonal floor-plan editor" })
    ).toBeVisible();
    await expectPageAtTop(page);
  }
  await auditPrimaryView(page);

  expect(apiRequests).toEqual([]);
  expect(cspViolations).toEqual([]);
});
