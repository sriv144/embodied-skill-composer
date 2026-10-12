import AxeBuilder from "@axe-core/playwright";
import { expect, type Locator, type Page } from "@playwright/test";

export async function expectNoSeriousAccessibilityViolations(
  page: Page
): Promise<void> {
  await expect
    .poll(() =>
      page.locator(".view-stage").evaluateAll((elements) =>
        elements.every(
          (element) => Number.parseFloat(getComputedStyle(element).opacity) >= 0.999
        )
      )
    )
    .toBe(true);
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const violations = audit.violations
    .filter((item) => item.impact === "serious" || item.impact === "critical")
    .map((item) => ({
      id: item.id,
      impact: item.impact,
      help: item.help,
      nodes: item.nodes.map((node) => ({
        target: node.target,
        html: node.html,
        failureSummary: node.failureSummary
      }))
    }));

  expect(
    violations,
    `Serious or critical accessibility violations:\n${JSON.stringify(
      violations,
      null,
      2
    )}`
  ).toEqual([]);
}

export async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth
  );
  expect(overflow).toBeLessThanOrEqual(1);
}

export async function expectDocumentScrollsToBottom(page: Page): Promise<void> {
  const extent = await page.evaluate(() => ({
    innerHeight,
    scrollHeight: document.documentElement.scrollHeight
  }));
  expect(extent.scrollHeight).toBeGreaterThan(extent.innerHeight);

  await page.evaluate(() => {
    window.scrollTo(0, document.documentElement.scrollHeight);
  });
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(0);
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          Math.ceil(window.scrollY + innerHeight) >=
          document.documentElement.scrollHeight - 1
      )
    )
    .toBe(true);
}

export async function expectPageAtTop(page: Page): Promise<void> {
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
}

export async function expectNoVisibleControlOcclusion(
  page: Page
): Promise<void> {
  const occluded = await page.evaluate(() => {
    const selector = [
      "button:not([disabled])",
      "a[href]",
      "input:not([disabled]):not([type='hidden'])",
      "select:not([disabled])",
      "[role='button'][tabindex='0']"
    ].join(",");
    return [...document.querySelectorAll<HTMLElement | SVGElement>(selector)]
      .flatMap((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          Number(style.opacity) <= 0 ||
          rect.width <= 2 ||
          rect.height <= 2
        ) {
          return [];
        }
        let left = Math.max(0, rect.left);
        let right = Math.min(innerWidth, rect.right);
        let topEdge = Math.max(0, rect.top);
        let bottom = Math.min(innerHeight, rect.bottom);
        const clipping = new Set(["auto", "scroll", "hidden", "clip"]);
        for (
          let ancestor = element.parentElement;
          ancestor;
          ancestor = ancestor.parentElement
        ) {
          const ancestorStyle = getComputedStyle(ancestor);
          const ancestorRect = ancestor.getBoundingClientRect();
          if (clipping.has(ancestorStyle.overflowX)) {
            left = Math.max(left, ancestorRect.left);
            right = Math.min(right, ancestorRect.right);
          }
          if (clipping.has(ancestorStyle.overflowY)) {
            topEdge = Math.max(topEdge, ancestorRect.top);
            bottom = Math.min(bottom, ancestorRect.bottom);
          }
        }
        if (right - left <= 2 || bottom - topEdge <= 2) return [];
        const x = (left + right) / 2;
        const y = (topEdge + bottom) / 2;
        const top = document.elementFromPoint(x, y);
        if (
          !top ||
          top === element ||
          element.contains(top) ||
          top.contains(element) ||
          element instanceof SVGElement
        ) {
          return [];
        }
        return [
          element.getAttribute("aria-label") ||
            element.textContent?.trim().slice(0, 80) ||
            element.tagName
        ];
      });
  });
  expect(occluded, `Visible controls obscured at their centre: ${occluded.join(", ")}`)
    .toEqual([]);
}

export async function expectVisibleFocusIndicator(
  locator: Locator
): Promise<void> {
  await locator.scrollIntoViewIfNeeded();
  await locator.focus();
  await expect(locator).toBeFocused();
  const indicator = await locator.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      boxShadow: style.boxShadow,
      outlineStyle: style.outlineStyle,
      outlineWidth: style.outlineWidth
    };
  });
  const outlined =
    indicator.outlineStyle !== "none" &&
    Number.parseFloat(indicator.outlineWidth) > 0;
  expect(
    outlined || indicator.boxShadow !== "none",
    `Focused element has no visible outline or box shadow: ${JSON.stringify(indicator)}`
  ).toBe(true);
}
