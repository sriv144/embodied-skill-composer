import { expect, type Page, type Response } from "@playwright/test";

export function watchForCspViolations(page: Page): string[] {
  const violations: string[] = [];
  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      /content security policy|violates the following directive/i.test(
        message.text()
      )
    ) {
      violations.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    if (/content security policy/i.test(error.message)) {
      violations.push(error.message);
    }
  });
  return violations;
}

export function expectHardenedDocumentHeaders(response: Response | null): void {
  expect(response).not.toBeNull();
  const headers = response?.headers() ?? {};
  expect(headers["content-security-policy"]).toContain(
    "frame-ancestors 'none'"
  );
  expect(headers["content-security-policy"]).toContain("connect-src 'self'");
  expect(headers["x-content-type-options"]).toBe("nosniff");
  expect(headers["x-frame-options"]).toBe("DENY");
  expect(headers["referrer-policy"]).toBe("no-referrer");
}
