import { describe, expect, it } from "vitest";
import {
  developmentSecurityHeaders,
  previewSecurityHeaders
} from "./securityHeaders";

describe("Vite security response headers", () => {
  it("denies framing and constrains preview resources to the same origin", () => {
    expect(previewSecurityHeaders).toEqual(
      expect.objectContaining({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer"
      })
    );
    expect(previewSecurityHeaders["Content-Security-Policy"]).toContain(
      "connect-src 'self'"
    );
    expect(previewSecurityHeaders["Content-Security-Policy"]).toContain(
      "frame-ancestors 'none'"
    );
    const previewScriptDirective = previewSecurityHeaders[
      "Content-Security-Policy"
    ]
      .split("; ")
      .find((directive) => directive.startsWith("script-src"));
    expect(previewScriptDirective).not.toContain(
      "'unsafe-inline'"
    );
    expect(previewSecurityHeaders["Content-Security-Policy"]).toContain(
      "script-src 'self' 'wasm-unsafe-eval'"
    );
    expect(previewSecurityHeaders["Content-Security-Policy"]).not.toContain(
      " 'unsafe-eval'"
    );
  });

  it("permits only the Vite development preamble concession", () => {
    expect(developmentSecurityHeaders["Content-Security-Policy"]).toContain(
      "script-src 'self' 'wasm-unsafe-eval' 'unsafe-inline'"
    );
    expect(developmentSecurityHeaders["Content-Security-Policy"]).toContain(
      "frame-ancestors 'none'"
    );
    expect(developmentSecurityHeaders["X-Frame-Options"]).toBe("DENY");
  });
});
