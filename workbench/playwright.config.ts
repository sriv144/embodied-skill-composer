import { defineConfig, devices } from "@playwright/test";

const inCi = Boolean(process.env.CI);
const browserChannel =
  process.env.PLAYWRIGHT_BUNDLED === "true" ? undefined : "chrome";

export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  globalTeardown: "./tests/e2e/global-teardown.ts",
  // The loopback lab is deliberately single-user in v1, so browser flows run
  // serially even though they cover four desktop/mobile runtime projects.
  fullyParallel: false,
  forbidOnly: inCi,
  retries: inCi ? 2 : 0,
  workers: 1,
  reporter: inCi
    ? [
        [
          "html",
          { open: "never", outputFolder: "../output/playwright/report" }
        ],
        ["list"]
      ]
    : "list",
  outputDir: "../output/playwright/test-results",
  expect: {
    timeout: 10_000
  },
  use: {
    channel: browserChannel,
    reducedMotion: "reduce",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure"
  },
  projects: [
    {
      name: "local-desktop",
      testMatch: /local\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        baseURL: "http://127.0.0.1:4173"
      }
    },
    {
      name: "local-mobile",
      testMatch: /local\.spec\.ts/,
      use: {
        ...devices["Pixel 7"],
        baseURL: "http://127.0.0.1:4173"
      }
    },
    {
      name: "static-desktop",
      testMatch: /static\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        baseURL: "http://127.0.0.1:4174"
      }
    },
    {
      name: "static-mobile",
      testMatch: /static\.spec\.ts/,
      use: {
        ...devices["Pixel 7"],
        baseURL: "http://127.0.0.1:4174"
      }
    }
  ]
});
