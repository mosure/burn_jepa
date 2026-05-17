import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const knownWasmPanicNeedles = [
  "Creating a wgpu setup synchronously is unsupported on wasm",
  "Failed to read tensor data synchronously",
  "condvar wait not supported",
  "Buffer is already mapped",
  "used in submit while mapped",
  "std::time::Instant",
];

function expectNoKnownWasmPanic(consoleLines, pageErrors) {
  const output = `${consoleLines.join("\n")}\n${pageErrors.join("\n")}`;
  for (const needle of knownWasmPanicNeedles) {
    expect(output).not.toContain(needle);
  }
}

test("boots wasm viewer in tiny smoke mode", async ({ page }) => {
  const consoleLines = [];
  const pageErrors = [];
  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          throw new Error("webcam should not be requested in static smoke mode");
        },
      },
    });
  });

  await page.goto(
    "/?source=static&load-model=false&image-size=256&static-width=256&static-height=256&static-fps=4&high-res-pca-every=100000",
    { waitUntil: "domcontentloaded" },
  );

  await expect
    .poll(async () => page.evaluate(() => window.__jepaFrameStats?.count || 0), {
      timeout: 90_000,
    })
    .toBeGreaterThanOrEqual(2);
  await expect
    .poll(async () => page.evaluate(() => window.__jepaPipelineMetrics?.completedFrames || 0), {
      timeout: 90_000,
    })
    .toBeGreaterThanOrEqual(1);

  const stats = await page.evaluate(() => window.__jepaFrameStats);
  expect(stats.lastWidth).toBe(256);
  expect(stats.lastHeight).toBe(256);
  const metrics = await page.evaluate(() => window.__jepaPipelineMetrics);
  expect(metrics.completedFrames).toBeGreaterThanOrEqual(1);
  expect(metrics.encoderSource).toBe("tiny-test");
  expect(metrics.gridHeight).toBe(16);
  expect(metrics.gridWidth).toBe(16);
  expect(metrics.contextTokens).toBeGreaterThan(0);
  expect(metrics.totalUs).toBeGreaterThan(0);
  expect(pageErrors).toEqual([]);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});

test("loads local sharded package metadata before init", async ({ page }) => {
  test.skip(
    process.env.BURN_JEPA_WASM_MODEL_E2E !== "1",
    "set BURN_JEPA_WASM_MODEL_E2E=1 and serve www/model/manifest.json or set BURN_JEPA_WASM_MODEL_DIR",
  );
  const modelDir = process.env.BURN_JEPA_WASM_MODEL_DIR;
  if (modelDir) {
    const baseDir = path.resolve(modelDir);
    await page.route("**/model/**", async (route) => {
      const url = new URL(route.request().url());
      const fileName = decodeURIComponent(url.pathname.split("/model/").at(-1) || "");
      const filePath = path.resolve(baseDir, fileName);
      if (
        !(filePath === baseDir || filePath.startsWith(`${baseDir}${path.sep}`)) ||
        !fs.existsSync(filePath)
      ) {
        await route.fulfill({ status: 404, body: "missing model fixture" });
        return;
      }
      await route.fulfill({ path: filePath });
    });
  }
  if (modelDir) {
    test.skip(
      !fs.existsSync(path.join(modelDir, "manifest.json")),
      "missing BURN_JEPA_WASM_MODEL_DIR/manifest.json",
    );
  } else {
    const response = await page.request.get("/model/manifest.json");
    test.skip(!response.ok(), "missing local www/model/manifest.json");
  }

  const consoleLines = [];
  const pageErrors = [];
  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
  });

  await page.goto(
    "/?preload-only=true&source=static&encoder-source=base-checkpoint&model-base=./model&image-size=256&static-width=256&static-height=256&static-fps=1&high-res-pca-every=100000",
    { waitUntil: "domcontentloaded" },
  );

  await expect
    .poll(async () => page.evaluate(() => window.__jepaModelPackageStats || null), {
      timeout: 120_000,
    })
    .not.toBeNull();
  const packageStats = await page.evaluate(() => window.__jepaModelPackageStats);
  expect(packageStats.parts).toBeGreaterThan(0);
  expect(packageStats.bytes).toBeGreaterThan(0);
  await expect
    .poll(async () => page.evaluate(() => window.__jepaPreloadOnly || false), {
      timeout: 120_000,
    })
    .toBe(true);
  expectNoKnownWasmPanic(consoleLines, pageErrors);
});
