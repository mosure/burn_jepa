import { chromium } from "@playwright/test";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const crateDir = path.resolve(import.meta.dirname, "..");
const repoDir = path.resolve(crateDir, "../..");
const apiDir = path.resolve(
  process.env.BURN_JEPA_WASM_API_DIR || path.join(repoDir, "target/burn-jepa-wasm-api"),
);
const modelDir = path.resolve(
  process.env.BURN_JEPA_WASM_MODEL_DIR || path.join(repoDir, "target/burn-jepa-wasm-model"),
);
const modelManifestUrl =
  process.env.BURN_JEPA_WASM_MODEL_MANIFEST_URL ||
  (process.env.BURN_JEPA_WASM_MODEL_BASE_URL
    ? new URL("manifest.json", withTrailingSlash(process.env.BURN_JEPA_WASM_MODEL_BASE_URL)).href
    : null);

for (const file of [path.join(apiDir, "out/burn_jepa.js"), path.join(apiDir, "out/burn_jepa_bg.wasm")]) {
  if (!fs.existsSync(file)) {
    throw new Error(`missing wasm API smoke fixture: ${file}`);
  }
}
if (!modelManifestUrl && !fs.existsSync(path.join(modelDir, "manifest.json"))) {
  throw new Error(`missing wasm API smoke fixture: ${path.join(modelDir, "manifest.json")}`);
}

const expectedEmbedDim = await readExpectedEmbedDim(modelManifestUrl, modelDir);
const badConsoleNeedles = [
  "condvar wait not supported",
  "cannot recursively acquire mutex",
  "usage (Storage(read-write)|Storage(read-only))",
  "CubeCL Tasks Encoder",
  "Invalid CommandBuffer",
];

const server = http.createServer((request, response) => {
  const url = new URL(request.url || "/", "http://127.0.0.1");
  const root = url.pathname.startsWith("/model/") ? modelDir : apiDir;
  const relativePath = url.pathname.startsWith("/model/")
    ? url.pathname.slice("/model/".length)
    : url.pathname.slice(1);
  const filePath = path.resolve(root, decodeURIComponent(relativePath || "index.html"));
  if (!(filePath === root || filePath.startsWith(`${root}${path.sep}`)) || !fs.existsSync(filePath)) {
    response.writeHead(404);
    response.end("missing");
    return;
  }
  response.writeHead(200, {
    "Cache-Control": "no-store",
    "Content-Type": contentType(filePath),
  });
  fs.createReadStream(filePath).pipe(response);
});

const port = await listen(server);
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu"] });
try {
  const page = await browser.newPage();
  const pageErrors = [];
  const consoleLines = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    const line = message.text();
    consoleLines.push(line);
    console.log(`browser: ${line}`);
  });

  await page.goto(`http://127.0.0.1:${port}/out/burn_jepa.js`, {
    waitUntil: "domcontentloaded",
  });
  const summary = await page.evaluate(async ({ modelManifestUrl, expectedEmbedDim }) => {
    console.log("wasm-api-smoke: importing wasm bindings");
    const mod = await import("/out/burn_jepa.js");
    await mod.default();
    let model;
    if (modelManifestUrl) {
      console.log(`wasm-api-smoke: constructing model from remote manifest ${modelManifestUrl}`);
      model = await mod.WasmVJepa.createFromManifestUrl(modelManifestUrl);
    } else {
      console.log("wasm-api-smoke: fetching model manifest");
      const manifest = await (await fetch("/model/manifest.json")).text();
      const manifestJson = JSON.parse(manifest);
      const partsManifestPath =
        manifestJson.parts_manifest ||
        manifestJson.partsManifest ||
        `${manifestJson.burnpack || "jepa.bpk"}.parts.json`;
      console.log(`wasm-api-smoke: fetching parts manifest ${partsManifestPath}`);
      const partsManifest = await (await fetch(`/model/${partsManifestPath}`)).json();
      const parts = [];
      for (const [index, part] of partsManifest.parts.entries()) {
        console.log(`wasm-api-smoke: fetching shard ${index + 1}/${partsManifest.parts.length}`);
        parts.push(new Uint8Array(await (await fetch(`/model/${part.path}`)).arrayBuffer()));
      }
      console.log("wasm-api-smoke: constructing model from bpk parts");
      model = await mod.WasmVJepa.createFromBpkParts(manifest, parts);
    }
    console.log("wasm-api-smoke: running RGBA embedding summary");
    const batch = 1;
    const frames = 4;
    const height = 32;
    const width = 32;
    const rgba = new Uint8Array(batch * frames * height * width * 4);
    for (let index = 0; index < rgba.length; index += 4) {
      const pixel = index / 4;
      rgba[index] = pixel % 256;
      rgba[index + 1] = (pixel >> 2) % 256;
      rgba[index + 2] = 127;
      rgba[index + 3] = 255;
    }
    const summary = JSON.parse(await model.embedRgbaSummaryJson(rgba, batch, frames, height, width));
    summary.expectedEmbedDim = expectedEmbedDim;
    return summary;
  }, { modelManifestUrl, expectedEmbedDim });

  if (pageErrors.length > 0) {
    throw new Error(`wasm page errors: ${pageErrors.join("; ")}`);
  }
  assertNoBadConsole(consoleLines, pageErrors);
  assertArrayEqual(summary.shape, [1, 8, summary.expectedEmbedDim], "summary.shape");
  assertArrayEqual(summary.grid, [2, 2, 2], "summary.grid");
  if (summary.sample_count <= 0 || !Number.isFinite(summary.sample_mean)) {
    throw new Error(`invalid token summary: ${JSON.stringify(summary)}`);
  }
  console.log(JSON.stringify(summary));
} finally {
  await browser.close();
  await close(server);
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("failed to bind smoke server"));
        return;
      }
      resolve(address.port);
    });
  });
}

function close(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

function assertArrayEqual(actual, expected, label) {
  if (
    !Array.isArray(actual) ||
    actual.length !== expected.length ||
    actual.some((value, index) => value !== expected[index])
  ) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertNoBadConsole(consoleLines, pageErrors) {
  const output = `${consoleLines.join("\n")}\n${pageErrors.join("\n")}`;
  for (const needle of badConsoleNeedles) {
    if (output.includes(needle)) {
      throw new Error(`unexpected wasm/browser error containing ${JSON.stringify(needle)}`);
    }
  }
}

async function readExpectedEmbedDim(modelManifestUrl, modelDir) {
  let manifest;
  if (modelManifestUrl) {
    manifest = await (await fetch(modelManifestUrl)).json();
  } else {
    manifest = JSON.parse(fs.readFileSync(path.join(modelDir, "manifest.json"), "utf8"));
  }
  return manifest.jepa_config?.encoder?.embed_dim ?? 32;
}

function withTrailingSlash(url) {
  return url.endsWith("/") ? url : `${url}/`;
}

function contentType(filePath) {
  switch (path.extname(filePath)) {
    case ".js":
      return "text/javascript";
    case ".json":
      return "application/json";
    case ".wasm":
      return "application/wasm";
    default:
      return "application/octet-stream";
  }
}
