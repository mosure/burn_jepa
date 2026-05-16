import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { strict as assert } from "node:assert";

const root = new URL("..", import.meta.url).pathname;
const index = readFileSync(join(root, "www", "index.html"), "utf8");
const manifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));

assert(index.includes("#bevy"), "index.html should mount the Bevy canvas");
assert(index.includes("frame_input"), "index.html should forward camera/static frames to wasm");
assert(index.includes("./out/bevy_jepa.js"), "index.html should load wasm-bindgen output");
assert(index.includes("bevy_jepa camera source running"), "index.html should expose camera runtime status");
assert(index.includes("bevy_jepa static source running"), "index.html should expose static runtime status");
assert(index.includes("getUserMedia"), "index.html should support browser camera capture");
assert(index.includes('queryNumber(["camera-width", "width"], 256'), "camera source should not request sub-256 frames by default");
assert(index.includes('queryNumber(["static-width", "width"], 256'), "static source should not generate sub-256 frames by default");
assert(index.includes('queryNumber(["static-width", "width"], 256, 3840, 512'), "static source should default to 512px frames");
assert(index.includes("usesStaticSource"), "index.html should support static-source smoke mode");
assert(index.includes("usesCameraSource"), "index.html should make camera-source selection explicit");
assert(index.includes("unsupported wasm frame source"), "index.html should reject unsupported source values");
assert(!index.includes("usesSyntheticSource"), "wasm page should not silently select generated frames");
assert(manifest.scripts["build:wasm"].includes("--out-name bevy_jepa"));
assert(existsSync(join(root, "src", "lib.rs")), "viewer source exists");
assert(existsSync(join(root, "src", "platform.rs")), "viewer camera platform source exists");
