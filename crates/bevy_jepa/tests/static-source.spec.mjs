import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { strict as assert } from "node:assert";

const root = new URL("..", import.meta.url).pathname;
const index = readFileSync(join(root, "www", "index.html"), "utf8");
const manifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));

assert(index.includes("#bevy"), "index.html should mount the Bevy canvas");
assert(index.includes("./out/bevy_jepa.js"), "index.html should load wasm-bindgen output");
assert(index.includes("bevy_jepa running"), "index.html should expose runtime status");
assert(manifest.scripts["build:wasm"].includes("--out-name bevy_jepa"));
assert(existsSync(join(root, "src", "lib.rs")), "viewer source exists");
