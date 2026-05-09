import { readFileSync } from "node:fs";
import { join } from "node:path";

const html = readFileSync(join("www", "index.html"), "utf8");
if (!html.includes("burn_jepa")) {
  throw new Error("www/index.html should identify burn_jepa");
}
