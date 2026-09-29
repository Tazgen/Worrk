// Render logos with the JS port, using the same list/options as run_all.sh.
// Usage: node test/render.mjs <outdir> [name-filter]
import fs from "node:fs";
import { PNG } from "pngjs";
import { stylize, hexRgb } from "../src/stylize.js";

const [outDir, filter] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const sh = fs.readFileSync(new URL("../../run_all.sh", import.meta.url), "utf8");
const list = sh.split("<<'LIST'")[1].split("\nLIST")[0].trim().split("\n");
for (const line of list) {
  const [src, dst, ...args] = line.trim().split(/\s+/);
  const name = dst.split("/").pop().replace(".png", "");
  if (filter && !name.includes(filter)) continue;
  const o = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--color") o.color = hexRgb(args[++i]);
    else if (a === "--tint") o.tint = hexRgb(args[++i]);
    else if (a === "--blend") o.blend = +args[++i];
    else if (a === "--knockout-dark") o.knockoutDark = true;
    else if (a === "--knockout-thin-dark") o.knockoutThinDark = +args[++i];
    else if (a === "--knockout-light") o.knockoutLight = +args[++i];
    else if (a === "--split-colors") o.splitColors = true;
    else throw new Error("unknown option " + a);
  }
  const png = PNG.sync.read(fs.readFileSync(new URL("../../" + src, import.meta.url)));
  const t = Date.now();
  const res = stylize({ w: png.width, h: png.height, data: new Uint8ClampedArray(png.data) }, o);
  const out = new PNG({ width: res.w, height: res.h });
  out.data = Buffer.from(res.data.buffer);
  fs.writeFileSync(`${outDir}/${name}.png`, PNG.sync.write(out));
  console.log(name, ((Date.now() - t) / 1000).toFixed(1) + "s", res.w + "x" + res.h);
}
