// Render one logo under several setting combinations at preview quality.
// Usage: node test/variants.mjs <logo.png> <outdir> <themeHex>
import fs from "node:fs";
import { PNG } from "pngjs";
import { stylize, hexRgb } from "../src/stylize.js";

const [src, outDir, theme] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const png = PNG.sync.read(fs.readFileSync(src));
const img = { w: png.width, h: png.height, data: new Uint8ClampedArray(png.data) };
const t = hexRgb(theme);
const variants = {
  "1-original": {},
  "2-split": { splitColors: true },
  "3-cutdark": { knockoutDark: true },
  "4-thin+split": { knockoutThinDark: 0.014, splitColors: true },
  "5-one+cutdark": { color: t, knockoutDark: true },
  "6-shaded+thin+split": { tint: t, knockoutThinDark: 0.014, splitColors: true },
};
for (const [name, o] of Object.entries(variants)) {
  const res = stylize(img, { size: 700, ss: 1, detailScale: 700 / 1500, ...o });
  const out = new PNG({ width: res.w, height: res.h });
  out.data = Buffer.from(res.data.buffer);
  fs.writeFileSync(`${outDir}/${name}.png`, PNG.sync.write(out));
}
