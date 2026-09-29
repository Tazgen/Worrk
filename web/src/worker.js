// Runs the styler off the main thread so the page stays responsive.
import { stylize } from "./stylize.js";

self.onmessage = (e) => {
  const { id, img, options } = e.data;
  try {
    const t = performance.now();
    const res = stylize(img, options);
    self.postMessage({ id, res, ms: performance.now() - t }, [res.data.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
