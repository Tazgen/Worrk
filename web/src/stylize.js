// Textured metallic badge style for flat transparent logos.
// JavaScript port of stylize.py; runs in the browser and in Node (no DOM needed).
// Images are {w, h, data: Uint8ClampedArray RGBA}.

export const SS = 2; // supersampling factor for smooth edges

export function hexRgb(s) {
  s = s.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16));
}

// ---------- small numeric helpers ----------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normals(rand, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

function std(a) {
  let s = 0, s2 = 0;
  for (let i = 0; i < a.length; i++) { s += a[i]; s2 += a[i] * a[i]; }
  const m = s / a.length;
  return Math.sqrt(Math.max(s2 / a.length - m * m, 0));
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher).
// Returns, for every pixel, the distance to the nearest pixel where `src` is 1,
// and optionally that pixel's index.
export function edt(src, w, h, wantIndex = false) {
  const INF = 1e20;
  const n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), z = new Float64Array(n + 1);
  const v = new Int32Array(n), fi = new Int32Array(n), di = new Int32Array(n);
  const dist = new Float32Array(w * h);
  const idx = wantIndex ? new Int32Array(w * h) : null;
  const colIdx = wantIndex ? new Int32Array(w * h) : null;

  function pass1d(len) {
    let k = 0;
    v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < len; q++) {
      if (f[q] >= INF) continue;
      if (f[v[k]] >= INF) { v[k] = q; continue; }
      let s;
      while (true) {
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        if (s <= z[k] && k > 0) k--; else break;
      }
      if (s <= z[k]) { v[k] = q; continue; }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++;
      const p = v[k];
      d[q] = f[p] >= INF ? INF : (q - p) * (q - p) + f[p];
      di[q] = p;
    }
  }

  // columns
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = src[y * w + x] ? 0 : INF;
    pass1d(h);
    for (let y = 0; y < h; y++) {
      dist[y * w + x] = d[y];
      if (wantIndex) colIdx[y * w + x] = di[y] * w + x;
    }
  }
  // rows
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) f[x] = dist[o + x];
    pass1d(w);
    for (let x = 0; x < w; x++) {
      dist[o + x] = Math.sqrt(d[x]);
      if (wantIndex) idx[o + x] = colIdx[o + di[x]];
    }
  }
  return wantIndex ? { dist, idx } : dist;
}

function not(m) {
  const o = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) o[i] = m[i] ? 0 : 1;
  return o;
}

// Round morphology via distance transforms.
function dilate(m, w, h, r) {
  const d = edt(m, w, h);
  const o = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) o[i] = d[i] <= r ? 1 : 0;
  return o;
}
function erode(m, w, h, r) {
  const d = edt(not(m), w, h);
  const o = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) o[i] = d[i] > r ? 1 : 0;
  return o;
}
const opening = (m, w, h, r) => dilate(erode(m, w, h, r), w, h, r);
const closing = (m, w, h, r) => erode(dilate(m, w, h, r), w, h, r);

// Separable Gaussian blur (reflect edges). Large sigmas use three box passes.
export function gauss(src, w, h, sigma) {
  if (sigma > 8) return boxGauss(src, w, h, sigma);
  const r = Math.ceil(sigma * 4);
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const refl = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let j = -r; j <= r; j++) acc += k[j + r] * src[o + refl(x + j, w)];
      tmp[o + x] = acc;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let j = -r; j <= r; j++) acc += k[j + r] * tmp[refl(y + j, h) * w + x];
      out[y * w + x] = acc;
    }
  }
  return out;
}

function boxGauss(src, w, h, sigma) {
  // Three box blurs approximate a Gaussian (sizes from Kovesi).
  const n = 3;
  const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
  const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
  let a = Float32Array.from(src), b = new Float32Array(w * h);
  for (let i = 0; i < n; i++) {
    const r = ((i < m ? wl : wl + 2) - 1) / 2;
    boxPass(a, b, w, h, r, true); boxPass(b, a, w, h, r, false);
  }
  return a;
}

function boxPass(src, dst, w, h, r, horiz) {
  const len = horiz ? w : h, lines = horiz ? h : w;
  const step = horiz ? 1 : w, lstep = horiz ? w : 1;
  const inv = 1 / (2 * r + 1);
  const get = (base, i) => src[base + clamp(i, 0, len - 1) * step];
  for (let l = 0; l < lines; l++) {
    const base = l * lstep;
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += get(base, i);
    for (let i = 0; i < len; i++) {
      dst[base + i * step] = acc * inv;
      acc += get(base, i + r + 1) - get(base, i - r);
    }
  }
}

// Sliding maximum over a square window of odd size k.
function maxFilter(src, w, h, k) {
  const r = (k - 1) >> 1;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  const dq = new Int32Array(Math.max(w, h) + k);
  const run = (get, set, len) => {
    let head = 0, tail = 0;
    for (let i = 0; i < len + r; i++) {
      if (i < len) {
        const val = get(i);
        while (tail > head && get(dq[tail - 1]) <= val) tail--;
        dq[tail++] = i;
      }
      const c = i - r;
      if (c >= 0) {
        while (dq[head] < c - r) head++;
        set(c, get(dq[head]));
      }
    }
  };
  for (let y = 0; y < h; y++) run((i) => src[y * w + i], (i, v) => { tmp[y * w + i] = v; }, w);
  for (let x = 0; x < w; x++) run((i) => tmp[i * w + x], (i, v) => { out[i * w + x] = v; }, h);
  return out;
}

// Keep only 4-connected pieces with at least minArea pixels.
function dropSmall(m, w, h, minArea) {
  const lab = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  const out = new Uint8Array(w * h);
  let n = 0;
  for (let s = 0; s < w * h; s++) {
    if (!m[s] || lab[s]) continue;
    n++;
    let top = 0, cnt = 0;
    stack[top++] = s; lab[s] = n;
    const members = [];
    while (top) {
      const p = stack[--top]; members.push(p); cnt++;
      const x = p % w, y = (p / w) | 0;
      if (x > 0 && m[p - 1] && !lab[p - 1]) { lab[p - 1] = n; stack[top++] = p - 1; }
      if (x < w - 1 && m[p + 1] && !lab[p + 1]) { lab[p + 1] = n; stack[top++] = p + 1; }
      if (y > 0 && m[p - w] && !lab[p - w]) { lab[p - w] = n; stack[top++] = p - w; }
      if (y < h - 1 && m[p + w] && !lab[p + w]) { lab[p + w] = n; stack[top++] = p + w; }
    }
    if (cnt >= minArea) for (const p of members) out[p] = 1;
  }
  return out;
}

// Separable Lanczos-3 resample of RGBA (premultiplied internally).
export function resize(img, W, H) {
  const { w, h, data } = img;
  const pm = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const a = data[i * 4 + 3] / 255;
    pm[i * 4] = data[i * 4] * a; pm[i * 4 + 1] = data[i * 4 + 1] * a;
    pm[i * 4 + 2] = data[i * 4 + 2] * a; pm[i * 4 + 3] = data[i * 4 + 3];
  }
  const tmp = resample1d(pm, w, h, W, true);
  const res = resample1d(tmp, W, h, H, false);
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const a = res[i * 4 + 3];
    const k = a > 0.5 ? 255 / a : 0;
    out[i * 4] = res[i * 4] * k; out[i * 4 + 1] = res[i * 4 + 1] * k;
    out[i * 4 + 2] = res[i * 4 + 2] * k; out[i * 4 + 3] = a;
  }
  return { w: W, h: H, data: out };
}

function resample1d(src, w, h, N, horiz) {
  const len = horiz ? w : h, other = horiz ? h : w;
  const scale = N / len, support = 3 * Math.max(1, 1 / scale);
  const fscale = Math.min(1, scale);
  const lanczos = (x) => {
    if (x === 0) return 1;
    if (Math.abs(x) >= 3) return 0;
    const px = Math.PI * x;
    return (3 * Math.sin(px) * Math.sin(px / 3)) / (px * px);
  };
  const out = new Float32Array((horiz ? N * h : w * N) * 4);
  for (let i = 0; i < N; i++) {
    const center = (i + 0.5) / scale;
    const lo = Math.max(0, Math.floor(center - support)), hi = Math.min(len - 1, Math.ceil(center + support));
    const ws = [];
    let sum = 0;
    for (let j = lo; j <= hi; j++) { const wt = lanczos((j + 0.5 - center) * fscale); ws.push(wt); sum += wt; }
    for (let o = 0; o < other; o++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let j = lo; j <= hi; j++) {
        const wt = ws[j - lo] / sum;
        const si = (horiz ? o * w + j : j * w + o) * 4;
        r += wt * src[si]; g += wt * src[si + 1]; b += wt * src[si + 2]; a += wt * src[si + 3];
      }
      const di = (horiz ? o * N + i : i * w + o) * 4;
      out[di] = r; out[di + 1] = g; out[di + 2] = b; out[di + 3] = clamp(a, 0, 255);
    }
  }
  return out;
}

// ---------- the style ----------

// Darken/lighten by factor f the way foil does: shadows get richer, not greyer.
function shadeInto(r, g, b, f, out, o) {
  const c = [clamp(r / 255, 1e-4, 1), clamp(g / 255, 1e-4, 1), clamp(b / 255, 1e-4, 1)];
  if (f < 1) {
    // Light colours (yellow, lime) need deeper shadows than rich ones (orange, red).
    const s = c.slice().sort((p, q) => p - q)[1];
    const k = 0.5 + 0.9 * s;
    const e = 1 + 1.6 * (1 - f), fk = Math.pow(f, k);
    for (let j = 0; j < 3; j++) out[o + j] = 255 * Math.pow(c[j], e) * fk;
  } else {
    const wv = clamp((f - 1) * 0.4, 0, 0.5);
    for (let j = 0; j < 3; j++) { const l = clamp(c[j] * f, 0, 1); out[o + j] = 255 * (l + (1 - l) * wv); }
  }
}

// Mean colour of the most common dark colour bucket, or null if the logo has no dark pixels.
export function darkCentre(img) {
  const { w, h, data } = img;
  const counts = new Map();
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    if (data[i * 4 + 3] <= 128 || (r + g + b) / 3 >= 90) continue;
    const k = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
    const e = counts.get(k) || [0, 0, 0, 0];
    e[0]++; e[1] += r; e[2] += g; e[3] += b;
    counts.set(k, e);
  }
  let best = null;
  for (const e of counts.values()) if (!best || e[0] > best[0]) best = e;
  return best && [best[1] / best[0], best[2] / best[0], best[3] / best[0]];
}

// Turn parts of a flat logo into holes (dark linework, thin dark strokes, light areas).
export function knockout(img, { dark = false, thinDark = 0, light = null } = {}) {
  const { w, h } = img;
  const data = new Uint8ClampedArray(img.data);
  if (!dark && !(thinDark > 0) && light == null) return { w, h, data };
  const isDark = new Uint8Array(w * h), darkish = new Uint8Array(w * h), holes = new Uint8Array(w * h);
  const centre = darkCentre(img);
  for (let i = 0; i < w * h; i++) {
    const op = data[i * 4 + 3] > 128;
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    const lum = (r + g + b) / 3;
    // Only the logo's main dark colour counts (its linework/black), not dark shades of
    // other colours such as shaded letters or a dark purple fill.
    darkish[i] = op && lum < 90 ? 1 : 0;
    isDark[i] = darkish[i] && centre && Math.hypot(r - centre[0], g - centre[1], b - centre[2]) < 48 ? 1 : 0;
    if (light != null && op && lum > light) holes[i] = 1;
  }
  // Anti-aliased edge pixels are blends, so let the core grow into dark neighbours.
  const grown = dilate(isDark, w, h, 2);
  for (let i = 0; i < w * h; i++) {
    isDark[i] = darkish[i] && grown[i] ? 1 : 0;
    if (dark && isDark[i]) holes[i] = 1;
  }
  if (thinDark > 0) {
    const r = Math.max(1, Math.round((thinDark * Math.max(w, h)) / 2));
    const thick = opening(isDark, w, h, r);
    for (let i = 0; i < w * h; i++) if (isDark[i] && !thick[i]) holes[i] = 1;
  }
  for (let i = 0; i < w * h; i++) if (holes[i]) data[i * 4 + 3] = 0;
  return { w, h, data };
}

export const DEFAULTS = {
  size: 1500, color: null, seed: 0, bevel: 20, bevelStrength: 0.3, sheen: 0.12, blend: 0,
  knockoutDark: false, knockoutThinDark: 0, knockoutLight: null, tint: null,
  splitColors: false, detailScale: null, ss: SS,
};

export function stylize(src, options = {}) {
  const o = { ...DEFAULTS, ...options };
  const ss = o.ss;
  const rand = mulberry32(o.seed * 2654435761 + 1);
  const im = knockout(src, { dark: o.knockoutDark, thinDark: o.knockoutThinDark, light: o.knockoutLight });

  // Padding so the outline never touches the canvas edge.
  const pad = Math.floor(Math.max(im.w, im.h) * 0.03);
  const cw = im.w + 2 * pad, ch = im.h + 2 * pad;
  const canvas = { w: cw, h: ch, data: new Uint8ClampedArray(cw * ch * 4) };
  for (let y = 0; y < im.h; y++) canvas.data.set(im.data.subarray(y * im.w * 4, (y + 1) * im.w * 4), ((y + pad) * cw + pad) * 4);
  const scale = (o.size * ss) / Math.max(cw, ch);
  const W = Math.round(cw * scale), H = Math.round(ch * scale), N = W * H;
  const big = resize(canvas, W, H).data;

  const a0 = new Float32Array(N);
  for (let i = 0; i < N; i++) a0[i] = big[i * 4 + 3] / 255;
  const alpha = gauss(a0, W, H, 1.5 * ss);
  let inside = new Uint8Array(N);
  for (let i = 0; i < N; i++) inside[i] = alpha[i] > 0.5 ? 1 : 0;

  // Un-premultiplied colours bleed at edges; fill them from the nearest solid pixel.
  const solid = new Uint8Array(N);
  let anySolid = false;
  for (let i = 0; i < N; i++) if (big[i * 4 + 3] > 250) { solid[i] = 1; anySolid = true; }
  if (!anySolid) for (let i = 0; i < N; i++) solid[i] = big[i * 4 + 3] > 0 ? 1 : 0;
  const { idx } = edt(solid, W, H, true);
  let rgb = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const s = idx[i] * 4;
    rgb[i * 3] = big[s]; rgb[i * 3 + 1] = big[s + 1]; rgb[i * 3 + 2] = big[s + 2];
  }

  if (o.splitColors) {
    // Cut a thin line wherever two different colours touch.
    const grad = new Float32Array(N);
    for (let c = 0; c < 3; c++) {
      const ch1 = new Float32Array(N);
      for (let i = 0; i < N; i++) ch1[i] = rgb[i * 3 + c];
      const sm = gauss(ch1, W, H, 1.0 * ss);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const at = (xx, yy) => sm[clamp(yy, 0, H - 1) * W + clamp(xx, 0, W - 1)];
        const gxv = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
        const gyv = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
        grad[y * W + x] += Math.hypot(gxv, gyv);
      }
    }
    const dIn0 = edt(not(inside), W, H);
    let edge = new Uint8Array(N);
    for (let i = 0; i < N; i++) edge[i] = grad[i] > 250 && dIn0[i] > 4 * ss ? 1 : 0;
    edge = closing(edge, W, H, ss);
    const cut = dilate(edge, W, H, ss);
    for (let i = 0; i < N; i++) if (cut[i]) inside[i] = 0;
    inside = opening(inside, W, H, ss);
  }

  if (o.color) for (let i = 0; i < N; i++) { rgb[i * 3] = o.color[0]; rgb[i * 3 + 1] = o.color[1]; rgb[i * 3 + 2] = o.color[2]; }
  if (o.tint) {
    // Recolour to one hue but keep the source's light/dark structure as shades.
    const out = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const lum = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3 / 255;
      shadeInto(o.tint[0], o.tint[1], o.tint[2], 0.92 + 0.35 * lum, out, i * 3);
    }
    rgb = out;
  }
  if (o.blend > 0) {
    // Soften hard colour splits into smooth gradients, weighted by saturation.
    const sigma = o.blend * Math.max(W, H);
    const sat = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
      sat[i] = ((Math.max(r, g, b) - Math.min(r, g, b)) / 255) * inside[i] + 1e-3;
    }
    const den = gauss(sat, W, H, sigma);
    for (let c = 0; c < 3; c++) {
      const num = new Float32Array(N);
      for (let i = 0; i < N; i++) num[i] = rgb[i * 3 + c] * sat[i];
      const bl = gauss(num, W, H, sigma);
      for (let i = 0; i < N; i++) rgb[i * 3 + c] = bl[i] / den[i];
    }
  }

  const px = ss * (o.detailScale == null ? o.size / 1500 : o.detailScale);
  // Drop specks: tiny slivers left over from knockouts and colour splits.
  inside = dropSmall(inside, W, H, (14 * px) ** 2);
  const dIn = edt(not(inside), W, H);
  const dOut = edt(inside, W, H);

  // Diagonal gradient, lit from the top-left, with soft warped sheen bands.
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (inside[y * W + x]) {
    const p = 0.55 * x + 0.83 * y; if (p < lo) lo = p; if (p > hi) hi = p;
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const gw = (W >> 3) + 1, gh = (H >> 3) + 1;
  let warpS = gauss(normals(rand, gw * gh), gw, gh, (Math.max(H, W) / 8) * 0.08);
  const wsd = std(warpS) + 1e-9;
  const col = new Float32Array(N * 3);
  const tArr = [0, 0.3, 0.75, 1], vArr = [1.18, 1.0, 0.72, 0.76];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const t = clamp((0.55 * x + 0.83 * y - lo) / Math.max(1, hi - lo), 0, 1);
    let k = 0; while (k < 2 && t > tArr[k + 1]) k++;
    const ramp = vArr[k] + ((vArr[k + 1] - vArr[k]) * (t - tArr[k])) / (tArr[k + 1] - tArr[k]);
    const fy = Math.min(y / 8, gh - 1), fx = Math.min(x / 8, gw - 1);
    const y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(y0 + 1, gh - 1), x1 = Math.min(x0 + 1, gw - 1);
    const ty = fy - y0, tx = fx - x0;
    const wv = ((warpS[y0 * gw + x0] * (1 - tx) + warpS[y0 * gw + x1] * tx) * (1 - ty) +
      (warpS[y1 * gw + x0] * (1 - tx) + warpS[y1 * gw + x1] * tx) * ty) / wsd;
    const bands = Math.sin(2 * Math.PI * (1.33 * (t + 0.05 * wv) - 0.15));
    shadeInto(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], ramp * (1 + o.sheen * bands), col, i * 3);
  }
  warpS = null;

  // Grain: fine noise + soft mottling + sparse bright specks.
  const fine = gauss(normals(rand, N), W, H, 0.9 * px);
  const fsd = std(fine);
  const mottle = gauss(normals(rand, N), W, H, 30 * px);
  const msd = std(mottle);
  const sp = new Float32Array(N);
  for (let i = 0; i < N; i++) sp[i] = rand() > 0.9994 ? 1 : 0;
  const specks = gauss(sp, W, H, 1.2 * px);
  let smax = 1e-9; for (let i = 0; i < N; i++) if (specks[i] > smax) smax = specks[i];
  for (let i = 0; i < N; i++) {
    const lum = 1 + (0.045 * fine[i]) / fsd + (0.01 * mottle[i]) / msd;
    const s = (90 * specks[i]) / smax;
    for (let c = 0; c < 3; c++) col[i * 3 + c] = col[i * 3 + c] * lum + (s * rgb[i * 3 + c]) / 255;
  }

  // Bevel: chamfer band inside the edge, lit from the top-left; thin strokes get a narrower one.
  const chamferMax = o.bevel * px;
  const reach = Math.floor(2 * chamferMax) | 1;
  const halfWidth = maxFilter(dIn, W, H, reach);
  const dS = gauss(dIn, W, H, 1.5 * px);
  const ow = 9 * px, olColor = [2, 22, 22];
  const out = new Uint8ClampedArray(N * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const gx = x === 0 ? dS[i + 1] - dS[i] : x === W - 1 ? dS[i] - dS[i - 1] : (dS[i + 1] - dS[i - 1]) / 2;
    const gy = y === 0 ? dS[i + W] - dS[i] : y === H - 1 ? dS[i] - dS[i - W] : (dS[i + W] - dS[i - W]) / 2;
    const norm = Math.hypot(gx, gy) + 1e-6;
    const light = (-gx / norm) * -0.6 + (-gy / norm) * -0.8;
    const chamfer = clamp(0.5 * halfWidth[i], 2.5 * px, chamferMax);
    const d = dIn[i];
    const band = clamp((chamfer - d) / px + 0.5, 0, 1);
    const slope = 1 + 0.25 * clamp(1 - d / chamfer, 0, 1);
    const face = light > 0 ? light : 0.7 * light;
    let shade = 1 + o.bevelStrength * face * band * slope;
    const crease = Math.exp(-(((d - chamfer) / (1.3 * px)) ** 2));
    shade *= 1 - 0.22 * light * crease - 0.06 * crease;
    // Dark outline centred on the edge.
    const sdf = d - dOut[i];
    const ol = clamp((ow * 0.4 - sdf) / (1.2 * px), 0, 1);
    for (let c = 0; c < 3; c++) out[i * 4 + c] = clamp(col[i * 3 + c] * shade * (1 - ol) + olColor[c] * ol, 0, 255);
    out[i * 4 + 3] = 255 * clamp((ow * 0.6 + sdf) / (1.2 * px) + 0.5, 0, 1);
  }

  const res = resize({ w: W, h: H, data: out }, Math.floor(W / ss), Math.floor(H / ss));
  return crop(res);
}

// Crop to the non-transparent bounding box.
export function crop(img) {
  const { w, h, data } = img;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (data[(y * w + x) * 4 + 3]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) return img;
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1, o = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) o.set(data.subarray(((y + y0) * w + x0) * 4, ((y + y0) * w + x1 + 1) * 4), y * cw * 4);
  return { w: cw, h: ch, data: o };
}
