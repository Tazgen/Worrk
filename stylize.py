"""Apply the textured metallic badge style to a flat transparent logo.

Usage: python stylize.py input.png output.png [--size 1500] [--color #RRGGBB]
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

SS = 2  # supersampling factor for smooth edges


def hex_rgb(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], float)


def shade_color(rgb, f):
    """Darken/lighten by factor f the way foil does: shadows get richer, not greyer."""
    f = f[..., None]
    c = np.clip(rgb / 255, 1e-4, 1)
    # Light colours (yellow, lime) need deeper shadows than rich ones (orange, red) to read as metal.
    # The second-strongest channel separates them: ~1 for yellow, ~0 for pure red.
    second = np.sort(c, axis=-1)[..., 1:2]
    k = 0.5 + 0.9 * second
    dark = c ** (1 + 1.6 * np.clip(1 - f, 0, None)) * np.minimum(f, 1) ** k
    # Highlights brighten and drift slightly towards white.
    lit = np.clip(c * f, 0, 1)
    lit = lit + (1 - lit) * np.clip((f - 1) * 0.4, 0, 0.5)
    return 255 * np.where(f < 1, dark, lit)


def knockout(im, dark=False, thin_dark=0.0, light=None):
    """Turn parts of a flat logo into holes, the way the style treats linework and white areas.

    dark:      every dark pixel becomes a hole (line-art logos).
    thin_dark: only dark strokes thinner than this fraction of the logo size become holes,
               so thin outlines turn into cut lines while thick dark text/rings stay filled.
    light:     pixels brighter than this (0-255) become holes (white backgrounds, pale stripes).
    """
    arr = np.array(im)
    opaque = arr[..., 3] > 128
    lum = arr[..., :3].astype(float).mean(-1)
    holes = np.zeros(opaque.shape, bool)
    is_dark = opaque & (lum < 90)
    if dark:
        holes |= is_dark
    if thin_dark > 0:
        r = max(1, round(thin_dark * max(im.size) / 2))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        holes |= is_dark & ~ndimage.binary_opening(is_dark, disk)
    if light is not None:
        holes |= opaque & (lum > light)
    arr[holes, 3] = 0
    return Image.fromarray(arr, "RGBA")


def stylize(src, size=1500, color=None, seed=0, bevel=20, bevel_strength=0.30,
            sheen=0.12, blend=0.0, knockout_dark=False, knockout_thin_dark=0.0,
            knockout_light=None, tint=None, split_colors=False):
    rng = np.random.default_rng(seed)
    im = knockout(src.convert("RGBA"), knockout_dark, knockout_thin_dark, knockout_light)

    # Padding so the outline never touches the canvas edge.
    pad = int(max(im.size) * 0.03)
    canvas = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0))
    canvas.paste(im, (pad, pad))
    scale = size * SS / max(canvas.size)
    W, H = round(canvas.width * scale), round(canvas.height * scale)
    big = canvas.resize((W, H), Image.LANCZOS)
    a = np.asarray(big).astype(float)

    alpha = ndimage.gaussian_filter(a[..., 3] / 255, 1.5 * SS)
    inside = alpha > 0.5
    rgb = a[..., :3]
    # Un-premultiplied colors bleed at edges; fill them from the nearest solid pixel.
    solid = a[..., 3] > 250
    idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    rgb = rgb[idx[0], idx[1]]
    if split_colors:
        # Cut a thin line wherever two different colours touch, so each colour region
        # becomes its own piece with its own outline and bevel.
        sm = np.dstack([ndimage.gaussian_filter(rgb[..., c], 1.0 * SS) for c in range(3)])
        grad = sum(np.hypot(ndimage.sobel(sm[..., c], 0), ndimage.sobel(sm[..., c], 1)) for c in range(3))
        edge = grad > 250  # well above the soft ramps inside a single colour
        # Ignore the soft band just inside outer edges, where colours bleed from the background.
        edge &= ndimage.distance_transform_edt(inside) > 4 * SS
        edge = ndimage.binary_closing(edge, iterations=SS)
        inside &= ~ndimage.binary_dilation(edge, iterations=SS)
        inside = ndimage.binary_opening(inside, iterations=SS)
    if color is not None:
        rgb[:] = color
    if tint is not None:
        # Recolour to one hue but keep the source's light/dark structure as shades.
        lum = rgb.mean(-1) / 255
        rgb = shade_color(np.broadcast_to(tint, rgb.shape), 0.92 + 0.35 * lum)
    if blend > 0:
        # Soften hard colour splits into smooth gradients (fraction of logo size).
        # Weighted by saturation so white/grey pieces pick up the colours around them.
        sigma = blend * max(W, H)
        sat = (rgb.max(-1) - rgb.min(-1)) / 255 * inside + 1e-3
        num = [ndimage.gaussian_filter(rgb[..., c] * sat, sigma) for c in range(3)]
        rgb = np.dstack(num) / ndimage.gaussian_filter(sat, sigma)[..., None]

    px = SS * size / 1500  # all sizes below are tuned for a 1500px output
    # Drop specks: tiny slivers left over from knockouts and colour splits.
    lab, n = ndimage.label(inside)
    if n:
        areas = ndimage.sum(inside, lab, range(1, n + 1))
        inside = np.isin(lab, 1 + np.nonzero(areas >= (14 * px) ** 2)[0])
    d_in = ndimage.distance_transform_edt(inside)
    d_out = ndimage.distance_transform_edt(~inside)
    sdf = d_in - d_out  # positive inside

    # Diagonal gradient, lit from the top-left: bright top-left, darkest around 75%,
    # slight lift in the far bottom-right corner.
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    proj = 0.55 * xx + 0.83 * yy
    lo, hi = proj[inside].min(), proj[inside].max()
    t = np.clip((proj - lo) / max(1.0, hi - lo), 0, 1)
    ramp = np.interp(t, [0, 0.3, 0.75, 1], [1.18, 1.0, 0.72, 0.76])
    # Sheen: soft light bands across the diagonal, warped so they don't look ruled.
    warp = ndimage.gaussian_filter(rng.standard_normal((H // 8 + 1, W // 8 + 1)), max(H, W) / 8 * 0.08)
    warp = ndimage.zoom(warp / (warp.std() + 1e-9), 8, order=1)[:H, :W]
    bands = np.sin(2 * np.pi * (1.33 * (t + 0.05 * warp) - 0.15))
    col = shade_color(rgb, ramp * (1 + sheen * bands))

    # Grain: fine noise + soft mottling + sparse bright specks.
    fine = ndimage.gaussian_filter(rng.standard_normal((H, W)), 0.9 * px)
    fine /= fine.std()
    mottle = ndimage.gaussian_filter(rng.standard_normal((H, W)), 30 * px)
    mottle /= mottle.std()
    specks = (rng.random((H, W)) > 0.9994).astype(float)
    specks = ndimage.gaussian_filter(specks, 1.2 * px)
    specks /= specks.max()
    lum = 1 + 0.045 * fine + 0.010 * mottle
    col = col * lum[..., None] + 90 * specks[..., None] * (rgb / 255)

    # Bevel: chamfer band inside the edge, lit from the top-left.
    # Thin strokes get a narrower bevel so they aren't swallowed by it.
    chamfer_max = bevel * px
    reach = int(2 * chamfer_max) | 1
    half_width = ndimage.maximum_filter(d_in, size=reach)
    chamfer = np.clip(0.5 * half_width, 2.5 * px, chamfer_max)
    gy, gx = np.gradient(ndimage.gaussian_filter(d_in, 1.5 * px))
    norm = np.hypot(gx, gy) + 1e-6
    # Gradient points inward; the face normal points outward.
    light = (-gx / norm) * -0.6 + (-gy / norm) * -0.8
    band = np.clip((chamfer - d_in) / (1.0 * px) + 0.5, 0, 1)  # crisp inner edge
    # Chamfer faces brighten slightly towards the outer edge, like a real slope catching light.
    slope = 1 + 0.25 * np.clip(1 - d_in / chamfer, 0, 1)
    # Shadowed faces get less contrast than lit ones so they stay clean rather than muddy.
    face = np.where(light > 0, light, 0.7 * light)
    shade = 1 + bevel_strength * face * band * slope
    # Crease where the chamfer meets the flat face; it contrasts with the face it borders.
    crease = np.exp(-((d_in - chamfer) / (1.3 * px)) ** 2)
    shade *= 1 - 0.22 * light * crease - 0.06 * crease
    col *= shade[..., None]

    # Dark outline centred on the edge.
    ow = 9 * px
    ol_color = np.array([2, 22, 22], float)
    ol = np.clip((ow * 0.4 - sdf) / (1.2 * px), 0, 1)  # inner part of the stroke
    col = col * (1 - ol[..., None]) + ol_color * ol[..., None]
    out_alpha = np.clip((ow * 0.6 + sdf) / (1.2 * px) + 0.5, 0, 1)

    out = np.dstack([np.clip(col, 0, 255), out_alpha * 255]).astype(np.uint8)
    res = Image.fromarray(out, "RGBA").resize((W // SS, H // SS), Image.LANCZOS)
    return res.crop(res.getbbox())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--size", type=int, default=1500)
    p.add_argument("--color", type=hex_rgb)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bevel", type=float, default=20, help="bevel width in px at 1500px output")
    p.add_argument("--bevel-strength", type=float, default=0.30)
    p.add_argument("--sheen", type=float, default=0.12, help="strength of diagonal light bands")
    p.add_argument("--blend", type=float, default=0.0, help="smooth hard colour splits, e.g. 0.08")
    p.add_argument("--knockout-dark", action="store_true", help="turn dark linework into holes")
    p.add_argument("--knockout-thin-dark", type=float, default=0.0,
                   help="turn dark strokes thinner than this fraction of logo size into holes, e.g. 0.015")
    p.add_argument("--knockout-light", type=float, help="turn pixels brighter than this (0-255) into holes")
    p.add_argument("--tint", type=hex_rgb, help="recolour to one hue, keeping light/dark shades")
    p.add_argument("--split-colors", action="store_true", help="draw outlines between touching colours")
    args = p.parse_args()
    stylize(Image.open(args.input), args.size, args.color, args.seed, args.bevel, args.bevel_strength,
            args.sheen, args.blend, args.knockout_dark, args.knockout_thin_dark,
            args.knockout_light, args.tint, args.split_colors).save(args.output)
