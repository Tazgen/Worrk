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


def stylize(src, size=1500, color=None, seed=0, bevel=20, bevel_strength=0.30):
    rng = np.random.default_rng(seed)
    im = src.convert("RGBA")

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
    if color is not None:
        rgb[:] = color

    px = SS * size / 1500  # all sizes below are tuned for a 1500px output
    d_in = ndimage.distance_transform_edt(inside)
    d_out = ndimage.distance_transform_edt(~inside)
    sdf = d_in - d_out  # positive inside

    # Vertical gradient: bright top, darkest around 75%, slight lift at the bottom.
    ys = np.nonzero(inside.any(1))[0]
    t = np.clip((np.arange(H) - ys[0]) / max(1, ys[-1] - ys[0]), 0, 1)
    ramp = np.interp(t, [0, 0.3, 0.75, 1], [0.90, 0.80, 0.62, 0.66])[:, None, None]
    col = rgb * ramp

    # Grain: fine noise + soft mottling + sparse bright specks.
    fine = ndimage.gaussian_filter(rng.standard_normal((H, W)), 0.9 * px)
    fine /= fine.std()
    mottle = ndimage.gaussian_filter(rng.standard_normal((H, W)), 30 * px)
    mottle /= mottle.std()
    specks = (rng.random((H, W)) > 0.9994).astype(float)
    specks = ndimage.gaussian_filter(specks, 1.2 * px)
    specks /= specks.max()
    lum = 1 + 0.045 * fine + 0.025 * mottle
    col = col * lum[..., None] + 90 * specks[..., None] * (rgb / 255)

    # Bevel: chamfer band inside the edge, lit from the top-left.
    chamfer = bevel * px
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
    args = p.parse_args()
    stylize(Image.open(args.input), args.size, args.color, args.seed, args.bevel, args.bevel_strength).save(args.output)
