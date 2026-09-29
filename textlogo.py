"""Generate a styled lettering logo for teams that have no logo of their own.

Usage: python textlogo.py "MYT" output.png [--color #D9D9D9] [--tracking 0.06] [--weight 0.045]

The letters are drawn in a heavy font, spaced so each letter is its own piece (later
letters sit on top when they overlap), then run through the same badge style as stylize.py.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from stylize import hex_rgb, stylize

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FONT = os.path.join(HERE, "fonts", "RussoOne.woff")
CAP = 700  # cap height of the flat source in px


def grow(mask, r):
    """Round dilation by r px (a distance transform is far faster than a big disk kernel)."""
    return ndimage.distance_transform_edt(~mask) <= r


def glyph_mask(ch, font, weight):
    """Mask of one character, cropped to its ink and thickened by `weight` × cap height."""
    pad = int(CAP * (0.2 + weight))
    size = CAP * 3
    im = Image.new("L", (size, size), 0)
    ImageDraw.Draw(im).text((pad, pad), ch, font=font, fill=255)
    m = np.asarray(im) > 127
    if weight > 0:
        m = grow(m, weight * CAP)
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], ys.min() - pad


def render_text(text, font_path=DEFAULT_FONT, tracking=0.06, weight=0.045, gap=0.035,
                color=(217, 217, 217)):
    """Flat RGBA source image of `text`, one separated piece per letter."""
    # Scale the font so capitals come out CAP px tall before thickening.
    probe = ImageFont.truetype(font_path, 100)
    x0, y0, x1, y1 = probe.getbbox("H")
    font = ImageFont.truetype(font_path, round(100 * CAP / (y1 - y0)))

    # Space letters by their shapes, not their boxes, so pairs like "VA" don't gape.
    # `edge` holds, per row of the line, the rightmost ink placed so far.
    rows = CAP * 3
    base = CAP
    edge = np.full(rows, -np.inf)
    glyphs, pending_space = [], 0
    for ch in text:
        if ch == " ":
            pending_space += int(CAP * 0.35)
            continue
        m, top = glyph_mask(ch, font, weight)
        y = base + top
        left = np.where(m.any(1), m.argmax(1), np.inf)
        prev = edge[y:y + m.shape[0]]
        # Rows where both sides have ink set the distance; tracking is the gap between them.
        x = np.max(prev - left) + 1 + tracking * CAP if np.isfinite(prev - left).any() else 0
        if not glyphs:
            x = 0
        x = int(x) + pending_space
        pending_space = 0
        right = np.where(m.any(1), m.shape[1] - 1 - m[:, ::-1].argmax(1), -np.inf)
        edge[y:y + m.shape[0]] = np.maximum(prev, x + right)
        glyphs.append((m, x, top))
    if not glyphs:
        raise ValueError("text has no visible characters")
    shift = -min(gx for _, gx, _ in glyphs)
    glyphs = [(m, gx + shift, t) for m, gx, t in glyphs]

    top_min = min(t for _, _, t in glyphs)
    W = max(gx + m.shape[1] for m, gx, _ in glyphs)
    H = max(t - top_min + m.shape[0] for m, _, t in glyphs)
    layers = []
    for m, gx, t in glyphs:
        full = np.zeros((H, W), bool)
        y = t - top_min
        full[y:y + m.shape[0], gx:gx + m.shape[1]] = m
        layers.append(full)

    # Later letters sit on top: carve a gap out of every earlier letter they overlap or touch.
    above = np.zeros((H, W), bool)
    for i in range(len(layers) - 1, -1, -1):
        mine = layers[i]
        if above.any():
            layers[i] = mine & ~grow(above, gap * CAP)
        above |= mine
    ink = np.logical_or.reduce(layers)

    out = np.zeros((H, W, 4), np.uint8)
    out[ink, :3] = color
    out[ink, 3] = 255
    return Image.fromarray(out, "RGBA")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("output")
    p.add_argument("--color", type=hex_rgb, default=hex_rgb("#D9D9D9"))
    p.add_argument("--font", default=DEFAULT_FONT)
    p.add_argument("--tracking", type=float, default=0.06,
                   help="space between letters as a fraction of cap height; negative overlaps them")
    p.add_argument("--weight", type=float, default=0.045, help="extra stroke thickness, fraction of cap height")
    p.add_argument("--height", type=int, default=690, help="output letter height in px")
    p.add_argument("--flat", help="also save the flat, unstyled lettering here")
    args = p.parse_args()
    flat = render_text(args.text.upper(), args.font, args.tracking, args.weight, color=args.color)
    if args.flat:
        flat.save(args.flat)
    # Size by letter height so long names keep the same outline/bevel/grain as short ones.
    size = round(args.height * max(flat.size) / flat.height)
    stylize(flat, size, detail_scale=args.height / 690).save(args.output)
