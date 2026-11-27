#!/usr/bin/env python3
"""ASCII portrait pipeline: image -> character grid -> preview PNG -> interactive page.

Three subcommands:
  render   <image> <out.txt>    image -> char grid (the deliverable)
  preview  <art.txt> <out.png>  char grid -> exact-cell PNG (canvas == picture)
  page     <art.txt> <out.html> char grid -> self-contained animated page

Design rules baked in (see SKILL.md for why each one exists):
  * regions from colour, never a geometric ellipse
  * per-region tone bands with a floor so highlights never vanish
  * feature channel = local darkening measured at a blur radius >= 2% of width
  * iris found by chroma, not luminance
  * cell aspect fixed at 0.5 (rows = h/w * cols * 0.5)
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAMP = " .'`:-=+*#%@"


# ---------------------------------------------------------------- render ----
def render(src, cols=300, crop=(0.07, 0.015, 0.80, 0.60), rows_f=0.5,
           smooth=0.5, skin_ink=0.28, hair_ink=0.62, coat_ink=0.78,
           feat_lo=0.16, feat_hi=0.52, hair_strand=0.30, coat_fold=0.34):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = crop
    im2 = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    bw, bh = im2.size
    rows = max(1, round(bh / bw * cols * rows_f))

    sm = im2.resize((cols, rows), Image.LANCZOS).filter(ImageFilter.GaussianBlur(smooth))
    a = np.asarray(sm, dtype=np.float32) / 255.0
    L = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    S = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1e-6)

    # feature channel at 2x, blur radius must exceed the stroke width
    pw, ph = cols * 2, max(1, round(bh / bw * cols * 2))
    big = im2.resize((pw, ph), Image.LANCZOS)
    ba = np.asarray(big, dtype=np.float32) / 255.0
    bL = ba @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    rad = max(2.0, pw * 0.020)
    blur = np.asarray(
        Image.fromarray((bL * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rad)),
        dtype=np.float32) / 255.0
    R = np.clip((blur - bL) * 3.0, 0, 1)
    cw, chh = pw / cols, ph / rows
    feat = np.zeros((rows, cols), dtype=np.float32)
    for ry in range(rows):
        for cx in range(cols):
            X0, X1 = int(cx * cw), min(pw, int((cx + 1) * cw) + 1)
            Y0, Y1 = int(ry * chh), min(ph, int((ry + 1) * chh) + 1)
            cell = R[Y0:Y1, X0:X1]
            feat[ry, cx] = np.percentile(cell, 96) if cell.size else 0.0

    yy0 = np.arange(rows)[:, None]
    bg = (b < 0.26) & (L < 0.25) & (S < 0.68)
    coat = (S > 0.36) & (b > r) & (L < 0.58) & ~bg & (yy0 > rows * 0.52)
    skin = (r > b + 0.012) & (L > 0.70) & ~bg & ~coat
    hair = ~bg & ~coat & ~skin

    def band(mask):
        out = np.zeros((rows, cols), dtype=np.float32)
        if mask.sum() == 0:
            return out
        v = L[mask]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out[mask] = np.clip((v - lo) / max(1e-6, hi - lo), 0, 1)
        return out

    t_h, t_c, t_s = band(hair), band(coat), band(skin)
    hair_ramp = "   ..''``::;;--=="
    coat_ramp = " .`:-=+*#%@"

    out = [[" "] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            if bg[y, x]:
                continue
            f = float(np.clip((feat[y, x] - feat_lo) / (feat_hi - feat_lo), 0, 1))
            if f > 0.62:                                  # strong stroke wins over tone
                k = min(5, int((f - 0.62) / 0.38 * 5) + 1)
                out[y][x] = "  .'`:"[k]
                continue
            if coat[y, x]:
                ink = coat_ink * (1 - t_c[y, x]) + coat_fold * f
                out[y][x] = coat_ramp[min(9, int(np.clip(ink, 0, 1) * 9))]
            elif skin[y, x]:
                if S[y, x] > 0.17 and (r[y, x] - b[y, x]) > 0.035 and L[y, x] < 0.97:
                    out[y][x] = "#" if S[y, x] > 0.26 else "o"   # iris by chroma
                elif f > 0.26:
                    kk = min(5, int((f - 0.26) / 0.74 * 5) + 1)
                    out[y][x] = "  .'`:"[kk]
                else:
                    k = int((1 - t_s[y, x]) * 4.2 * skin_ink)
                    if k > 0:
                        out[y][x] = "  ..''`"[min(6, k)]
            else:
                ink = 0.10 + hair_ink * (1 - t_h[y, x]) + hair_strand * f
                out[y][x] = hair_ramp[min(15, int(np.clip(ink, 0, 1) * 15))]

    lines = ["".join(q).rstrip() for q in out]
    return trim(lines)


def trim(lines):
    w = max(len(l) for l in lines)
    lines = [l.ljust(w) for l in lines]
    cols = [x for x in range(w) if any(l[x] != " " for l in lines)]
    rows = [y for y, l in enumerate(lines) if l.strip()]
    c0, c1 = min(cols), max(cols)
    r0, r1 = min(rows), max(rows)
    return [lines[y][c0:c1 + 1].rstrip() for y in range(r0, r1 + 1)]


# --------------------------------------------------------------- preview ----
def preview(art, out_png, cell=4, font_path=None, fg=(226, 234, 242), bg=(11, 13, 17)):
    lines = [l.rstrip("\n") for l in open(art, encoding="utf-8")]
    while lines and not lines[-1].strip():
        lines.pop()
    cols = max(len(l) for l in lines)
    lines = [l.ljust(cols) for l in lines]
    ch = cell * 2                                  # cell aspect 0.5
    font = ImageFont.truetype(font_path or _default_font(), int(round(cell * 1.45)))
    img = Image.new("RGB", (cols * cell, len(lines) * ch), bg)
    d = ImageDraw.Draw(img)
    for j, line in enumerate(lines):               # per character: canvas == picture
        for i, c in enumerate(line):
            if c != " ":
                d.text((i * cell, j * ch), c, font=font, fill=fg)
    img.save(out_png)
    return img.size


def _default_font():
    for p in (r"C:\Windows\Fonts\consola.ttf",
              "/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            return p
    raise SystemExit("pass --font with a monospace ttf")


# ------------------------------------------------------------------ page ----
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>character portrait</title>
<style>
  :root { --bg:#0a0d12; --ink:#b9cfe0; --dim:#3f4d5c; --hot:#7ff0ff; --cool:#cfe9ff; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin:0; background: radial-gradient(120% 80% at 50% 0%, #121c26 0%, var(--bg) 62%);
         color: var(--ink); display:grid; place-items:center; padding:2vmin;
         font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
  .plate { display:grid; gap:.8rem; justify-items:center; }
  pre#art { margin:0; white-space:pre; line-height:0.94; color:var(--ink);
            cursor:crosshair; user-select:none; }
  pre#art i { font-style:normal; color:var(--cool); text-shadow:0 0 6px rgba(207,233,255,.35); }
  pre#art em { font-style:normal; color:var(--hot); text-shadow:0 0 8px rgba(127,240,255,.55); }
  .cap { letter-spacing:.18em; text-transform:uppercase; font-size:clamp(9px,1vmin,12px); color:var(--dim); }
</style>
</head>
<body>
<div class="plate">
  <pre id="art" role="img"></pre>
  <div class="cap">character portrait &middot; __COLS__ &times; __ROWS__ cells</div>
</div>
<script>
(function () {
  var grid = __DATA__.map(function (r) { return r.split(''); });
  var COLS = grid[0].length, ROWS = grid.length;
  var NOISE = "01#%*+=:/\\|-_";
  var DECODE = 27000, CYCLE = 144000, SWEEP = 42000, LINGER = 18000, HOT_R = 15;
  var pre = document.getElementById('art');
  var still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var rnd = function (s) { return s[(Math.random() * s.length) | 0]; };
  var plain = function () { return grid.map(function (r) { return r.join(''); }).join('\n'); };

  pre.textContent = plain();          // first paint; empty pre is invisible to IntersectionObserver
  function fit() {                     // canvas == picture, whatever the font metrics are
    pre.style.fontSize = '100px';
    var cw = pre.scrollWidth / COLS, ch = pre.scrollHeight / ROWS;
    var k = Math.min((innerWidth * 0.96) / (cw * COLS), (innerHeight * 0.94) / (ch * ROWS));
    pre.style.fontSize = Math.max(2, 100 * k).toFixed(2) + 'px';
  }
  fit();
  addEventListener('resize', fit);
  if (still) return;

  var mx = -99, my = -99, stir = 0, start = performance.now();
  var lastFrame = 0, raf = 0, visible = true;
  function track(e) {
    var b = pre.getBoundingClientRect();
    mx = ((e.clientX - b.left) / b.width) * COLS;
    my = ((e.clientY - b.top) / b.height) * ROWS;
    stir = performance.now();
    wake();
  }
  pre.addEventListener('pointermove', track);
  pre.addEventListener('pointerdown', track);
  pre.addEventListener('pointerleave', function () { mx = my = -99; });
  if ('IntersectionObserver' in window)
    new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible) wake(); }).observe(pre);
  document.addEventListener('visibilitychange', wake);
  function wake() { if (!raf) raf = requestAnimationFrame(frame); }

  function frame(now) {
    raf = 0;
    if (!visible || document.hidden) return;
    if (now - lastFrame < 50) { wake(); return; }        // 20 fps, so motion is even
    lastFrame = now;
    var t = now - start, decoding = t < DECODE;
    var cycle = (t - DECODE) % CYCLE;
    var wave = cycle < SWEEP ? (cycle / SWEEP) * (COLS + ROWS) * 1.15 - 10 : -99;
    var hot = now - stir < LINGER;
    var heat = hot ? 1 - (now - stir) / LINGER : 0;      // sparks fade out
    var out = '';
    for (var r = 0; r < ROWS; r++) {
      var row = grid[r];
      for (var k = 0; k < COLS; k++) {
        var chr = row[k];
        if (decoding) {
          var edge = (t / DECODE) * (COLS + 20) - 10;
          var me = k + ((r * 7 + k * 13) % 9) - 4;
          if (me > edge) out += chr === ' ' ? ' ' : rnd(NOISE);
          else if (me > edge - 4 && chr !== ' ') out += '<em>' + rnd(NOISE) + '</em>';
          else out += chr;
          continue;
        }
        if (chr !== ' ') {
          if (hot) {
            var dx = k - mx, dy = (r - my) * 1.55, d = Math.sqrt(dx * dx + dy * dy);
            if (d < HOT_R && Math.random() < (1 - d / HOT_R) * heat * 0.16) {
              out += '<em>' + rnd(NOISE) + '</em>';
              continue;
            }
          }
          if (Math.abs(k * 0.62 + r - wave) < 3 && Math.random() < 0.6) {
            out += '<i>' + rnd(NOISE) + '</i>';
            continue;
          }
        }
        out += chr;
      }
      if (r < ROWS - 1) out += '\n';
    }
    pre.innerHTML = out;
    if (decoding || wave > -99 || hot) wake();
    else setTimeout(wake, Math.max(400, CYCLE - cycle));
  }
  wake();
})();
</script>
</body>
</html>
"""


def page(art, out_html):
    rows = [l.rstrip("\n") for l in open(art, encoding="utf-8")]
    while rows and not rows[-1].strip():
        rows.pop()
    cols = max(len(r) for r in rows)
    rows = [r.ljust(cols) for r in rows]
    html = (PAGE.replace("__DATA__", json.dumps(rows, ensure_ascii=True))
                .replace("__COLS__", str(cols))
                .replace("__ROWS__", str(len(rows))))
    open(out_html, "w", encoding="utf-8").write(html)
    return len(html), cols, len(rows)


# ------------------------------------------------------------------- cli ----
def _ensure(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="ASCII portrait pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("render")
    p1.add_argument("src"); p1.add_argument("out")
    p1.add_argument("--cols", type=int, default=300)
    p1.add_argument("--crop", default="0.07,0.015,0.80,0.60")
    p1.add_argument("--rows-factor", type=float, default=0.5)

    p2 = sub.add_parser("preview")
    p2.add_argument("art"); p2.add_argument("out")
    p2.add_argument("--cell", type=int, default=4)
    p2.add_argument("--font")

    p3 = sub.add_parser("page")
    p3.add_argument("art"); p3.add_argument("out")

    a = ap.parse_args()
    _ensure(a.out)
    if a.cmd == "render":
        lines = render(a.src, cols=a.cols,
                       crop=tuple(float(v) for v in a.crop.split(",")),
                       rows_f=a.rows_factor)
        open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"{len(lines)} rows x {max(len(l) for l in lines)} cols -> {a.out}")
    elif a.cmd == "preview":
        print(preview(a.art, a.out, a.cell, a.font))
    else:
        print(page(a.art, a.out))


if __name__ == "__main__":
    main()
