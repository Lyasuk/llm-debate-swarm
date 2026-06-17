"""Render a calibration reliability diagram as SVG (stdlib only) from an eval
results JSON. Points on the diagonal = perfectly calibrated.

Usage:
    python eval/plot_calibration.py eval/results/consensus.json eval/results/calibration.svg
"""
from __future__ import annotations

import json
import sys

COLORS = {"consensus": "#2563eb", "swarm": "#dc2626", "combined": "#16a34a"}


def _xy(p: float, W: int, H: int, pad: int) -> tuple[float, float]:
    x = pad + p * (W - 2 * pad)
    y = (H - pad) - p * (H - 2 * pad)  # invert y
    return x, y


def render(report: dict, W: int = 460, H: int = 460, pad: int = 50) -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # axes box
    x0, y0 = pad, H - pad
    x1, y1 = W - pad, pad
    s.append(f'<rect x="{x0}" y="{y1}" width="{x1-x0}" height="{y0-y1}" fill="none" stroke="#ccc"/>')
    # perfect-calibration diagonal
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#999" '
             f'stroke-dasharray="5,4"/>')
    # grid + ticks
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx, _ = _xy(t, W, H, pad)
        _, gy = _xy(t, W, H, pad)
        s.append(f'<line x1="{gx}" y1="{y0}" x2="{gx}" y2="{y1}" stroke="#f0f0f0"/>')
        s.append(f'<line x1="{x0}" y1="{gy}" x2="{x1}" y2="{gy}" stroke="#f0f0f0"/>')
        s.append(f'<text x="{gx}" y="{y0+16}" text-anchor="middle" fill="#666">{t:.2f}</text>')
        s.append(f'<text x="{x0-8}" y="{gy+4}" text-anchor="end" fill="#666">{t:.2f}</text>')
    # axis labels
    s.append(f'<text x="{(x0+x1)/2}" y="{H-12}" text-anchor="middle" fill="#333">'
             f'predicted P(YES)</text>')
    s.append(f'<text x="14" y="{(y0+y1)/2}" text-anchor="middle" fill="#333" '
             f'transform="rotate(-90 14 {(y0+y1)/2})">observed YES frequency</text>')
    # series
    legend_y = y1 + 6
    for col in ("consensus", "swarm", "combined"):
        if col not in report:
            continue
        color = COLORS[col]
        pts = []
        for b in report[col]["calibration"]:
            if b["n"] == 0 or b["avg_pred"] is None:
                continue
            px, py = _xy(b["avg_pred"], W, H, pad)
            pts.append((px, py, b["n"]))
        if len(pts) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
            s.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                     f'stroke-width="2" opacity="0.8"/>')
        for x, y, n in pts:
            r = 3 + min(n, 20) ** 0.5
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" opacity="0.75"/>')
        brier = report[col]["brier"]
        s.append(f'<circle cx="{x1-150}" cy="{legend_y}" r="5" fill="{color}"/>')
        s.append(f'<text x="{x1-140}" y="{legend_y+4}" fill="#333">'
                 f'{col} (Brier {brier:.3f}, n={report[col]["n"]})</text>')
        legend_y += 18
    s.append('</svg>')
    return "\n".join(s)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "eval/results/consensus.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "eval/results/calibration.svg"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    report = data["report"]
    with open(dst, "w", encoding="utf-8") as f:
        f.write(render(report))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
