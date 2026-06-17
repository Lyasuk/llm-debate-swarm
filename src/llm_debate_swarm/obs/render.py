"""Render captured spans as a waterfall trace diagram (SVG, stdlib only).

A span's horizontal position/length encodes when it ran and for how long;
indentation encodes nesting — i.e. a normal APM/trace waterfall.
"""
from __future__ import annotations

import json

_COLORS = {
    "forecast": "#111827",
    "classify": "#6b7280",
    "research": "#16a34a",
    "consensus": "#2563eb",
    "llm.query": "#3b82f6",
    "swarm": "#dc2626",
}


def _color(name: str) -> str:
    return _COLORS.get(name, "#3b82f6")


def render(spans: list[dict], width: int = 900, row_h: int = 26, label_w: int = 230) -> str:
    if not spans:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'

    by_id = {s["span_id"]: s for s in spans}

    def depth(s: dict) -> int:
        d, p = 0, s.get("parent_id")
        while p and p in by_id:
            d += 1
            p = by_id[p].get("parent_id")
        return d

    spans = sorted(spans, key=lambda s: s["start_ns"])
    t0 = min(s["start_ns"] for s in spans)
    t1 = max(s["end_ns"] for s in spans)
    span_total = max(t1 - t0, 1)
    total_ms = span_total / 1e6

    pad_l, pad_r, pad_t = 20, 20, 46
    plot_w = width - pad_l - pad_r - label_w
    axis_x0 = pad_l + label_w
    height = pad_t + row_h * len(spans) + 30

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{pad_l}" y="22" font-size="14" font-weight="bold" fill="#111">'
        f'debate-swarm forecast trace — {total_ms:.0f} ms, {len(spans)} spans</text>',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = axis_x0 + frac * plot_w
        out.append(f'<line x1="{gx:.0f}" y1="{pad_t-6}" x2="{gx:.0f}" y2="{height-22}" stroke="#eee"/>')
        out.append(f'<text x="{gx:.0f}" y="{height-8}" text-anchor="middle" fill="#999">'
                   f'{frac*total_ms:.0f}ms</text>')

    for i, s in enumerate(spans):
        y = pad_t + i * row_h
        x = axis_x0 + (s["start_ns"] - t0) / span_total * plot_w
        w = max((s["end_ns"] - s["start_ns"]) / span_total * plot_w, 2)
        attrs = s.get("attributes", {})
        err = s.get("status") == "ERROR" or bool(attrs.get("llm.error"))
        label = ("  " * depth(s)) + s["name"]
        out.append(f'<text x="{pad_l}" y="{y+row_h/2+4:.0f}" fill="#333">{label[:34]}</text>')
        out.append(f'<rect x="{x:.0f}" y="{y+5:.0f}" width="{w:.0f}" height="{row_h-12}" rx="3" '
                   f'fill="{_color(s["name"])}" opacity="{0.45 if err else 0.85}"/>')
        meta = f'{s["duration_ms"]:.0f}ms'
        if s["name"] == "llm.query":
            meta += f' {attrs.get("llm.model", "")}'
            if attrs.get("llm.probability") is not None:
                meta += f' p={float(attrs["llm.probability"]):.2f}'
            if attrs.get("llm.error"):
                meta += ' ERR'
        out.append(f'<text x="{x+w+6:.0f}" y="{y+row_h/2+4:.0f}" fill="#555" font-size="11">'
                   f'{meta[:44]}</text>')

    out.append('</svg>')
    return "\n".join(out)


def render_file(src: str, dst: str) -> int:
    with open(src, encoding="utf-8") as f:
        spans = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(render(spans))
    return len(spans)
