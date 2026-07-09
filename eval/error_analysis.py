"""Offline error analysis of a committed eval run — diagnosis before fixing.

Reads the per-question rows and buckets the mistakes into named failure modes,
names the dominant one, and surfaces the worst calls. This is the "open-code the
failures, fix the dominant category, report the delta" discipline — done from
committed data, no API calls.

Usage (from repo root)::

    python -m eval.error_analysis --results eval/results/consensus.json \
        --out eval/results/error_analysis.md
"""
from __future__ import annotations

import argparse
import json
import statistics

# Named failure buckets for a binary probabilistic forecaster.
_BUCKET_LABELS = {
    "missed_yes": "missed YES (resolved YES, forecast < 0.40 — under-forecast the event)",
    "missed_no": "missed NO (resolved NO, forecast > 0.60 — over-forecast the event)",
    "hedged_wrong": "hedged wrong (0.40–0.60, landed on the wrong side of 50%)",
    "ok": "correct side of 50%",
}


def analyze_errors(rows: list[dict], col: str = "consensus", worst_k: int = 8) -> dict:
    """Bucket per-question errors and summarize the dominant failure mode."""
    scored = [
        (r, (float(r[col]) - float(r["outcome"])) ** 2)
        for r in rows
        if r.get(col) is not None
    ]
    scored.sort(key=lambda x: -x[1])

    buckets: dict[str, list] = {k: [] for k in _BUCKET_LABELS}
    for r, _ in scored:
        p, o = float(r[col]), float(r["outcome"])
        if o == 1.0 and p < 0.40:
            buckets["missed_yes"].append(r)
        elif o == 0.0 and p > 0.60:
            buckets["missed_no"].append(r)
        elif (p - 0.5) * (o - 0.5) < 0:  # wrong side of 50%, milder
            buckets["hedged_wrong"].append(r)
        else:
            buckets["ok"].append(r)

    preds = [float(r[col]) for r, _ in scored]
    yes_preds = [float(r[col]) for r, _ in scored if float(r["outcome"]) == 1.0]
    no_preds = [float(r[col]) for r, _ in scored if float(r["outcome"]) == 0.0]
    error_counts = {k: len(v) for k, v in buckets.items() if k != "ok"}
    dominant = max(error_counts, key=lambda k: error_counts[k]) if any(error_counts.values()) else None

    return {
        "col": col,
        "n": len(scored),
        "min_pred": round(min(preds), 3) if preds else None,
        "max_pred": round(max(preds), 3) if preds else None,
        "avg_pred_on_yes": round(statistics.mean(yes_preds), 3) if yes_preds else None,
        "avg_pred_on_no": round(statistics.mean(no_preds), 3) if no_preds else None,
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "dominant_failure": dominant,
        "worst": [
            {
                "question": r["question"],
                "pred": round(float(r[col]), 3),
                "outcome": float(r["outcome"]),
                "sq_err": round(e, 3),
            }
            for r, e in scored[:worst_k]
        ],
    }


def to_markdown(a: dict) -> str:
    lines = [
        f"# Error analysis — `{a['col']}` (n={a['n']})",
        "",
        "Diagnosis of *where* the forecaster is wrong, from committed predictions "
        "(no API calls). Fix the dominant category, then re-measure the delta.",
        "",
        "## Prediction range",
        f"- forecasts span **{a['min_pred']} – {a['max_pred']}**",
        f"- mean forecast on questions that resolved **YES**: **{a['avg_pred_on_yes']}**",
        f"- mean forecast on questions that resolved **NO**: **{a['avg_pred_on_no']}**",
        "",
        "## Failure buckets",
    ]
    for k, label in _BUCKET_LABELS.items():
        lines.append(f"- **{a['bucket_counts'].get(k, 0)}** — {label}")
    dom = a["dominant_failure"]
    lines += [
        "",
        f"**Dominant failure mode:** `{dom}` — {_BUCKET_LABELS.get(dom, 'n/a')}"
        if dom else "**Dominant failure mode:** none (no wrong-side calls)",
        "",
        "## Worst calls (highest squared error)",
        "",
        "| squared err | forecast | resolved | question |",
        "|---:|---:|---:|---|",
    ]
    for w in a["worst"]:
        lines.append(
            f"| {w['sq_err']:.3f} | {w['pred']:.3f} | {int(w['outcome'])} | {w['question']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="eval/results/consensus.json")
    p.add_argument("--col", default="consensus")
    p.add_argument("--out", default="eval/results/error_analysis.md")
    args = p.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    a = analyze_errors(rows, col=args.col)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(to_markdown(a))
    print(f"dominant failure: {a['dominant_failure']}  buckets={a['bucket_counts']}")
    print(f"pred range {a['min_pred']}–{a['max_pred']}  "
          f"avg on YES={a['avg_pred_on_yes']} on NO={a['avg_pred_on_no']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
