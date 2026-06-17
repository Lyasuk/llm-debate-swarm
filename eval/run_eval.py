"""Run the debate-swarm engine over the ground-truth questions and report
calibration: Brier score, log-loss, and a reliability table — for the multi-LLM
consensus, the debate swarm, and/or their combination.

A single `combined` run yields all three predictions per question (the Verdict
carries consensus_probability, swarm_probability and the combined probability),
so the consensus-vs-swarm comparison costs one engine run per question, not three.

Usage (from repo root):
    python -m eval.run_eval --config consensus            # cheap: all questions
    python -m eval.run_eval --config combined --limit 12  # swarm comparison on a subset
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import yaml

from eval.metrics import base_rate_brier, brier, calibration_table, log_loss
from llm_debate_swarm.config import load_config
from llm_debate_swarm.engine import DebateSwarmEngine


def load_questions(path: str, limit: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    qs = data["questions"]
    return qs[:limit] if limit else qs


async def run(args) -> None:
    qs = load_questions(args.questions, args.limit)
    use_swarm = args.config in ("swarm", "combined")
    use_consensus = args.config in ("consensus", "combined")
    cfg = load_config(args.config_yaml) if args.config_yaml else None
    engine = DebateSwarmEngine(cfg, use_consensus=use_consensus, use_swarm=use_swarm)

    rows: list[dict] = []
    for i, q in enumerate(qs, 1):
        t0 = time.time()
        try:
            v = await engine.forecast(q["question"])
            rows.append({
                "question": q["question"], "outcome": float(q["outcome"]),
                "consensus": v.consensus_probability, "swarm": v.swarm_probability,
                "combined": v.probability, "cost": v.cost_usd,
            })
            print(f"[{i}/{len(qs)}] {time.time()-t0:4.1f}s  "
                  f"comb={v.probability:.2f} cons={v.consensus_probability} sw={v.swarm_probability} "
                  f"true={int(q['outcome'])}  {q['question'][:46]}")
        except Exception as e:
            print(f"[{i}/{len(qs)}] FAILED: {e}  {q['question'][:46]}")

    report: dict = {"n_questions": len(rows), "config": args.config}
    for col in ("consensus", "swarm", "combined"):
        preds = [r[col] for r in rows if r[col] is not None]
        outs = [r["outcome"] for r in rows if r[col] is not None]
        if len(preds) >= 5:
            report[col] = {
                "n": len(preds),
                "brier": round(brier(preds, outs), 4),
                "log_loss": round(log_loss(preds, outs), 4),
                "base_rate_brier": round(base_rate_brier(outs), 4),
                "calibration": calibration_table(preds, outs),
            }
    report["total_cost_usd"] = round(sum(r["cost"] for r in rows), 4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "rows": rows}, f, indent=2)

    print("\n=== REPORT ===")
    for col in ("consensus", "swarm", "combined"):
        if col in report:
            m = report[col]
            skill = "better" if m["brier"] < m["base_rate_brier"] else "worse"
            print(f"{col:9s} n={m['n']:3d}  Brier={m['brier']:.4f}  "
                  f"(base-rate {m['base_rate_brier']:.4f} -> {skill})  log-loss={m['log_loss']:.4f}")
    print(f"saved -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/questions.yaml")
    p.add_argument("--config", default="consensus", choices=["consensus", "swarm", "combined"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--config-yaml", default="", help="Optional config file (e.g. eval/config.eval.yaml).")
    p.add_argument("--out", default="eval/results/eval.json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
