"""Fetch resolved binary questions from the public Polymarket Gamma API as eval
ground truth.

Public market data only — each item is just (question, resolved outcome, date,
volume). No positions, sizing, or strategy. The resulting questions.yaml is the
committed, reproducible ground-truth set for the eval harness.

Filters for *substantive* questions: real Yes/No markets with meaningful traded
volume, recently resolved (by ``closedTime``), excluding high-frequency micro
markets (5-minute "Up or Down", exact-score sports, etc.).

Usage:
    python eval/fetch_questions.py                 # writes eval/questions.yaml
    python eval/fetch_questions.py 50 2026-01-01 30000
"""
from __future__ import annotations

import json
import re
import sys

import httpx
import yaml

GAMMA = "https://gamma-api.polymarket.com/markets"

# High-frequency / micro-market noise we do NOT want as forecasting questions.
JUNK = re.compile(
    r"\b(up or down|exact score|map handicap|games total|o/u|set \d+ winner|"
    r"moneyline|1st half|halftime|race to|winning margin|correct score)\b",
    re.IGNORECASE,
)


def _as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return None
    return v


def fetch(target: int = 50, min_close: str = "2026-01-01", min_volume: float = 30000.0):
    out: list[dict] = []
    seen: set[str] = set()
    prefix_count: dict[str, int] = {}
    offset = 0
    with httpx.Client(timeout=30.0, headers={"User-Agent": "llm-debate-swarm-eval"}) as c:
        while len(out) < target and offset < 10000:
            r = c.get(GAMMA, params={
                "closed": "true", "limit": 100, "offset": offset,
                "order": "volumeNum", "ascending": "false",
            })
            r.raise_for_status()
            markets = r.json()
            if not markets:
                break
            offset += 100
            for m in markets:
                try:
                    outcomes = _as_list(m.get("outcomes"))
                    prices = _as_list(m.get("outcomePrices"))
                    if not outcomes or [str(o).lower() for o in outcomes] != ["yes", "no"]:
                        continue
                    if not prices or len(prices) != 2:
                        continue
                    yes_p = float(prices[0])
                    if yes_p > 0.97:
                        outcome = 1.0
                    elif yes_p < 0.03:
                        outcome = 0.0
                    else:
                        continue
                    closed = (m.get("closedTime") or "")[:10]
                    if not closed or closed < min_close:
                        continue
                    try:
                        vol = float(m.get("volumeNum") or 0)
                    except (ValueError, TypeError):
                        continue
                    if vol < min_volume:
                        continue
                    q = (m.get("question") or "").strip()
                    if len(q) < 12 or q in seen or JUNK.search(q):
                        continue
                    pref = " ".join(q.lower().split()[:2])
                    if prefix_count.get(pref, 0) >= 2:  # topic diversity
                        continue
                    prefix_count[pref] = prefix_count.get(pref, 0) + 1
                    seen.add(q)
                    out.append({
                        "question": q,
                        "outcome": outcome,
                        "resolution_date": closed,
                        "volume_usd": round(vol),
                        "market_id": str(m.get("id") or ""),
                    })
                except Exception:
                    continue
    return out[:target]


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    min_close = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    min_volume = float(sys.argv[3]) if len(sys.argv) > 3 else 30000.0
    qs = fetch(target=target, min_close=min_close, min_volume=min_volume)
    with open("eval/questions.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"_meta": {
                "source": "Polymarket Gamma API (public resolved markets)",
                "min_resolution_date": min_close,
                "min_volume_usd": min_volume,
                "count": len(qs),
                "note": "Public question+outcome only. Some questions may predate model "
                        "training cutoffs (leakage); the consensus-vs-swarm comparison is "
                        "leakage-robust since all configs see the same questions.",
            }, "questions": qs},
            f, sort_keys=False, allow_unicode=True,
        )
    yes = sum(1 for q in qs if q["outcome"] == 1.0)
    print(f"wrote {len(qs)} -> eval/questions.yaml  (YES={yes}, NO={len(qs)-yes})")
    print("date range:",
          min((q["resolution_date"] for q in qs), default="-"), "..",
          max((q["resolution_date"] for q in qs), default="-"))
    for q in qs[:18]:
        print(f"  [{int(q['outcome'])}] ({q['resolution_date']}) {q['question'][:62]}")


if __name__ == "__main__":
    main()
