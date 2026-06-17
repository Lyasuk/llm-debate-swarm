"""Classifies prediction market questions into types for specialized handling.

Question types:
- BARRIER: "Will X reach/hit/dip to Y level?" (price/value targets — first-hit option)
- DEADLINE_EVENT: "Will X happen by DATE?" (probabilistic event within a window)
- FIXED_DATE_EVENT: "Who wins the election?" (scheduled event on fixed date)
- HEAD_TO_HEAD: "A vs B" (discrete sports/competition outcome)

Each type has different time-sensitivity and reasoning requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("analysis.classifier")


class QuestionType(str, Enum):
    BARRIER = "barrier"              # price/value hits a level
    DEADLINE_EVENT = "deadline_event"  # event happens by date
    FIXED_DATE_EVENT = "fixed_date_event"  # election, scheduled event
    HEAD_TO_HEAD = "head_to_head"     # sports, 1v1 competition
    UNKNOWN = "unknown"


@dataclass
class QuestionClassification:
    """Result of classifying a prediction market question."""

    question_type: QuestionType
    confidence: float  # 0-1, how sure we are
    reasoning: str
    # Extracted entities
    asset: str | None = None  # "Bitcoin", "WTI Crude Oil", etc
    level: float | None = None  # target price/level
    direction: str | None = None  # "above", "below", "hit"
    deadline: str | None = None  # "April", "by end of April", etc


# ---------------------------------------------------------------------------
# Rule-based classifier (fast, deterministic, no LLM calls)
# ---------------------------------------------------------------------------

# BARRIER: "reach X", "hit X", "dip to X", "drop below X", "rise above X"
# Requires a price-like level ($ or >=2 digit number representing a price)
BARRIER_PATTERNS = [
    r"\breach\b.*\$\d",
    r"\bhit\b.*\$\d",
    r"\bdip\s+to\b.*\$\d",
    r"\bdrop\s+(below|to)\b.*\$\d",
    r"\brise\s+above\b.*\$\d",
    r"\bcross\b.*\$\d",
    r"\bexceed\b.*\$\d",
    r"\bfall\s+below\b.*\$\d",
    r"\bclimb\s+above\b.*\$\d",
    r"\btrade\s+(above|below|at)\b.*\$\d",
    # Price level queries: "$75,000 in April", "$1,800", "$130 in April"
    r"\$\d[\d,]*\s*(in|by|before)\b",
    r"price\s+of.*\$\d",
    r"\(HIGH\)\s+\$\d",  # Polymarket HIGH/LOW format
    r"\(LOW\)\s+\$\d",
]

# HEAD_TO_HEAD: Team A vs Team B, A vs. B, A v B
HEAD_TO_HEAD_PATTERNS = [
    r"\w+\s+vs\.?\s+\w+",  # "Dodgers vs. Blue Jays"
    r"\w+\s+v\s+\w+",  # "A v B"
    r"\bagainst\b.*\bwin\b",
    r"\bwins?\s+(the\s+)?(match|game|fight|set)",
]

# FIXED_DATE_EVENT: elections, scheduled events
FIXED_DATE_PATTERNS = [
    r"\b(presidential|parliamentary|general|mayoral)\s+election",
    r"\b(election|primary|runoff|referendum)\b",
    r"\bwin(s|ning)?\s+(the\s+)?(election|primary|seat|nomination)",
    r"\b(fomc|fed\s+meeting|rate\s+decision|fed.*rates?.*meeting)\b",
    r"\b(oscar|emmy|grammy|nobel|pulitzer)\s+award",
    r"\bfed\s+interest\s+rates?\b",
    r"\binterest\s+rates?.*after.*(fomc|meeting)\b",
    r"\b(rate\s+cut|rate\s+hike)\b",
    r"\bafter\s+the\s+\w+\s+\d{4}\s+(fomc|meeting)",
    r"\bwin\s+the\s+most\s+seats?\b",
    r"\b(parliament|parliamentary|senate|congress|bundestag|knesset|diet)\b",
    r"\bwon\s+by\b.*\b(party|coalition|candidate)\b",
]

# DEADLINE_EVENT: "by April", "before end of", "announces by"
DEADLINE_PATTERNS = [
    r"\bby\s+(end\s+of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})",
    r"\bbefore\s+(end\s+of|\w+\s+\d)",
    r"\bannounce[sd]?\b",
    r"\bsign[sd]?\s+(a|an|the)",
    r"\bpass(es|ed)?\b.*\b(bill|law|act|resolution)",
    r"\bapprove[sd]?\b",
    r"\bceasefire\b",
    r"\bresume[sd]?\b",
    r"\breturn[sd]?\s+to\b",
]


def _match_patterns(text: str, patterns: list[str]) -> int:
    """Count how many patterns match. Returns 0 if none."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _extract_price_level(text: str) -> float | None:
    """Extract a price/value level from the question.

    Examples:
    - "Will Bitcoin reach $75,000 in April?" -> 75000.0
    - "WTI Crude hit $130" -> 130.0
    - "Ethereum dip to $1,800" -> 1800.0
    """
    # Match $ followed by number with optional commas
    m = re.search(r"\$?([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_direction(text: str) -> str | None:
    """Detect direction of barrier: above/below/hit."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["reach", "rise above", "climb above", "exceed", "hit (high)", "cross above"]):
        return "above"
    if any(w in text_lower for w in ["dip to", "drop below", "fall below", "hit (low)", "decline to"]):
        return "below"
    if "hit" in text_lower:
        return "hit"
    return None


def _extract_asset(text: str) -> str | None:
    """Extract the main asset/entity being asked about."""
    text_lower = text.lower()
    crypto = {"bitcoin": "Bitcoin", "btc": "Bitcoin", "ethereum": "Ethereum",
              "eth": "Ethereum", "solana": "Solana", "sol": "Solana"}
    for k, v in crypto.items():
        if k in text_lower:
            return v

    commodities = {"wti crude": "WTI Crude Oil", "wti": "WTI Crude Oil",
                   "brent": "Brent Crude", "gold": "Gold", "silver": "Silver",
                   "natural gas": "Natural Gas", "copper": "Copper"}
    for k, v in commodities.items():
        if k in text_lower:
            return v

    return None


def classify_question(question: str, category: str = "") -> QuestionClassification:
    """Classify a prediction market question using rule-based heuristics.

    Returns:
        QuestionClassification with type, confidence, and extracted entities.
    """
    q = question.strip()
    cat_lower = (category or "").lower()

    # Count pattern matches for each type
    barrier_score = _match_patterns(q, BARRIER_PATTERNS)
    h2h_score = _match_patterns(q, HEAD_TO_HEAD_PATTERNS)
    fixed_score = _match_patterns(q, FIXED_DATE_PATTERNS)
    deadline_score = _match_patterns(q, DEADLINE_PATTERNS)

    # Apply category biases
    if "sport" in cat_lower:
        h2h_score += 2
    if "polit" in cat_lower or "election" in cat_lower:
        fixed_score += 2

    # Determine type (barrier takes precedence if price level detected)
    scores = {
        QuestionType.BARRIER: barrier_score,
        QuestionType.HEAD_TO_HEAD: h2h_score,
        QuestionType.FIXED_DATE_EVENT: fixed_score,
        QuestionType.DEADLINE_EVENT: deadline_score,
    }

    # Barrier detection has priority if we find a price level AND barrier patterns
    if barrier_score >= 1 and _extract_price_level(q) is not None:
        best = QuestionType.BARRIER
        total = sum(scores.values())
        confidence = min(0.95, 0.6 + (barrier_score / (total + 1)) * 0.35)
    elif h2h_score >= 2:  # strong H2H signal
        best = QuestionType.HEAD_TO_HEAD
        confidence = 0.85
    elif fixed_score >= 2:  # strong election signal
        best = QuestionType.FIXED_DATE_EVENT
        confidence = 0.85
    else:
        # Pick highest-scoring type
        best = max(scores, key=lambda k: scores[k])
        max_score = scores[best]
        if max_score == 0:
            return QuestionClassification(
                question_type=QuestionType.UNKNOWN,
                confidence=0.0,
                reasoning="No patterns matched",
            )
        total = sum(scores.values())
        confidence = min(0.90, 0.5 + (max_score / (total + 1)) * 0.40)

    # Extract entities
    asset = _extract_asset(q) if best == QuestionType.BARRIER else None
    level = _extract_price_level(q) if best == QuestionType.BARRIER else None
    direction = _extract_direction(q) if best == QuestionType.BARRIER else None

    reasoning_parts = []
    if best == QuestionType.BARRIER:
        reasoning_parts.append(f"Price barrier question ({direction} {level} on {asset or 'asset'})")
    elif best == QuestionType.HEAD_TO_HEAD:
        reasoning_parts.append("Head-to-head competition (sports/1v1)")
    elif best == QuestionType.FIXED_DATE_EVENT:
        reasoning_parts.append("Fixed-date event (election/scheduled)")
    elif best == QuestionType.DEADLINE_EVENT:
        reasoning_parts.append("Deadline-based event (happens by date)")

    result = QuestionClassification(
        question_type=best,
        confidence=confidence,
        reasoning=" | ".join(reasoning_parts) or "Default classification",
        asset=asset,
        level=level,
        direction=direction,
    )

    log.info(
        f"Classified '{question[:60]}...' as {best.value} "
        f"(conf={confidence:.2f})"
    )

    return result


# ---------------------------------------------------------------------------
# Type-specific guidance blocks (injected into prompts)
# ---------------------------------------------------------------------------

BARRIER_GUIDANCE_TEMPLATE = """\
## QUESTION TYPE: BARRIER (price/value reaches a target)
This question asks: will {asset} {direction} {level} ONCE (any time) during the {days:.0f}-day window?
This is a FIRST-HIT option, NOT an end-state measurement.

**CRITICAL time-decay reasoning:**
- Current price: check research for current {asset} price
- Target: {level} ({direction})
- Time window: {days:.0f} days remaining
- The probability grows NON-LINEARLY with time — more days = more volatility = more chances to hit
- BUT each day without movement reduces remaining probability (theta decay)

**DO:**
- Think about typical DAILY volatility of {asset}
- Calculate how far from target we are in standard deviation terms
- Longer time windows favor reaching the target; short windows favor no-hit
- Use geometric Brownian motion intuition: P(hit) ≈ 2 * P(end above target)

**DO NOT:**
- Treat this as "will price be at target on day {days:.0f}" (that's end-state, wrong framing)
- Ignore volatility or time window
- Assume linear probability scaling"""

DEADLINE_EVENT_GUIDANCE_TEMPLATE = """\
## QUESTION TYPE: DEADLINE EVENT (does X happen by DATE)
This question asks: will the specific event occur within the next {days:.0f} days?

**CRITICAL time-decay reasoning:**
- Time remaining: {days:.0f} days
- Does this type of event typically happen quickly or slowly?
- What's the BASE RATE for similar events occurring in a ~{days:.0f}-day window?
- Is there a forcing function (deadline, election, meeting) that concentrates probability?

**DO:**
- Estimate P(event | unlimited time) first — the unconditional probability
- Then estimate P(event happens in next {days:.0f} days | event happens eventually)
- Multiply: P(event by deadline) = P(event eventually) × P(timing | eventually)
- For {days:.0f} days, consider news cycle, diplomatic pace, institutional timing

**DO NOT:**
- Assume event is imminent without evidence
- Ignore that {days:.0f} days may be insufficient for slow processes (legislation, treaties)
- Use full unconditional probability without time adjustment"""

FIXED_DATE_GUIDANCE_TEMPLATE = """\
## QUESTION TYPE: FIXED-DATE EVENT (scheduled election/meeting)
This question asks about a SCHEDULED outcome (election, FOMC meeting, etc) that occurs on a fixed date.

**Time context:**
- Days until resolution: {days:.0f}
- The EVENT date is fixed — time only affects information uncertainty, not event probability
- With {days:.0f} days remaining, polling/signals should be {signal_quality}

**DO:**
- Focus on fundamentals: polls, betting markets, expert consensus, base rates
- Weight RECENT information more heavily (last 7-14 days)
- If {days:.0f} > 30: more uncertainty, many things can change
- If {days:.0f} < 7: rely heavily on latest polls/signals

**DO NOT:**
- Overweight early polls when date is far away
- Ignore institutional dynamics (incumbency, coalition politics)
- Assume time decay — event will happen regardless"""

HEAD_TO_HEAD_GUIDANCE_TEMPLATE = """\
## QUESTION TYPE: HEAD-TO-HEAD (sports/1v1 competition)
This question is a DISCRETE competition outcome between two parties.

**Time context:**
- Days until match: {days:.0f}
- Time is LESS important here — match happens at fixed time
- Focus on FORM, not time decay

**DO:**
- Look up recent head-to-head records
- Consider Elo ratings / form / injuries / conditions
- Weight last 10 matches of each competitor heavily
- For {days:.0f} days out: consider injury recovery, weather, venue effects

**DO NOT:**
- Apply time decay reasoning
- Overthink — in sports, form > narrative
- Assume underdog wins unless strong specific reason"""

UNKNOWN_GUIDANCE_TEMPLATE = """\
## QUESTION TYPE: UNCLASSIFIED
Could not automatically classify this question type. Use general probabilistic reasoning.

**Time context:** {days:.0f} days until resolution

**DO:**
- Establish a base rate from reference class
- Update on specific evidence
- Consider whether {days:.0f} days is enough time for the outcome
- Balance time constraints against the event's typical duration"""


def build_time_guidance(
    classification: QuestionClassification,
    days_to_resolution: float,
) -> str:
    """Build the type-specific time-aware guidance block for prompts."""
    days = max(0.5, days_to_resolution)  # avoid 0

    if classification.question_type == QuestionType.BARRIER:
        return BARRIER_GUIDANCE_TEMPLATE.format(
            asset=classification.asset or "the asset",
            direction=classification.direction or "reach",
            level=f"${classification.level:,.0f}" if classification.level else "the target",
            days=days,
        )
    elif classification.question_type == QuestionType.DEADLINE_EVENT:
        return DEADLINE_EVENT_GUIDANCE_TEMPLATE.format(days=days)
    elif classification.question_type == QuestionType.FIXED_DATE_EVENT:
        signal_quality = (
            "very high (event is imminent)" if days <= 7
            else "moderately reliable (polls matter)" if days <= 30
            else "lower quality (many things can change)"
        )
        return FIXED_DATE_GUIDANCE_TEMPLATE.format(
            days=days, signal_quality=signal_quality
        )
    elif classification.question_type == QuestionType.HEAD_TO_HEAD:
        return HEAD_TO_HEAD_GUIDANCE_TEMPLATE.format(days=days)
    else:
        return UNKNOWN_GUIDANCE_TEMPLATE.format(days=days)
