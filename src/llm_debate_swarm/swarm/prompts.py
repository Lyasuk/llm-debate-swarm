"""Prompt templates for the 7-round swarm debate with debiasing and time-awareness."""

# ---------------------------------------------------------------------------
# Round 1 — BLIND (no market price shown)
# ---------------------------------------------------------------------------

ROUND1_BLIND_PROMPT = """\
## PREDICTION QUESTION
{question}

## TIME CONTEXT (CRITICAL)
- **Days until resolution: {days_to_resolution:.1f}**
- You MUST reference this in your reasoning.

## RESOLUTION CRITERIA
{resolution_source}

{type_guidance}

## RESEARCH CONTEXT
{research_summary}

---

You have NOT been shown the current market price. \
Form your estimate purely from the research and the time window above.

Steps:
1. Identify the BASE RATE for similar historical events (in similar time windows).
2. Assess whether {days_to_resolution:.0f} days is sufficient for the outcome.
3. List 2 strongest reasons FOR (YES) and 2 AGAINST (NO).
4. Give your calibrated probability for YES.

CRITICAL — PROBABILITY CALIBRATION RULES:
- Use FINE-GRAINED probabilities like 0.234, 0.487, 0.628 — NOT round numbers like 0.10, 0.25, 0.50.
- Round numbers (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75) signal lazy thinking. AVOID them.
- Your number should reflect your specific confidence — if you're "around 30%", commit to 0.287 or 0.314, not 0.30.
- Valid range: 0.020 to 0.980. Three decimal places required.

Output ONLY this JSON (no markdown, no explanation outside JSON):
```json
{{
  "base_rate": 0.NNN,
  "time_sufficient": true/false,
  "arguments_for": ["arg1", "arg2"],
  "arguments_against": ["arg1", "arg2"],
  "probability": 0.NNN,
  "confidence": "low|medium|high",
  "reasoning": "1-2 sentences (must reference time window)",
  "key_factor": "single biggest unknown"
}}
```
"""

# ---------------------------------------------------------------------------
# Rounds 2-3 — DEBATE without market price
# ---------------------------------------------------------------------------

DEBATE_BLIND_PROMPT = """\
## PREDICTION QUESTION
{question}

## Round {round_num}/7 — DEBATE (market price still hidden)
## Days until resolution: {days_to_resolution:.1f}

{type_guidance_compact}

## Previous Round — Agent Positions
{debate_summary}

## Group Statistics
Mean: {mean:.1%} | Median: {median:.1%} | Range: [{min_est:.1%} – {max_est:.1%}]
Above 50%: {count_above} agents | Below 50%: {count_below} agents

{devils_advocate_instruction}

You still have NOT seen the market price. \
Update your estimate based on the debate above.
Do NOT simply move toward the group mean — independent thinking is valued.
If you change your estimate, explain WHY.

CRITICAL — PROBABILITY CALIBRATION:
- Use FINE-GRAINED probabilities (0.NNN with 3 decimals) like 0.342, 0.617.
- AVOID round numbers (0.10, 0.25, 0.50). They indicate lazy thinking.
- Your number must reflect specific evidence strength, not a rough bucket.

Output ONLY this JSON:
```json
{{
  "probability": 0.NNN,
  "reasoning": "1-2 sentences (consider time window)",
  "changed": true/false,
  "change_reason": "why you changed or held firm"
}}
```
"""

# ---------------------------------------------------------------------------
# Rounds 4-5 — DEBATE with market price revealed
# ---------------------------------------------------------------------------

PRICE_AWARE_PROMPT = """\
## PREDICTION QUESTION
{question}

## Round {round_num}/7 — Market Price Revealed
## Days until resolution: {days_to_resolution:.1f}

{type_guidance_compact}

## ADDITIONAL DATA: Current Market Price
The Polymarket consensus currently prices this at {yes_price:.1%} YES / {no_price:.1%} NO.
This is one data point among many — neither anchor on it nor automatically oppose it.

## Previous Round — Agent Positions
{debate_summary}

## Your Previous Estimate: {prev_estimate:.1%}

{devils_advocate_instruction}

YOUR TASK:
Give your BEST CALIBRATED probability estimate. The market price is just \
information about what other traders believe — it has no special authority. \
Most of the time you should be CLOSE to the market (markets are usually right), \
sometimes you'll find genuine mispricing (your edge). \
Move from your previous estimate ONLY if the market price reveals new information \
you hadn't considered (e.g., other traders may know something you don't).

CRITICAL — PROBABILITY CALIBRATION:
- Output a FINE-GRAINED probability (3 decimals): 0.342, 0.587, 0.781.
- Do NOT output extreme values (0.99, 0.01) unless evidence is overwhelming.
- Do NOT output round numbers (0.10, 0.25, 0.50).
- Most realistic estimates are between 0.15 and 0.85.

Output ONLY this JSON:
```json
{{
  "probability": 0.NNN,
  "reasoning": "1-2 sentences (reference time window)",
  "agree_with_market": true/false,
  "market_diff_explanation": "if your estimate differs from market, why; if close, what evidence supports agreement"
}}
```
"""

# ---------------------------------------------------------------------------
# Round 6 — PRE-MORTEM
# ---------------------------------------------------------------------------

PRE_MORTEM_PROMPT = """\
## PREDICTION QUESTION
{question}

## Your Current Estimate: {prev_estimate:.1%}
## Market Price: {yes_price:.1%} YES
## Days until resolution: {days_to_resolution:.1f}

## PRE-MORTEM EXERCISE

{premortem_direction}

1. Write a brief, PLAUSIBLE explanation of HOW this happened (2-3 sentences).
2. What did you MISS or UNDERWEIGHT in your analysis?
3. Now update your probability estimate. It is OK to change significantly.

CRITICAL — PROBABILITY CALIBRATION:
- Use FINE-GRAINED probability (3 decimals): 0.234, 0.617.
- Avoid round numbers (0.10, 0.25, 0.50).

Output ONLY this JSON:
```json
{{
  "premortem_explanation": "how the opposite happened",
  "missed_factor": "what you underweighted",
  "probability": 0.NNN,
  "reasoning": "why your estimate changed (or didn't)",
  "confidence": "low|medium|high"
}}
```
"""

# ---------------------------------------------------------------------------
# Round 7 — FINAL
# ---------------------------------------------------------------------------

FINAL_PROMPT = """\
## PREDICTION QUESTION
{question}

## Market Price: {yes_price:.1%} YES
## Days until resolution: {days_to_resolution:.1f}
## Your Estimate Trajectory: {trajectory}

## Pre-Mortem Arguments From All Agents
{premortem_summary}

Give your FINAL calibrated probability.
Consider:
- Did the pre-mortem arguments reveal blind spots in your reasoning?
- Does the {days_to_resolution:.0f}-day time window affect your final estimate?

Rate confidence 0-100 (where 80 means you'd be WRONG 20% of the time).

CRITICAL — PROBABILITY CALIBRATION:
- Use FINE-GRAINED probability (3 decimals): 0.357, 0.682.
- Most realistic estimates are 0.10 to 0.90 — extreme values need overwhelming evidence.
- Avoid round numbers (0.10, 0.25, 0.50).

Output ONLY this JSON:
```json
{{
  "probability": 0.NNN,
  "confidence_score": 0-100,
  "final_reasoning": "2-3 sentences (reference time window)",
  "key_uncertainty": "single factor that could change outcome"
}}
```
"""

# ---------------------------------------------------------------------------
# Meta-synthesis — GPT-4o (single call)
# ---------------------------------------------------------------------------

META_SYNTHESIS_SYSTEM = """\
You are a calibration expert and superforecaster reviewing a structured \
prediction debate among 40 AI agents. Your job is to produce a FINAL \
calibrated probability that corrects for systematic biases in the debate.

You ALWAYS consider the time-to-resolution window when calibrating. For barrier \
(price-target) questions you apply first-hit probability reasoning; for fixed-date \
events you focus on fundamentals; for head-to-head competitions you focus on form."""

META_SYNTHESIS_PROMPT = """\
## Question
{question}

## Question Type: {question_type}
## Days Until Resolution: {days_to_resolution:.1f}

{type_guidance}

## Market Price: {yes_price:.1%} YES / {no_price:.1%} NO
## Resolution Criteria: {resolution_source}

## Agent Final Estimates (40 agents, sorted by probability)
{agent_finals}

## Debate Statistics
- Blind estimate mean (R1-3, no market price): {blind_mean:.1%}
- Price-aware mean (R4-5 + R7, EXCLUDES pre-mortem R6): {aware_mean:.1%}
- **ROBUST TRIMMED MEAN** (R7, excludes top 2 & bottom 2 outliers): {trimmed_mean:.1%}
- **MEDIAN** (R7): {median:.1%}
- Anchoring shift: {anchoring_shift:.1%} (blind→aware, toward market)
- Std dev Round 1: {std_r1:.3f} → Round 7: {std_r7:.3f} (convergence: {convergence:.0%})
- Pre-mortem impact: {premortem_changed} of {agent_count} agents changed >5% after pre-mortem

## ⚠️ OUTLIER WARNING
Agent estimates may include model-specific outliers (e.g., one model family saying 1%, another saying 95%).
**DO NOT anchor on extreme ends of the distribution.** Use TRIMMED MEAN ({trimmed_mean:.1%}) or MEDIAN ({median:.1%})
as your starting point. Outliers are often model hallucination, not signal.

Bimodal distributions (std > 0.20) indicate DISAGREEMENT across model families, not insight.
When bimodal, prefer the cluster that aligns with base rates and type_guidance.

## Strongest Arguments FOR (YES) — from agents
{top_arguments_for}

## Strongest Arguments AGAINST (NO) — from agents
{top_arguments_against}

## Pre-mortem Failure Modes (most cited)
{premortem_modes}

---

## BIAS CHECKLIST (evaluate each before giving your estimate)

1. **ANCHORING**: Shift of {anchoring_shift:.1%} from blind to price-aware. \
If >10%, agents anchored to market. Trust blind estimates more.
2. **GROUPTHINK**: Std dev change of {convergence:.0%} from R1→R7. \
If <40%, convergence may be artificial — agents herded.
3. **NARRATIVE vs DATA**: Which arguments above rely on compelling stories \
vs hard data? Downweight narrative-driven arguments.
4. **OVERCONFIDENCE**: Pre-mortem revealed {premortem_changed} plausible \
failure modes. If agents ignored them, the group is overconfident.
5. **BASE RATE NEGLECT**: Is the group estimate far from historical base rate \
without extraordinary justification?
6. **TIME MISCALIBRATION**: Given {days_to_resolution:.0f} days remaining, is the \
group's estimate consistent with the question type? \
Barrier questions: longer windows = higher probability (non-linear). \
Deadline events: shorter windows may be insufficient.

## YOUR TASK
Correct for identified biases. Give FINAL calibrated probability with 3 decimals (0.NNN).

Output ONLY this JSON:
```json
{{
  "probability": 0.NNN,
  "confidence": 0.NNN,
  "reasoning": "2-3 sentences explaining your calibrated estimate (must reference time window)",
  "bias_corrections": {{
    "anchoring_detected": true/false,
    "anchoring_correction": 0.NNN,
    "groupthink_detected": true/false,
    "groupthink_correction": 0.NNN,
    "overconfidence_detected": true/false,
    "time_miscalibration_detected": true/false,
    "base_rate_used": 0.NNN
  }}
}}
```
"""

# ---------------------------------------------------------------------------
# Devil's advocate instruction (injected into debate prompts)
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_INSTRUCTION = """\
**SPECIAL ROLE THIS ROUND: DEVIL'S ADVOCATE**
Regardless of your personal estimate, your job this round is to find the \
STRONGEST possible argument AGAINST the current group consensus ({consensus_direction}). \
Steel-man the minority position. Be intellectually rigorous, not contrarian for its own sake. \
Still give your honest probability (do NOT artificially shift it), but ensure your \
reasoning challenges the majority."""


# ---------------------------------------------------------------------------
# Compact type guidance for debate rounds (2-5) — shorter to save tokens
# ---------------------------------------------------------------------------

def compact_type_guidance(question_type: str, days: float) -> str:
    """Shorter type guidance for debate rounds to save tokens."""
    if question_type == "barrier":
        return (
            f"**TYPE: BARRIER** — first-hit over {days:.0f} days. "
            f"Longer time window = more volatility = higher hit probability (non-linear). "
            f"Think daily volatility × sqrt(days)."
        )
    elif question_type == "deadline_event":
        return (
            f"**TYPE: DEADLINE EVENT** — must happen within {days:.0f} days. "
            f"P(event by deadline) = P(eventually) × P(timing | eventually). "
            f"Is {days:.0f} days enough for this type of event?"
        )
    elif question_type == "fixed_date_event":
        return (
            f"**TYPE: FIXED-DATE EVENT** — scheduled outcome in {days:.0f} days. "
            f"Time affects signal quality, not event probability. "
            f"Focus on polls/fundamentals, weight recent info more."
        )
    elif question_type == "head_to_head":
        return (
            f"**TYPE: HEAD-TO-HEAD** — discrete competition in {days:.0f} days. "
            f"Time decay does NOT apply. Focus on form, Elo, recent results."
        )
    else:
        return f"**TYPE: UNCLASSIFIED** — {days:.0f} days remaining. Use standard probabilistic reasoning."
