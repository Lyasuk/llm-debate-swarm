"""Calibrated prompts for probability estimation from LLMs."""

SUPERFORECASTER_SYSTEM = """\
You are an expert superforecaster trained in calibrated probability estimation.
You make predictions by:
1. Identifying the base rate from similar historical events
2. Updating based on specific evidence for this case
3. Considering both sides of the argument thoroughly
4. Avoiding anchoring on the current market price
5. Expressing genuine uncertainty when appropriate
6. ALWAYS reasoning explicitly about the TIME WINDOW — how many days remain
   until resolution and whether that window is sufficient for the outcome

You are well-calibrated: when you say 70%, the event happens ~70% of the time.
You think in terms of probability distributions, not certainties.
You NEVER give round numbers like 0.50, 0.25 — you commit to fine-grained
estimates like 0.437, 0.681 that reflect specific reasoning.
"""

PROBABILITY_ESTIMATION_PROMPT = """\
Analyze this prediction market question and provide a calibrated probability estimate.

## MARKET QUESTION
{question}

## CURRENT MARKET PRICE
{yes_price:.1%} YES / {no_price:.1%} NO

This price represents the aggregated view of 1000+ traders with real money.
Markets are usually right. If your estimate differs from market by >20%,
you MUST cite specific concrete evidence for WHY you know better.
Most of the time your estimate should be WITHIN 15% of the market price.

## TIME CONTEXT (CRITICAL)
- **Days until resolution: {days_to_resolution:.1f}**
- You MUST reference this time window in your reasoning.
- Explain whether this window is SUFFICIENT for the event to resolve.

## RESOLUTION CRITERIA
{resolution_source}

{type_guidance}

## RESEARCH CONTEXT
{research_document}

---

## YOUR TASK

Provide a calibrated probability estimate by following these steps:

1. **Base Rate**: What is the historical base rate for similar events in a \
similar time window? If unsure, use 50%.

2. **Time Adjustment**: How does the {days_to_resolution:.0f}-day window affect the base rate?
   - Is this enough time for the outcome to occur?
   - Does the question type call for time decay or time-independent reasoning?

3. **Arguments FOR (YES)**: List the 3 strongest arguments supporting YES.

4. **Arguments AGAINST (NO)**: List the 3 strongest arguments supporting NO.

5. **Information Asymmetry**:
   - What information might the market be missing that you can see?
   - What information might the market know that you don't?

6. **Key Uncertainties**: What is the single biggest unknown that could swing the outcome?

7. **Final Estimate**: Your calibrated probability for YES (use 3 decimals).

CRITICAL: Form your estimate INDEPENDENTLY before comparing to the market price.
Do NOT simply agree with the market. The whole point is to find where the market is wrong.

CALIBRATION RULES:
- Use FINE-GRAINED probabilities with 3 decimals: 0.234, 0.487, 0.628
- NEVER use round numbers (0.10, 0.25, 0.50, 0.75, 0.90) — they signal lazy thinking
- Valid range: 0.020 to 0.980
- Most realistic estimates are between 0.10 and 0.90

## OUTPUT FORMAT (strict JSON)

```json
{{
  "base_rate": 0.NNN,
  "time_assessment": "1 sentence on whether {days_to_resolution:.0f} days is enough",
  "arguments_for": ["arg1", "arg2", "arg3"],
  "arguments_against": ["arg1", "arg2", "arg3"],
  "probability": 0.NNN,
  "confidence": "low|medium|high",
  "reasoning": "2-3 sentence summary of your reasoning (must reference time window)",
  "key_uncertainty": "The main factor that could change the outcome"
}}
```

Output ONLY the JSON block, no other text.
"""


def build_probability_prompt(
    question: str,
    yes_price: float,
    no_price: float,
    resolution_source: str,
    research_document: str,
    days_to_resolution: float,
    type_guidance: str,
) -> str:
    """Build the complete probability estimation prompt with time + type context."""
    return PROBABILITY_ESTIMATION_PROMPT.format(
        question=question,
        yes_price=yes_price,
        no_price=no_price,
        resolution_source=resolution_source or "Not specified",
        research_document=research_document[:8000],  # Truncate for token limits
        days_to_resolution=days_to_resolution,
        type_guidance=type_guidance,
    )
