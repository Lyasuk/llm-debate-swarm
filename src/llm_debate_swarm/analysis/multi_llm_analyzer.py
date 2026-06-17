"""Multi-LLM consensus analyzer — queries multiple models and aggregates predictions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from llm_debate_swarm.analysis.prompts.probability_prompts import (
    SUPERFORECASTER_SYSTEM,
    build_probability_prompt,
)
from llm_debate_swarm.config import AppConfig, LLMModelConfig, get_optional_api_key
from llm_debate_swarm.types import Question as Market
from llm_debate_swarm.utils.logger import get_logger
from llm_debate_swarm.utils.retry import retry_async

log = get_logger("analysis.multi_llm")


@dataclass
class LLMPrediction:
    """A single LLM's prediction for a market."""

    model_name: str
    provider: str
    probability: float  # 0-1
    confidence: str  # low / medium / high
    reasoning: str = ""
    base_rate: float = 0.5
    key_uncertainty: str = ""
    arguments_for: list[str] = field(default_factory=list)
    arguments_against: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ConsensusResult:
    """Aggregated consensus from multiple LLMs."""

    predictions: list[LLMPrediction]
    consensus_probability: float  # weighted average
    confidence: float  # 0-1 based on agreement
    spread: float  # max - min probability
    models_queried: int
    models_responded: int

    @property
    def is_valid(self) -> bool:
        return self.models_responded > 0


class MultiLLMAnalyzer:
    """Queries multiple LLMs for probability estimates and builds consensus."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models = config.analysis.models
        self.min_models = config.analysis.min_models_required

    async def analyze(
        self,
        market: Market,
        research_document: str,
        type_guidance: str = "",
    ) -> ConsensusResult:
        """Query all configured LLMs and return consensus prediction.

        Args:
            market: Market to analyze
            research_document: Compiled research brief
            type_guidance: Type-specific guidance block from QuestionClassifier
        """
        prompt = build_probability_prompt(
            question=market.question,
            yes_price=market.yes_price,
            no_price=market.no_price,
            resolution_source=market.resolution_source,
            research_document=research_document,
            days_to_resolution=market.days_to_resolution,
            type_guidance=type_guidance,
        )

        # Query all models concurrently
        import time
        start_times = [time.time() for _ in self.models]
        tasks = [
            self._query_model(model, prompt) for model in self.models
        ]
        predictions = await asyncio.gather(*tasks, return_exceptions=True)

        # Log to decision DB
        decision_id = getattr(market, "_decision_id", None)
        if decision_id is not None:
            try:
                from llm_debate_swarm.tracking.decision_logger import get_logger_instance
                from llm_debate_swarm.utils.cost_tracker import get_tracker
                dl = get_logger_instance()
                ct = get_tracker()
                for model_cfg, pred, t0 in zip(self.models, predictions, start_times):
                    latency_ms = int((time.time() - t0) * 1000)
                    if isinstance(pred, LLMPrediction):
                        dl.log_llm_prediction(
                            decision_id=decision_id,
                            model_name=model_cfg.name,
                            role="consensus",
                            weight=model_cfg.weight,
                            probability=pred.probability,
                            confidence=0.5 if isinstance(pred.confidence, str) else pred.confidence,
                            reasoning=pred.reasoning or "",
                            latency_ms=latency_ms,
                            error=pred.error or "",
                        )
                        # Approximate cost by token count if available
                        # (actual tokens logged by provider-specific methods is preferable)
                        ct.record(
                            provider=model_cfg.provider,
                            model=model_cfg.name,
                            role="consensus",
                            input_tokens=len(prompt) // 4,  # rough estimate
                            output_tokens=len(pred.reasoning or "") // 4,
                        )
                    elif isinstance(pred, Exception):
                        dl.log_llm_prediction(
                            decision_id=decision_id,
                            model_name=model_cfg.name,
                            role="consensus",
                            weight=model_cfg.weight,
                            probability=0.5,
                            confidence=0.0,
                            reasoning="",
                            latency_ms=latency_ms,
                            error=str(pred)[:500],
                        )
            except Exception as log_exc:
                log.warning(f"LLM decision logging failed: {log_exc}")

        # Process results
        valid_predictions: list[LLMPrediction] = []
        for pred in predictions:
            if isinstance(pred, LLMPrediction) and pred.error is None:
                valid_predictions.append(pred)
            elif isinstance(pred, Exception):
                log.warning(f"LLM query failed: {pred}")

        if len(valid_predictions) < self.min_models:
            log.warning(
                f"Only {len(valid_predictions)} models responded "
                f"(need {self.min_models})"
            )

        consensus = self._build_consensus(valid_predictions, len(self.models))
        log.info(
            f"Consensus: {consensus.consensus_probability:.1%} "
            f"(spread={consensus.spread:.1%}, confidence={consensus.confidence:.2f}, "
            f"models={consensus.models_responded}/{consensus.models_queried})"
        )
        return consensus

    async def _query_model(
        self, model_config: LLMModelConfig, prompt: str
    ) -> LLMPrediction:
        """Query a single LLM model (OpenTelemetry-traced as an ``llm.query`` span)."""
        from llm_debate_swarm.obs.tracing import get_tracer

        provider = model_config.provider.lower()
        model_name = model_config.name

        with get_tracer().start_as_current_span("llm.query") as span:
            span.set_attribute("llm.model", model_name)
            span.set_attribute("llm.provider", provider)
            try:
                if provider == "anthropic":
                    pred = await self._query_anthropic(model_name, prompt)
                elif provider == "openai":
                    pred = await self._query_openai(model_name, prompt)
                elif provider == "google":
                    pred = await self._query_google(model_name, prompt)
                elif provider == "groq":
                    pred = await self._query_groq(model_name, prompt)
                else:
                    pred = LLMPrediction(
                        model_name=model_name,
                        provider=provider,
                        probability=0.5,
                        confidence="low",
                        error=f"Unknown provider: {provider}",
                    )
            except Exception as exc:
                log.error(f"Error querying {model_name}: {exc}")
                pred = LLMPrediction(
                    model_name=model_name,
                    provider=provider,
                    probability=0.5,
                    confidence="low",
                    error=str(exc),
                )
            span.set_attribute("llm.probability", float(pred.probability))
            if pred.error:
                span.set_attribute("llm.error", str(pred.error)[:200])
            return pred

    @retry_async(max_retries=2, base_delay=3.0)
    async def _query_anthropic(self, model: str, prompt: str) -> LLMPrediction:
        """Query Anthropic's Claude API."""
        import anthropic

        api_key = get_optional_api_key("ANTHROPIC_API_KEY")
        if not api_key:
            return LLMPrediction(
                model_name=model, provider="anthropic",
                probability=0.5, confidence="low",
                error="ANTHROPIC_API_KEY not set",
            )

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=1500,
            system=SUPERFORECASTER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.content[0].text
        return self._parse_response(text, model, "anthropic")

    @retry_async(max_retries=2, base_delay=3.0)
    async def _query_openai(self, model: str, prompt: str) -> LLMPrediction:
        """Query OpenAI's GPT API."""
        import openai

        api_key = get_optional_api_key("OPENAI_API_KEY")
        if not api_key:
            return LLMPrediction(
                model_name=model, provider="openai",
                probability=0.5, confidence="low",
                error="OPENAI_API_KEY not set",
            )

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUPERFORECASTER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.3,
        )

        text = response.choices[0].message.content or ""
        return self._parse_response(text, model, "openai")

    @retry_async(max_retries=2, base_delay=3.0)
    async def _query_groq(self, model: str, prompt: str) -> LLMPrediction:
        """Query Groq API (OpenAI-compatible)."""
        import openai

        api_key = get_optional_api_key("GROQ_API_KEY")
        if not api_key:
            return LLMPrediction(
                model_name=model, provider="groq",
                probability=0.5, confidence="low",
                error="GROQ_API_KEY not set",
            )

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=60.0,
        )

        # Qwen 3: disable reasoning trace to fit JSON output in budget
        if "qwen" in model.lower():
            prompt = "/no_think\n\n" + prompt

        # Try JSON mode first for cleaner parses
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUPERFORECASTER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "response_format" in str(exc) or "json_object" in str(exc):
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SUPERFORECASTER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.3,
                )
            else:
                raise

        text = response.choices[0].message.content or ""
        return self._parse_response(text, model, "groq")

    @retry_async(max_retries=2, base_delay=3.0)
    async def _query_google(self, model: str, prompt: str) -> LLMPrediction:
        """Query Google's Gemini/Gemma API.

        Gemma models don't support system_instruction — inline в user prompt.
        Gemini models accept system_instruction natively.
        """
        api_key = get_optional_api_key("GOOGLE_API_KEY")
        if not api_key:
            return LLMPrediction(
                model_name=model, provider="google",
                probability=0.5, confidence="low",
                error="GOOGLE_API_KEY not set",
            )

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Gemma не підтримує system_instruction — merge в user prompt
        is_gemma = model.startswith("gemma")
        if is_gemma:
            gen_model = genai.GenerativeModel(model)
            full_prompt = f"{SUPERFORECASTER_SYSTEM}\n\n---\n\n{prompt}"
        else:
            gen_model = genai.GenerativeModel(
                model,
                system_instruction=SUPERFORECASTER_SYSTEM,
            )
            full_prompt = prompt

        response = await gen_model.generate_content_async(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            ),
        )

        text = response.text
        return self._parse_response(text, model, "google")

    def _parse_response(
        self, text: str, model_name: str, provider: str
    ) -> LLMPrediction:
        """Parse JSON response from any LLM."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            prob = float(data.get("probability", 0.5))
            # Sanity check: probability must be between 0.01 and 0.99
            prob = max(0.01, min(0.99, prob))

            return LLMPrediction(
                model_name=model_name,
                provider=provider,
                probability=prob,
                confidence=data.get("confidence", "medium"),
                reasoning=data.get("reasoning", ""),
                base_rate=float(data.get("base_rate", 0.5)),
                key_uncertainty=data.get("key_uncertainty", ""),
                arguments_for=data.get("arguments_for", []),
                arguments_against=data.get("arguments_against", []),
            )
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
            log.warning(f"Failed to parse {model_name} response: {exc}")
            return LLMPrediction(
                model_name=model_name,
                provider=provider,
                probability=0.5,
                confidence="low",
                error=f"Parse error: {exc}",
            )

    def _build_consensus(
        self, predictions: list[LLMPrediction], total_queried: int
    ) -> ConsensusResult:
        """Build weighted consensus from multiple LLM predictions."""
        # Filter out predictions with invalid probabilities (issue #41)
        predictions = [
            p for p in predictions
            if 0.01 <= p.probability <= 0.99
        ]

        if not predictions:
            return ConsensusResult(
                predictions=[],
                consensus_probability=0.5,
                confidence=0.0,
                spread=0.0,
                models_queried=total_queried,
                models_responded=0,
            )

        # Build weight map from config
        weight_map = {m.name: m.weight for m in self.models}

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for pred in predictions:
            w = weight_map.get(pred.model_name, 1.0 / len(predictions))
            weighted_sum += pred.probability * w
            total_weight += w

        consensus_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Calculate spread (max - min)
        probs = [p.probability for p in predictions]
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0

        # Confidence based on agreement (low spread = high confidence)
        # Also factor in model-reported confidence
        conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
        avg_reported_conf = sum(
            conf_map.get(p.confidence, 0.5) for p in predictions
        ) / len(predictions)

        agreement_conf = max(0, 1.0 - spread * 3)  # spread of 0.33 = 0 confidence
        confidence = (agreement_conf * 0.6 + avg_reported_conf * 0.4)

        return ConsensusResult(
            predictions=predictions,
            consensus_probability=consensus_prob,
            confidence=confidence,
            spread=spread,
            models_queried=total_queried,
            models_responded=len(predictions),
        )
