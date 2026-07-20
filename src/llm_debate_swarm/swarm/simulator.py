"""Lightweight multi-agent swarm simulator with 5-layer debiasing."""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass, field

import openai

from llm_debate_swarm.analysis.question_classifier import (
    QuestionClassification,
    build_time_guidance,
    classify_question,
)
from llm_debate_swarm.config import AppConfig, get_optional_api_key
from llm_debate_swarm.types import Question as Market
from llm_debate_swarm.swarm.aggregator import (
    adjusted_probability,
    compute_confidence,
    convergence_ratio,
    detect_anchoring,
    extract_valid_estimates,
    premortem_impact,
    trimmed_mean,
)
from llm_debate_swarm.swarm.personas import PERSONAS, AgentPersona
from llm_debate_swarm.swarm.prompts import (
    DEBATE_BLIND_PROMPT,
    DEVILS_ADVOCATE_INSTRUCTION,
    FINAL_PROMPT,
    META_SYNTHESIS_PROMPT,
    META_SYNTHESIS_SYSTEM,
    PRE_MORTEM_PROMPT,
    PRICE_AWARE_PROMPT,
    ROUND1_BLIND_PROMPT,
    compact_type_guidance,
)
from llm_debate_swarm.utils.logger import get_logger
from llm_debate_swarm.utils.retry import retry_async

log = get_logger("swarm.simulator")


@dataclass
class AgentState:
    """Tracks one agent across all debate rounds."""

    persona: AgentPersona
    estimates: list[float] = field(default_factory=list)
    confidences: list = field(default_factory=list)
    reasonings: list[str] = field(default_factory=list)
    key_factors: list[str] = field(default_factory=list)
    is_devils_advocate: list[bool] = field(default_factory=list)
    premortem_explanation: str = ""
    missed_factor: str = ""
    parse_failures: int = 0

    @property
    def last_estimate(self) -> float:
        return self.estimates[-1] if self.estimates else 0.5

    @property
    def last_reasoning(self) -> str:
        return self.reasonings[-1] if self.reasonings else ""


@dataclass
class SwarmResult:
    """Final output of a swarm simulation."""

    probability: float
    confidence: float
    raw_trimmed_mean: float = 0.5
    blind_mean: float = 0.5
    aware_mean: float = 0.5
    anchoring_shift: float = 0.0
    convergence_ratio: float = 1.0
    premortem_changed_frac: float = 0.0
    agent_count: int = 0
    rounds_completed: int = 0
    std_per_round: list[float] = field(default_factory=list)
    agent_finals: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    meta_bias_corrections: dict = field(default_factory=dict)
    swarm_model: str = ""  # for A/B tracking: "gpt-4o-mini" vs "gemini-2.5-flash-lite"
    error: str | None = None


class SwarmSimulator:
    """Multi-agent debate simulator with 7-round debiasing protocol."""

    def __init__(self, config: AppConfig):
        self.config = config.swarm
        self._swarm_client: openai.AsyncOpenAI | None = None
        self._groq_client: openai.AsyncOpenAI | None = None
        self._meta_client: openai.AsyncOpenAI | None = None
        self._gemini_model = None
        self._gemini_models_cache: dict = {}  # model_name → GenerativeModel
        self._bucket_usage_today: dict = {}  # model_name → count today
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # Provider determination: explicit config.provider overrides model name detection
        explicit = getattr(self.config, "provider", None)
        if explicit and explicit.lower() != "auto":
            self._swarm_provider = explicit.lower()
        else:
            self._swarm_provider = self._detect_provider(self.config.swarm_model)
        self._meta_provider = self._detect_provider(self.config.meta_model)

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        """Detect LLM provider from model name."""
        m = model_name.lower()
        if "gemini" in m or "flash" in m:
            return "google"
        if "claude" in m:
            return "anthropic"
        if "llama" in m or "qwen" in m or "gpt-oss" in m or "gemma" in m or "mixtral" in m:
            return "groq"
        return "openai"

    def _get_swarm_client(self) -> openai.AsyncOpenAI:
        if self._swarm_client is None:
            api_key = get_optional_api_key("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            self._swarm_client = openai.AsyncOpenAI(api_key=api_key)
        return self._swarm_client

    def _get_groq_client(self) -> openai.AsyncOpenAI:
        """Lazy init Groq client (OpenAI-compatible)."""
        if self._groq_client is None:
            api_key = get_optional_api_key("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not set")
            self._groq_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=60.0,
            )
        return self._groq_client

    def _get_gemini_model(self):
        """Lazy init Gemini model for swarm agents."""
        if self._gemini_model is None:
            import google.generativeai as genai
            api_key = get_optional_api_key("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.config.swarm_model)
        return self._gemini_model

    def _get_meta_client(self) -> openai.AsyncOpenAI:
        if self._meta_client is None:
            api_key = get_optional_api_key("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            self._meta_client = openai.AsyncOpenAI(api_key=api_key)
        return self._meta_client

    async def simulate(
        self,
        market: Market,
        research_doc: str,
        classification: QuestionClassification | None = None,
    ) -> SwarmResult:
        """Run full 7-round swarm simulation with meta-synthesis.

        Args:
            market: Market to analyze
            research_doc: Compiled research brief
            classification: QuestionClassifier output (if None, auto-classifies)
        """
        start = time.time()
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Auto-classify if not provided
        if classification is None:
            classification = classify_question(market.question, market.category)

        # Build type-specific guidance blocks
        full_guidance = build_time_guidance(classification, market.days_to_resolution)
        compact_guidance = compact_type_guidance(
            classification.question_type.value, market.days_to_resolution
        )

        # Store on self for access from round methods
        self._classification = classification
        self._full_guidance = full_guidance
        self._compact_guidance = compact_guidance

        try:
            agents = [
                AgentState(persona=p)
                for p in PERSONAS[: self.config.agent_count]
            ]
            research_summary = research_doc[:6000]

            # Extract decision_id from market for DB logging
            decision_id = getattr(market, "_decision_id", None)

            # Round 1: BLIND — no market price (uses full guidance)
            log.info(f"Swarm Round 1/7 [BLIND] type={classification.question_type.value}...")
            await self._run_round_blind(agents, market, research_summary)
            self._log_round_stats(agents, 1, round_type="BLIND", decision_id=decision_id)

            # Diversity check: if R1 has too few unique values, re-run with higher temp
            r1_ests = extract_valid_estimates([a.last_estimate for a in agents])
            r1_unique = len(set(round(e, 3) for e in r1_ests))
            min_unique = max(12, len(r1_ests) // 3)
            if r1_unique < min_unique and len(r1_ests) > 10:
                log.warning(
                    f"R1 diversity too low: {r1_unique}/{len(r1_ests)} unique. "
                    f"Re-running blind round with temperature +0.3"
                )
                old_temp = self.config.temperature
                self.config.temperature = min(1.5, old_temp + 0.3)
                # Скинути останній раунд і перезапустити
                for agent in agents:
                    if agent.estimates:
                        agent.estimates.pop()
                    if agent.reasonings:
                        agent.reasonings.pop()
                    if agent.key_factors:
                        agent.key_factors.pop()
                    if agent.is_devils_advocate:
                        agent.is_devils_advocate.pop()
                await self._run_round_blind(agents, market, research_summary)
                self.config.temperature = old_temp
                self._log_round_stats(
                    agents, 1, round_type="BLIND_RETRY",
                    decision_id=decision_id, retry_triggered=True,
                )

            # Rounds 2-3: DEBATE without price (uses compact guidance)
            for r in range(2, 4):
                log.info(f"Swarm Round {r}/7 [DEBATE-BLIND]...")
                await self._run_round_debate(agents, market, r, show_price=False)
                self._log_round_stats(agents, r, round_type="DEBATE-BLIND", decision_id=decision_id)

            # Rounds 4-5: DEBATE with price revealed
            for r in range(4, 6):
                log.info(f"Swarm Round {r}/7 [PRICE-AWARE]...")
                await self._run_round_debate(agents, market, r, show_price=True)
                self._log_round_stats(agents, r, round_type="PRICE-AWARE", decision_id=decision_id)

            # Round 6: PRE-MORTEM
            log.info("Swarm Round 6/7 [PRE-MORTEM]...")
            await self._run_premortem(agents, market)
            self._log_round_stats(agents, 6, round_type="PRE-MORTEM", decision_id=decision_id)

            # Round 7: FINAL
            log.info("Swarm Round 7/7 [FINAL]...")
            await self._run_final(agents, market)
            self._log_round_stats(agents, 7, round_type="FINAL", decision_id=decision_id)

            # Meta-synthesis: statistical aggregator (replaces Claude/LLM meta)
            # Motivation: LLM meta anchored на outliers (D28: 0.125 vs trimmed 0.32).
            # Statistical methods robust to outliers + zero API cost.
            log.info("Swarm meta-synthesis [statistical]...")
            meta = self._run_statistical_meta(agents, market)

            result = self._build_result(agents, meta)
            result.duration_sec = time.time() - start
            result.cost_usd = self._estimate_cost()

            log.info(
                f"Swarm complete: prob={result.probability:.1%} "
                f"conf={result.confidence:.2f} "
                f"anchor={result.anchoring_shift:.1%} "
                f"cost=${result.cost_usd:.3f} "
                f"time={result.duration_sec:.1f}s"
            )
            return result

        except Exception as exc:
            log.error(f"Swarm simulation failed: {exc}")
            return SwarmResult(
                probability=0.5,
                confidence=0.0,
                duration_sec=time.time() - start,
                cost_usd=self._estimate_cost(),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Round implementations
    # ------------------------------------------------------------------

    async def _run_round_blind(
        self, agents: list[AgentState], market: Market, research_summary: str
    ) -> None:
        """Round 1: Independent estimates without market price."""
        prompt = ROUND1_BLIND_PROMPT.format(
            question=market.question,
            resolution_source=market.resolution_source or "Not specified",
            research_summary=research_summary,
            days_to_resolution=market.days_to_resolution,
            type_guidance=self._full_guidance,
        )

        tasks = []
        for agent in agents:
            agent.is_devils_advocate.append(False)
            messages = [
                {"role": "system", "content": agent.persona.system_prompt},
                {"role": "user", "content": prompt},
            ]
            tasks.append(self._query_agent(agent, messages))

        await self._run_batched(tasks)

    async def _run_round_debate(
        self,
        agents: list[AgentState],
        market: Market,
        round_num: int,
        show_price: bool,
    ) -> None:
        """Rounds 2-5: Debate with optional price reveal."""
        debate_summary = self._build_debate_summary(agents)
        estimates = extract_valid_estimates([a.last_estimate for a in agents])
        mean_est = statistics.mean(estimates) if estimates else 0.5
        median_est = statistics.median(estimates) if estimates else 0.5

        # Select devil's advocates — pick conformists
        da_agents = self._select_devils_advocates(agents, round_num)
        da_ids = {a.persona.id for a in da_agents}

        consensus_dir = "YES (above 50%)" if mean_est > 0.5 else "NO (below 50%)"

        tasks = []
        for agent in agents:
            is_da = agent.persona.id in da_ids
            agent.is_devils_advocate.append(is_da)

            da_instruction = ""
            if is_da:
                da_instruction = DEVILS_ADVOCATE_INSTRUCTION.format(
                    consensus_direction=consensus_dir
                )

            if show_price:
                prompt = PRICE_AWARE_PROMPT.format(
                    question=market.question,
                    round_num=round_num,
                    days_to_resolution=market.days_to_resolution,
                    type_guidance_compact=self._compact_guidance,
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    debate_summary=debate_summary,
                    prev_estimate=agent.last_estimate,
                    devils_advocate_instruction=da_instruction,
                )
            else:
                prompt = DEBATE_BLIND_PROMPT.format(
                    question=market.question,
                    round_num=round_num,
                    days_to_resolution=market.days_to_resolution,
                    type_guidance_compact=self._compact_guidance,
                    debate_summary=debate_summary,
                    mean=mean_est,
                    median=median_est,
                    min_est=min(estimates) if estimates else 0,
                    max_est=max(estimates) if estimates else 1,
                    count_above=sum(1 for e in estimates if e > 0.5),
                    count_below=sum(1 for e in estimates if e <= 0.5),
                    devils_advocate_instruction=da_instruction,
                )

            messages = [
                {"role": "system", "content": agent.persona.system_prompt},
                {"role": "user", "content": prompt},
            ]
            tasks.append(self._query_agent(agent, messages))

        await self._run_batched(tasks)

    async def _run_premortem(
        self, agents: list[AgentState], market: Market
    ) -> None:
        """Round 6: Pre-mortem exercise."""
        tasks = []
        for agent in agents:
            agent.is_devils_advocate.append(False)

            if agent.last_estimate > 0.5:
                direction = (
                    "Assume this question resolved as NO (the opposite of your estimate). "
                    "The event did NOT happen."
                )
            else:
                direction = (
                    "Assume this question resolved as YES (the opposite of your estimate). "
                    "The event DID happen."
                )

            prompt = PRE_MORTEM_PROMPT.format(
                question=market.question,
                prev_estimate=agent.last_estimate,
                yes_price=market.yes_price,
                days_to_resolution=market.days_to_resolution,
                premortem_direction=direction,
            )

            messages = [
                {"role": "system", "content": agent.persona.system_prompt},
                {"role": "user", "content": prompt},
            ]
            tasks.append(self._query_agent_premortem(agent, messages))

        await self._run_batched(tasks)

    async def _run_final(
        self, agents: list[AgentState], market: Market
    ) -> None:
        """Round 7: Final calibrated estimates."""
        premortem_summary = self._build_premortem_summary(agents)

        tasks = []
        for agent in agents:
            agent.is_devils_advocate.append(False)

            trajectory = " → ".join(
                f"{e:.0%}" for e in agent.estimates[-4:]  # last 4 rounds
            )

            prompt = FINAL_PROMPT.format(
                question=market.question,
                yes_price=market.yes_price,
                days_to_resolution=market.days_to_resolution,
                trajectory=trajectory,
                premortem_summary=premortem_summary,
            )

            messages = [
                {"role": "system", "content": agent.persona.system_prompt},
                {"role": "user", "content": prompt},
            ]
            tasks.append(self._query_agent(agent, messages))

        await self._run_batched(tasks)

    # ------------------------------------------------------------------
    # Meta-synthesis (GPT-4o)
    # ------------------------------------------------------------------

    def _run_statistical_meta(
        self,
        agents: list[AgentState],
        market: Market,
    ) -> dict:
        """Statistical meta synthesis — replaces LLM meta call.

        Uses trimmed mean + MAD outlier rejection + bimodal detection.
        Output shape матчиться з LLM meta для downstream compatibility.
        """
        from llm_debate_swarm.swarm.statistical_meta import aggregate

        estimates = [a.last_estimate for a in agents if a.estimates]
        if not estimates:
            return {
                "probability": 0.5,
                "confidence": 0.3,
                "reasoning": "No valid agent estimates",
                "bias_corrections": {},
            }

        result = aggregate(
            estimates=estimates,
            confidences=None,  # Future: weight by agent.confidences
            market_price=market.yes_price,
        )

        # Log для monitoring
        log.info(
            f"StatMeta: prob={result.probability:.3f} conf={result.confidence:.2f} "
            f"outliers={result.outliers_rejected} bimodal={result.is_bimodal} "
            f"trimmed={result.trimmed_mean:.3f} median={result.median:.3f}"
        )

        return {
            "probability": result.probability,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "bias_corrections": {
                "outliers_rejected": result.outliers_rejected,
                "is_bimodal": result.is_bimodal,
                "bimodal_low": result.bimodal_low,
                "bimodal_high": result.bimodal_high,
                "trimmed_mean": result.trimmed_mean,
                "median": result.median,
                "mad": result.mad,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                # Keep shape compat with LLM meta output
                "anchoring_detected": False,
                "anchoring_correction": 0.0,
                "groupthink_detected": False,
                "groupthink_correction": 0.0,
                "overconfidence_detected": False,
                "time_miscalibration_detected": False,
                "base_rate_used": result.median,
            },
        }

    async def _run_meta_synthesis(
        self,
        agents: list[AgentState],
        market: Market,
        research_doc: str,
    ) -> dict:
        """Single GPT-4o call to synthesize and debias the swarm output."""
        # Build agent finals summary
        sorted_agents = sorted(agents, key=lambda a: a.last_estimate)
        agent_finals_lines = []
        for a in sorted_agents:
            agent_finals_lines.append(
                f"- {a.persona.name} ({a.persona.category}): "
                f"{a.last_estimate:.1%} [{a.confidences[-1] if a.confidences else '?'}] "
                f"— {a.last_reasoning[:120]}"
            )
        agent_finals = "\n".join(agent_finals_lines)

        # Compute stats
        # IMPORTANT: aware_mean must EXCLUDE pre-mortem round (R6, index 5)
        # because pre-mortem asks agents to "assume opposite" — those are NOT
        # real beliefs, just thought experiments.
        all_estimates = extract_valid_estimates([a.last_estimate for a in agents])
        blind_ests = []
        aware_ests = []
        pm_idx = self.config.premortem_round - 1  # 0-indexed (R6 -> 5)
        for a in agents:
            for i, est in enumerate(a.estimates):
                if i < self.config.blind_rounds:
                    blind_ests.append(est)
                elif i == pm_idx:
                    continue  # skip pre-mortem in aware_mean
                else:
                    aware_ests.append(est)

        blind_mean = statistics.mean(blind_ests) if blind_ests else 0.5
        aware_mean = statistics.mean(aware_ests) if aware_ests else 0.5
        anch_shift = detect_anchoring(blind_ests, aware_ests)

        std_r1 = statistics.stdev([a.estimates[0] for a in agents if a.estimates]) if len(agents) > 1 else 0
        std_r7 = statistics.stdev(all_estimates) if len(all_estimates) > 1 else 0
        conv = convergence_ratio(std_r1, std_r7)

        # Pre-mortem stats — compare R5 (pre) vs R6 (premortem) estimates
        # Fix #28: require BOTH rounds present before comparing
        pm_idx_local = self.config.premortem_round - 1  # R6 in 0-indexed = 5
        pre_idx = pm_idx_local - 1  # R5 = 4
        pre_ests = []
        post_ests = []
        for a in agents:
            if len(a.estimates) > pm_idx_local:  # has both R5 and R6
                pre_ests.append(a.estimates[pre_idx])
                post_ests.append(a.estimates[pm_idx_local])
        pm_changed = sum(
            1 for pre, post in zip(pre_ests, post_ests) if abs(pre - post) > 0.05
        )

        # Top arguments
        for_args = []
        against_args = []
        for a in agents:
            if a.last_estimate > 0.6 and a.last_reasoning:
                for_args.append(f"[{a.persona.name}] {a.last_reasoning[:150]}")
            elif a.last_estimate < 0.4 and a.last_reasoning:
                against_args.append(f"[{a.persona.name}] {a.last_reasoning[:150]}")

        # Pre-mortem modes — Fix #30: filter empty strings
        pm_modes = []
        for a in agents:
            if a.premortem_explanation and a.premortem_explanation.strip():
                pm_modes.append(f"[{a.persona.name}] {a.premortem_explanation[:150]}")

        # Robust stats to prevent Claude anchoring on outliers (e.g., 0.99 extremes)
        r7_ests = [a.last_estimate for a in agents if a.estimates]
        r7_trimmed_mean = trimmed_mean(r7_ests, 0.10) if r7_ests else 0.5
        r7_median = statistics.median(r7_ests) if r7_ests else 0.5

        prompt = META_SYNTHESIS_PROMPT.format(
            question=market.question,
            question_type=self._classification.question_type.value,
            days_to_resolution=market.days_to_resolution,
            trimmed_mean=r7_trimmed_mean,
            median=r7_median,
            type_guidance=self._full_guidance,
            yes_price=market.yes_price,
            no_price=market.no_price,
            resolution_source=market.resolution_source or "Not specified",
            agent_finals=agent_finals,
            blind_mean=blind_mean,
            aware_mean=aware_mean,
            anchoring_shift=anch_shift,
            std_r1=std_r1,
            std_r7=std_r7,
            convergence=conv * 100,
            premortem_changed=pm_changed,
            agent_count=len(agents),
            top_arguments_for="\n".join(for_args[:5]) or "None prominent",
            top_arguments_against="\n".join(against_args[:5]) or "None prominent",
            premortem_modes="\n".join(pm_modes[:5]) or "None recorded",
        )

        try:
            if self._meta_provider == "anthropic":
                text = await self._call_meta_anthropic(prompt)
            elif self._meta_provider == "google":
                text = await self._call_meta_gemini(prompt)
            else:
                text = await self._call_meta_openai(prompt)
            return self._parse_json_response(text)

        except Exception as exc:
            log.warning(f"Meta-synthesis failed: {exc}")
            # Fallback to aggregator
            raw = trimmed_mean(all_estimates, self.config.trim_pct)
            return {
                "probability": adjusted_probability(
                    raw, blind_mean, aware_mean, anch_shift, conv,
                    premortem_impact(pre_ests, post_ests),
                ),
                "confidence": compute_confidence(all_estimates),
                "reasoning": "Meta-synthesis failed, using statistical aggregation",
                "bias_corrections": {},
            }

    async def _call_meta_openai(self, prompt: str) -> str:
        """Meta-synthesis via OpenAI (GPT-4o)."""
        client = self._get_meta_client()
        response = await client.chat.completions.create(
            model=self.config.meta_model,
            messages=[
                {"role": "system", "content": META_SYNTHESIS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""

    async def _call_meta_anthropic(self, prompt: str) -> str:
        """Meta-synthesis via Anthropic (Claude Sonnet)."""
        import anthropic
        api_key = get_optional_api_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self.config.meta_model,
            max_tokens=1000,
            system=META_SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.content[0].text if response.content else ""
        if response.usage:
            self._total_input_tokens += response.usage.input_tokens
            self._total_output_tokens += response.usage.output_tokens
        return text

    async def _call_meta_gemini(self, prompt: str) -> str:
        """Meta-synthesis via Gemini."""
        import google.generativeai as genai
        api_key = get_optional_api_key("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            self.config.meta_model,
            system_instruction=META_SYNTHESIS_SYSTEM,
        )
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
            ),
        )
        return response.text or ""

    # ------------------------------------------------------------------
    # Agent query helpers
    # ------------------------------------------------------------------

    @retry_async(max_retries=2, base_delay=2.0)
    async def _query_single_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> str:
        """Query a single agent. Supports OpenAI, Gemini, Groq, Gemini-multi providers."""
        from llm_debate_swarm.obs.tracing import get_tracer

        # One span per attempt (the retry decorator re-enters), so 429 retries
        # are visible in the trace. `is_devils_advocate` grows once per round
        # before queries launch, so its length IS the current round number.
        with get_tracer().start_as_current_span("swarm.agent") as span:
            span.set_attribute("gen_ai.provider.name", str(self._swarm_provider))
            span.set_attribute("gen_ai.request.model", str(self.config.swarm_model))
            # Raw attrs show up in Langfuse/Jaeger; LangSmith's OTLP ingest drops
            # unrecognized keys, so mirror them under its langsmith.metadata.* namespace.
            span.set_attribute("swarm.persona", str(agent.persona.id))
            span.set_attribute("swarm.round", len(agent.is_devils_advocate))
            span.set_attribute("langsmith.metadata.swarm_persona", str(agent.persona.id))
            span.set_attribute("langsmith.metadata.swarm_round", len(agent.is_devils_advocate))
            if self._swarm_provider == "gemini_multi":
                return await self._query_gemini_multi_agent(agent, messages)
            elif self._swarm_provider == "google":
                return await self._query_gemini_agent(agent, messages)
            elif self._swarm_provider == "groq":
                return await self._query_groq_agent(agent, messages)
            else:
                return await self._query_openai_agent(agent, messages)

    def _get_gemini_model_by_name(self, model_name: str):
        """Cache Gemini model instances by name."""
        if model_name not in self._gemini_models_cache:
            import google.generativeai as genai
            api_key = get_optional_api_key("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._gemini_models_cache[model_name] = genai.GenerativeModel(model_name)
        return self._gemini_models_cache[model_name]

    def _resolve_bucket_model(self, bucket: str) -> str:
        """Resolve bucket → model name with failover across 5 buckets.

        Fallback order: requested → standard_27b → standard_4b → nano_llama → nano_qwen
        """
        bucket_models = getattr(self.config, "bucket_models", {})
        bucket_limits = getattr(self.config, "bucket_rpd_limits", {})

        fallback_order = [bucket, "standard_27b", "standard_4b", "nano_llama", "nano_qwen"]
        # Dedupe while preserving order
        seen = set()
        unique = [b for b in fallback_order if not (b in seen or seen.add(b))]

        for b in unique:
            model = bucket_models.get(b)
            if model is None:
                continue
            limit = bucket_limits.get(b, 14400)
            used = self._bucket_usage_today.get(model, 0)
            if used < int(limit * 0.95):
                return model

        # Last resort: any available model
        return bucket_models.get("nano_e2b", "gemma-3n-e2b-it")

    async def _query_gemini_multi_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> str:
        """Query model based on agent's bucket. Routes Google or Groq by model name."""
        bucket = getattr(agent.persona, "model_bucket", "standard_27b")
        model_name = self._resolve_bucket_model(bucket)

        # Route to Groq for llama/qwen/gpt-oss models, Google for Gemini/Gemma
        if model_name.startswith("llama") or model_name.startswith("qwen") or "gpt-oss" in model_name:
            return await self._query_groq_bucket_agent(agent, messages, model_name)

        import google.generativeai as genai
        model = self._get_gemini_model_by_name(model_name)

        # Extract system + user з messages
        system_text = ""
        user_text = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content

        # Reinforce contrarian bias — ALE slabshe for small models.
        # Проблема: nano моделі (e2b, e4b) беруть instruction буквально і видають
        # extreme values (3+ agents дали 0.99 на D20 R3). Softer для nano.
        category = getattr(agent.persona, "category", "")
        is_nano = bucket.startswith("nano_")
        if "contrarian" in str(category).lower() or "devil" in str(category).lower():
            if is_nano:
                user_text += (
                    "\n\nNote: Consider alternative perspectives where evidence supports them. "
                    "Stay calibrated — don't go extreme unless data strongly supports it."
                )
            else:
                user_text += (
                    "\n\nIMPORTANT: You are a contrarian voice. "
                    "Challenge the consensus direction."
                )

        full_prompt = f"{system_text}\n\n---\n\n{user_text}" if system_text else user_text

        response = await model.generate_content_async(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens_per_response,
            ),
        )

        # Track usage для failover
        self._bucket_usage_today[model_name] = self._bucket_usage_today.get(model_name, 0) + 1

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self._total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
            self._total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)

        # Store model_name in agent для tracking
        agent._last_model = model_name  # type: ignore

        return response.text or ""

    async def _query_groq_bucket_agent(
        self, agent: AgentState, messages: list[dict], model_name: str
    ) -> str:
        """Query Groq for multi-bucket routing (e.g., nano_llama, nano_qwen)."""
        client = self._get_groq_client()

        category = getattr(agent.persona, "category", "")
        if "contrarian" in str(category).lower() or "devil" in str(category).lower():
            if len(messages) > 0 and messages[-1].get("role") == "user":
                messages[-1]["content"] += (
                    "\n\nNote: Offer an alternative perspective if evidence supports it."
                )

        # Qwen 3 specific: disable reasoning trace via /no_think directive
        # Otherwise Qwen spends max_tokens on <think> blocks and never reaches JSON
        if "qwen" in model_name.lower():
            if len(messages) > 0 and messages[-1].get("role") == "user":
                messages[-1]["content"] = "/no_think\n\n" + messages[-1]["content"]

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=self.config.max_tokens_per_response,
            temperature=self.config.temperature,
        )

        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens

        self._bucket_usage_today[model_name] = self._bucket_usage_today.get(model_name, 0) + 1
        agent._last_model = model_name  # type: ignore

        return response.choices[0].message.content or ""

    async def _query_groq_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> str:
        """Query via Groq API (OpenAI-compatible endpoint).

        Groq supports JSON mode on most models. Use response_format json_object
        when possible for zero parse failures.
        """
        client = self._get_groq_client()

        # Reinforce contrarian bias для contrarian agents (як в Gemini)
        category = getattr(agent.persona, "category", "")
        if "contrarian" in str(category).lower() or "devil" in str(category).lower():
            if len(messages) > 0 and messages[-1].get("role") == "user":
                messages[-1]["content"] += (
                    "\n\nIMPORTANT: You are a contrarian voice. "
                    "Challenge the consensus direction."
                )

        # NO JSON mode — змушує модель форматувати до кінця, і якщо tokens
        # закінчуються в середині JSON — failure. Parser handles markdown JSON.
        response = await client.chat.completions.create(
            model=self.config.swarm_model,
            messages=messages,
            max_tokens=self.config.max_tokens_per_response,
            temperature=self.config.temperature,
        )

        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content or ""

    async def _query_openai_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> str:
        """Query via OpenAI SDK (gpt-4o-mini)."""
        client = self._get_swarm_client()
        response = await client.chat.completions.create(
            model=self.config.swarm_model,
            messages=messages,
            max_tokens=self.config.max_tokens_per_response,
            temperature=self.config.temperature,
        )

        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content or ""

    async def _query_gemini_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> str:
        """Query via Google Gemini API (gemini-2.5-flash-lite).

        Uses response_mime_type="application/json" to FORCE valid JSON output,
        eliminating parse failures from prose responses.

        For contrarian-category agents, adds reinforcement instruction to
        ensure they actually disagree with the group (Fix Priority 2).
        """
        import google.generativeai as genai

        model = self._get_gemini_model()

        # Extract system + user from messages
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

        # Fix Priority 2: reinforce contrarian behavior for Gemini
        if agent.persona.category == "contrarian" or agent.persona.bias_direction == "contrarian":
            system_text += (
                "\n\nCRITICAL REMINDER: You are a CONTRARIAN agent. "
                "Your probability estimate MUST differ from the group average "
                "by at least 10 percentage points. If the group says 40%, "
                "you should say 30% or 50%+. NEVER agree with the majority."
            )

        full_prompt = f"{system_text}\n\n---\n\n{user_text}" if system_text else user_text

        response = await model.generate_content_async(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens_per_response,
                response_mime_type="application/json",  # Fix Priority 1: force JSON
            ),
        )

        # Rough token estimation (Gemini doesn't always return usage)
        self._total_input_tokens += len(full_prompt) // 4
        self._total_output_tokens += len(response.text) // 4 if response.text else 0

        return response.text or ""

    async def _query_agent(
        self, agent: AgentState, messages: list[dict]
    ) -> None:
        """Query agent and update state with parsed response."""
        try:
            text = await self._query_single_agent(agent, messages)
            data = self._parse_json_response(text)

            prob = float(data.get("probability", agent.last_estimate))
            prob = max(0.01, min(0.99, prob))

            agent.estimates.append(prob)
            agent.confidences.append(
                data.get("confidence_score", data.get("confidence", "medium"))
            )
            agent.reasonings.append(
                data.get("final_reasoning", data.get("reasoning", ""))
            )
            agent.key_factors.append(
                data.get("key_factor", data.get("key_uncertainty", ""))
            )

        except Exception as exc:
            agent.parse_failures += 1
            agent.estimates.append(agent.last_estimate if agent.estimates else 0.5)
            agent.confidences.append("low")
            agent.reasonings.append(f"Error: {exc}")
            agent.key_factors.append("")
            log.warning(f"Agent {agent.persona.id} parse failure: {str(exc)[:100]}")

    async def _query_agent_premortem(
        self, agent: AgentState, messages: list[dict]
    ) -> None:
        """Query agent for pre-mortem and store explanation."""
        try:
            text = await self._query_single_agent(agent, messages)
            data = self._parse_json_response(text)

            prob = float(data.get("probability", agent.last_estimate))
            prob = max(0.01, min(0.99, prob))

            agent.estimates.append(prob)
            agent.confidences.append(data.get("confidence", "medium"))
            agent.reasonings.append(data.get("reasoning", ""))
            agent.key_factors.append(data.get("missed_factor", ""))
            agent.premortem_explanation = data.get("premortem_explanation", "")
            agent.missed_factor = data.get("missed_factor", "")

        except Exception as exc:
            agent.parse_failures += 1
            agent.estimates.append(agent.last_estimate)
            agent.confidences.append("low")
            agent.reasonings.append(f"Error: {exc}")
            agent.key_factors.append("")

    def _log_round_stats(
        self,
        agents: list[AgentState],
        round_num: int,
        round_type: str = "",
        decision_id: int | None = None,
        retry_triggered: bool = False,
    ) -> dict:
        """Log statistics for current round to detect issues like std=0.
        Also writes to swarm_rounds and swarm_agent_estimates tables."""
        round_idx = round_num - 1
        ests = [a.estimates[round_idx] for a in agents if len(a.estimates) > round_idx]
        if len(ests) < 2:
            return {}

        mean_e = statistics.mean(ests)
        median_e = statistics.median(ests)
        std_e = statistics.stdev(ests)
        min_e = min(ests)
        max_e = max(ests)
        unique_count = len(set(round(e, 3) for e in ests))
        parse_failures_this_round = sum(
            1 for a in agents if a.parse_failures > 0
            and len(a.estimates) > round_idx
            and "Error:" in (a.reasonings[round_idx] if len(a.reasonings) > round_idx else "")
        )
        da_count = sum(
            1 for a in agents
            if len(a.is_devils_advocate) > round_idx
            and a.is_devils_advocate[round_idx]
        )

        log.info(
            f"  R{round_num} stats: mean={mean_e:.1%} std={std_e:.3f} "
            f"range=[{min_e:.1%}-{max_e:.1%}] unique={unique_count}/{len(ests)} "
            f"parse_fail={parse_failures_this_round}"
        )

        # Warn if suspiciously low diversity
        if unique_count <= 3 and len(ests) > 10:
            log.warning(
                f"  R{round_num} LOW DIVERSITY: only {unique_count} unique values "
                f"among {len(ests)} agents! Possible parsing/prompt issue."
            )

        stats = {
            "mean": mean_e,
            "median": median_e,
            "std": std_e,
            "min": min_e,
            "max": max_e,
            "unique": unique_count,
            "parse_failures": parse_failures_this_round,
            "agents_count": len(ests),
            "devils_advocates": da_count,
            "diversity_retry": retry_triggered,
            "temperature": self.config.temperature,
        }

        # Write to DB
        if decision_id is not None:
            try:
                from llm_debate_swarm.tracking.decision_logger import get_logger_instance
                dl = get_logger_instance()
                dl.log_swarm_round(
                    decision_id=decision_id,
                    round_num=round_num,
                    round_type=round_type or "unknown",
                    stats=stats,
                )
                # Batch agent estimates
                agents_data = []
                for a in agents:
                    if len(a.estimates) <= round_idx:
                        continue
                    # Model name from persona bucket (для gemini_multi tracking)
                    bucket = getattr(a.persona, "model_bucket", "standard")
                    model_used = getattr(a, "_last_model", "") or bucket
                    agents_data.append({
                        "agent_id": a.persona.id,
                        "category": getattr(a.persona, "category", "unknown"),
                        "is_devils_advocate": (
                            len(a.is_devils_advocate) > round_idx
                            and a.is_devils_advocate[round_idx]
                        ),
                        "estimate": a.estimates[round_idx],
                        "confidence": 0.5,
                        "reasoning": a.reasonings[round_idx] if len(a.reasonings) > round_idx else "",
                        "key_factor": f"model={model_used}|" + (
                            a.key_factors[round_idx] if len(a.key_factors) > round_idx else ""
                        ),
                    })
                dl.log_swarm_agents_batch(decision_id, round_num, agents_data)
            except Exception as exc:
                log.warning(f"swarm round DB log failed: {exc}")

        return stats

    async def _run_batched(self, tasks: list) -> None:
        """Run tasks in batches to respect rate limits.

        Delay залежить від provider: Groq має tight TPM (30k/min) → довший delay.
        """
        batch_size = self.config.batch_size
        # Batch delay per provider/model TPM budget.
        # gemini_multi: interleaved personas → consecutive batches touch
        # DIFFERENT models (5-bucket rotation). Кожен bucket hit once per 5 batches.
        # batch_size=2 × 3000 = 6000 tokens per batch, but split 2 different models
        # → 3000 tokens per model per batch.
        # Per bucket: 3 hits per round × 3000 = 9000 tokens; spread across 2.5 min
        # → 3600 tokens/min per bucket << 15K TPM. Huge margin.
        # Can be aggressive: 15s between batches.
        if self._swarm_provider == "gemini_multi":
            batch_delay = 15.0
        elif self._swarm_provider == "groq":
            batch_delay = 22.0
        else:
            batch_delay = 1.0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            if i + batch_size < len(tasks):
                await asyncio.sleep(batch_delay)

    # ------------------------------------------------------------------
    # Devil's advocate selection
    # ------------------------------------------------------------------

    def _select_devils_advocates(
        self, agents: list[AgentState], round_num: int
    ) -> list[AgentState]:
        """Select conformist agents as devil's advocates for this round."""
        count = self.config.devils_advocate_count

        # Exclude agents who were DA in the previous round
        prev_round_idx = round_num - 2  # 0-indexed, previous round
        eligible = []
        for a in agents:
            was_da = (
                len(a.is_devils_advocate) > prev_round_idx
                and prev_round_idx >= 0
                and a.is_devils_advocate[prev_round_idx]
            )
            # Also exclude permanent contrarians — they already do this
            if not was_da and a.persona.category != "contrarian":
                eligible.append(a)

        if not eligible:
            eligible = list(agents)

        # Pick agents closest to consensus (most conformist)
        estimates = [a.last_estimate for a in agents if a.estimates]
        if not estimates:
            return eligible[:count]

        consensus = statistics.median(estimates)
        eligible.sort(key=lambda a: abs(a.last_estimate - consensus))
        return eligible[:count]

    # ------------------------------------------------------------------
    # Debate summaries
    # ------------------------------------------------------------------

    def _build_debate_summary(self, agents: list[AgentState]) -> str:
        """Build LOW/MID/HIGH group summary for debate prompt."""
        if not agents or not agents[0].estimates:
            return "No previous estimates."

        low, mid, high = [], [], []
        for a in agents:
            est = a.last_estimate
            if est < 0.35:
                low.append(a)
            elif est > 0.65:
                high.append(a)
            else:
                mid.append(a)

        lines = []

        if high:
            lines.append(f"**HIGH probability ({len(high)} agents, >65% YES):**")
            for a in sorted(high, key=lambda x: x.last_estimate, reverse=True)[:3]:
                lines.append(
                    f"  - {a.persona.name} ({a.last_estimate:.0%}): "
                    f"{a.last_reasoning[:150]}"
                )

        if mid:
            lines.append(f"\n**UNCERTAIN ({len(mid)} agents, 35-65%):**")
            for a in sorted(mid, key=lambda x: abs(x.last_estimate - 0.5))[:3]:
                lines.append(
                    f"  - {a.persona.name} ({a.last_estimate:.0%}): "
                    f"{a.last_reasoning[:150]}"
                )

        if low:
            lines.append(f"\n**LOW probability ({len(low)} agents, <35% YES):**")
            for a in sorted(low, key=lambda x: x.last_estimate)[:3]:
                lines.append(
                    f"  - {a.persona.name} ({a.last_estimate:.0%}): "
                    f"{a.last_reasoning[:150]}"
                )

        return "\n".join(lines)

    def _build_premortem_summary(self, agents: list[AgentState]) -> str:
        """Summarize pre-mortem failure modes from all agents."""
        explanations = []
        for a in agents:
            if a.premortem_explanation:
                explanations.append(
                    f"- {a.persona.name}: {a.premortem_explanation[:200]}"
                )
        if not explanations:
            return "No pre-mortem arguments available."
        return "\n".join(explanations[:15])  # top 15 to keep prompt manageable

    # ------------------------------------------------------------------
    # JSON parsing (4-level fallback)
    # ------------------------------------------------------------------

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response with 4-level fallback + Qwen <think> strip."""
        # Strip Qwen/DeepSeek R1 reasoning trace: <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also strip unclosed <think> at start
        if text.strip().startswith("<think>"):
            end = text.find("</think>")
            if end > 0:
                text = text[end + len("</think>"):]
            else:
                # Unclosed — drop up to last }
                last_brace = text.rfind("}")
                if last_brace > 0:
                    text = text[:last_brace + 1]

        # Level 1: ```json ... ``` block
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass

        # Level 2: bare ``` ... ``` block
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                try:
                    return json.loads(parts[1].strip())
                except json.JSONDecodeError:
                    pass

        # Level 3: raw JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Level 4: regex extraction for probability (Fix #26: clamp to valid range)
        prob_match = re.search(r'"probability"\s*:\s*([\d.]+)', text)
        if prob_match:
            try:
                raw_prob = float(prob_match.group(1))
            except ValueError:
                raise ValueError(f"Could not parse probability from regex: {prob_match.group(1)}")
            clamped = max(0.01, min(0.99, raw_prob))
            return {
                "probability": clamped,
                "reasoning": text[:200],
                "confidence": "low",
            }

        raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, agents: list[AgentState], meta: dict) -> SwarmResult:
        """Assemble SwarmResult from agents and meta-synthesis."""
        all_finals = extract_valid_estimates([a.last_estimate for a in agents])

        # Compute per-round std
        std_per_round = []
        max_rounds = max(len(a.estimates) for a in agents) if agents else 0
        for r in range(max_rounds):
            round_ests = [a.estimates[r] for a in agents if len(a.estimates) > r]
            if len(round_ests) > 1:
                std_per_round.append(statistics.stdev(round_ests))
            else:
                std_per_round.append(0.0)

        # Blind vs aware means
        # IMPORTANT: aware_mean must EXCLUDE pre-mortem (R6) — it's a thought
        # experiment, not a real estimate.
        blind_ests, aware_ests = [], []
        pm_idx_local = self.config.premortem_round - 1
        for a in agents:
            for i, est in enumerate(a.estimates):
                if i < self.config.blind_rounds:
                    blind_ests.append(est)
                elif i == pm_idx_local:
                    continue  # skip pre-mortem
                else:
                    aware_ests.append(est)

        blind_mean = statistics.mean(blind_ests) if blind_ests else 0.5
        aware_mean = statistics.mean(aware_ests) if aware_ests else 0.5
        anch = detect_anchoring(blind_ests, aware_ests)
        conv = convergence_ratio(std_per_round[0], std_per_round[-1]) if len(std_per_round) >= 2 else 1.0

        # Pre-mortem impact — require BOTH R5 and R6 for comparison (Fix #28)
        pm_idx = self.config.premortem_round - 1
        pre_idx = pm_idx - 1
        pre, post = [], []
        for a in agents:
            if len(a.estimates) > pm_idx:
                pre.append(a.estimates[pre_idx])
                post.append(a.estimates[pm_idx])
        pm_frac = premortem_impact(pre, post)

        # Agent final summaries
        agent_finals = [
            {"id": a.persona.id, "name": a.persona.name,
             "probability": a.last_estimate,
             "category": a.persona.category,
             "bias": a.persona.bias_direction}
            for a in agents
        ]

        # Meta result
        meta_prob = float(meta.get("probability", 0.5))
        meta_prob = max(0.01, min(0.99, meta_prob))
        meta_conf = float(meta.get("confidence", compute_confidence(all_finals)))
        meta_conf = max(0.05, min(0.95, meta_conf))

        return SwarmResult(
            probability=meta_prob,
            confidence=meta_conf,
            raw_trimmed_mean=trimmed_mean(all_finals, self.config.trim_pct),
            blind_mean=blind_mean,
            aware_mean=aware_mean,
            anchoring_shift=anch,
            convergence_ratio=conv,
            premortem_changed_frac=pm_frac,
            agent_count=len(agents),
            rounds_completed=max_rounds,
            std_per_round=std_per_round,
            agent_finals=agent_finals,
            meta_bias_corrections=meta.get("bias_corrections", {}),
            swarm_model=self.config.swarm_model,
        )

    def _estimate_cost(self) -> float:
        """Estimate cost in USD from tracked token usage."""
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        mini_input_cost = self._total_input_tokens * 0.15 / 1_000_000
        mini_output_cost = self._total_output_tokens * 0.60 / 1_000_000
        # Rough split: meta-synthesis is ~2% of total tokens but at GPT-4o price
        # This is an approximation; exact tracking would need per-call model tracking
        return mini_input_cost + mini_output_cost
