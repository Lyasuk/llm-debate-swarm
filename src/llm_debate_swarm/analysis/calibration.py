"""Calibration tracking via Brier score and related metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from llm_debate_swarm.utils.logger import get_logger

log = get_logger("analysis.calibration")


@dataclass
class PredictionRecord:
    """A single prediction + outcome for calibration tracking."""

    market_id: str
    question: str
    predicted_probability: float
    actual_outcome: float | None = None  # 1.0 = YES, 0.0 = NO, None = unresolved
    model_source: str = ""  # "claude", "gpt4o", "consensus", "mirofish"
    timestamp: str = ""


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""

    brier_score: float  # lower is better, 0 = perfect, 0.25 = random
    num_resolved: int
    win_rate: float  # % of correct directional calls
    avg_confidence: float
    overconfidence_ratio: float  # how often we're overconfident


class CalibrationTracker:
    """Tracks prediction accuracy and calculates calibration metrics."""

    def __init__(self, data_dir: str = "data/calibration"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[PredictionRecord] = []
        self._load()

    def _load(self) -> None:
        """Load existing records from disk."""
        path = self.data_dir / "predictions.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.records.append(PredictionRecord(**data))
                        except (json.JSONDecodeError, TypeError):
                            continue
        log.info(f"Loaded {len(self.records)} calibration records")

    def add_prediction(self, record: PredictionRecord) -> None:
        """Add a new prediction record."""
        self.records.append(record)
        self._save_record(record)

    def resolve(self, market_id: str, outcome: float) -> None:
        """Resolve a prediction (1.0 = YES won, 0.0 = NO won)."""
        for record in self.records:
            if record.market_id == market_id and record.actual_outcome is None:
                record.actual_outcome = outcome
        self._save_all()  # Rewrite file with resolved records

    def get_metrics(self, model_source: str | None = None) -> CalibrationMetrics:
        """Calculate calibration metrics, optionally filtered by model."""
        resolved = [
            r for r in self.records
            if r.actual_outcome is not None
            and (model_source is None or r.model_source == model_source)
        ]

        if not resolved:
            return CalibrationMetrics(
                brier_score=0.25,
                num_resolved=0,
                win_rate=0.5,
                avg_confidence=0.5,
                overconfidence_ratio=0.5,
            )

        # Brier score: mean squared error
        brier = sum(
            (r.predicted_probability - r.actual_outcome) ** 2
            for r in resolved
        ) / len(resolved)

        # Win rate: did we predict the right direction?
        correct = sum(
            1 for r in resolved
            if (r.predicted_probability > 0.5 and r.actual_outcome == 1.0)
            or (r.predicted_probability < 0.5 and r.actual_outcome == 0.0)
            or (r.predicted_probability == 0.5)  # 50/50 doesn't count wrong
        )
        win_rate = correct / len(resolved)

        # Average confidence (distance from 0.5)
        avg_conf = sum(
            abs(r.predicted_probability - 0.5) for r in resolved
        ) / len(resolved)

        # Overconfidence: % of high-confidence predictions that were wrong
        high_conf = [r for r in resolved if abs(r.predicted_probability - 0.5) > 0.2]
        if high_conf:
            wrong = sum(
                1 for r in high_conf
                if (r.predicted_probability > 0.5 and r.actual_outcome == 0.0)
                or (r.predicted_probability < 0.5 and r.actual_outcome == 1.0)
            )
            overconf = wrong / len(high_conf)
        else:
            overconf = 0.0

        return CalibrationMetrics(
            brier_score=brier,
            num_resolved=len(resolved),
            win_rate=win_rate,
            avg_confidence=avg_conf,
            overconfidence_ratio=overconf,
        )

    def get_brier_adjustment(self, model_source: str = "consensus") -> float:
        """Return Kelly fraction adjustment based on Brier score.

        Returns 0.5-1.5 multiplier:
        - Brier < 0.15 (excellent): 1.5x
        - Brier 0.15-0.20 (good): 1.2x
        - Brier 0.20-0.25 (random): 1.0x
        - Brier > 0.25 (bad): 0.5x
        """
        metrics = self.get_metrics(model_source)
        if metrics.num_resolved < 10:
            return 1.0  # Not enough data, use default

        brier = metrics.brier_score
        if brier < 0.15:
            return 1.5
        elif brier < 0.20:
            return 1.2
        elif brier < 0.25:
            return 1.0
        else:
            return 0.5

    def _save_record(self, record: PredictionRecord) -> None:
        path = self.data_dir / "predictions.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def _save_all(self) -> None:
        path = self.data_dir / "predictions.jsonl"
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.__dict__) + "\n")
