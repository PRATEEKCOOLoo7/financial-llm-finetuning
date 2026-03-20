"""Outcome-based feedback tracker.

Tracks which AI-generated recommendations users actually followed,
creating a natural reward signal for model improvement. High-outcome
examples become training data for the next fine-tuning iteration.

The feedback loop:
1. Model generates recommendation
2. Delivered to user
3. Track if user acted on it (clicked, invested, scheduled)
4. Aggregate outcomes per model version
5. High-outcome examples → next training set
6. A/B test new model vs incumbent
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


class Outcome(str, Enum):
    FOLLOWED = "followed"
    IGNORED = "ignored"
    DISMISSED = "dismissed"
    PARTIAL = "partial"


class Action(str, Enum):
    CLICKED = "clicked"
    INVESTED = "invested"
    SCHEDULED = "scheduled"
    SHARED = "shared"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    NONE = "none"


@dataclass
class TrackedRecommendation:
    rec_id: str
    model_version: str
    text: str
    user_id: str
    delivered_at: datetime
    outcome: Optional[Outcome] = None
    actions: list[Action] = field(default_factory=list)
    outcome_at: Optional[datetime] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.outcome in (Outcome.FOLLOWED, Outcome.PARTIAL)

    @property
    def time_to_action(self) -> Optional[timedelta]:
        if self.outcome_at and self.delivered_at:
            return self.outcome_at - self.delivered_at
        return None


class OutcomeTracker:
    def __init__(self, drift_threshold: float = 0.1):
        self.records: list[TrackedRecommendation] = []
        self.drift_threshold = drift_threshold

    def record_delivery(self, rec_id: str, model_version: str,
                        text: str, user_id: str,
                        meta: dict = None) -> TrackedRecommendation:
        rec = TrackedRecommendation(
            rec_id=rec_id, model_version=model_version,
            text=text, user_id=user_id,
            delivered_at=datetime.utcnow(), meta=meta or {},
        )
        self.records.append(rec)
        return rec

    def record_outcome(self, rec_id: str, outcome: Outcome,
                       actions: list[Action] = None) -> Optional[TrackedRecommendation]:
        for rec in reversed(self.records):
            if rec.rec_id == rec_id:
                rec.outcome = outcome
                rec.actions = actions or []
                rec.outcome_at = datetime.utcnow()
                log.info(f"outcome: {rec_id} → {outcome.value}")
                return rec
        log.warning(f"recommendation not found: {rec_id}")
        return None

    def model_outcome_rate(self, model: str, window_days: int = None) -> dict:
        filtered = [r for r in self.records if r.model_version == model and r.outcome]
        if window_days:
            cutoff = datetime.utcnow() - timedelta(days=window_days)
            filtered = [r for r in filtered if r.delivered_at >= cutoff]

        if not filtered:
            return {"model": model, "total": 0, "rate": 0.0}

        success = sum(1 for r in filtered if r.success)
        return {
            "model": model,
            "total": len(filtered),
            "successful": success,
            "rate": round(success / len(filtered), 4),
            "by_outcome": {
                o.value: sum(1 for r in filtered if r.outcome == o)
                for o in Outcome
            },
        }

    def compare_models(self, model_a: str, model_b: str,
                       window_days: int = 7) -> dict:
        a = self.model_outcome_rate(model_a, window_days)
        b = self.model_outcome_rate(model_b, window_days)
        rate_a, rate_b = a["rate"], b["rate"]
        lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0
        winner = model_b if rate_b > rate_a else model_a

        return {
            "model_a": a, "model_b": b,
            "lift_pct": round(lift, 2),
            "winner": winner,
            "recommendation": (
                f"Promote {winner}" if abs(lift) > 5
                else "Continue testing — no significant difference"
            ),
        }

    def extract_training_examples(self, model: str,
                                  limit: int = 1000) -> list[dict]:
        """Extract high-outcome examples for next training iteration."""
        successful = [r for r in self.records
                      if r.model_version == model and r.success]
        successful.sort(key=lambda r: r.delivered_at, reverse=True)

        examples = [
            {
                "text": r.text,
                "outcome": r.outcome.value,
                "actions": [a.value for a in r.actions],
                "meta": r.meta,
            }
            for r in successful[:limit]
        ]
        log.info(f"extracted {len(examples)} high-outcome examples from {model}")
        return examples

    def detect_drift(self, model: str, baseline_days: int = 30,
                     recent_days: int = 7) -> dict:
        baseline = self.model_outcome_rate(model, baseline_days)
        recent = self.model_outcome_rate(model, recent_days)

        b_rate = baseline["rate"]
        r_rate = recent["rate"]
        drift_mag = b_rate - r_rate
        drifted = b_rate > 0 and drift_mag > self.drift_threshold

        return {
            "model": model,
            "baseline_rate": b_rate,
            "recent_rate": r_rate,
            "drift": round(drift_mag, 4),
            "drifted": drifted,
            "action": "RETRAIN" if drifted else "OK",
        }
