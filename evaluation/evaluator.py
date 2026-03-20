"""Model evaluation framework for financial LLMs.

Runs multiple evaluation dimensions and produces a structured
report with pass/fail decisions tied to configurable thresholds.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from evaluation.tone_scorer import ToneScorer

log = logging.getLogger(__name__)


@dataclass
class EvalMetric:
    name: str
    value: float
    threshold: float
    passed: bool

    def __str__(self):
        icon = "✓" if self.passed else "✗"
        return f"[{icon}] {self.name}: {self.value:.3f} (threshold: {self.threshold})"


@dataclass
class EvalReport:
    model: str
    task: str
    metrics: list[EvalMetric]
    passed: bool
    overall_score: float
    total_examples: int
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Evaluation: {self.model} / {self.task}"]
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'} ({self.overall_score:.3f})")
        lines.append(f"Examples: {self.total_examples}")
        for m in self.metrics:
            lines.append(f"  {m}")
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluates fine-tuned models across multiple dimensions."""

    def __init__(self):
        self.tone_scorer = ToneScorer()

    def evaluate_dataset(self, model_name: str, task: str,
                         examples: list[dict],
                         thresholds: dict[str, float] = None) -> EvalReport:
        thresholds = thresholds or {
            "compliance_rate": 0.90,
            "avg_tone_score": 0.70,
            "hedge_rate": 0.60,
            "aggression_rate": 0.10,  # Max acceptable
        }

        compliant_count = 0
        tone_scores = []
        hedge_scores = []
        agg_scores = []
        violation_counts = []

        for ex in examples:
            response = ex.get("response", "")
            label = ex.get("label", "unknown")

            result = self.tone_scorer.score(response)
            tone_scores.append(result.score)
            hedge_scores.append(result.hedge_score)
            agg_scores.append(result.aggression)
            violation_counts.append(len(result.violations))

            if result.compliant:
                compliant_count += 1

        n = len(examples)
        if n == 0:
            return EvalReport(
                model=model_name, task=task, metrics=[],
                passed=False, overall_score=0.0, total_examples=0,
            )

        compliance_rate = compliant_count / n
        avg_tone = sum(tone_scores) / n
        avg_hedge = sum(hedge_scores) / n
        avg_agg = sum(agg_scores) / n

        metrics = [
            EvalMetric("compliance_rate", compliance_rate,
                       thresholds["compliance_rate"],
                       compliance_rate >= thresholds["compliance_rate"]),
            EvalMetric("avg_tone_score", avg_tone,
                       thresholds["avg_tone_score"],
                       avg_tone >= thresholds["avg_tone_score"]),
            EvalMetric("hedge_rate", avg_hedge,
                       thresholds["hedge_rate"],
                       avg_hedge >= thresholds["hedge_rate"]),
            EvalMetric("aggression_rate", avg_agg,
                       thresholds["aggression_rate"],
                       avg_agg <= thresholds["aggression_rate"]),
        ]

        all_pass = all(m.passed for m in metrics)
        overall = sum(m.value if m.name != "aggression_rate" else (1 - m.value)
                      for m in metrics) / len(metrics)

        report = EvalReport(
            model=model_name, task=task, metrics=metrics,
            passed=all_pass, overall_score=round(overall, 4),
            total_examples=n,
            details={
                "total_violations": sum(violation_counts),
                "worst_tone_score": round(min(tone_scores), 3),
                "best_tone_score": round(max(tone_scores), 3),
            },
        )

        log.info(f"eval complete: {model_name}/{task} — {'PASS' if all_pass else 'FAIL'} ({overall:.3f})")
        return report
