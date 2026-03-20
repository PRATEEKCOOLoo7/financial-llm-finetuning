"""Financial LLM Fine-Tuning Pipeline — Demo

Demonstrates:
1. Load training data (financial advisory JSONL)
2. Train/eval split
3. Fine-tune with mock trainer (or real GPU trainer with --live)
4. Evaluate model outputs with tone scorer
5. Track outcomes for feedback loop
6. Compare model versions

Run:
  python main.py           # mock mode (no GPU needed)
  python main.py --live    # real training (requires GPU + model weights)
"""

import logging
import sys
from datetime import datetime, timedelta

from training.trainer import FTConfig, FTMethod, FinancialTrainer, load_dataset, split_dataset
from evaluation.tone_scorer import ToneScorer
from evaluation.evaluator import ModelEvaluator
from evaluation.outcome_tracker import OutcomeTracker, Outcome, Action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def demo_training(use_mock: bool = True):
    print(f"\n{'='*60}")
    print("  1. TRAINING PIPELINE")
    print(f"{'='*60}\n")

    config = FTConfig(
        model_name="mistralai/Mistral-7B-v0.3",
        method=FTMethod.LORA,
        dataset_path="data/samples/tone_training.jsonl",
        task="tone_compliance",
    )

    data = load_dataset(config.dataset_path)
    train, eval_set = split_dataset(data, train_ratio=0.75)

    trainer = FinancialTrainer(config, use_mock=use_mock)
    trainer.setup()
    result = trainer.train(train, eval_set)

    print(f"\n  Training result:")
    for k, v in result.items():
        print(f"    {k}: {v}")


def demo_evaluation():
    print(f"\n{'='*60}")
    print("  2. MODEL EVALUATION")
    print(f"{'='*60}\n")

    data = load_dataset("data/samples/tone_training.jsonl")
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_dataset(
        model_name="mistral-7b-tone-v1",
        task="tone_compliance",
        examples=data,
    )

    print(f"\n{report.summary()}")
    print(f"\n  Details:")
    for k, v in report.details.items():
        print(f"    {k}: {v}")


def demo_tone_scoring():
    print(f"\n{'='*60}")
    print("  3. TONE SCORING EXAMPLES")
    print(f"{'='*60}\n")

    scorer = ToneScorer()

    examples = [
        {
            "name": "Good (compliant, hedged)",
            "text": (
                "Based on historical data, NVIDIA may offer growth potential given "
                "strong AI infrastructure demand. However, the stock trades at a premium "
                "valuation and past performance does not guarantee future results. "
                "Consider your risk tolerance and consult with your financial advisor."
            ),
        },
        {
            "name": "Bad (violations + aggressive)",
            "text": (
                "Buy NVIDIA now! Guaranteed returns of 50% this year. This is a "
                "risk-free investment and you'd be crazy not to get in. Don't miss "
                "out on this once-in-a-lifetime opportunity. Can't lose!"
            ),
        },
        {
            "name": "Mediocre (generic, no hedge)",
            "text": (
                "In general, most investors should consider diversification. "
                "It depends on many factors. Everyone's situation is different. "
                "There are many factors to consider."
            ),
        },
    ]

    for ex in examples:
        result = scorer.score(ex["text"])
        icon = "✓" if result.compliant else "✗"
        print(f"  [{icon}] {ex['name']}")
        print(f"      Score: {result.score:.3f}")
        print(f"      Compliant: {result.compliant}")
        if result.violations:
            print(f"      Violations: {result.violations}")
        print(f"      Hedge: {result.hedge_score:.2f} | Personal: {result.personalization:.2f} | Aggression: {result.aggression:.2f}")
        print()


def demo_outcome_tracking():
    print(f"\n{'='*60}")
    print("  4. OUTCOME TRACKING & MODEL COMPARISON")
    print(f"{'='*60}\n")

    tracker = OutcomeTracker()

    # Simulate model v1 recommendations
    for i in range(20):
        tracker.record_delivery(
            rec_id=f"rec_v1_{i}", model_version="tone-v1",
            text=f"Recommendation {i} from v1",
            user_id=f"user_{i % 5}",
        )
    # Simulate outcomes: 12/20 followed
    for i in range(20):
        outcome = Outcome.FOLLOWED if i < 12 else Outcome.IGNORED
        actions = [Action.CLICKED, Action.INVESTED] if i < 12 else [Action.NONE]
        tracker.record_outcome(f"rec_v1_{i}", outcome, actions)

    # Simulate model v2 (better)
    for i in range(20):
        tracker.record_delivery(
            rec_id=f"rec_v2_{i}", model_version="tone-v2",
            text=f"Recommendation {i} from v2",
            user_id=f"user_{i % 5}",
        )
    for i in range(20):
        outcome = Outcome.FOLLOWED if i < 15 else Outcome.IGNORED
        actions = [Action.CLICKED, Action.SCHEDULED] if i < 15 else [Action.NONE]
        tracker.record_outcome(f"rec_v2_{i}", outcome, actions)

    # Compare
    comparison = tracker.compare_models("tone-v1", "tone-v2")
    print(f"  Model v1: {comparison['model_a']['rate']:.0%} outcome rate ({comparison['model_a']['total']} recs)")
    print(f"  Model v2: {comparison['model_b']['rate']:.0%} outcome rate ({comparison['model_b']['total']} recs)")
    print(f"  Lift: {comparison['lift_pct']:.1f}%")
    print(f"  Winner: {comparison['winner']}")
    print(f"  Recommendation: {comparison['recommendation']}")

    # Extract training examples
    examples = tracker.extract_training_examples("tone-v2", limit=5)
    print(f"\n  High-outcome training examples extracted: {len(examples)}")

    # Drift detection
    drift = tracker.detect_drift("tone-v1")
    print(f"\n  Drift check (v1): {drift['action']} (baseline={drift['baseline_rate']:.2f}, recent={drift['recent_rate']:.2f})")


def main():
    live = "--live" in sys.argv
    if not live:
        print("\nRunning in mock mode (no GPU needed). Use --live for real training.\n")

    demo_training(use_mock=not live)
    demo_evaluation()
    demo_tone_scoring()
    demo_outcome_tracking()

    print(f"\n{'='*60}")
    print("  All demos complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
