import pytest
from datetime import datetime

from evaluation.tone_scorer import ToneScorer, ToneResult
from evaluation.outcome_tracker import OutcomeTracker, Outcome, Action
from evaluation.evaluator import ModelEvaluator
from training.trainer import FTConfig, FTMethod, FinancialTrainer, load_dataset, split_dataset


class TestToneScorer:
    def setup_method(self):
        self.s = ToneScorer()

    def test_compliant_hedged_text(self):
        r = self.s.score(
            "Based on historical data, this sector may offer growth potential. "
            "Past performance does not guarantee future results. Consider your "
            "risk tolerance and consult with a financial advisor."
        )
        assert r.compliant
        assert r.hedge_score > 0.5
        assert r.aggression == 0.0
        assert r.score > 0.7

    def test_noncompliant_text(self):
        r = self.s.score("Guaranteed returns! 100% safe. Can't lose. Buy now!")
        assert not r.compliant
        assert len(r.violations) >= 3
        assert r.score < 0.3

    def test_aggressive_language(self):
        r = self.s.score(
            "Act now! Limited time offer. Don't miss out on this huge opportunity!"
        )
        assert r.aggression > 0.5

    def test_generic_penalized(self):
        r = self.s.score(
            "In general, most investors should consider diversification. "
            "It depends on many factors."
        )
        assert r.personalization < 0.5

    def test_personalized_with_context(self):
        r = self.s.score(
            "Given your moderate risk tolerance and 20-year horizon, "
            "a 70/30 equity-bond split may be appropriate for your $50K portfolio.",
            user_ctx={"risk": "moderate", "horizon": "20-year"},
        )
        assert r.personalization > 0.3

    def test_score_range(self):
        for text in [
            "Buy now! Guaranteed!",
            "Consider diversification based on your risk tolerance.",
            "In general it depends.",
        ]:
            r = self.s.score(text)
            assert 0.0 <= r.score <= 1.0


class TestOutcomeTracker:
    def setup_method(self):
        self.t = OutcomeTracker()

    def _seed_data(self, model, total, successes):
        for i in range(total):
            self.t.record_delivery(f"r_{model}_{i}", model, f"rec {i}", f"u_{i}")
            outcome = Outcome.FOLLOWED if i < successes else Outcome.IGNORED
            self.t.record_outcome(f"r_{model}_{i}", outcome)

    def test_delivery_and_outcome(self):
        rec = self.t.record_delivery("r1", "v1", "Buy AAPL", "u1")
        assert rec.rec_id == "r1"
        updated = self.t.record_outcome("r1", Outcome.FOLLOWED, [Action.INVESTED])
        assert updated.success
        assert updated.outcome == Outcome.FOLLOWED

    def test_outcome_rate(self):
        self._seed_data("v1", 10, 7)
        stats = self.t.model_outcome_rate("v1")
        assert stats["rate"] == 0.7
        assert stats["total"] == 10

    def test_compare_models(self):
        self._seed_data("v1", 20, 12)
        self._seed_data("v2", 20, 16)
        comp = self.t.compare_models("v1", "v2")
        assert comp["winner"] == "v2"
        assert comp["lift_pct"] > 0

    def test_extract_training_examples(self):
        self._seed_data("v1", 10, 7)
        examples = self.t.extract_training_examples("v1", limit=5)
        assert len(examples) == 5
        assert all(e["outcome"] == "followed" for e in examples)

    def test_missing_recommendation(self):
        result = self.t.record_outcome("nonexistent", Outcome.FOLLOWED)
        assert result is None

    def test_drift_detection_no_drift(self):
        self._seed_data("v1", 20, 14)
        drift = self.t.detect_drift("v1")
        assert not drift["drifted"]


class TestModelEvaluator:
    def test_evaluate_compliant_dataset(self):
        data = [
            {"response": "Based on available data, this may offer growth potential. Consider your risk tolerance.", "label": "compliant"},
            {"response": "Historically this sector could provide returns. Past performance is not guaranteed.", "label": "compliant"},
            {"response": "You might consider diversification depending on your timeline and risk tolerance.", "label": "compliant"},
        ]
        ev = ModelEvaluator()
        report = ev.evaluate_dataset("test-model", "tone", data)
        assert report.passed
        assert report.overall_score > 0.5

    def test_evaluate_mixed_dataset(self):
        data = [
            {"response": "Consider your risk tolerance. Past performance is not guaranteed.", "label": "compliant"},
            {"response": "Guaranteed returns! Buy now! Can't lose!", "label": "non_compliant"},
        ]
        ev = ModelEvaluator()
        report = ev.evaluate_dataset("test-model", "tone", data)
        assert not report.passed  # compliance rate below 90%

    def test_empty_dataset(self):
        ev = ModelEvaluator()
        report = ev.evaluate_dataset("test", "tone", [])
        assert not report.passed


class TestTrainer:
    def test_mock_training(self):
        config = FTConfig(
            model_name="test-model",
            method=FTMethod.LORA,
            dataset_path="data/samples/tone_training.jsonl",
            task="test_task",
            epochs=2,
        )
        trainer = FinancialTrainer(config, use_mock=True)
        trainer.setup()

        data = load_dataset(config.dataset_path)
        train, eval_set = split_dataset(data)
        result = trainer.train(train, eval_set)

        assert result["task"] == "test_task"
        assert result["method"] == "lora"
        assert result["mock"] is True
        assert result["train_loss"] > 0

    def test_load_dataset(self):
        data = load_dataset("data/samples/tone_training.jsonl")
        assert len(data) > 5
        assert all("instruction" in d for d in data)
        assert all("response" in d for d in data)

    def test_split_dataset(self):
        data = load_dataset("data/samples/tone_training.jsonl")
        train, eval_set = split_dataset(data, train_ratio=0.8)
        assert len(train) + len(eval_set) == len(data)
        assert len(train) > len(eval_set)

    def test_config_from_values(self):
        config = FTConfig(
            model_name="test", method=FTMethod.QLORA,
            lora_r=32, epochs=5,
        )
        assert config.method == FTMethod.QLORA
        assert config.lora_r == 32
        assert config.epochs == 5
