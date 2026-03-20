# Financial LLM Fine-Tuning Pipeline

Production-grade pipeline for fine-tuning open-source LLMs (Llama, Mistral) on financial advisory tasks with evaluation, A/B testing, and outcome-based model selection.

Built for regulated financial environments where tone compliance, factual accuracy, and personalization matter.

## Use Cases

| Model Variant | Task | Base Model | Method |
|---|---|---|---|
| `pearl-advisor-tone` | Financial advice tone & SEC compliance | Mistral-7B | LoRA |
| `pearl-portfolio` | Portfolio recommendation personalization | Llama-3.1-8B | QLoRA |
| `pearl-risk` | Risk assessment scoring | Mistral-7B | LoRA |
| `pearl-intent` | User intent classification | Llama-3.1-8B | Full FT (4-bit) |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Data Prep   │────▶│  Training    │────▶│  Evaluation  │────▶│  A/B Deploy  │
│              │     │              │     │              │     │              │
│ • Curate     │     │ • LoRA/QLoRA │     │ • Accuracy   │     │ • Canary     │
│ • Clean      │     │ • Checkpts   │     │ • Tone score │     │ • Traffic %  │
│ • Split      │     │ • W&B track  │     │ • Compliance │     │ • Outcome    │
│ • Tokenize   │     │ • Early stop │     │ • Latency    │     │   tracking   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                     │
                                              ┌──────────────────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │ Outcome Feedback  │
                                    │ Loop              │
                                    │                   │
                                    │ • User ratings    │
                                    │ • Action tracking │
                                    │ • Retrain trigger │
                                    └──────────────────┘
```

## Project Structure

```
financial-llm-finetuning/
├── README.md
├── requirements.txt
├── config/
│   ├── base_config.yaml
│   ├── tone_compliance.yaml        # Advisor tone fine-tuning config
│   ├── portfolio_personalization.yaml
│   ├── risk_scoring.yaml
│   └── intent_classification.yaml
├── data/
│   ├── prepare_dataset.py          # Data curation & cleaning
│   ├── tokenizer_utils.py          # Tokenization with financial vocab
│   └── samples/
│       ├── tone_samples.jsonl      # Sample training data
│       └── intent_samples.jsonl
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # Unified training loop (LoRA/QLoRA/Full)
│   ├── lora_config.py              # LoRA hyperparameter configs
│   └── callbacks.py                # W&B logging, early stopping, checkpointing
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py                # Multi-metric evaluation framework
│   ├── tone_scorer.py              # Financial tone & compliance scoring
│   ├── accuracy_checker.py         # Factual accuracy vs ground truth
│   ├── ab_comparator.py            # Head-to-head model comparison
│   └── outcome_tracker.py          # Track which outputs users acted on
├── serving/
│   ├── __init__.py
│   ├── model_registry.py           # Version management & rollback
│   ├── ab_router.py                # Traffic splitting for A/B tests
│   └── inference.py                # Production inference with batching
└── tests/
    ├── test_trainer.py
    ├── test_evaluator.py
    ├── test_tone_scorer.py
    └── test_outcome_tracker.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Prepare dataset
python data/prepare_dataset.py --config config/tone_compliance.yaml

# Fine-tune with LoRA
python -m training.trainer --config config/tone_compliance.yaml --method lora

# Evaluate
python -m evaluation.evaluator --model checkpoints/tone-v1 --test-set data/test.jsonl

# Run A/B comparison
python -m evaluation.ab_comparator --model-a tone-v1 --model-b tone-v2

# Track outcomes
python -m evaluation.outcome_tracker --model tone-v2 --window 7d
```

## Training Details

### LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Evaluation Metrics
- **Tone Score**: Custom classifier trained on SEC-compliant vs non-compliant advisory language
- **Factual Accuracy**: Claims extracted and verified against financial data sources
- **Personalization Score**: Measures specificity to user portfolio vs generic advice
- **Latency**: P50/P95/P99 inference latency under production load
- **User Outcome Rate**: % of recommendations users actually followed (tracked post-delivery)

### Outcome-Based Feedback Loop
The pipeline tracks which AI recommendations users actually act on. Models with higher "outcome rate" (user followed the recommendation) get promoted in the A/B router. This creates a continuous improvement cycle:

```
Generate recommendation → Deliver to user → Track if user acted on it
→ Aggregate outcome data → Retrain model on high-outcome examples
→ Deploy new version → A/B test against incumbent
```

## Design Decisions

- **LoRA over full fine-tuning for most tasks**: 10x cheaper, faster iteration, easy to swap adapters per use case while keeping the same base model in memory
- **QLoRA for portfolio personalization**: This task needs more model capacity than LoRA provides — QLoRA with 4-bit quantization gives us full fine-tuning quality at manageable GPU cost
- **Outcome tracking over RLHF**: For financial advisory, we care about whether users *acted* on recommendations — that's a cleaner signal than abstract preference ratings. The outcome tracker provides a natural reward signal without needing a separate reward model
- **A/B routing at inference**: New models are deployed to 10% traffic first, promoted to 50% if metrics hold, and fully rolled out after 7 days. Automatic rollback if tone compliance drops below threshold

