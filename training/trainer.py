"""Fine-tuning pipeline for financial advisory LLMs.

Supports LoRA, QLoRA, and full fine-tuning with:
- Configurable training via YAML
- W&B experiment tracking
- Early stopping
- Checkpoint management

When running without GPU (e.g., CI/testing), set use_mock=True
to validate the full pipeline flow without actual training.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class FTMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


@dataclass
class FTConfig:
    model_name: str = "mistralai/Mistral-7B-v0.3"
    method: FTMethod = FTMethod.LORA
    dataset_path: str = "data/samples/tone_training.jsonl"
    output_dir: str = "checkpoints"
    task: str = "tone_compliance"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_len: int = 2048
    fp16: bool = True

    # Early stopping
    patience: int = 3
    min_delta: float = 0.01

    @classmethod
    def from_yaml(cls, path: str) -> "FTConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        method = data.pop("method", "lora")
        data["method"] = FTMethod(method)
        return cls(**data)


def load_dataset(path: str) -> list[dict]:
    """Load JSONL training data."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"loaded {len(records)} training examples from {path}")
    return records


def split_dataset(records: list[dict], train_ratio: float = 0.8) -> tuple[list, list]:
    """Split into train/eval sets."""
    split_idx = int(len(records) * train_ratio)
    train = records[:split_idx]
    eval_set = records[split_idx:]
    log.info(f"split: {len(train)} train, {len(eval_set)} eval")
    return train, eval_set


class FinancialTrainer:
    """Handles the full fine-tuning lifecycle.
    
    For real training (GPU required):
        trainer = FinancialTrainer(config)
        trainer.setup()
        result = trainer.train(train_data, eval_data)
        
    For mock/testing (no GPU):
        trainer = FinancialTrainer(config, use_mock=True)
        trainer.setup()
        result = trainer.train(train_data, eval_data)
    """

    def __init__(self, config: FTConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None

    def setup(self):
        if self.use_mock:
            log.info(f"[mock] setup: {self.config.model_name} method={self.config.method.value}")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        except ImportError as e:
            log.error(f"Missing dependency: {e}. Install with: pip install torch transformers peft bitsandbytes")
            raise

        log.info(f"loading {self.config.model_name} method={self.config.method.value}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if self.config.method == FTMethod.QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
        )

        if self.config.method in (FTMethod.LORA, FTMethod.QLORA):
            if self.config.method == FTMethod.QLORA:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self._log_params()

    def train(self, train_data: list[dict], eval_data: list[dict]) -> dict[str, Any]:
        if self.use_mock:
            return self._mock_train(train_data, eval_data)

        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

        out_dir = os.path.join(self.config.output_dir, self.config.task)
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum,
            learning_rate=self.config.lr,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.patience,
                early_stopping_threshold=self.config.min_delta,
            )],
        )

        result = trainer.train()
        final_path = os.path.join(out_dir, "final")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)

        return {
            "task": self.config.task,
            "method": self.config.method.value,
            "model": self.config.model_name,
            "train_loss": result.training_loss,
            "steps": result.global_step,
            "path": final_path,
        }

    def _mock_train(self, train_data, eval_data) -> dict[str, Any]:
        """Simulate training for pipeline testing without GPU."""
        log.info(f"[mock] training {len(train_data)} examples, eval {len(eval_data)}")

        # Simulate epochs
        for epoch in range(self.config.epochs):
            fake_loss = 2.5 / (epoch + 1) + 0.15
            log.info(f"[mock] epoch {epoch+1}/{self.config.epochs} loss={fake_loss:.4f}")
            time.sleep(0.01)

        return {
            "task": self.config.task,
            "method": self.config.method.value,
            "model": self.config.model_name,
            "train_loss": 0.42,
            "steps": len(train_data) * self.config.epochs,
            "path": f"{self.config.output_dir}/{self.config.task}/final",
            "mock": True,
        }

    def _log_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = 100 * trainable / total
        log.info(f"trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
