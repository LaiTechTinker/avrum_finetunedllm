import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[1]

# Locally defaults to ./outputs. Beam overrides this with its mounted volume.
OUTPUT_DIR = Path(os.environ.get("AVRUM_OUTPUT_DIR", str(BASE_DIR / "outputs")))
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "qwen_agriculture_lora"

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
DATASET_NAME = "KisanVaani/agriculture-qa-english-only"

DATASET_SAMPLE_SIZE = 8000
DATASET_SEED = 42
EVAL_SPLIT_SIZE = 0.1


@dataclass(slots=True)
class QLoRAConfig:
    model_name: str = MODEL_NAME
    quantization_config: Optional[Dict[str, Any]] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = str(CHECKPOINT_DIR)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 2
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    optim: str = "paged_adamw_8bit"
    remove_unused_columns: bool = False
    report_to: list[str] = field(default_factory=lambda: ["mlflow"])
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    gradient_checkpointing: bool = True
