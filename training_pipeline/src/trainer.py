
from typing import Any
from functools import partial

from transformers import TrainingArguments
from trl import SFTTrainer

from configs.training_config import TrainingConfig
from src.data import format_example


class QLoRATrainer:
    """
    Build the TRL SFTTrainer using the model,
    tokenizer, datasets and training configuration.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        validation_dataset: Any,
    ) -> None:

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def build_training_args(self) -> TrainingArguments:
        """
        Create Hugging Face TrainingArguments.

        Training values are preserved from the original notebook.
        """

        return TrainingArguments(
            output_dir=self.config.output_dir,

            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,

            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,

            lr_scheduler_type=self.config.lr_scheduler_type,
            num_train_epochs=self.config.num_train_epochs,

            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,

            bf16=self.config.bf16,
            fp16=self.config.fp16,

            optim=self.config.optim,
            remove_unused_columns=self.config.remove_unused_columns,

            report_to=self.config.report_to,

            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,

            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,

            gradient_checkpointing=self.config.gradient_checkpointing,

            logging_dir="logs",
            save_safetensors=True,
        )

    def build_trainer(self) -> SFTTrainer:
        """
        Construct and return the TRL SFTTrainer.

        Training is not started inside this method.
        """

        training_args = self.build_training_args()

        formatting_function = partial(
            format_example,
            tokenizer=self.tokenizer,
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            args=training_args,
            formatting_func=formatting_function,
        )

        return trainer
