
import os
from pathlib import Path
import logging
from configs.training_config import (
    QLoRAConfig,
    TrainingConfig,
    MODEL_OUTPUT_DIR,
)

from src.data import get_prepared_datasets
from src.model import QLoRAModelLoader
from src.trainer import QLoRATrainer
import logging
from src.logger import setup_logger

logger=logging.getLogger(__name__)

def create_output_directories() -> None:
    """
    Create directories for checkpoints and final model artifacts.
    """

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer():
    """
    Load the QLoRA model and its tokenizer.
    """

    print("\nLoading Qwen model and tokenizer...")

    config = QLoRAConfig()

    loader = QLoRAModelLoader(config)

    model, tokenizer = loader.load()

    model.print_trainable_parameters()

    print("Model and tokenizer loaded successfully.")

    return model, tokenizer


def prepare_training_data():
    """
    Load and prepare the agricultural training dataset.
    """

    print("\nPreparing agricultural dataset...")

    train_dataset, validation_dataset = get_prepared_datasets()

    print("Dataset preparation completed.")

    return train_dataset, validation_dataset


def start_training(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
):
    """
    Build the trainer and start the fine-tuning process.
    """

    print("\nBuilding training configuration...")

    config = TrainingConfig()

    trainer_builder = QLoRATrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    trainer = trainer_builder.build_trainer()

    print("\nStarting QLoRA fine-tuning...")

    if config.resume_from_checkpoint:
        trainer.train(
            resume_from_checkpoint=config.resume_from_checkpoint
        )
    else:
        trainer.train()

    print("\nTraining completed successfully.")

    return trainer


def save_model_artifacts(trainer, tokenizer) -> None:
    """
    Save the trained LoRA adapter, tokenizer and trainer state.
    """

    print("\nSaving model artifacts...")

    adapter_dir = MODEL_OUTPUT_DIR / "adapter"
    tokenizer_dir = MODEL_OUTPUT_DIR / "tokenizer"

    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(adapter_dir))

    tokenizer.save_pretrained(str(tokenizer_dir))

    trainer.state.save_to_json(
        str(MODEL_OUTPUT_DIR / "trainer_state.json")
    )

    expected_files = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_config.json",
        tokenizer_dir / "tokenizer_config.json",
        MODEL_OUTPUT_DIR / "trainer_state.json",
    ]

    print("\nSaved artifact verification:")

    for file_path in expected_files:
        status = "OK" if file_path.exists() else "MISSING"

        print(f"[{status}] {file_path}")

    missing_files = [
        file_path
        for file_path in expected_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Some model artifacts are missing: {missing_files}"
        )

    print("\nAll expected artifacts saved successfully.")


def train() -> None:
    """
    Main training pipeline entry point.
    """

    print("=" * 60)
    print("AVRUM AGRICULTURAL LLM TRAINING PIPELINE")
    print("=" * 60)

    create_output_directories()

    # Step 1: Prepare the dataset
    train_dataset, validation_dataset = prepare_training_data()

    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Step 3: Start fine-tuning
    trainer = start_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    # Step 4: Save trained model artifacts
    save_model_artifacts(
        trainer=trainer,
        tokenizer=tokenizer,
    )

    print("\nTraining pipeline finished successfully.")


if __name__ == "__main__":
    train()
