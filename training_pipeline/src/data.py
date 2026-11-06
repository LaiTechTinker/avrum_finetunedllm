
from datasets import load_dataset, DatasetDict
from typing import Any

from configs.training_config import (
    DATASET_NAME,
    DATASET_SAMPLE_SIZE,
    DATASET_SEED,
    EVAL_SPLIT_SIZE,
)


def load_agricultural_dataset() -> DatasetDict:
    """
    Download the agricultural question-answering dataset
    from Hugging Face.
    """

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    return dataset


def prepare_dataset(dataset: DatasetDict) -> Any:
    """
    Shuffle the original training dataset and select
    the first 8,000 samples.

    This preserves the original notebook's logic.
    """

    print("Shuffling and selecting dataset samples...")

    train_dataset = (
        dataset["train"]
        .shuffle(seed=DATASET_SEED)
        .select(range(DATASET_SAMPLE_SIZE))
    )

    print(f"Selected samples: {len(train_dataset)}")

    return train_dataset


def split_dataset(dataset: Any) -> tuple[Any, Any]:
    """
    Split the prepared dataset into training
    and validation datasets.

    Original split: 90% training, 10% validation.
    """

    print("Splitting dataset into training and validation...")

    split = dataset.train_test_split(
        test_size=EVAL_SPLIT_SIZE,
        seed=DATASET_SEED,
    )

    train_dataset = split["train"]
    validation_dataset = split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    return train_dataset, validation_dataset


def format_example(example: dict, tokenizer: Any) -> str:
    """
    Format a question-answer pair using Qwen's
    built-in chat template.

    This preserves the original notebook's formatting.
    """

    messages = [
        {
            "role": "user",
            "content": example["question"],
        },
        {
            "role": "assistant",
            "content": example["answers"],
        },
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return formatted_text


def get_prepared_datasets() -> tuple[Any, Any]:
    """
    Main dataset preparation function.

    Downloads, prepares and splits the dataset.
    """

    dataset = load_agricultural_dataset()

    prepared_dataset = prepare_dataset(dataset)

    train_dataset, validation_dataset = split_dataset(
        prepared_dataset
    )

    return train_dataset, validation_dataset
