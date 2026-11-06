"""Beam resource settings for Avrum QLoRA training."""
GPU = ["RTX4090", "A10G"]  # Both support the notebook's BF16 mode.
CPU = 4
MEMORY = "32Gi"
TIMEOUT_SECONDS = 12 * 60 * 60
HEADLESS = True

VOLUME_NAME = "avrum-training-storage"
VOLUME_MOUNT_PATH = "./beam-storage"

PYTHON_PACKAGES = [
    "torch", "transformers", "datasets", "accelerate",
    "bitsandbytes", "peft", "trl", "huggingface_hub",
    "mlflow", "safetensors", "sentencepiece",
]
