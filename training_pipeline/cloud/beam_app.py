"""Run the existing Avrum training pipeline on a Beam GPU."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beam import Image, Volume, function

from cloud.beam_config import (
    CPU, GPU, HEADLESS, MEMORY, PYTHON_PACKAGES,
    TIMEOUT_SECONDS, VOLUME_MOUNT_PATH, VOLUME_NAME,
)

training_volume = Volume(
    name=VOLUME_NAME,
    mount_path=VOLUME_MOUNT_PATH,
)

training_image = (
    Image(python_version="python3.11")
    .add_python_packages(PYTHON_PACKAGES)
)


@function(
    name="avrum-qlora-training",
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY,
    image=training_image,
    volumes=[training_volume],
    secrets=["HF_TOKEN"],
    timeout=TIMEOUT_SECONDS,
    headless=HEADLESS,
)
def run_avrum_training():
    """Execute src.train.train() inside Beam's remote GPU container."""
    import os
    import torch

    volume_path = Path(VOLUME_MOUNT_PATH).resolve()
    output_path = volume_path / "outputs"
    hf_cache_path = volume_path / "hf-cache"

    output_path.mkdir(parents=True, exist_ok=True)
    hf_cache_path.mkdir(parents=True, exist_ok=True)

    # Set paths before importing the training modules and Hugging Face libraries.
    os.environ["AVRUM_OUTPUT_DIR"] = str(output_path)
    os.environ["HF_HOME"] = str(hf_cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_path / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_path / "transformers")

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN was not injected. Create the Beam secret named HF_TOKEN."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Beam did not provide a CUDA-enabled GPU to this job.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Persistent output directory: {output_path}")
    print(f"Hugging Face cache: {hf_cache_path}")

    # Import after setting environment variables.
    from src.train import train

    train()
    print("Avrum training completed successfully.")
    return "Training completed. Artifacts are saved on the Beam volume."


if __name__ == "__main__":
    run_avrum_training.remote()
