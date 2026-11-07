
import modal
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

APP_DIR = BASE_DIR / "app"
MODEL_DIR = PROJECT_DIR / "qwen_agriculture_lora"

# Modal application
app = modal.App("avrum-inference")

# Persistent storage for downloaded Hugging Face model files
model_cache = modal.Volume.from_name(
    "avrum-model-cache",
    create_if_missing=True,
)

# Cloud Python environment

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "fastapi",
        "uvicorn[standard]",
    )
    .env({
        "AVRUM_MODEL_DIR": "/root/qwen_agriculture_lora",
        "HF_HOME": "/root/.cache/huggingface",
    })
    # Keep all local file additions at the end
    .add_local_dir(
        str(APP_DIR),
        remote_path="/root/app",
    )
    .add_local_dir(
        str(MODEL_DIR),
        remote_path="/root/qwen_agriculture_lora",
    )
)



@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=1800,
    startup_timeout=1800,
    scaledown_window=60,
    volumes={
        "/root/.cache/huggingface": model_cache,
    },
    secrets=[
        modal.Secret.from_name("avrum-inference-secret")
    ],
)
@modal.asgi_app()
def serve():
    # Import inside the cloud container
    from app.main import app as fastapi_app
    from app.model_loader import load_model

    print("Starting Avrum cloud inference service...")

    # Load model once when the container starts
    load_model()

    # Persist downloaded Hugging Face model files
    model_cache.commit()

    print("Avrum is ready to receive requests!")

    return fastapi_app
