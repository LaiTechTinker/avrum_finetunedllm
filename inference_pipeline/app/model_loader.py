
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Project directories
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent.parent

MODEL_DIR = Path(
    os.getenv(
        "AVRUM_MODEL_DIR",
        str(PROJECT_DIR / "qwen_agriculture_lora")
    )
)
ADAPTER_PATH = MODEL_DIR / "adapter"
TOKENIZER_PATH = MODEL_DIR / "tokenizer"

BASE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

_model = None
_tokenizer = None


def load_model():
    """
    Load the original Qwen model and attach
    the trained Avrum agricultural LoRA adapter.
    """

    global _model, _tokenizer

    # Prevent loading the model multiple times.
    if _model is not None:
        return _model, _tokenizer

    print("Starting Avrum model loading...")

    # Check artifact directories.
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {TOKENIZER_PATH}"
        )

    # Check GPU availability.
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        allow_cpu = os.getenv(
            "ALLOW_CPU_INFERENCE", "false"
        ).lower() == "true"

        if not allow_cpu:
            raise RuntimeError(
                "CUDA GPU unavailable. "
                "Deploy this service on a GPU machine "
                "or explicitly enable CPU inference."
            )

        dtype = torch.float32
        device_map = None

    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        device_map = "auto"

        print("GPU detected:", torch.cuda.get_device_name(0))

    print("Loading tokenizer...")

    _tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        local_files_only=True
    )

    # Ensure padding is configured.
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    print("Attaching Avrum LoRA adapter...")

    _model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False
    )

    _model.eval()

    print("Avrum model loaded successfully!")

    return _model, _tokenizer
