
import torch

from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from configs.training_config import QLoRAConfig


class QLoRAModelLoader:
    """
    Load a base language model, tokenizer,
    and attach LoRA adapters.
    """

    def __init__(self, config: QLoRAConfig) -> None:
        self.config = config

    def _build_quantization_config(self) -> Any:
        """
        Build the 4-bit NF4 quantization configuration.
        """

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

    def _resolve_torch_dtype(self) -> Any:
        """
        Resolve the configured PyTorch data type.
        """

        return getattr(torch, self.config.torch_dtype, None)

    def load_tokenizer(self) -> Any:
        """
        Load the tokenizer for the selected base model.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(self) -> Any:
        """
        Load the base model with 4-bit quantization
        and attach LoRA adapters.
        """

        quantization_config = (
            self.config.quantization_config
            or self._build_quantization_config()
        )

        torch_dtype = self._resolve_torch_dtype()

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
        )

        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        target_modules = self.config.target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias=self.config.bias,
            task_type=self.config.task_type,
        )

        model = get_peft_model(model, lora_config)

        return model

    def load(self) -> tuple[Any, Any]:
        """
        Load both the tokenizer and QLoRA-enabled model.
        """

        tokenizer = self.load_tokenizer()
        model = self.load_model()

        return model, tokenizer
