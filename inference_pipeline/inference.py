import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"


def model_fn(model_dir):

    print("===================================")
    print("Loading Qwen Agriculture Model")
    print("===================================")

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,

        torch_dtype=torch.float16,

        device_map="auto",

        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
    )

    model.eval()

    print("Model loaded successfully.")

    return {
        "model": model,
        "tokenizer": tokenizer,
    }


def predict_fn(data, model_data):

    model = model_data["model"]

    tokenizer = model_data["tokenizer"]

    question = data.get(
        "question",
        "",
    )

    if not question:

        return {
            "error": "Question is required."
        }

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,

        tokenize=False,

        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,

        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=256,

            temperature=0.7,

            do_sample=True,

            top_p=0.9,
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,

        skip_special_tokens=True,
    )

    return {
        "answer": answer.strip(),
    }