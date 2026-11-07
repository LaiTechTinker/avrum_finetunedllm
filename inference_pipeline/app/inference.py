
import torch

from app.model_loader import load_model


SYSTEM_PROMPT = """
You are Avrum, an agricultural AI assistant.

Your role is to help farmers and agricultural
professionals understand agricultural problems.

Provide:
- Clear and practical explanations.
- Evidence-based agricultural recommendations.
- Relevant crop management advice.
- Simple language that farmers can understand.

If you do not have enough information to identify
an agricultural problem, ask relevant follow-up questions.

Never present uncertain diagnoses as confirmed facts.
"""


def generate_response(
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> str:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    model, tokenizer = load_model()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Use Qwen's chat format.
    chat_template = tokenizer.chat_template

    if not chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' "
            "+ message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = inputs.to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove the original prompt from generated tokens.
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()
