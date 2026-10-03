from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    # Load the tokenizer and causal language model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Format a simple English prompt with the model's chat template.
    messages = [
        {
            "role": "user",
            "content": "Explain in one sentence what large language model inference is.",
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate at most 32 tokens beyond the prompt.
    outputs = model.generate(**inputs, max_new_tokens=32)

    # Decode only the newly generated tokens and print the response.
    prompt_length = inputs["input_ids"].shape[-1]
    generated_text = tokenizer.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
