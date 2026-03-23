from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig


def get_special_token_values(tokenizer=None, model_name=None):
    if model_name is None:
        assert tokenizer is not None, (
            "Provide either tokenizer or model_name"
        )
        model_name = tokenizer.name_or_path
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, 'model_type', None)

    if model_type == "llama" and "smollm" in model_name.lower():
        model_type = "smollm"

    if model_type == 'llama':
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = (
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif ('qwen2' in model_type or 'qwen3' in model_type or
          'smollm' in model_type):
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
    else:
        raise NotImplementedError

    return instruction_template, response_template


def get_custom_start_of_response(tokenizer=None, model_name=None):
    instruction_template, response_template = (
        get_special_token_values(tokenizer=tokenizer, model_name=model_name))
    return response_template


def override_system_prompt(tokenizer_or_tokenizer_name, new_system_prompt):
    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
        tokenizer.apply_chat_template
    elif isinstance(tokenizer_or_tokenizer_name, PreTrainedTokenizerBase):
        tokenizer = tokenizer_or_tokenizer_name
    else:
        raise NotImplementedError

    if hasattr(tokenizer, "default_system_prompt"):
        tokenizer.default_system_prompt = new_system_prompt
    elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{system_prompt}", new_system_prompt
        )
    else:
        print("This tokenizer does not support system prompts.")
        raise NotImplementedError

    return tokenizer
