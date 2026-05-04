"""Small Hydra helpers used by training and tokenizer/model setup."""

import hydra
import torch
import trl
import transformers
from omegaconf import DictConfig
from transformers import AutoConfig


def fix_pad_token(tokenizer, model_name=None, force_override_pad_token=False):
    if model_name is None:
        model_name = tokenizer.name_or_path
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", None)
    if tokenizer.pad_token is None or force_override_pad_token:
        if model_type == "llama":
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        elif (
            "qwen2" in model_type
            or "qwen3" in model_type
            or "smollm3" in model_type
        ):
            tokenizer.pad_token = "<|fim_pad|>"
        elif model_type == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
            print(
                "WARNING: Setting pad_token to eos_token for gpt2, this will"
                " cause issues for SFT."
            )
        else:
            raise NotImplementedError
    else:
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, "Issue!"
    return tokenizer


def load_model(
    model_args,
    config=None,
    from_pretrained=False,
    custom_class=None,
):
    if isinstance(model_args, DictConfig):
        model_args = hydra.utils.instantiate(model_args)
    if isinstance(config, DictConfig):
        config = hydra.utils.instantiate(config)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    attn_implementation = model_args.attn_implementation

    if from_pretrained:
        assert model_args.model_name_or_path is not None, (
            "Model name or path must be provided for loading a pretrained model."
        )
        print(f"Loading model from {model_args.model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        # SparseMLP.__init__ transposes down_proj.weight to [intermediate, hidden]
        # at construction time when sparsity_use_hybrid_kernel is set. from_pretrained
        # then loads the HF-layout [hidden, intermediate] checkpoint on top of that,
        # clobbering the transpose. Re-apply it once weights are loaded.
        if getattr(model.config, "sparsity_use_hybrid_kernel", False):
            for layer in model.model.layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None and hasattr(mlp, "down_proj"):
                    mlp.down_proj.weight.data = (
                        mlp.down_proj.weight.data.t().contiguous()
                    )
        return model

    if config is None:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)

    if custom_class is not None:
        config._attn_implementation = attn_implementation
        if isinstance(custom_class, DictConfig):
            model = hydra.utils.instantiate(custom_class, config)
        else:
            model = custom_class(config=config)
    else:
        model = transformers.AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=model_args.trust_remote_code,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    dtype = getattr(model, "dtype", None)
    print(
        f"Training new model from scratch. Total size={n_params / 1e6:.2f}M "
        f"params, in {dtype}"
    )
    return model
