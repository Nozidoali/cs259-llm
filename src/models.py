import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from config import MODEL_CONFIGS, MODELS_DIR
import os

logger = logging.getLogger(__name__)

def freeze_all_except_mlp(model):
    for param in model.parameters():
        param.requires_grad = False
    
    mlp_params_count = 0
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama, Qwen, etc.
        layers = model.model.layers
        for layer in layers:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
    elif hasattr(model, 'layers'):  # Some models have layers directly
        layers = model.layers
        for layer in layers:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # GPT-2
        layers = model.transformer.h
        for layer in layers:
            if hasattr(layer, 'mlp'):  # Some GPT-2 variants
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
            else:  # Standard GPT-2 uses c_fc and c_proj
                if hasattr(layer, 'c_fc'):
                    for param in layer.c_fc.parameters():
                        param.requires_grad = True
                        mlp_params_count += 1
                if hasattr(layer, 'c_proj'):
                    for param in layer.c_proj.parameters():
                        param.requires_grad = True
                        mlp_params_count += 1
    else:
        raise ValueError(f"Unsupported model architecture. Model type: {type(model).__name__}, attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    logger.info(f"Unfrozen {mlp_params_count} MLP parameters")
    
    return mlp_params_count

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    return {"total": total_params, "trainable": trainable_params, "frozen": frozen_params, "trainable_percentage": 100 * trainable_params / total_params}

def load_model_and_tokenizer(model_key_or_path=None, model_path=None, return_config_info=False, dtype=None, enable_gradient_checkpointing=False):
    if model_key_or_path is None and model_path is None:
        raise ValueError("Either model_key_or_path or model_path must be provided")
    if model_key_or_path and model_key_or_path in MODEL_CONFIGS:
        model_key = model_key_or_path
        config = MODEL_CONFIGS[model_key]
        from data import download_model
        model_path_resolved = download_model(model_key) if model_path is None else Path(model_path)
        model_id = str(model_path_resolved)
        model_type = config.get("model_type", "causal")
        output_dir = MODELS_DIR / config["finetuned_dir"]
        logger.info(f"Loading {config['display_name']} from: {model_id}")
    else:
        if model_path:
            model_id = str(model_path)
        elif model_key_or_path:
            model_id = str(model_key_or_path)
        else:
            raise ValueError("Either model_key_or_path or model_path must be provided")
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            logger.info(f"Loading model from local path: {model_id}")
        else:
            logger.info(f"Loading model from HuggingFace: {model_id}")
        model_type = "causal"
        output_dir = None

    model_path_obj = Path(model_id)
    is_local = model_path_obj.exists() and model_path_obj.is_dir()
    if is_local and model_path_obj.is_absolute():
        try:
            model_id_for_load = os.path.relpath(model_id)
        except ValueError:
            model_id_for_load = str(model_path_obj)
    else:
        model_id_for_load = model_id
    if is_local:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_for_load,
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dtype is not None:
        model_dtype = dtype
    elif torch.cuda.is_available():
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    if is_local:
        model_load_kwargs = {"local_files_only": True, "trust_remote_code": False}
    else:
        model_load_kwargs = {}
    
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id_for_load, dtype=model_dtype, **model_load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id_for_load, dtype=model_dtype, **model_load_kwargs)
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    if return_config_info:
        return model, tokenizer, output_dir, model_type
    else:
        return model, tokenizer

