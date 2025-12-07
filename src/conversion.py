import json
import logging
import shutil
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def convert_to_gguf(model_path: Path, output_file: Path, quantize_level: str = "f16") -> None:
    model_path = Path(model_path).resolve()
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    quantize_map = {"Q4_0": "f16", "Q8_0": "q8_0"}
    quantize_level = quantize_map.get(quantize_level, quantize_level)
    logger.info(f"Converting: {model_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Quantization: {quantize_level}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    from config import LLAMA_CPP_DIR
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}.\n"
            f"Please ensure llama.cpp submodule is initialized:\n"
            f"  git submodule update --init --recursive"
        )
    
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["convert_hf_to_gguf.py", str(model_path), "--outfile", str(output_file), "--outtype", quantize_level]
        sys.path.insert(0, str(LLAMA_CPP_DIR))
        try:
            import convert_hf_to_gguf
            convert_hf_to_gguf.main()
        finally:
            sys.path.remove(str(LLAMA_CPP_DIR))
        logger.info(f"Complete: {output_file}")
    finally:
        sys.argv = original_argv

def prepare_moe_for_gguf(model_path: Path, output_path: Path):
    """
    Prepare sparse MoE model for GGUF conversion without merging experts.
    Keeps expert weights side-by-side and includes router weights.
    """
    logger.info("Preparing sparse MoE model for GGUF conversion (no merging)")
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    
    # Load MoE config
    moe_config_path = model_path / "moe_config.json"
    if moe_config_path.exists():
        with open(moe_config_path) as f:
            moe_config = json.load(f)
        logger.info(f"MoE config: {moe_config}")
    else:
        logger.warning("No moe_config.json found, assuming standard model")
        moe_config = {}
    
    state_dict = {}
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        logger.info("Loading from sharded safetensors format")
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        for safetensors_file in set(weight_map.values()):
            with safe_open(str(model_path / safetensors_file), framework="pt", device="cpu") as f:
                state_dict.update({key: f.get_tensor(key) for key in f.keys()})
    elif (model_path / "model.safetensors").exists():
        logger.info("Loading from single safetensors file")
        with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as f:
            keys = f.keys()
            state_dict.update({key: f.get_tensor(key) for key in keys})
            logger.info(f"Loaded {len(keys)} tensors from model.safetensors")
    elif (model_path / "pytorch_model.bin.index.json").exists():
        logger.info("Loading from sharded pytorch format")
        with open(model_path / "pytorch_model.bin.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        for bin_file in set(weight_map.values()):
            state_dict.update(torch.load(str(model_path / bin_file), map_location="cpu", weights_only=True))
    elif (model_path / "pytorch_model.bin").exists():
        logger.info("Loading from single pytorch file")
        state_dict = torch.load(str(model_path / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}. Expected one of: model.safetensors[.index.json], pytorch_model.bin[.index.json]")
    
    logger.info(f"Total state dict keys: {len(state_dict)}")
    
    # Count experts in the new format (experts.0., experts.1., etc.)
    expert_keys = [k for k in state_dict.keys() if ".mlp.experts." in k]
    logger.info(f"Found {len(expert_keys)} expert keys with new naming convention")
    if expert_keys:
        logger.info(f"Sample expert keys (first 5): {expert_keys[:5]}")
    
    # Load router weights if available
    router_path = model_path / "router.pt"
    if router_path.exists():
        logger.info("Loading router weights")
        router_state = torch.load(router_path, map_location="cpu", weights_only=True)
        logger.info(f"Router state keys: {list(router_state.keys())}")
    else:
        logger.warning("No router.pt found")
    
    # Save the model with MoE structure intact
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save state dict directly to preserve expert structure
    # Don't load into a model as that would drop the expert weights
    from safetensors.torch import save_file
    save_file(state_dict, str(output_path / "model.safetensors"))
    
    # Copy config and tokenizer
    import shutil
    shutil.copy2(model_path / "config.json", output_path / "config.json")
    AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True).save_pretrained(str(output_path))
    
    # Copy MoE and router config files
    for fname in ["moe_config.json", "router.pt", "gating_config.json", "generation_config.json", "chat_template.jinja"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            logger.info(f"Copied {fname}")
    
    logger.info(f"MoE model prepared for GGUF (experts kept separate): {output_path}")
    return output_path

def merge_experts_to_standard_mlp(model_path: Path, output_path: Path, merge_mode: str = "average"):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).to("cpu")
    
    state_dict = {}
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        logger.info("Loading from sharded safetensors format")
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        for safetensors_file in set(weight_map.values()):
            with safe_open(str(model_path / safetensors_file), framework="pt", device="cpu") as f:
                state_dict.update({key: f.get_tensor(key) for key in f.keys()})
    elif (model_path / "model.safetensors").exists():
        logger.info("Loading from single safetensors file")
        with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as f:
            keys = f.keys()
            state_dict.update({key: f.get_tensor(key) for key in keys})
            logger.info(f"Loaded {len(keys)} tensors from model.safetensors")
    elif (model_path / "pytorch_model.bin.index.json").exists():
        logger.info("Loading from sharded pytorch format")
        with open(model_path / "pytorch_model.bin.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        for bin_file in set(weight_map.values()):
            state_dict.update(torch.load(str(model_path / bin_file), map_location="cpu", weights_only=True))
    elif (model_path / "pytorch_model.bin").exists():
        logger.info("Loading from single pytorch file")
        state_dict = torch.load(str(model_path / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}. Expected one of: model.safetensors[.index.json], pytorch_model.bin[.index.json]")
    
    logger.info(f"Total state dict keys: {len(state_dict)}")
    expert_indices = set()
    expert_keys = []
    for key in state_dict.keys():
        if ".mlp.expert" in key:
            expert_keys.append(key)
            if ".weight" in key:
                parts = key.split(".mlp.expert")
                if len(parts) > 1:
                    rest = parts[1].split("_mlp.")[0]
                    try:
                        expert_idx = int(rest)
                        expert_indices.add(expert_idx)
                    except ValueError:
                        pass
    
    num_experts = len(expert_indices) if expert_indices else 0
    logger.info(f"Found {num_experts} experts with indices: {sorted(expert_indices)}")
    logger.info(f"Total expert keys: {len(expert_keys)}")
    if expert_keys:
        logger.info(f"Sample expert keys (first 5): {expert_keys[:5]}")
    new_state_dict = {k: v for k, v in state_dict.items() if ".mlp.expert" not in k}
    logger.info(f"Base state dict (non-expert) keys: {len(new_state_dict)}")
    
    if num_experts > 0:
        logger.info(f"Merging {num_experts} experts using mode: {merge_mode}")
        merged_layers = 0
        for layer_idx in range(config.num_hidden_layers):
            layer_merged = False
            for proj_type in ["gate_proj", "up_proj", "down_proj"]:
                standard_key = f"model.layers.{layer_idx}.mlp.{proj_type}.weight"
                expert_weights = []
                for expert_idx in sorted(expert_indices):
                    expert_key = f"model.layers.{layer_idx}.mlp.expert{expert_idx}_mlp.{proj_type}.weight"
                    if expert_key in state_dict:
                        expert_weights.append(state_dict[expert_key])
                        if layer_idx == 0 and not layer_merged:
                            logger.info(f"Layer {layer_idx}: Found expert {expert_idx} for {proj_type}, shape: {state_dict[expert_key].shape}")
                if expert_weights:
                    if merge_mode == "average":
                        merged_weight = sum(expert_weights) / len(expert_weights)
                    else:
                        merged_weight = expert_weights[0].clone()
                    new_state_dict[standard_key] = merged_weight
                    layer_merged = True
                    if layer_idx == 0:
                        logger.info(f"Layer {layer_idx}: Merged {len(expert_weights)} experts for {proj_type}, output shape: {merged_weight.shape}")
            if layer_merged:
                merged_layers += 1
        logger.info(f"Successfully merged experts in {merged_layers}/{config.num_hidden_layers} layers")
    else:
        logger.warning("No experts found in model - skipping expert merging")
    model.load_state_dict(new_state_dict, strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True).save_pretrained(str(output_path))
    for fname in ["moe_config.json", "gating_config.json", "generation_config.json", "chat_template.jinja"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
    return output_path
