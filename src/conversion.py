import json
import logging
import shutil
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import importlib
from config import LLAMA_CPP_DIR

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
    sys.path.append(str(LLAMA_CPP_DIR))
    convert_module = importlib.import_module("convert_hf_to_gguf")
    sys.argv = ["convert_hf_to_gguf.py", str(model_path), "--outfile", str(output_file), "--outtype", quantize_level]
    convert_module.main()
    logger.info(f"Complete: {output_file}")

def merge_experts_to_standard_mlp(model_path: Path, output_path: Path, merge_mode: str = "average"):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).to("cpu")
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    state_dict = {}
    for safetensors_file in set(weight_map.values()):
        with safe_open(str(model_path / safetensors_file), framework="pt", device="cpu") as f:
            state_dict.update({key: f.get_tensor(key) for key in f.keys()})
    expert_indices = set()
    for key in state_dict.keys():
        if ".mlp.expert" in key and ".weight" in key:
            parts = key.split(".mlp.expert")
            if len(parts) > 1:
                rest = parts[1].split("_mlp.")[0]
                try:
                    expert_idx = int(rest)
                    expert_indices.add(expert_idx)
                except ValueError:
                    pass
    num_experts = len(expert_indices) if expert_indices else 0
    logger.info(f"Found {num_experts} experts in model")
    new_state_dict = {k: v for k, v in state_dict.items() if ".mlp.expert" not in k}
    if num_experts > 0:
        for layer_idx in range(config.num_hidden_layers):
            for proj_type in ["gate_proj", "up_proj", "down_proj"]:
                standard_key = f"model.layers.{layer_idx}.mlp.{proj_type}.weight"
                expert_weights = []
                for expert_idx in sorted(expert_indices):
                    expert_key = f"model.layers.{layer_idx}.mlp.expert{expert_idx}_mlp.{proj_type}.weight"
                    if expert_key in state_dict:
                        expert_weights.append(state_dict[expert_key])
                if expert_weights:
                    if merge_mode == "average":
                        merged_weight = sum(expert_weights) / len(expert_weights)
                    else:
                        merged_weight = expert_weights[0].clone()
                    new_state_dict[standard_key] = merged_weight
    model.load_state_dict(new_state_dict, strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True).save_pretrained(str(output_path))
    for fname in ["moe_config.json", "gating_config.json", "generation_config.json", "chat_template.jinja"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
    return output_path

