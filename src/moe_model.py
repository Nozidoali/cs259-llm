#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Union, Literal
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_file_dir = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.dirname(_file_dir)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from model_utils import load_model_and_tokenizer
from gating.gating_model import GatingNetwork

logger = logging.getLogger(__name__)


class MoEFFN(nn.Module):
    
    def __init__(self, expert1_mlp, expert2_mlp, parent_model, routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum"):
        super().__init__()
        self.expert1_mlp = expert1_mlp
        self.expert2_mlp = expert2_mlp
        self.parent_model = parent_model
        self.routing_mode = routing_mode
    
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        
        if hasattr(self.parent_model, '_current_gating_probs') and self.parent_model._current_gating_probs is not None:
            if self.parent_model._current_gating_probs.dim() == 3:
                gating_probs = self.parent_model._current_gating_probs[:, 0, :].to(hidden_states.dtype)
            else:
                gating_probs = self.parent_model._current_gating_probs.to(hidden_states.dtype)
        else:
            logger.warning("_current_gating_probs not set, using equal probabilities")
            gating_probs = torch.ones(batch_size, 2, device=hidden_states.device, dtype=hidden_states.dtype) * 0.5
        
        expert1_out = self.expert1_mlp(hidden_states)
        expert2_out = self.expert2_mlp(hidden_states)
        
        if self.routing_mode == "weighted_sum":
            prob1 = gating_probs[:, 0].unsqueeze(-1).unsqueeze(-1)
            prob2 = gating_probs[:, 1].unsqueeze(-1).unsqueeze(-1)
            output = prob1 * expert1_out + prob2 * expert2_out
        else:
            selected_expert = gating_probs.argmax(dim=-1)
            selected_expert = selected_expert.unsqueeze(-1).unsqueeze(-1)
            output = torch.where(
                selected_expert == 0,
                expert1_out,
                expert2_out
            )
        
        return output


class MoEModel(nn.Module):
    
    def __init__(
        self,
        model1_path: Union[str, Path],
        model2_path: Union[str, Path],
        gating_model_path: Union[str, Path],
        routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device
        
        self.routing_mode = routing_mode
        
        self._model1_path = str(model1_path)
        self._model2_path = str(model2_path)
        self._gating_model_path = str(gating_model_path)
        
        logger.info("Loading expert model 1...")
        self.model1, self.tokenizer1 = load_model_and_tokenizer(model_path=model1_path)
        self.model1 = self.model1.to(self.device)
        self.model1.eval()
        
        logger.info("Loading expert model 2...")
        self.model2, self.tokenizer2 = load_model_and_tokenizer(model_path=model2_path)
        self.model2 = self.model2.to(self.device)
        self.model2.eval()
        
        self.base_model = self.model1
        self.tokenizer = self.tokenizer1
        
        if not self._verify_model_compatibility():
            raise ValueError("Models have incompatible architectures")
        
        logger.info("Loading gating network...")
        gating_model_path = Path(gating_model_path)
        
        training_info_path = gating_model_path / "training_info.json"
        if not training_info_path.exists():
            raise FileNotFoundError(f"training_info.json not found in {gating_model_path}")
        
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        
        self._gating_training_info = training_info
        
        embedding_dim = training_info["embedding_dim"]
        model_config = training_info.get("config", {})
        hidden_dims = model_config.get("hidden_dims", [512, 256])
        dropout = model_config.get("dropout", 0.1)
        
        model_dtype = next(self.model1.parameters()).dtype
        self.gating_network = GatingNetwork(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device).to(model_dtype)
        
        best_model_path = gating_model_path / "best_model.pt"
        final_model_path = gating_model_path / "final_model.pt"
        
        if best_model_path.exists():
            self.gating_network.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )
        elif final_model_path.exists():
            self.gating_network.load_state_dict(
                torch.load(final_model_path, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Neither best_model.pt nor final_model.pt found in {gating_model_path}")
        
        self.gating_network = self.gating_network.to(model_dtype)
        self.gating_network.eval()
        
        logger.info("Using model1's embedding layer for gating")
        self.embedding_model = self.model1
        self.embedding_tokenizer = self.tokenizer1
        
        self._replace_ffn_layers()
        
        logger.info(f"MoE model initialized with routing mode: {routing_mode}")
    
    def _verify_model_compatibility(self):
        if not hasattr(self.model1, 'model') or not hasattr(self.model1.model, 'layers'):
            return False
        if not hasattr(self.model2, 'model') or not hasattr(self.model2.model, 'layers'):
            return False
        
        layers1 = self.model1.model.layers
        layers2 = self.model2.model.layers
        
        if len(layers1) != len(layers2):
            logger.warning(f"Model layer count mismatch: {len(layers1)} vs {len(layers2)}")
            return False
        
        for i, (layer1, layer2) in enumerate(zip(layers1, layers2)):
            if not hasattr(layer1, 'mlp') or not hasattr(layer2, 'mlp'):
                logger.warning(f"Layer {i} missing mlp attribute")
                return False
        
        return True
    
    def _replace_ffn_layers(self):
        layers = self.base_model.model.layers
        model1_layers = self.model1.model.layers
        model2_layers = self.model2.model.layers
        
        for i, (base_layer, expert1_layer, expert2_layer) in enumerate(zip(layers, model1_layers, model2_layers)):
            moe_ffn = MoEFFN(
                expert1_mlp=expert1_layer.mlp,
                expert2_mlp=expert2_layer.mlp,
                parent_model=self,
                routing_mode=self.routing_mode,
            )
            base_layer.mlp = moe_ffn
        
        logger.info(f"Replaced {len(layers)} FFN layers with MoE FFN layers")
    
    def _get_gating_probs(self, input_ids, attention_mask):
        with torch.no_grad():
            if hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, 'embed_tokens'):
                embeddings = self.embedding_model.model.embed_tokens(input_ids)
            elif hasattr(self.embedding_model, 'transformer') and hasattr(self.embedding_model.transformer, 'wte'):
                embeddings = self.embedding_model.transformer.wte(input_ids)
            else:
                raise ValueError("Could not find embedding layer in model1")
            
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9).to(embeddings.dtype)
                mean_embeddings = sum_embeddings / sum_mask
            else:
                mean_embeddings = embeddings.mean(dim=1)
        
        model_dtype = next(self.model1.parameters()).dtype
        embedding_tensor = mean_embeddings.to(model_dtype)
        
        with torch.no_grad():
            gating_probs = self.gating_network(embedding_tensor)
        
        return gating_probs.to(model_dtype)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        gating_probs = self._get_gating_probs(input_ids, attention_mask)
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        model_dtype = next(self.base_model.parameters()).dtype
        self._current_gating_probs = gating_probs.to(model_dtype).unsqueeze(1).expand(batch_size, seq_len, 2)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            gating_probs = self._get_gating_probs(input_ids, attention_mask)
            
            batch_size = input_ids.shape[0]
            self._current_gating_probs = gating_probs.unsqueeze(1)
        else:
            logger.warning("No input_ids provided to generate, using default gating probabilities")
            batch_size = 1
            model_dtype = next(self.base_model.parameters()).dtype
            self._current_gating_probs = torch.ones(batch_size, 1, 2, device=self.device, dtype=model_dtype) * 0.5
        
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_gating_probs(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self._get_gating_probs(input_ids, attention_mask)
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving MoE model to {save_directory}")
        
        parent_refs = []
        layers = self.base_model.model.layers
        for layer in layers:
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, MoEFFN):
                parent_refs.append(layer.mlp.parent_model)
                layer.mlp.parent_model = None
        
        try:
            logger.info("Saving base model with MoE FFN layers...")
            self.base_model.save_pretrained(str(save_directory))
            self.tokenizer.save_pretrained(str(save_directory))
        finally:
            idx = 0
            for layer in layers:
                if hasattr(layer, 'mlp') and isinstance(layer.mlp, MoEFFN):
                    layer.mlp.parent_model = parent_refs[idx]
                    idx += 1
        
        gating_dir = save_directory / "gating_network"
        gating_dir.mkdir(exist_ok=True)
        logger.info("Saving gating network...")
        torch.save(self.gating_network.state_dict(), gating_dir / "gating_model.pt")
        
        moe_config = {
            "model1_path": str(getattr(self, '_model1_path', None)),
            "model2_path": str(getattr(self, '_model2_path', None)),
            "gating_model_path": str(getattr(self, '_gating_model_path', None)),
            "routing_mode": self.routing_mode,
        }
        
        with open(save_directory / "moe_config.json", "w") as f:
            json.dump(moe_config, f, indent=2)
        
        gating_config_path = save_directory / "gating_config.json"
        if hasattr(self, '_gating_training_info'):
            with open(gating_config_path, "w") as f:
                json.dump(self._gating_training_info, f, indent=2)
        
        logger.info(f"MoE model saved successfully to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        moe_config_path = model_path / "moe_config.json"
        if not moe_config_path.exists():
            raise FileNotFoundError(f"moe_config.json not found in {model_path}")
        
        with open(moe_config_path, "r") as f:
            moe_config = json.load(f)
        
        gating_config_path = model_path / "gating_config.json"
        gating_training_info = None
        if gating_config_path.exists():
            with open(gating_config_path, "r") as f:
                gating_training_info = json.load(f)
        
        if moe_config.get("model1_path") and moe_config.get("model2_path") and moe_config.get("gating_model_path"):
            logger.info("Reconstructing MoE model from original components...")
            routing_mode = moe_config.get("routing_mode", "weighted_sum")
            if routing_mode not in ["weighted_sum", "select_one"]:
                routing_mode = "weighted_sum"
            
            instance = cls(
                model1_path=moe_config["model1_path"],
                model2_path=moe_config["model2_path"],
                gating_model_path=moe_config["gating_model_path"],
                routing_mode=routing_mode,
                device=device,
            )
        else:
            logger.info("Loading MoE model from saved checkpoint...")
            if device is None:
                device = torch.device(
                    "cuda" if torch.cuda.is_available() 
                    else "mps" if torch.backends.mps.is_available() 
                    else "cpu"
                )
            
            base_model, tokenizer = load_model_and_tokenizer(model_path=model_path)
            base_model = base_model.to(device)
            base_model.eval()
            
            instance = cls.__new__(cls)
            instance.device = device
            instance.routing_mode = moe_config.get("routing_mode", "weighted_sum")
            instance.base_model = base_model
            instance.tokenizer = tokenizer
            
            if gating_training_info:
                embedding_dim = gating_training_info["embedding_dim"]
                model_config = gating_training_info.get("config", {})
                hidden_dims = model_config.get("hidden_dims", [512, 256])
                dropout = model_config.get("dropout", 0.1)
                
                model_dtype = next(base_model.parameters()).dtype
                instance.gating_network = GatingNetwork(
                    input_dim=embedding_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                ).to(device).to(model_dtype)
                
                gating_model_path = model_path / "gating_network" / "gating_model.pt"
                if gating_model_path.exists():
                    instance.gating_network.load_state_dict(
                        torch.load(gating_model_path, map_location=device, weights_only=True)
                    )
                    instance.gating_network = instance.gating_network.to(model_dtype)
                    instance.gating_network.eval()
                else:
                    raise FileNotFoundError(f"Gating model not found: {gating_model_path}")
                
                logger.info("Using model1's embedding layer for gating")
                instance.embedding_model = instance.base_model
                instance.embedding_tokenizer = instance.tokenizer
            else:
                raise ValueError("Cannot load model without gating training info")
        
        return instance


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and save MoE model")
    parser.add_argument("--model1_path", type=str, required=True,
                       help="Path to first finetuned model")
    parser.add_argument("--model2_path", type=str, required=True,
                       help="Path to second finetuned model")
    parser.add_argument("--gating_model_path", type=str, required=True,
                       help="Path to gating network")
    parser.add_argument("--routing_mode", type=str, default="weighted_sum",
                       choices=["weighted_sum", "select_one"],
                       help="Routing mode: weighted_sum or select_one")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory to save the merged MoE model")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/mps/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        if args.device == "cuda":
            device = torch.device("cuda")
        elif args.device == "mps":
            device = torch.device("mps")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            logger.warning(f"Unknown device: {args.device}, using auto-detect")
    
    logger.info("=" * 60)
    logger.info("Creating MoE Model")
    logger.info("=" * 60)
    logger.info(f"Model 1: {args.model1_path}")
    logger.info(f"Model 2: {args.model2_path}")
    logger.info(f"Gating Network: {args.gating_model_path}")
    logger.info(f"Routing Mode: {args.routing_mode}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 60)
    
    moe_model = MoEModel(
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        gating_model_path=args.gating_model_path,
        routing_mode=args.routing_mode,
        device=device,
    )
    
    logger.info(f"Saving MoE model to {args.output_dir}...")
    moe_model.save_pretrained(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("✓ MoE model saved successfully!")
    logger.info(f"  Saved to: {args.output_dir}")
    logger.info("=" * 60)
    logger.info("\nTo load the model later, use:")
    logger.info(f"  from moe_model import MoEModel")
    logger.info(f"  model = MoEModel.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()

