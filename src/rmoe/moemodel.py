import os
import sys
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Union, Literal, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_file_dir = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.dirname(_file_dir)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from models import load_model_and_tokenizer
from gating.gatingmodel import GatingNetwork, SharedExpertGating
from gating.gatingdataset import load_base_model_for_embeddings

logger = logging.getLogger(__name__)

class MoEFFN(nn.Module):
    def __init__(self, expert_mlps: List, parent_model, routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum"):
        super().__init__()
        self.expert_mlps = nn.ModuleList(expert_mlps)
        object.__setattr__(self, 'parent_model', parent_model)
        self.routing_mode = routing_mode
        self.num_experts = len(expert_mlps)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = {}
            destination._metadata = {}
        
        destination._metadata[prefix[:-1]] = {}
        
        for name, module in self.named_children():
            if name == 'expert_mlps':  # Only include expert_mlps
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        
        # Include parameters and buffers
        for name, param in self.named_parameters(recurse=False):
            destination[prefix + name] = param if keep_vars else param.detach()
        for name, buffer in self.named_buffers(recurse=False):
            destination[prefix + name] = buffer if keep_vars else buffer.detach()
        
        return destination
    
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        if hasattr(self.parent_model, '_current_gating_probs') and self.parent_model._current_gating_probs is not None:
            if self.parent_model._current_gating_probs.dim() == 3:
                gating_probs = self.parent_model._current_gating_probs[:, 0, :].to(hidden_states.dtype)
            else:
                gating_probs = self.parent_model._current_gating_probs.to(hidden_states.dtype)
        else:
            logger.warning("_current_gating_probs not set, using equal probabilities")
            gating_probs = torch.ones(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype) / self.num_experts
        expert_outputs = [expert_mlp(hidden_states) for expert_mlp in self.expert_mlps]
        if self.routing_mode == "weighted_sum":
            output = torch.zeros_like(expert_outputs[0])
            for idx, expert_out in enumerate(expert_outputs):
                prob = gating_probs[:, idx].unsqueeze(-1).unsqueeze(-1)
                output += prob * expert_out
            return output
        else:
            selected_expert = gating_probs.argmax(dim=-1)
            output = torch.zeros_like(expert_outputs[0])
            for idx, expert_out in enumerate(expert_outputs):
                mask = (selected_expert == idx).unsqueeze(-1).unsqueeze(-1)
                output += mask.float() * expert_out
            return output

class MoEModel(nn.Module):
    def __init__(self, expert_paths: List[Union[str, Path]], gating_model_path: Union[str, Path], base_model_path: Optional[Union[str, Path]] = None, routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum", device=None, target_architecture: Optional[Literal["qwen2moe", "qwen3moe", "auto"]] = "auto", use_shared_expert: bool = False, shared_expert_path: Optional[Union[str, Path]] = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.routing_mode = routing_mode
        self.target_architecture = target_architecture
        self.use_shared_expert = use_shared_expert
        expert_paths = [Path(p) for p in expert_paths]
        gating_model_path = Path(gating_model_path)
        if base_model_path is None:
            base_model_path = expert_paths[0]
        else:
            base_model_path = Path(base_model_path)
        logger.info(f"Loading base model from: {base_model_path}")
        base_model, base_tokenizer = load_model_and_tokenizer(model_path=str(base_model_path), dtype=torch.float32)
        self.config = base_model.config
        self.tokenizer = base_tokenizer
        base_model = base_model.to(self.device)
        logger.info(f"Loading {len(expert_paths)} expert models...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        expert_mlps_by_layer = {}  # {layer_idx: [mlp0, mlp1, mlp2, ...]}
        
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            num_layers = len(base_model.model.layers)
        elif hasattr(base_model, 'layers'):
            num_layers = len(base_model.layers)
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            num_layers = len(base_model.transformer.h)
        else:
            raise ValueError(f"Unsupported model architecture")
        
        for layer_idx in range(num_layers):
            expert_mlps_by_layer[layer_idx] = []
        
        for idx, expert_path in enumerate(expert_paths):
            logger.info(f"Loading expert {idx} from: {expert_path}")
            expert_model, _ = load_model_and_tokenizer(model_path=str(expert_path), dtype=torch.float32)
            
            if hasattr(expert_model, 'model') and hasattr(expert_model.model, 'layers'):
                expert_layers = expert_model.model.layers
            elif hasattr(expert_model, 'layers'):
                expert_layers = expert_model.layers
            elif hasattr(expert_model, 'transformer') and hasattr(expert_model.transformer, 'h'):
                expert_layers = expert_model.transformer.h
            else:
                raise ValueError(f"Unsupported expert model architecture")
            
            for layer_idx in range(num_layers):
                expert_mlp = expert_layers[layer_idx].mlp
                # Create a copy of the MLP and move to device
                expert_mlp_copy = expert_mlp.to(self.device)
                expert_mlps_by_layer[layer_idx].append(expert_mlp_copy)
            
            del expert_model, expert_layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"Extracted MLP layers from {len(expert_paths)} experts")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Loading gating network from: {gating_model_path}")
        base_model_for_emb, _, embedding_dim = load_base_model_for_embeddings(str(base_model_path))
        base_model_for_emb = base_model_for_emb.to(self.device)
        base_model_for_emb.eval()
        
        num_classes = len(expert_paths)
        
        gating_checkpoint_path = gating_model_path / "gating_network.pt"
        if not gating_checkpoint_path.exists():
            gating_checkpoint_path = gating_model_path / "expert_router_best.pt"
        if not gating_checkpoint_path.exists():
            gating_checkpoint_path = gating_model_path / "best_model.pt"
        if not gating_checkpoint_path.exists():
            raise FileNotFoundError(f"Expert router gating checkpoint not found at {gating_model_path}")
        
        gating_network = GatingNetwork(
            input_dim=embedding_dim,
            hidden_dims=[512, 256],
            dropout=0.1,
            num_classes=num_classes,
        )
        
        state_dict = torch.load(gating_checkpoint_path, map_location=self.device, weights_only=True)
        gating_network.load_state_dict(state_dict)
        gating_network = gating_network.to(self.device)
        
        self.gating_network = gating_network
        self.gating_network.eval()
        self.embedding_model = base_model_for_emb
        self.embedding_model.eval()
        self._current_gating_probs = None
        
        self.shared_expert_model = None
        self.shared_expert_gating = None
        if use_shared_expert:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if shared_expert_path is None:
                shared_expert_path = base_model_path
            shared_expert_path = Path(shared_expert_path)
            logger.info(f"Loading shared expert from: {shared_expert_path}")
            shared_expert_model, _ = load_model_and_tokenizer(model_path=str(shared_expert_path), dtype=torch.float32)
            self.shared_expert_model = shared_expert_model.to(self.device)
            self.shared_expert_model.eval()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            shared_gating_checkpoint = gating_model_path / "shared_expert_gating.pt"
            if not shared_gating_checkpoint.exists():
                raise FileNotFoundError(f"Shared expert gating checkpoint not found at {shared_gating_checkpoint}")
            
            logger.info(f"Loading shared expert gating network from: {shared_gating_checkpoint}")
            self.shared_expert_gating = SharedExpertGating(input_dim=embedding_dim)
            shared_gating_state = torch.load(shared_gating_checkpoint, map_location=self.device, weights_only=True)
            self.shared_expert_gating.load_state_dict(shared_gating_state)
            self.shared_expert_gating = self.shared_expert_gating.to(self.device)
            self.shared_expert_gating.eval()
            logger.info("Shared expert and gating network loaded successfully")
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        elif hasattr(base_model, 'layers'):
            layers = base_model.layers
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
        else:
            raise ValueError(f"Unsupported model architecture")
        num_layers = len(layers)
        logger.info(f"Found {num_layers} layers in base model")
        
        if self.target_architecture == "auto":
            if use_shared_expert:
                self.target_architecture = "qwen2moe"
                logger.info(f"Auto-detected architecture: qwen2moe (with shared expert)")
            else:
                self.target_architecture = "qwen3moe"
                logger.info(f"Auto-detected architecture: qwen3moe (no shared expert requirement)")
        else:
            logger.info(f"Using configured architecture: {self.target_architecture}")
        
        expert_mlps_list = []
        for layer_idx in range(num_layers):
            layer = layers[layer_idx]
            if hasattr(layer, 'mlp'):
                expert_mlps = expert_mlps_by_layer[layer_idx]
                moe_ffn = MoEFFN(expert_mlps, self, routing_mode=routing_mode)
                layer.mlp = moe_ffn
                expert_mlps_list.append(expert_mlps)
        self.model = base_model
        if hasattr(base_model, 'model'):
            self.model = base_model.model
        self.lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else None
        if self.lm_head is None and hasattr(base_model, 'model') and hasattr(base_model.model, 'lm_head'):
            self.lm_head = base_model.model.lm_head
        logger.info(f"MoE model initialized with {len(expert_paths)} experts, routing_mode={routing_mode}")
    
    def _get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            if hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, 'embed_tokens'):
                embeddings = self.embedding_model.model.embed_tokens(input_ids)
            elif hasattr(self.embedding_model, 'transformer') and hasattr(self.embedding_model.transformer, 'wte'):
                embeddings = self.embedding_model.transformer.wte(input_ids)
            else:
                raise ValueError("Could not find embedding layer in embedding model")
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9).to(embeddings.dtype)
            mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.model.embed_tokens(input_ids) if hasattr(self.model, 'embed_tokens') else self.model.embeddings(input_ids)
        hidden_states = embeddings
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)
        elif hasattr(self.model, 'layer_norm'):
            hidden_states = self.model.layer_norm(hidden_states)
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            gating_input = self._get_embeddings(input_ids, attention_mask)
            gating_probs = self.gating_network(gating_input)
        self._current_gating_probs = gating_probs
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)[0] if isinstance(layer(hidden_states, attention_mask=attention_mask, **kwargs), tuple) else layer(hidden_states, attention_mask=attention_mask, **kwargs)
        else:
            for layer in self.model.h:
                hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)[0] if isinstance(layer(hidden_states, attention_mask=attention_mask, **kwargs), tuple) else layer(hidden_states, attention_mask=attention_mask, **kwargs)
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)
        elif hasattr(self.model, 'layer_norm'):
            hidden_states = self.model.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states) if self.lm_head is not None else None
        if logits is None:
            raise ValueError("lm_head not found")
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        self.eval()
        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            gating_input = self._get_embeddings(input_ids, attention_mask)
            gating_probs = self.gating_network(gating_input)
            self._current_gating_probs = gating_probs
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self.forward(generated, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=-1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        return generated
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving MoE model to: {save_directory}")
        
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'h'):
            layers = self.model.h
        else:
            layers = []
        
        num_experts = 0
        moe_intermediate_size = None
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            if hasattr(self.model.layers[0].mlp, 'expert_mlps'):
                num_experts = len(self.model.layers[0].mlp.expert_mlps)
                first_expert = self.model.layers[0].mlp.expert_mlps[0]
                if hasattr(first_expert, 'gate_proj'):
                    moe_intermediate_size = first_expert.gate_proj.out_features
        
        state_dict = {}
        
        if hasattr(self.model, 'embed_tokens'):
            state_dict['model.embed_tokens.weight'] = self.model.embed_tokens.weight.detach()
        if hasattr(self.model, 'norm'):
            state_dict['model.norm.weight'] = self.model.norm.weight.detach()
        
        gating_weight = self.gating_network.network.weight.detach()
        
        for layer_idx, layer in enumerate(layers):
            # Save attention weights
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'q_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = attn.q_proj.weight.detach()
                if hasattr(attn, 'k_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = attn.k_proj.weight.detach()
                if hasattr(attn, 'v_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = attn.v_proj.weight.detach()
                if hasattr(attn, 'o_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = attn.o_proj.weight.detach()
                if hasattr(attn, 'q_norm'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.q_norm.weight'] = attn.q_norm.weight.detach()
                if hasattr(attn, 'k_norm'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.k_norm.weight'] = attn.k_norm.weight.detach()
            
            if hasattr(layer, 'input_layernorm'):
                state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = layer.input_layernorm.weight.detach()
            if hasattr(layer, 'post_attention_layernorm'):
                state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight.detach()
            
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, MoEFFN):
                for expert_idx, expert_mlp in enumerate(layer.mlp.expert_mlps):
                    if hasattr(expert_mlp, 'gate_proj'):
                        state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight'] = expert_mlp.gate_proj.weight.detach()
                    if hasattr(expert_mlp, 'up_proj'):
                        state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight'] = expert_mlp.up_proj.weight.detach()
                    if hasattr(expert_mlp, 'down_proj'):
                        state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight'] = expert_mlp.down_proj.weight.detach()
                
                state_dict[f'model.layers.{layer_idx}.mlp.gate.weight'] = gating_weight.clone()
                
                if self.use_shared_expert and self.shared_expert_model is not None:
                    if hasattr(self.shared_expert_model, 'model') and hasattr(self.shared_expert_model.model, 'layers'):
                        shared_layers = self.shared_expert_model.model.layers
                    elif hasattr(self.shared_expert_model, 'layers'):
                        shared_layers = self.shared_expert_model.layers
                    else:
                        raise ValueError("Unsupported shared expert model architecture")
                    
                    shared_layer = shared_layers[layer_idx]
                    if hasattr(shared_layer, 'mlp'):
                        shared_mlp = shared_layer.mlp
                        if hasattr(shared_mlp, 'gate_proj'):
                            state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight'] = shared_mlp.gate_proj.weight.detach()
                        if hasattr(shared_mlp, 'up_proj'):
                            state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight'] = shared_mlp.up_proj.weight.detach()
                        if hasattr(shared_mlp, 'down_proj'):
                            state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight'] = shared_mlp.down_proj.weight.detach()
                    
                    if self.shared_expert_gating is not None:
                        shared_gating_weight = self.shared_expert_gating.network.weight.detach()
                        shared_gating_weight_zeroed = shared_gating_weight.squeeze(0).clone()
                        shared_gating_weight_zeroed.fill_(-100.0)
                        state_dict[f'model.layers.{layer_idx}.mlp.shared_expert_gate.weight'] = shared_gating_weight_zeroed
        
        if self.lm_head is not None:
            state_dict['lm_head.weight'] = self.lm_head.weight.detach().clone()
        
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(save_directory / "model.safetensors"))
            logger.info("Saved model weights in safetensors format")
        except ImportError:
            logger.warning("safetensors not available, saving as PyTorch .bin file")
            torch.save(state_dict, str(save_directory / "pytorch_model.bin"))
        
        self.tokenizer.save_pretrained(str(save_directory))
        self.gating_network.save_pretrained(str(save_directory / "gating"))
        
        if self.use_shared_expert and self.shared_expert_gating is not None:
            self.shared_expert_gating.save_pretrained(str(save_directory / "gating"))
            logger.info("Saved shared expert gating network")
        
        base_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
        
        if self.target_architecture == "qwen3moe":
            base_config['architectures'] = ['Qwen3MoeForCausalLM']
            base_config['model_type'] = 'qwen3moe'
            logger.info("Saving as Qwen3MoeForCausalLM (no shared experts, optional k_norm/q_norm)")
        elif self.target_architecture == "qwen2moe":
            base_config['architectures'] = ['Qwen2MoeForCausalLM']
            base_config['model_type'] = 'qwen2moe'
            logger.info("Saving as Qwen2MoeForCausalLM (requires shared experts)")
        else:
            raise ValueError(f"Unknown target_architecture: {self.target_architecture}")
        
        base_config['num_experts'] = num_experts
        
        if self.routing_mode == "select_one":
            base_config['num_experts_per_tok'] = 1
        elif self.routing_mode == "weighted_sum":
            base_config['num_experts_per_tok'] = min(2, num_experts)
        else:
            base_config['num_experts_per_tok'] = num_experts
        if moe_intermediate_size is not None:
            base_config['moe_intermediate_size'] = moe_intermediate_size
        
        if self.use_shared_expert:
            if hasattr(self.shared_expert_model, 'model') and hasattr(self.shared_expert_model.model, 'layers'):
                shared_layer = self.shared_expert_model.model.layers[0]
            elif hasattr(self.shared_expert_model, 'layers'):
                shared_layer = self.shared_expert_model.layers[0]
            else:
                shared_layer = None
            
            if shared_layer and hasattr(shared_layer, 'mlp') and hasattr(shared_layer.mlp, 'gate_proj'):
                shared_expert_intermediate_size = shared_layer.mlp.gate_proj.out_features
                base_config['shared_expert_intermediate_size'] = shared_expert_intermediate_size
                logger.info(f"Shared expert intermediate size: {shared_expert_intermediate_size}")
        
        if 'tie_word_embeddings' not in base_config:
            base_config['tie_word_embeddings'] = False
        
        with open(save_directory / "config.json", "w") as f:
            json.dump(base_config, f, indent=2)
        
        rmoe_metadata = {
            "original_model_type": "rmoe",
            "routing_mode": self.routing_mode,
            "per_sequence_routing": True,
            "gating_network_type": "linear",
            "num_experts": num_experts,
        }
        with open(save_directory / "rmoe_metadata.json", "w") as f:
            json.dump(rmoe_metadata, f, indent=2)
            
        moe_config = {
            "model_type": "moe",
            "num_experts": num_experts,
            "routing_mode": self.routing_mode,
        }
        with open(save_directory / "moe_config.json", "w") as f:
            json.dump(moe_config, f, indent=2)
        
        logger.info(f"MoE model saved to: {save_directory}")
        logger.info(f"Model can now be converted using: python convert_hf_to_gguf.py {save_directory} --outtype f16")

