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
    def __init__(self, expert_mlps: List, parent_model, shared_expert_mlp: nn.Module, routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum"):
        super().__init__()
        self.expert_mlps = nn.ModuleList(expert_mlps)
        self.shared_expert_mlp = shared_expert_mlp
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
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        use_per_token = getattr(self.parent_model, 'use_per_token_routing', False)
        
        # Check if forced expert routing is enabled
        if hasattr(self.parent_model, 'forced_expert_idx') and self.parent_model.forced_expert_idx is not None:
            # Force routing to specific expert (bypass gating network)
            forced_idx = self.parent_model.forced_expert_idx
            if forced_idx < 0 or forced_idx >= self.num_experts:
                raise ValueError(f"forced_expert_idx {forced_idx} is out of range [0, {self.num_experts-1}]")
            if use_per_token:
                gating_probs = torch.zeros(batch_size, seq_len, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
                gating_probs[:, :, forced_idx] = 1.0
            else:
                gating_probs = torch.zeros(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
                gating_probs[:, forced_idx] = 1.0
        elif hasattr(self.parent_model, '_current_gating_probs') and self.parent_model._current_gating_probs is not None:
            gating_probs = self.parent_model._current_gating_probs.to(hidden_states.dtype)
            # Handle shape: could be (batch_size, num_experts) or (batch_size, seq_len, num_experts)
            if gating_probs.dim() == 2 and use_per_token:
                # Expand per-sequence to per-token: [batch_size, num_experts] -> [batch_size, seq_len, num_experts]
                gating_probs = gating_probs.unsqueeze(1).expand(-1, seq_len, -1)
            elif gating_probs.dim() == 3 and not use_per_token:
                # Use first token's probabilities for per-sequence routing
                gating_probs = gating_probs[:, 0, :]
        else:
            logger.warning("_current_gating_probs not set, using equal probabilities")
            if use_per_token:
                gating_probs = torch.ones(batch_size, seq_len, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype) / self.num_experts
            else:
                gating_probs = torch.ones(batch_size, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype) / self.num_experts
        
        # Get shared expert gating probability (should be ~0 after training)
        if hasattr(self.parent_model, '_current_shared_expert_gate_prob') and self.parent_model._current_shared_expert_gate_prob is not None:
            shared_gate_prob = self.parent_model._current_shared_expert_gate_prob.to(hidden_states.dtype)
            # Handle shape: could be (batch_size, 1) or (batch_size, seq_len, 1)
            if shared_gate_prob.dim() == 2 and use_per_token:
                # Expand per-sequence to per-token: [batch_size, 1] -> [batch_size, seq_len, 1]
                shared_gate_prob = shared_gate_prob.unsqueeze(1).expand(-1, seq_len, -1)
            elif shared_gate_prob.dim() == 3 and not use_per_token:
                # Use first token's probability for per-sequence routing
                shared_gate_prob = shared_gate_prob[:, 0, :]
        else:
            logger.warning("_current_shared_expert_gate_prob not set, using 0 (no shared expert contribution)")
            if use_per_token:
                shared_gate_prob = torch.zeros(batch_size, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
            else:
                shared_gate_prob = torch.zeros(batch_size, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Compute expert outputs
        expert_outputs = [expert_mlp(hidden_states) for expert_mlp in self.expert_mlps]
        
        # Compute expert output (weighted sum or select one)
        if self.routing_mode == "weighted_sum":
            expert_output = torch.zeros_like(expert_outputs[0])
            if use_per_token:
                # Per-token weighted sum: gating_probs is [batch_size, seq_len, num_experts]
                for idx, expert_out in enumerate(expert_outputs):
                    prob = gating_probs[:, :, idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
                    expert_output += prob * expert_out
            else:
                # Per-sequence weighted sum: gating_probs is [batch_size, num_experts]
                for idx, expert_out in enumerate(expert_outputs):
                    prob = gating_probs[:, idx].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                    expert_output += prob * expert_out
        else:
            # select_one routing
            if use_per_token:
                # Per-token selection: gating_probs is [batch_size, seq_len, num_experts]
                selected_expert = gating_probs.argmax(dim=-1)  # [batch_size, seq_len]
                expert_output = torch.zeros_like(expert_outputs[0])
                for idx, expert_out in enumerate(expert_outputs):
                    # Create mask: [batch_size, seq_len, hidden_dim]
                    mask = (selected_expert == idx).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                    expert_output += mask * expert_out
            else:
                # Per-sequence selection: gating_probs is [batch_size, num_experts]
                selected_expert = gating_probs.argmax(dim=-1)  # [batch_size]
                expert_output = torch.zeros_like(expert_outputs[0])
                for idx, expert_out in enumerate(expert_outputs):
                    mask = (selected_expert == idx).unsqueeze(-1).unsqueeze(-1).float()  # [batch_size, 1, 1]
                    expert_output += mask * expert_out
        
        # Compute shared expert output
        shared_expert_output = self.shared_expert_mlp(hidden_states)
        
        # Combine: output = (1 - shared_prob) * expert_output + shared_prob * shared_expert_output
        # Since shared_prob should be ~0 and shared_expert is zero-initialized, this effectively uses only expert_output
        if use_per_token:
            # shared_gate_prob is [batch_size, seq_len, 1]
            output = (1.0 - shared_gate_prob) * expert_output + shared_gate_prob * shared_expert_output
        else:
            # shared_gate_prob is [batch_size, 1]
            output = (1.0 - shared_gate_prob.unsqueeze(-1)) * expert_output + shared_gate_prob.unsqueeze(-1) * shared_expert_output
        
        return output

class MoEModel(nn.Module):
    def __init__(self, expert_paths: List[Union[str, Path]], gating_model_path: Union[str, Path], base_model_path: Optional[Union[str, Path]] = None, routing_mode: Literal["weighted_sum", "select_one"] = "weighted_sum", device=None, shared_expert_path: Optional[Union[str, Path]] = None, num_experts_per_tok: Optional[int] = None, use_zero_shared_expert: bool = True, forced_expert_idx: Optional[int] = None, use_per_token_routing: Optional[bool] = None, shared_expert_intermediate_size: Optional[int] = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.routing_mode = routing_mode
        self.num_experts_per_tok = num_experts_per_tok
        self.use_zero_shared_expert = use_zero_shared_expert
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        # Qwen2MoE always requires shared expert
        self.use_shared_expert = True
        # Force routing to a specific expert (bypasses gating network)
        self.forced_expert_idx = forced_expert_idx
        # Per-token routing: default True for select_one, False for weighted_sum
        if use_per_token_routing is None:
            self.use_per_token_routing = (routing_mode == "select_one")
        else:
            self.use_per_token_routing = use_per_token_routing
        logger.info(f"Using per-token routing: {self.use_per_token_routing} (routing_mode={routing_mode})")
        expert_paths = [Path(p) for p in expert_paths]
        self.num_experts_total = len(expert_paths)
        if forced_expert_idx is not None:
            if forced_expert_idx < 0 or forced_expert_idx >= self.num_experts_total:
                raise ValueError(f"forced_expert_idx {forced_expert_idx} is out of range [0, {self.num_experts_total-1}]")
            logger.info(f"FORCED ROUTING: Will always use expert {forced_expert_idx} (bypassing gating network)")
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
        self._current_shared_expert_gate_prob = None
        
        # Qwen2MoE always requires shared expert
        logger.info("Qwen2MoE requires shared expert - initializing...")
        if use_zero_shared_expert:
            logger.info("Using zero-initialized shared expert (no model loaded)")
            self.shared_expert_model = None  # Will create zero tensors in save_pretrained
        else:
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
        logger.info("Shared expert gating network loaded successfully")
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
        logger.info("Using Qwen2MoE architecture (requires shared expert)")
        
        # Extract shared expert MLPs by layer
        shared_expert_mlps_by_layer = {}
        if use_zero_shared_expert or self.shared_expert_model is None:
            # Will create zero-initialized shared expert MLPs later
            for layer_idx in range(num_layers):
                shared_expert_mlps_by_layer[layer_idx] = None
        else:
            if hasattr(self.shared_expert_model, 'model') and hasattr(self.shared_expert_model.model, 'layers'):
                shared_layers = self.shared_expert_model.model.layers
            elif hasattr(self.shared_expert_model, 'layers'):
                shared_layers = self.shared_expert_model.layers
            else:
                raise ValueError("Unsupported shared expert model architecture")
            
            for layer_idx in range(num_layers):
                shared_layer = shared_layers[layer_idx]
                if hasattr(shared_layer, 'mlp'):
                    shared_expert_mlp = shared_layer.mlp.to(self.device)
                    shared_expert_mlps_by_layer[layer_idx] = shared_expert_mlp
        
        expert_mlps_list = []
        for layer_idx in range(num_layers):
            layer = layers[layer_idx]
            if hasattr(layer, 'mlp'):
                expert_mlps = expert_mlps_by_layer[layer_idx]
                
                # Get or create shared expert MLP for this layer
                if shared_expert_mlps_by_layer[layer_idx] is not None:
                    shared_expert_mlp = shared_expert_mlps_by_layer[layer_idx]
                else:
                    # Create zero-initialized shared expert MLP
                    first_expert = expert_mlps[0]
                    if hasattr(first_expert, 'gate_proj') and hasattr(first_expert, 'up_proj') and hasattr(first_expert, 'down_proj'):
                        # Create a zero-initialized MLP with same structure
                        class ZeroMLP(nn.Module):
                            def __init__(self, gate_shape, up_shape, down_shape, dtype):
                                super().__init__()
                                self.gate_proj = nn.Linear(gate_shape[1], gate_shape[0], bias=False)
                                self.up_proj = nn.Linear(up_shape[1], up_shape[0], bias=False)
                                self.down_proj = nn.Linear(down_shape[1], down_shape[0], bias=False)
                                # Zero initialize
                                nn.init.zeros_(self.gate_proj.weight)
                                nn.init.zeros_(self.up_proj.weight)
                                nn.init.zeros_(self.down_proj.weight)
                            
                            def forward(self, x):
                                gate = self.gate_proj(x)
                                up = self.up_proj(x)
                                return self.down_proj(torch.nn.functional.silu(gate) * up)
                        
                        gate_shape = first_expert.gate_proj.weight.shape
                        up_shape = first_expert.up_proj.weight.shape
                        down_shape = first_expert.down_proj.weight.shape
                        shared_expert_mlp = ZeroMLP(gate_shape, up_shape, down_shape, first_expert.gate_proj.weight.dtype).to(self.device)
                    else:
                        raise ValueError("Cannot create zero-initialized shared expert MLP - unsupported expert structure")
                
                moe_ffn = MoEFFN(expert_mlps, self, shared_expert_mlp, routing_mode=routing_mode)
                layer.mlp = moe_ffn
                expert_mlps_list.append(expert_mlps)
        self.model = base_model
        if hasattr(base_model, 'model'):
            self.model = base_model.model
        self.lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else None
        if self.lm_head is None and hasattr(base_model, 'model') and hasattr(base_model.model, 'lm_head'):
            self.lm_head = base_model.model.lm_head
        logger.info(f"MoE model initialized with {len(expert_paths)} experts, routing_mode={routing_mode}")
    
    def get_per_token_embeddings(self, input_ids, attention_mask=None):
        """
        Extract per-token embeddings for fine-tuning the gating network.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Per-token embeddings [batch_size, seq_len, embedding_dim]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self._get_embeddings(input_ids, attention_mask, per_token=True)
    
    def _get_embeddings(self, input_ids, attention_mask, per_token=None):
        """
        Get embeddings for gating network.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            per_token: If True, return per-token embeddings [batch_size, seq_len, embedding_dim]
                      If False, return averaged embeddings [batch_size, embedding_dim]
                      If None, use self.use_per_token_routing
        
        Returns:
            Embeddings for gating network
        """
        if per_token is None:
            per_token = self.use_per_token_routing
            
        with torch.no_grad():
            if hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, 'embed_tokens'):
                embeddings = self.embedding_model.model.embed_tokens(input_ids)
            elif hasattr(self.embedding_model, 'transformer') and hasattr(self.embedding_model.transformer, 'wte'):
                embeddings = self.embedding_model.transformer.wte(input_ids)
            else:
                raise ValueError("Could not find embedding layer in embedding model")
            
            if per_token:
                # Return per-token embeddings: [batch_size, seq_len, embedding_dim]
                # Apply attention mask to zero out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
                return embeddings * mask_expanded
            else:
                # Return averaged embeddings: [batch_size, embedding_dim]
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9).to(embeddings.dtype)
                mean_embeddings = sum_embeddings / sum_mask
                return mean_embeddings
    
    def forward(self, input_ids, attention_mask=None, labels=None, recompute_gating=True, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.model.embed_tokens(input_ids) if hasattr(self.model, 'embed_tokens') else self.model.embeddings(input_ids)
        hidden_states = embeddings
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)
        elif hasattr(self.model, 'layer_norm'):
            hidden_states = self.model.layer_norm(hidden_states)
        batch_size, seq_len = input_ids.shape
        # Skip gating computation if forced expert routing is enabled
        if self.forced_expert_idx is not None:
            # Create one-hot gating probabilities for forced expert
            if self.use_per_token_routing:
                gating_probs = torch.zeros(batch_size, seq_len, self.num_experts_total, device=self.device)
                gating_probs[:, :, self.forced_expert_idx] = 1.0
            else:
                gating_probs = torch.zeros(batch_size, self.num_experts_total, device=self.device)
                gating_probs[:, self.forced_expert_idx] = 1.0
            self._current_gating_probs = gating_probs
            # Shared expert should not be used when forcing a specific expert
            if self.use_per_token_routing:
                self._current_shared_expert_gate_prob = torch.zeros(batch_size, seq_len, 1, device=self.device)
            else:
                self._current_shared_expert_gate_prob = torch.zeros(batch_size, 1, device=self.device)
        # Only recompute gating if not already set or if explicitly requested
        elif recompute_gating or self._current_gating_probs is None:
            with torch.no_grad():
                gating_input = self._get_embeddings(input_ids, attention_mask, per_token=self.use_per_token_routing)
                
                if self.use_per_token_routing:
                    # Per-token routing: gating_input is [batch_size, seq_len, embedding_dim]
                    batch_size_pt, seq_len_pt, embedding_dim = gating_input.shape
                    # Reshape to [batch_size * seq_len, embedding_dim] for batch processing
                    gating_input_flat = gating_input.view(-1, embedding_dim)
                    gating_probs_flat = self.gating_network(gating_input_flat)  # [batch_size * seq_len, num_experts]
                    # Reshape back to [batch_size, seq_len, num_experts]
                    gating_probs = gating_probs_flat.view(batch_size_pt, seq_len_pt, self.num_experts_total)
                    
                    # Shared expert gating: [batch_size * seq_len, 1] -> [batch_size, seq_len, 1]
                    shared_gate_prob_flat = self.shared_expert_gating(gating_input_flat)
                    shared_gate_prob = shared_gate_prob_flat.view(batch_size_pt, seq_len_pt, 1)
                else:
                    # Per-sequence routing: gating_input is [batch_size, embedding_dim]
                    gating_probs = self.gating_network(gating_input)  # [batch_size, num_experts]
                    # Always compute shared expert gating probability (required for Qwen2MoE)
                    shared_gate_prob = self.shared_expert_gating(gating_input)  # Shape: (batch_size, 1)
            self._current_gating_probs = gating_probs
            self._current_shared_expert_gate_prob = shared_gate_prob
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
            generated = input_ids.clone()
            # Recompute routing for each generation step to adapt to growing sequence
            for _ in range(max_new_tokens):
                # Skip gating computation if forced expert routing is enabled
                if self.forced_expert_idx is None:
                    # Recompute gating probabilities based on current sequence state
                    gating_input = self._get_embeddings(generated, attention_mask, per_token=self.use_per_token_routing)
                    
                    if self.use_per_token_routing:
                        # Per-token routing: gating_input is [batch_size, seq_len, embedding_dim]
                        batch_size_pt, seq_len_pt, embedding_dim = gating_input.shape
                        gating_input_flat = gating_input.view(-1, embedding_dim)
                        gating_probs_flat = self.gating_network(gating_input_flat)
                        gating_probs = gating_probs_flat.view(batch_size_pt, seq_len_pt, self.num_experts_total)
                        
                        shared_gate_prob_flat = self.shared_expert_gating(gating_input_flat)
                        shared_gate_prob = shared_gate_prob_flat.view(batch_size_pt, seq_len_pt, 1)
                    else:
                        # Per-sequence routing: gating_input is [batch_size, embedding_dim]
                        gating_probs = self.gating_network(gating_input)
                        shared_gate_prob = self.shared_expert_gating(gating_input)
                    
                    self._current_gating_probs = gating_probs
                    self._current_shared_expert_gate_prob = shared_gate_prob
                    # Forward pass with updated routing (recompute_gating=False since we just computed it)
                    outputs = self.forward(generated, attention_mask=attention_mask, recompute_gating=False)
                else:
                    # Forced expert routing - gating will be handled in forward()
                    outputs = self.forward(generated, attention_mask=attention_mask, recompute_gating=True)
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
        
        # Modify gating weights if forced expert routing is enabled
        if self.forced_expert_idx is not None:
            logger.info(f"Modifying gating weights to force routing to expert {self.forced_expert_idx}")
            # Get original gating weights shape: [num_experts, embedding_dim]
            original_gating_weight = self.gating_network.network.weight.detach()
            gating_weight = torch.zeros_like(original_gating_weight)
            # Set the forced expert's weights to a large positive value so it always wins
            # The logit for this expert will be: large_value * sum(embedding) = large positive
            # Set all other experts' weights to zero so their logits are 0
            # After softmax, the forced expert will have probability ~1.0
            gating_weight[self.forced_expert_idx, :] = 100.0  # Large positive value for all embedding dimensions
            # All other experts already have zero weights (will output 0 logits)
            logger.info(f"Gating weights modified: expert {self.forced_expert_idx} will always be selected (logit ~100*sum(emb), others ~0)")
        else:
            gating_weight = self.gating_network.network.weight.detach()
        
        for layer_idx, layer in enumerate(layers):
            # Save attention weights and biases
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'q_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = attn.q_proj.weight.detach()
                    if attn.q_proj.bias is not None:
                        state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.bias'] = attn.q_proj.bias.detach()
                if hasattr(attn, 'k_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = attn.k_proj.weight.detach()
                    if attn.k_proj.bias is not None:
                        state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.bias'] = attn.k_proj.bias.detach()
                if hasattr(attn, 'v_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = attn.v_proj.weight.detach()
                    if attn.v_proj.bias is not None:
                        state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.bias'] = attn.v_proj.bias.detach()
                if hasattr(attn, 'o_proj'):
                    state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = attn.o_proj.weight.detach()
                    if attn.o_proj.bias is not None:
                        state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.bias'] = attn.o_proj.bias.detach()
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

                # Qwen2MoE always requires shared expert - force all weights to exactly zero
                first_expert = layer.mlp.expert_mlps[0]
                if hasattr(first_expert, 'gate_proj'):
                    gate_shape = first_expert.gate_proj.weight.shape
                    state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight'] = torch.zeros(gate_shape, dtype=first_expert.gate_proj.weight.dtype)
                if hasattr(first_expert, 'up_proj'):
                    up_shape = first_expert.up_proj.weight.shape
                    state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight'] = torch.zeros(up_shape, dtype=first_expert.up_proj.weight.dtype)
                if hasattr(first_expert, 'down_proj'):
                    down_shape = first_expert.down_proj.weight.shape
                    state_dict[f'model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight'] = torch.zeros(down_shape, dtype=first_expert.down_proj.weight.dtype)
                logger.info(f"Created zero-initialized shared expert for layer {layer_idx}" if layer_idx == 0 else "")

                # Always save shared expert gating (required for Qwen2MoE)
                # Set to very negative values so sigmoid outputs ~0 (never use shared expert)
                shared_gating_weight = self.shared_expert_gating.network.weight.detach()
                # Force to very negative to ensure sigmoid(x) ≈ 0
                shared_gating_weight_forced = torch.full_like(shared_gating_weight, -20.0)
                state_dict[f'model.layers.{layer_idx}.mlp.shared_expert_gate.weight'] = shared_gating_weight_forced
        
        if self.lm_head is not None:
            state_dict['lm_head.weight'] = self.lm_head.weight.detach().clone()
        
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(save_directory / "model.safetensors"))
            logger.info("Saved model weights in safetensors format")
        except ImportError:
            logger.warning("safetensors not available, saving as PyTorch .bin file")
            torch.save(state_dict, str(save_directory / "pytorch_model.bin"))
        
        # Save tokenizer with all configuration files
        logger.info("Saving tokenizer...")
        self.tokenizer.save_pretrained(str(save_directory))

        # Ensure tokenizer_config.json includes fix_mistral_regex if applicable
        tokenizer_config_path = save_directory / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            import json
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            # Ensure clean_up_tokenization_spaces is set for compatibility
            if 'clean_up_tokenization_spaces' not in tokenizer_config:
                tokenizer_config['clean_up_tokenization_spaces'] = False
            with open(tokenizer_config_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            logger.info("Updated tokenizer configuration")

        self.gating_network.save_pretrained(str(save_directory / "gating"))
        
        # Always save shared expert gating (required for Qwen2MoE)
        self.shared_expert_gating.save_pretrained(str(save_directory / "gating"))
        logger.info("Saved shared expert gating network")
        
        base_config = self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
        
        # Qwen2MoE always requires shared experts
        # Use qwen2_moe (with underscore) for transformers compatibility
        base_config['architectures'] = ['Qwen2MoeForCausalLM']
        base_config['model_type'] = 'qwen2_moe'
        logger.info("Saving as Qwen2MoeForCausalLM (requires shared experts)")
        
        base_config['num_experts'] = num_experts
        
        if self.num_experts_per_tok is not None:
            base_config['num_experts_per_tok'] = self.num_experts_per_tok
        elif self.routing_mode == "select_one":
            base_config['num_experts_per_tok'] = 1
        elif self.routing_mode == "weighted_sum":
            base_config['num_experts_per_tok'] = min(2, num_experts)
        else:
            base_config['num_experts_per_tok'] = num_experts
        if moe_intermediate_size is not None:
            base_config['moe_intermediate_size'] = moe_intermediate_size
        
        # Qwen2MoE always requires shared expert
        # Check if shared_expert_intermediate_size is explicitly configured
        if self.shared_expert_intermediate_size is not None:
            shared_expert_intermediate_size = self.shared_expert_intermediate_size
            base_config['shared_expert_intermediate_size'] = shared_expert_intermediate_size
            logger.info(f"Shared expert intermediate size (configured): {shared_expert_intermediate_size}")
        elif self.use_zero_shared_expert or self.shared_expert_model is None:
            # Use same intermediate size as regular experts
            shared_expert_intermediate_size = moe_intermediate_size
            base_config['shared_expert_intermediate_size'] = shared_expert_intermediate_size
            logger.info(f"Shared expert intermediate size (zero-initialized): {shared_expert_intermediate_size}")
        else:
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
            "per_token_routing": self.use_per_token_routing,
            "per_sequence_routing": not self.use_per_token_routing,
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

