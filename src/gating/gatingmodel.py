import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256], dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        logger.info(f"GatingNetwork (simplified, no bias): input_dim={input_dim}, num_classes={num_classes}")
    
    def forward(self, x):
        logits = self.network(x)
        probs = self.softmax(logits)
        return probs
    
    def get_logits(self, x):
        return self.network(x)
    
    def predict(self, x):
        with torch.no_grad():
            probs = self.forward(x)
            return probs.argmax(dim=-1)
    
    def save_pretrained(self, save_directory):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "gating_network.pt")
        logger.info(f"Saved gating network to: {save_directory}")


class SharedExpertGating(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Single linear layer (no bias) to match Qwen2MoE format
        # This will be saved as ffn_gate_inp_shexp.weight [n_embd]
        self.network = nn.Linear(input_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        logger.info(f"SharedExpertGating (no bias): input_dim={input_dim}, output=1 (binary)")
        
    def forward(self, x):
        """Returns probability of using shared expert [0,1]"""
        logits = self.network(x)
        return self.sigmoid(logits)  # Shape: (batch_size, 1)
    
    def get_logits(self, x):
        """Returns raw logits for training"""
        return self.network(x)
    
    def predict(self, x, threshold=0.5):
        """Binary decision: use shared expert (1.0) or not (0.0)"""
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > threshold).float()
    
    def save_pretrained(self, save_directory):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "shared_expert_gating.pt")
        logger.info(f"Saved shared expert gating to: {save_directory}")

