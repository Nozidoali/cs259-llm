import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class GatingModelWrapper(nn.Module):
    def __init__(self, base_model, gating_network):
        super().__init__()
        self.base_model = base_model
        self.gating_network = gating_network
    
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
                embeddings = self.base_model.model.embed_tokens(input_ids)
            elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
                embeddings = self.base_model.transformer.wte(input_ids)
            else:
                raise ValueError("Could not find embedding layer in base model")
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9).to(embeddings.dtype)
            mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    def forward(self, x):
        return self.gating_network(x)
    
    def save_pretrained(self, save_directory):
        from pathlib import Path
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.gating_network.state_dict(), save_directory / "gating_network.pt")
        logger.info(f"Saved gating network to: {save_directory}")

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256], dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        logger.info(f"GatingNetwork: input_dim={input_dim}, hidden_dims={hidden_dims}, dropout={dropout}, num_classes={num_classes}")
    
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

