#!/usr/bin/env python3

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        
        logger.info(f"Initialized GatingNetwork: input_dim={input_dim}, "
                   f"hidden_dims={hidden_dims}, dropout={dropout}, output_classes=2")
    
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

