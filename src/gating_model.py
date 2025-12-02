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
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized GatingNetwork: input_dim={input_dim}, "
                   f"hidden_dims={hidden_dims}, dropout={dropout}")
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= threshold).long()
