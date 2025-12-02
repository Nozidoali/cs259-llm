#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Union, Optional
import torch

from config import GATING_MODEL_DIR
from gating_dataset import load_base_model_for_embeddings, extract_embeddings
from gating_model import GatingNetwork

logger = logging.getLogger(__name__)


class GatingNetworkInference:
    def __init__(self, model_path: Union[str, Path], base_model: str, device: Optional[torch.device] = None):
        model_path = Path(model_path)
        
        training_info_path = model_path / "training_info.json"
        if not training_info_path.exists():
            raise FileNotFoundError(f"training_info.json not found in {model_path}")
        
        with open(training_info_path, "r") as f:
            self.training_info = json.load(f)
        
        self.embedding_dim = self.training_info["embedding_dim"]
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device
        
        logger.info(f"Loading base model: {base_model}")
        self.base_model, self.tokenizer, _ = load_base_model_for_embeddings(base_model)
        self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        
        model_config = self.training_info.get("config", {})
        hidden_dims = model_config.get("hidden_dims", [512, 256])
        dropout = model_config.get("dropout", 0.1)
        
        self.gating_model = GatingNetwork(
            input_dim=self.embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)
        
        best_model_path = model_path / "best_model.pt"
        final_model_path = model_path / "final_model.pt"
        
        if best_model_path.exists():
            self.gating_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        elif final_model_path.exists():
            self.gating_model.load_state_dict(torch.load(final_model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Neither best_model.pt nor final_model.pt found in {model_path}")
        
        self.gating_model.eval()
        logger.info("Gating network loaded successfully")
    
    def classify(self, text: str, threshold: float = 0.5) -> dict:
        embeddings = extract_embeddings(
            self.base_model, self.tokenizer, [text], max_length=512, batch_size=1, device=self.device
        )
        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            probability = self.gating_model(embedding_tensor).item()
        
        prediction = 1 if probability >= threshold else 0
        return {
            "probability": probability,
            "prediction": prediction,
            "is_summarization": prediction == 1,
        }
    
    def classify_batch(self, texts: list, threshold: float = 0.5) -> list:
        embeddings = extract_embeddings(
            self.base_model, self.tokenizer, texts, max_length=512, batch_size=8, device=self.device
        )
        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            probabilities = self.gating_model(embedding_tensor).cpu().numpy().flatten()
        
        results = []
        for prob in probabilities:
            prediction = 1 if prob >= threshold else 0
            results.append({
                "probability": float(prob),
                "prediction": prediction,
                "is_summarization": prediction == 1,
            })
        return results


def load_gating_network(model_path: Union[str, Path], base_model: Optional[str] = None, device: Optional[torch.device] = None):
    model_path = Path(model_path)
    
    if base_model is None:
        training_info_path = model_path / "training_info.json"
        if training_info_path.exists():
            with open(training_info_path, "r") as f:
                training_info = json.load(f)
            base_model = training_info.get("base_model")
            if base_model is None:
                raise ValueError("base_model not found in training_info.json")
        else:
            raise ValueError("base_model not specified and training_info.json not found")
    
    return GatingNetworkInference(model_path, base_model, device)
