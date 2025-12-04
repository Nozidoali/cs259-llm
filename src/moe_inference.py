#!/usr/bin/env python3

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union, List

import torch

_file_dir = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.dirname(_file_dir)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from moe_model import MoEModel
from config import MOE_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MoEInference:
    
    def __init__(
        self,
        saved_model_path: Optional[Union[str, Path]] = None,
        model1_path: Optional[Union[str, Path]] = None,
        model2_path: Optional[Union[str, Path]] = None,
        gating_model_path: Optional[Union[str, Path]] = None,
        base_model_path: Optional[Union[str, Path]] = None,
        routing_mode: str = "weighted_sum",
        device: Optional[torch.device] = None,
    ):
        if saved_model_path is not None:
            logger.info(f"Loading MoE model from saved checkpoint: {saved_model_path}")
            self.model = MoEModel.from_pretrained(saved_model_path, device=device)
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
            self.routing_mode = self.model.routing_mode
        else:
            if model1_path is None:
                model1_path = MOE_CONFIG.get("model1_path")
            if model2_path is None:
                model2_path = MOE_CONFIG.get("model2_path")
            if gating_model_path is None:
                gating_model_path = MOE_CONFIG.get("gating_model_path")
            if routing_mode == "weighted_sum" and "routing_mode" in MOE_CONFIG:
                routing_mode = MOE_CONFIG.get("routing_mode", "weighted_sum")
            
            if model1_path is None or model2_path is None or gating_model_path is None:
                raise ValueError("Either saved_model_path must be provided, or model1_path, model2_path, and gating_model_path must be provided or set in MOE_CONFIG")
            
            logger.info("Initializing MoE model from components...")
            self.model = MoEModel(
                model1_path=model1_path,
                model2_path=model2_path,
                gating_model_path=gating_model_path,
                base_model_path=base_model_path,
                routing_mode=routing_mode,
                device=device,
            )
            
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
            self.routing_mode = routing_mode
        
        logger.info("MoE model ready for inference")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_gating_probs: bool = True,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        gating_probs = None
        if return_gating_probs:
            gating_probs = self.model.get_gating_probs(input_ids, attention_mask)
            gating_probs = gating_probs.cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        result = {"text": generated_text}
        if return_gating_probs and gating_probs is not None:
            result["gating_probs"] = {
                "model1_prob": float(gating_probs[0, 0]),
                "model2_prob": float(gating_probs[0, 1]),
            }
        
        return result
    
    def get_gating_probs(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        with torch.no_grad():
            gating_probs = self.model.get_gating_probs(input_ids, attention_mask)
            gating_probs = gating_probs.cpu().numpy()
        
        return {
            "model1_prob": float(gating_probs[0, 0]),
            "model2_prob": float(gating_probs[0, 1]),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Model Inference")
    
    parser.add_argument("--saved_model_path", type=str, default=None,
                       help="Path to saved MoE model directory (if provided, loads from saved model)")
    
    parser.add_argument("--model1_path", type=str, default=None,
                       help="Path to first finetuned model (required if saved_model_path not provided)")
    parser.add_argument("--model2_path", type=str, default=None,
                       help="Path to second finetuned model (required if saved_model_path not provided)")
    parser.add_argument("--gating_model_path", type=str, default=None,
                       help="Path to gating network (required if saved_model_path not provided)")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base model for embeddings (optional)")
    parser.add_argument("--routing_mode", type=str, default="weighted_sum",
                       choices=["weighted_sum", "select_one"],
                       help="Routing mode: weighted_sum or select_one")
    
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--no_sample", action="store_true",
                       help="Disable sampling (use greedy decoding)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file")
    
    return parser.parse_args()


def load_config(args):
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            config = json.load(f)
    
    if args.saved_model_path:
        config["saved_model_path"] = args.saved_model_path
    if args.model1_path:
        config["model1_path"] = args.model1_path
    if args.model2_path:
        config["model2_path"] = args.model2_path
    if args.gating_model_path:
        config["gating_model_path"] = args.gating_model_path
    if args.base_model_path:
        config["base_model_path"] = args.base_model_path
    if args.routing_mode:
        config["routing_mode"] = args.routing_mode
    
    return config


def main():
    args = parse_args()
    config = load_config(args)
    
    inference = MoEInference(
        saved_model_path=config.get("saved_model_path"),
        model1_path=config.get("model1_path"),
        model2_path=config.get("model2_path"),
        gating_model_path=config.get("gating_model_path"),
        base_model_path=config.get("base_model_path"),
        routing_mode=config.get("routing_mode", "weighted_sum"),
    )
    
    result = inference.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        return_gating_probs=True,
    )
    
    print("=" * 60)
    print("Input Prompt:")
    print(args.prompt)
    print("=" * 60)
    print("Generated Text:")
    print(result["text"])
    print("=" * 60)
    if "gating_probs" in result:
        print("Gating Probabilities:")
        print(f"  Model 1: {result['gating_probs']['model1_prob']:.4f}")
        print(f"  Model 2: {result['gating_probs']['model2_prob']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()


