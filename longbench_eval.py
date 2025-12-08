#!/usr/bin/env python3
"""
LongBench Evaluation Script

Evaluates a HuggingFace model on the LongBench qmsum dataset using ROUGE-L scoring.
Runs on the first 50 samples from the LongBench qmsum test set.

Usage:
    python longbench_eval.py --model_dir <path_to_model_dir>

Example:
    python longbench_eval.py --model_dir ./models/qwen2-0.5b-instruct-finetuned
"""

import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_dir):
    """Load HuggingFace model and tokenizer from safetensors."""
    import json
    import shutil

    logger.info(f"Loading model from {model_dir}...")
    model_path = Path(model_dir)

    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    # Load and potentially modify config for compatibility
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    original_model_type = config.get("model_type")
    config_modified = False

    # Handle qwen2moe/qwen3moe by temporarily changing to qwen2
    if original_model_type in ["qwen2moe", "qwen3moe"]:
        logger.info(f"Detected {original_model_type} model, temporarily converting config to qwen2 for loading...")
        backup_path = model_path / "config.json.backup"
        shutil.copy(config_path, backup_path)

        config["model_type"] = "qwen2"
        config["architectures"] = ["Qwen2ForCausalLM"]

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        config_modified = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model with safetensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
    finally:
        # Restore original config if modified
        if config_modified:
            backup_path = model_path / "config.json.backup"
            if backup_path.exists():
                shutil.move(backup_path, config_path)
                logger.info("Restored original config")

    if device == "cpu":
        model = model.to(device)

    model.eval()

    return model, tokenizer, device


def generate_summary(model, tokenizer, prompt, device, max_new_tokens=200):
    """Generate a summary for a given prompt."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return summary


def load_longbench_references(n_samples=50):
    """Load LongBench qmsum dataset references."""
    logger.info("Loading LongBench qmsum dataset...")
    ds = load_dataset("THUDM/LongBench", "qmsum", split="test", trust_remote_code=True)

    ref_map = {}
    for i, rec in enumerate(ds):
        if i >= n_samples:
            break

        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()

    logger.info(f"Loaded {len(ref_map)} references from LongBench qmsum")
    return ref_map, ds


def evaluate_longbench(model_dir, n_samples=50, max_new_tokens=200):
    """
    Evaluate a model on LongBench qmsum using ROUGE-L scoring.

    Args:
        model_dir: Path to the model directory containing safetensors
        n_samples: Number of samples to evaluate (default: 50)
        max_new_tokens: Maximum number of tokens to generate per summary

    Returns:
        Dictionary with ROUGE-L score and per-sample results
    """
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    # Load LongBench dataset and references
    ref_map, ds = load_longbench_references(n_samples)

    # Generate predictions for all samples
    all_predictions = []
    all_references = []
    sample_ids = []

    logger.info(f"Generating summaries for {n_samples} samples...")
    start_time = time.time()

    for i in range(n_samples):
        if i >= len(ds):
            logger.warning(f"Only {len(ds)} samples available in dataset")
            break

        record = ds[i]

        # Extract the input context and instruction
        context = record.get("context", "")
        input_text = record.get("input", "")

        # LongBench qmsum format: context is the meeting transcript, input is the question
        # Construct the prompt
        if input_text:
            prompt = f"{context}\n\n{input_text}"
        else:
            prompt = context

        if not prompt:
            logger.warning(f"Skipping sample {i} due to missing prompt")
            continue

        # Generate summary
        sample_start = time.time()
        try:
            prediction = generate_summary(model, tokenizer, prompt, device, max_new_tokens)
        except Exception as e:
            logger.error(f"Error generating summary for sample {i}: {e}")
            prediction = ""
        sample_time = time.time() - sample_start

        # Get reference
        if i not in ref_map:
            logger.warning(f"No reference for sample {i}, skipping")
            continue

        reference = ref_map[i]

        all_predictions.append(prediction)
        all_references.append(reference)
        sample_ids.append(i)

        # Log progress
        if (i + 1) % 5 == 0:
            avg_time = (time.time() - start_time) / (i + 1)
            remaining = (n_samples - i - 1) * avg_time
            logger.info(
                f"Progress: {i+1}/{n_samples} | "
                f"Sample time: {sample_time:.2f}s | "
                f"Avg: {avg_time:.2f}s | "
                f"ETA: {remaining/60:.1f}min"
            )

        if (i + 1) % 10 == 0:
            logger.info(f"Sample {i} - Input length: {len(prompt)} chars")
            logger.info(f"Sample {i} - Prediction: {prediction[:150]}...")
            logger.info(f"Sample {i} - Reference: {reference[:150]}...")

    generation_time = time.time() - start_time
    logger.info(f"Generated {len(all_predictions)} summaries in {generation_time:.1f}s")

    if not all_predictions:
        logger.error("No valid predictions generated")
        return None

    # Evaluate with ROUGE
    logger.info("Computing ROUGE-L scores...")
    rouge = evaluate.load("rouge")

    # Compute aggregate ROUGE-L score
    result = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True
    )
    rougeL = result.get("rougeL", 0.0)

    # Compute per-sample ROUGE-L scores
    per_sample_result = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True,
        use_aggregator=False
    )
    per_sample_rougeL = per_sample_result.get("rougeL", [])

    # Collect per-sample scores
    per_sample_scores = []
    for i, sample_id in enumerate(sample_ids):
        rl = per_sample_rougeL[i] if i < len(per_sample_rougeL) else 0.0
        per_sample_scores.append({
            "sample_id": sample_id,
            "rougeL": float(rl)
        })

    results = {
        "rougeL": float(rougeL),
        "num_samples": len(all_predictions),
        "generation_time": generation_time,
        "avg_time_per_sample": generation_time / len(all_predictions),
        "per_sample_scores": per_sample_scores
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace model on LongBench qmsum using ROUGE-L"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing HuggingFace safetensors"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 200)"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("LongBench qmsum Evaluation")
    logger.info("="*80)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Number of samples: {args.n_samples}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("="*80)

    try:
        results = evaluate_longbench(
            model_dir=args.model_dir,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens
        )

        if results:
            logger.info("="*80)
            logger.info("RESULTS")
            logger.info("="*80)
            logger.info(f"Number of samples evaluated: {results['num_samples']}")
            logger.info(f"ROUGE-L Score: {results['rougeL']:.4f}")
            logger.info(f"Generation time: {results['generation_time']:.1f}s")
            logger.info(f"Avg time per sample: {results['avg_time_per_sample']:.2f}s")
            logger.info("="*80)

            # Show some per-sample scores
            logger.info("Sample ROUGE-L scores (first 10):")
            for score_info in results['per_sample_scores'][:10]:
                logger.info(f"  Sample {score_info['sample_id']}: {score_info['rougeL']:.4f}")
            logger.info("="*80)
        else:
            logger.error("Evaluation failed")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
