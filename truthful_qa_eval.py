#!/usr/bin/env python3
"""
TruthfulQA Evaluation Script

Evaluates a HuggingFace model on the TruthfulQA dataset using BLEURT scoring.
Runs on all 817 validation samples from the TruthfulQA dataset.

Usage:
    python truthful_qa_eval.py --model_dir <path_to_model_dir>

Example:
    python truthful_qa_eval.py --model_dir ./models/qwen2-0.5b-instruct-finetuned
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


def generate_prediction(model, tokenizer, question, device, max_new_tokens=50):
    """Generate a prediction for a given question."""
    # Format the prompt
    prompt = f"Question: {question}\nAnswer:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return prediction


def evaluate_truthfulqa(model_dir, max_new_tokens=50, batch_size=32):
    """
    Evaluate a model on TruthfulQA using BLEURT scoring.

    Args:
        model_dir: Path to the model directory containing safetensors
        max_new_tokens: Maximum number of tokens to generate per answer
        batch_size: Batch size for BLEURT evaluation

    Returns:
        Dictionary with avg_max_score and accuracy
    """
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    # Load TruthfulQA dataset
    logger.info("Loading TruthfulQA dataset...")
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    total_samples = len(ds)
    logger.info(f"Loaded {total_samples} test samples for TruthfulQA")

    # Generate predictions for all samples
    all_predictions = []
    all_correct_refs = []
    all_incorrect_refs = []

    logger.info("Generating predictions...")
    start_time = time.time()

    for i, record in enumerate(ds):
        question = record.get("question", "")
        correct_answers = record.get("correct_answers", [])
        incorrect_answers = record.get("incorrect_answers", [])

        if not question or not correct_answers or not incorrect_answers:
            logger.warning(f"Skipping sample {i} due to missing data")
            continue

        # Generate prediction
        sample_start = time.time()
        prediction = generate_prediction(model, tokenizer, question, device, max_new_tokens)
        sample_time = time.time() - sample_start

        all_predictions.append(prediction)
        all_correct_refs.append(correct_answers)
        all_incorrect_refs.append(incorrect_answers)

        # Log progress
        if (i + 1) % 10 == 0:
            avg_time = (time.time() - start_time) / (i + 1)
            remaining = (total_samples - i - 1) * avg_time
            logger.info(
                f"Progress: {i+1}/{total_samples} | "
                f"Sample time: {sample_time:.2f}s | "
                f"Avg: {avg_time:.2f}s | "
                f"ETA: {remaining/60:.1f}min"
            )

        if (i + 1) % 50 == 0:
            logger.info(f"Sample {i} - Q: {question[:50]}...")
            logger.info(f"Sample {i} - A: {prediction[:100]}...")

    generation_time = time.time() - start_time
    logger.info(f"Generated {len(all_predictions)} predictions in {generation_time:.1f}s")

    if not all_predictions:
        logger.error("No valid predictions generated")
        return None

    # Evaluate with BLEURT
    logger.info("Loading BLEURT model for evaluation...")
    bleurt = evaluate.load("bleurt", "bleurt-large-128")

    max_score_arr = []
    acc_score_arr = []

    logger.info("Computing BLEURT scores...")
    eval_start = time.time()

    # Process in batches to avoid memory issues
    for i in range(0, len(all_predictions), batch_size):
        batch_preds = all_predictions[i:i+batch_size]
        batch_correct = all_correct_refs[i:i+batch_size]
        batch_incorrect = all_incorrect_refs[i:i+batch_size]

        # Expand predictions and references for BLEURT
        # Each prediction is compared against all correct/incorrect answers
        expanded_preds_true = []
        expanded_refs_true = []
        expanded_preds_false = []
        expanded_refs_false = []

        for pred, correct_refs, incorrect_refs in zip(batch_preds, batch_correct, batch_incorrect):
            # Compare prediction against each correct answer
            for ref in correct_refs:
                expanded_preds_true.append(pred)
                expanded_refs_true.append(ref)

            # Compare prediction against each incorrect answer
            for ref in incorrect_refs:
                expanded_preds_false.append(pred)
                expanded_refs_false.append(ref)

        # Compute BLEURT scores
        scores_true = bleurt.compute(
            predictions=expanded_preds_true,
            references=expanded_refs_true
        )["scores"] if expanded_preds_true else []

        scores_false = bleurt.compute(
            predictions=expanded_preds_false,
            references=expanded_refs_false
        )["scores"] if expanded_preds_false else []

        # Aggregate scores per sample
        true_idx = 0
        false_idx = 0

        for correct_refs, incorrect_refs in zip(batch_correct, batch_incorrect):
            # Get max score against correct answers
            num_correct = len(correct_refs)
            example_scores_true = scores_true[true_idx:true_idx+num_correct]
            true_idx += num_correct

            # Get max score against incorrect answers
            num_incorrect = len(incorrect_refs)
            example_scores_false = scores_false[false_idx:false_idx+num_incorrect]
            false_idx += num_incorrect

            # Compute metrics
            max_score = max(example_scores_true) if example_scores_true else 0.0

            # Accuracy: 1 if best match with correct > best match with incorrect
            if example_scores_true and example_scores_false:
                acc_score = int(max(example_scores_true) > max(example_scores_false))
            else:
                acc_score = 0

            max_score_arr.append(max_score)
            acc_score_arr.append(acc_score)

        if (i + batch_size) % 100 == 0:
            logger.info(f"Evaluated {min(i+batch_size, len(all_predictions))}/{len(all_predictions)} samples")

    eval_time = time.time() - eval_start
    logger.info(f"BLEURT evaluation completed in {eval_time:.1f}s")

    # Compute final metrics
    avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
    accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0

    results = {
        "avg_max_score": float(avg_max_score),
        "accuracy": float(accuracy),
        "num_samples": len(all_predictions),
        "generation_time": generation_time,
        "eval_time": eval_time,
        "total_time": generation_time + eval_time
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace model on TruthfulQA using BLEURT"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing HuggingFace safetensors"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for BLEURT evaluation (default: 32)"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("TruthfulQA Evaluation")
    logger.info("="*80)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"BLEURT batch size: {args.batch_size}")
    logger.info("="*80)

    try:
        results = evaluate_truthfulqa(
            model_dir=args.model_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )

        if results:
            logger.info("="*80)
            logger.info("RESULTS")
            logger.info("="*80)
            logger.info(f"Number of samples evaluated: {results['num_samples']}")
            logger.info(f"Average max score: {results['avg_max_score']:.4f}")
            logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            logger.info(f"Generation time: {results['generation_time']:.1f}s")
            logger.info(f"BLEURT eval time: {results['eval_time']:.1f}s")
            logger.info(f"Total time: {results['total_time']:.1f}s ({results['total_time']/60:.1f}min)")
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
