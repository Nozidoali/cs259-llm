#!/usr/bin/env python3
import os
import random
import subprocess
import logging
import numpy as np

from datasets import load_dataset
import evaluate

logger = logging.getLogger(__name__)


def get_truthfulqa_score(script_path="./scripts/run-cli.sh", num_samples=100, num_tokens=25, random_seed=42, extra_args=None):
    extra_args = extra_args or []

    logger.info(f"Loading TruthfulQA dataset (samples={num_samples}, tokens={num_tokens}, seed={random_seed})...")
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except (TypeError, AttributeError) as e:
        logger.warning(f"Cache error: {e}, clearing cache...")
        import shutil
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/datasets/truthful_qa"),
            os.path.expanduser("~/.cache/huggingface/datasets/truthfulqa___truthful_qa"),
            "./data/truthfulqa_cache"
        ]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir=None)
    
    total_samples = len(ds)
    if num_samples is not None and num_samples < total_samples:
        random.seed(random_seed)
        indices = random.sample(range(total_samples), num_samples)
        indices.sort()
        ds = ds.select(indices)
        n = num_samples
        logger.info(f"Using {n}/{total_samples} samples (seed={random_seed})")
    else:
        n = total_samples
        logger.info(f"Using all {n} TruthfulQA samples")

    logger.info("Loading BLEURT evaluator...")
    bleurt = evaluate.load("bleurt", "bleurt-large-128")

    max_score_arr = []
    acc_score_arr = []

    for i, rec in enumerate(ds):
        question = rec["question"].replace('"', " ").replace("'", " ")
        correct_answers = rec["correct_answers"]
        incorrect_answers = rec["incorrect_answers"]

        cmd = ["bash", script_path, "-no-cnv", "-p", f"'{question}'", "-n", str(num_tokens)] + extra_args
        logger.debug(f"Running command for sample {i}: {' '.join(cmd)}")
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.stdout:
            logger.debug(f"Sample {i} stdout:\n{proc.stdout}")
        if proc.stderr:
            logger.debug(f"Sample {i} stderr:\n{proc.stderr}")
        
        if proc.returncode != 0:
            logger.error(f"CLI failed for sample {i} with return code {proc.returncode}")
            if proc.stderr:
                logger.error(f"Error output: {proc.stderr}")
            return None, None

        pred = proc.stdout.strip()

        predictions_true = [pred] * len(correct_answers)
        predictions_false = [pred] * len(incorrect_answers)
        score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)["scores"]
        score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)["scores"]
        max_score = max(score_true)
        acc_score = int(max(score_true) > max(score_false))

        if (i + 1) % 10 == 0 or i == n - 1:
            logger.info(f"Progress: {i+1}/{n} - max_score: {max_score:.3f}, acc: {acc_score}")
        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)

    accuracy = sum(acc_score_arr) / n
    avg_max_score = np.mean(np.array(max_score_arr))
    logger.info(f"TruthfulQA complete - Avg max_score: {avg_max_score:.3f}, Avg accuracy: {accuracy:.3f}")
    return avg_max_score, accuracy

