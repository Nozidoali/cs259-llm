#!/usr/bin/env python3

import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_dataset
from huggingface_hub import snapshot_download
from config import (
    TRUTHFULQA_CACHE_DIR,
    MODELS_DIR,
    MODEL_CONFIGS,
    DATASET_CONFIG,
)


def download_model(model_key, output_dir=None):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]
    
    output_dir = MODELS_DIR / config["base_dir"] if output_dir is None else Path(output_dir)
    
    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"✓ Model already exists at: {output_dir}")
        return output_dir
    
    print(f"Downloading {model_id}...")
    print(f"Output directory: {output_dir}\n")
    
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,
    )
    print(f"✓ Successfully downloaded to: {output_dir}")
    return output_dir


def prepare_truthfulqa_dataset(tokenizer, max_length=512, keep_metadata=False, model_type="causal"):
    try:
        ds = load_dataset(
            DATASET_CONFIG["name"],
            DATASET_CONFIG["config"],
            split=DATASET_CONFIG["split"],
            cache_dir=str(TRUTHFULQA_CACHE_DIR),
        )
    except (TypeError, AttributeError, ValueError) as e:
        import shutil
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Cache error loading TruthfulQA: {e}, clearing cache and retrying...")
        
        cache_dirs = [
            str(TRUTHFULQA_CACHE_DIR),
            os.path.expanduser("~/.cache/huggingface/datasets/truthful_qa"),
            os.path.expanduser("~/.cache/huggingface/datasets/truthfulqa___truthful_qa"),
        ]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Cleared cache directory: {cache_dir}")
                except Exception as clear_error:
                    logger.warning(f"Failed to clear {cache_dir}: {clear_error}")
        
        ds = load_dataset(
            DATASET_CONFIG["name"],
            DATASET_CONFIG["config"],
            split=DATASET_CONFIG["split"],
            cache_dir=str(TRUTHFULQA_CACHE_DIR),
        )
    
    def format_example(ex):
        question = ex['question']
        best_answer = ex['best_answer']
        if model_type == "seq2seq":
            result = {"input": question, "target": best_answer}
        else:
            text = DATASET_CONFIG["format_template"].format(
                question=question,
                best_answer=best_answer
            )
            result = {"text": text}
        if keep_metadata:
            result["question"] = question
            result["best_answer"] = best_answer
            result["correct_answers"] = ex.get("correct_answers", [])
            result["incorrect_answers"] = ex.get("incorrect_answers", [])
        return result
    
    formatted = ds.map(format_example)
    
    if model_type == "seq2seq":
        def tokenize_seq2seq(examples):
            inputs = tokenizer(examples["input"], truncation=True, max_length=max_length, padding="max_length")
            targets = tokenizer(examples["target"], truncation=True, max_length=max_length, padding="max_length")
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        tokenized = formatted.map(
            tokenize_seq2seq,
            batched=True,
            remove_columns=["input", "target"] if not keep_metadata else ["input", "target"],
            desc="Tokenizing dataset",
        )
    else:
        tokenized = formatted.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"] if not keep_metadata else [],
            desc="Tokenizing dataset",
        )
    return tokenized


def prepare_qmsum_dataset(tokenizer, max_length=512, keep_metadata=False, model_type="causal", split=None, num_samples=None):
    if split is None:
        try:
            ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
        except Exception:
            ds = load_dataset("zai-org/LongBench", "qmsum", trust_remote_code=True)
            if isinstance(ds, dict):
                ds = ds.get("test") or ds.get("train") or list(ds.values())[0]
    else:
        ds = load_dataset("zai-org/LongBench", "qmsum", split=split, trust_remote_code=True)
    
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))
        ds = ds.select(range(num_samples))
    
    def format_example(ex):
        context = ex.get("context", "")
        input_text = ex.get("input", "")
        answers = ex.get("answers", [])
        
        if isinstance(answers, list) and len(answers) > 0:
            answer = answers[0]
        else:
            answer = answers if isinstance(answers, str) else ""
        
        answer = answer.strip() if isinstance(answer, str) else ""
        
        if context and input_text:
            prompt = f"{context}\n\n{input_text}"
        else:
            prompt = input_text or context
        
        if model_type == "seq2seq":
            result = {"input": prompt, "target": answer}
        else:
            result = {"text": f"{prompt}\n\nSummary: {answer}"}
        
        if keep_metadata:
            result.update({
                "context": context,
                "input": input_text,
                "answer": answer
            })
        return result
    
    formatted = ds.map(format_example, desc="Formatting QMSum examples")
    
    if model_type == "seq2seq":
        def tokenize_seq2seq(examples):
            inputs = tokenizer(
                examples["input"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            targets = tokenizer(
                examples["target"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        tokenized = formatted.map(
            tokenize_seq2seq,
            batched=True,
            remove_columns=["input", "target"] if not keep_metadata else ["input", "target"],
            desc="Tokenizing QMSum dataset (seq2seq)"
        )
    else:
        tokenized = formatted.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            ),
            batched=True,
            remove_columns=["text"] if not keep_metadata else [],
            desc="Tokenizing QMSum dataset (causal)"
        )
    return tokenized
