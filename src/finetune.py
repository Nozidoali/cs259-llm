#!/usr/bin/env python3

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


def prepare_qmsum_dataset(tokenizer, max_length=512, keep_metadata=False, model_type="causal", split="test", num_samples=None):
    ds = load_dataset("zai-org/LongBench", "qmsum", split=split, trust_remote_code=True)
    
    if num_samples is not None:
        num_samples = min(num_samples, len(ds))
        ds = ds.select(range(num_samples))
    
    def format_example(ex):
        context = ex.get("context", "")
        input_text = ex.get("input", "")
        answers = ex.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and answers else (answers if isinstance(answers, str) else "")
        prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
        
        if model_type == "seq2seq":
            result = {"input": prompt, "target": answer}
        else:
            result = {"text": f"{prompt}\n\nSummary: {answer}"}
        
        if keep_metadata:
            result.update({"context": context, "input": input_text, "answer": answer})
        return result
    
    formatted = ds.map(format_example)
    
    if model_type == "seq2seq":
        def tokenize_seq2seq(examples):
            inputs = tokenizer(examples["input"], truncation=True, max_length=max_length, padding="max_length")
            inputs["labels"] = tokenizer(examples["target"], truncation=True, max_length=max_length, padding="max_length")["input_ids"]
            return inputs
        
        tokenized = formatted.map(tokenize_seq2seq, batched=True, remove_columns=["input", "target"] if not keep_metadata else ["input", "target"])
    else:
        tokenized = formatted.map(
            lambda examples: tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length"),
            batched=True,
            remove_columns=["text"] if not keep_metadata else []
        )
    return tokenized
