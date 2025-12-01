#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from huggingface_hub import snapshot_download
from config import (
    TRAINING_CONFIG,
    LOGS_DIR,
    TRUTHFULQA_CACHE_DIR,
    MODELS_DIR,
    MODEL_CONFIGS,
    DATASET_CONFIG,
)
from bleurt_trainer import BLEURTTrainer


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


def load_model_and_tokenizer(model_key, model_path=None):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    
    model_path = download_model(model_key) if model_path is None else Path(model_path)
    
    print(f"Loading {config['display_name']} from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    model_type = config.get("model_type", "causal")
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    output_dir = MODELS_DIR / config["finetuned_dir"]
    return model, tokenizer, output_dir, model_type


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on TruthfulQA")
    parser.add_argument("--model", type=str, default="qwen2-0.5b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--use-bleurt", action="store_true", help="Use BLEURT score for model selection")
    
    args = parser.parse_args()
    
    model, tokenizer, default_output_dir, model_type = load_model_and_tokenizer(args.model, args.model_path)
    
    if args.download_only:
        print("\n✓ Download complete. Run without --download-only to fine-tune.")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = TRAINING_CONFIG.copy()
    if args.max_length: config["max_length"] = args.max_length
    if args.num_epochs: config["num_epochs"] = args.num_epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps: config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate: config["learning_rate"] = args.learning_rate
    
    model_display_name = MODEL_CONFIGS[args.model]["display_name"]
    print(f"\nFine-tuning {model_display_name} on TruthfulQA")
    print(f"Output: {output_dir}")
    print(f"Max length: {config['max_length']}, Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}, Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {config['batch_size'] * config.get('gradient_accumulation_steps', 1)}")
    print(f"LR: {config['learning_rate']}, Using BLEURT: {args.use_bleurt}\n")
    
    dataset = prepare_truthfulqa_dataset(tokenizer, max_length=config["max_length"], keep_metadata=args.use_bleurt, model_type=model_type)
    dataset = dataset.train_test_split(test_size=config["eval_split"], seed=config["seed"])
    
    eval_dataset_with_answers = dataset["test"] if args.use_bleurt else None
    
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}\n")
    
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_dir=str(LOGS_DIR),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleurt_score" if args.use_bleurt else "eval_loss",
        greater_is_better=True if args.use_bleurt else False,
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to="none",
        seed=config["seed"],
    )
    
    trainer_class = BLEURTTrainer if args.use_bleurt else Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) if model_type == "seq2seq" else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["test"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }
    if args.use_bleurt:
        trainer_kwargs["eval_dataset_with_answers"] = eval_dataset_with_answers
        trainer_kwargs["model_type"] = model_type
    trainer = trainer_class(**trainer_kwargs)
    
    print("Starting training...\n")
    trainer.train()
    
    print(f"\nSaving to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    model_config = MODEL_CONFIGS[args.model]
    if model_config.get("supports_gguf", True):
        print(f"\n✓ Complete! Convert with: python src/convert.py --model {output_dir}")
    else:
        print(f"\n✓ Complete! Note: {model_config['display_name']} does not support GGUF conversion.")


if __name__ == "__main__":
    main()
