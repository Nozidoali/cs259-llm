#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from huggingface_hub import snapshot_download
import evaluate
import numpy as np
from config import (
    TRAINING_CONFIG,
    LOGS_DIR,
    TRUTHFULQA_CACHE_DIR,
    MODELS_DIR,
)


def download_qwen2(model_size="0.5B", output_dir=None):
    if model_size not in ["0.5B", "1.5B"]:
        raise ValueError("model_size must be '0.5B' or '1.5B'")
    
    model_id = f"Qwen/Qwen2-{model_size}-Instruct"
    
    if output_dir is None:
        output_dir = MODELS_DIR / f"qwen2-{model_size.lower()}-instruct"
    else:
        output_dir = Path(output_dir)
    
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
    )
    print(f"✓ Successfully downloaded to: {output_dir}")
    return output_dir


def prepare_truthfulqa_dataset(tokenizer, max_length=512, keep_metadata=False):
    ds = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
        cache_dir=str(TRUTHFULQA_CACHE_DIR),
    )
    
    def format_example(ex):
        question = ex['question']
        best_answer = ex['best_answer']
        text = f"Question: {question}\nAnswer: {best_answer}"
        result = {"text": text}
        if keep_metadata:
            result["question"] = question
            result["best_answer"] = best_answer
        return result
    
    formatted = ds.map(format_example)
    
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


def load_model_and_tokenizer(model_size, model_path=None):
    size = "0.5B" if "0.5" in model_size else "1.5B"
    if model_path is None:
        model_path = download_qwen2(size)
    else:
        model_path = Path(model_path)
    
    print(f"Loading Qwen2 {size} Instruct from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    output_dir = MODELS_DIR / f"qwen2-{size.lower()}-instruct-finetuned"
    return model, tokenizer, output_dir


class BLEURTTrainer(Trainer):
    def __init__(self, *args, eval_dataset_with_answers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_with_answers = eval_dataset_with_answers
        self.bleurt = evaluate.load("bleurt", "bleurt-large-128")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if self.eval_dataset_with_answers is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        self.model.eval()
        bleurt_scores = []
        
        print("Computing BLEURT scores...")
        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(eval_dataset)}")
            
            question = example.get("question", "")
            best_answer = example.get("best_answer", "")
            
            if not question or not best_answer:
                continue
            
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in generated:
                answer = generated.split("Answer:")[-1].strip()
            else:
                answer = generated[len(prompt):].strip()
            
            if answer and best_answer:
                score = self.bleurt.compute(predictions=[answer], references=[best_answer])["scores"][0]
                bleurt_scores.append(score)
        
        avg_bleurt = np.mean(bleurt_scores) if bleurt_scores else 0.0
        
        metrics = {f"{metric_key_prefix}_bleurt_score": avg_bleurt}
        self.log(metrics)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 models on TruthfulQA")
    parser.add_argument("--model", type=str, default="qwen2-0.5b", choices=["qwen2-0.5b", "qwen2-1.5b"])
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
    
    model, tokenizer, default_output_dir = load_model_and_tokenizer(args.model, args.model_path)
    
    if args.download_only:
        print("\n✓ Download complete. Run without --download-only to fine-tune.")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = TRAINING_CONFIG.copy()
    if args.max_length:
        config["max_length"] = args.max_length
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    
    print(f"\nFine-tuning {args.model.upper()} on TruthfulQA")
    print(f"Output: {output_dir}")
    print(f"Max length: {config['max_length']}, Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}, Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    print(f"Effective batch size: {config['batch_size'] * config.get('gradient_accumulation_steps', 1)}")
    print(f"LR: {config['learning_rate']}, Using BLEURT: {args.use_bleurt}\n")
    
    dataset = prepare_truthfulqa_dataset(tokenizer, max_length=config["max_length"], keep_metadata=args.use_bleurt)
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
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset_with_answers=eval_dataset_with_answers if args.use_bleurt else None,
        tokenizer=tokenizer,
    )
    
    print("Starting training...\n")
    trainer.train()
    
    print(f"\nSaving to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    print("\n✓ Complete! Convert with: python convert.py --model", output_dir)


if __name__ == "__main__":
    main()
