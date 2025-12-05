#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling

sys.path.insert(0, os.path.dirname(__file__))

from config import TRAINING_CONFIG, MODELS_DIR, DATASET_CONFIG
from finetune import prepare_truthfulqa_dataset, prepare_qmsum_dataset
from dataset_eval_trainer import DatasetEvalTrainer
from model_utils import freeze_all_except_mlp, count_parameters, load_model_and_tokenizer
from datasets import concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama model with only FFN/MLP layers trainable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Local path or HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path + '-ffn-finetuned')")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file")
    
    for key in ["max_length", "num_epochs", "batch_size", "gradient_accumulation_steps",
                "learning_rate", "weight_decay", "eval_split", "seed"]:
        default = TRAINING_CONFIG.get(key)
        parser.add_argument(f"--{key}", type=type(default) if default is not None else str,
                           default=None, help=f"Default: {default}")
    
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                       choices=["truthfulqa", "qmsum", "both"],
                       help="Dataset to use for fine-tuning (default: truthfulqa)")
    parser.add_argument("--qmsum_num_samples", type=int, default=None,
                       help="Number of QMSum samples to load (default: all samples)")
    parser.add_argument("--qmsum_max_new_tokens", type=int, default=200,
                       help="Maximum number of new tokens to generate for QMSum evaluation (default: 200)")
    
    return parser.parse_args()


def load_config(args):
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            config = json.load(f)
    
    training_config = TRAINING_CONFIG.copy()
    training_config.update(config)
    
    for key in ["max_length", "num_epochs", "batch_size", "gradient_accumulation_steps",
                "learning_rate", "weight_decay", "eval_split", "seed"]:
        value = getattr(args, key, None)
        if value is not None:
            training_config[key] = value
    
    if args.model_path:
        config["model_path"] = args.model_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.dataset:
        config["dataset"] = args.dataset
    if args.qmsum_num_samples is not None:
        config["qmsum_num_samples"] = args.qmsum_num_samples
    if args.qmsum_max_new_tokens is not None:
        config["qmsum_max_new_tokens"] = args.qmsum_max_new_tokens
    
    return config, training_config


def get_output_dir(config, model_path):
    if config.get("output_dir"):
        return Path(config["output_dir"])
    
    if "/" in model_path and not Path(model_path).exists():
        model_name = model_path.split("/")[-1]
        return MODELS_DIR / f"{model_name}-ffn-finetuned"
    else:
        return Path(model_path).parent / f"{Path(model_path).name}-ffn-finetuned"


def get_metric_for_best_model(dataset_name):
    if dataset_name == "truthfulqa":
        return "eval_bleurt_max_score"
    elif dataset_name == "qmsum":
        return "eval_rougeL"
    elif dataset_name == "both":
        return "eval_bleurt_max_score"
    else:
        return "eval_loss"


def get_greater_is_better(dataset_name):
    if dataset_name in ["truthfulqa", "qmsum", "both"]:
        return True
    else:
        return False


def main():
    args = parse_args()
    config, training_config = load_config(args)
    
    model_path = config["model_path"]
    output_dir = get_output_dir(config, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FFN-Only Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}\n")
    
    model, tokenizer = load_model_and_tokenizer(model_path=model_path, dtype=torch.float32)
    
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    
    model = model.to(device)
    
    logger.info("Freezing all parameters except MLP layers and output head...")
    freeze_all_except_mlp(model)
    
    model.train()
    
    logger.info("Parameter counts:")
    param_counts = count_parameters(model)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check model architecture.")
    logger.info("")
    
    dataset_name = config.get("dataset", "truthfulqa")
    logger.info(f"Preparing dataset: {dataset_name}...")
    
    keep_metadata_for_eval = True
    if dataset_name == "truthfulqa":
        dataset = prepare_truthfulqa_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=keep_metadata_for_eval,
            model_type="causal"
        )
    elif dataset_name == "qmsum":
        dataset = prepare_qmsum_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=keep_metadata_for_eval,
            model_type="causal",
            num_samples=config.get("qmsum_num_samples")
        )
    elif dataset_name == "both":
        truthfulqa_ds = prepare_truthfulqa_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=keep_metadata_for_eval,
            model_type="causal"
        )
        qmsum_ds = prepare_qmsum_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=keep_metadata_for_eval,
            model_type="causal",
            num_samples=config.get("qmsum_num_samples")
        )
        dataset = concatenate_datasets([truthfulqa_ds, qmsum_ds])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = dataset.train_test_split(
        test_size=training_config["eval_split"],
        seed=training_config["seed"]
    )
    
    logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['test'])}\n")
    
    cs259_llm_dir = Path(__file__).parent.parent
    tmp_base_dir = cs259_llm_dir / "tmp"
    tmp_base_dir.mkdir(parents=True, exist_ok=True)
    
    if "/" in model_path and not Path(model_path).exists():
        model_name = model_path.split("/")[-1]
    else:
        model_name = Path(model_path).name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tmp_base_dir / f"{model_name}-{timestamp}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping inputs and outputs to: {tmp_dir}")
    
    def dump_dataset_samples(dataset_split, split_name):
        dump_dir = tmp_dir / split_name
        dump_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, example in enumerate(dataset_split):
            sample_data = {}
            
            if dataset_name == "qmsum":
                sample_data["context"] = example.get("context", "")
                sample_data["input"] = example.get("input", "")
                sample_data["answer"] = example.get("answer", "")
                sample_data["text"] = example.get("text", "")
            elif dataset_name == "truthfulqa":
                sample_data["question"] = example.get("question", "")
                sample_data["best_answer"] = example.get("best_answer", "")
                sample_data["text"] = example.get("text", "")
            elif dataset_name == "both":
                if "context" in example:
                    sample_data["context"] = example.get("context", "")
                    sample_data["input"] = example.get("input", "")
                    sample_data["answer"] = example.get("answer", "")
                if "question" in example:
                    sample_data["question"] = example.get("question", "")
                    sample_data["best_answer"] = example.get("best_answer", "")
                sample_data["text"] = example.get("text", "")
            
            sample_data["input_ids"] = example.get("input_ids", []).tolist() if hasattr(example.get("input_ids", []), "tolist") else example.get("input_ids", [])
            sample_data["attention_mask"] = example.get("attention_mask", []).tolist() if hasattr(example.get("attention_mask", []), "tolist") else example.get("attention_mask", [])
            
            with open(dump_dir / f"sample_{idx}.json", "w") as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dumped {len(dataset_split)} {split_name} samples to {dump_dir}")
    
    dump_dataset_samples(dataset["train"], "train")
    dump_dataset_samples(dataset["test"], "eval")
    
    logger.info(f"Using device: {'cuda' if use_cuda else 'mps' if use_mps else 'cpu'}\n")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        logging_dir=str(Path("logs")),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=get_metric_for_best_model(dataset_name),
        greater_is_better=get_greater_is_better(dataset_name),
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to="none",
        seed=training_config["seed"],
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = DatasetEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        eval_dataset_with_answers=dataset["test"],
        model_type="causal",
        dataset_type=dataset_name,
        qmsum_max_new_tokens=config.get("qmsum_max_new_tokens", 200),
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {training_config['num_epochs']}, "
                f"Batch size: {training_config['batch_size']}, "
                f"LR: {training_config['learning_rate']}\n")
    
    logger.info("Starting training...")
    logger.info("=" * 60)
    trainer.train()
    
    logger.info("=" * 60)
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("=" * 60)
    logger.info("Generating outputs for evaluation dataset...")
    model.eval()
    output_dir_txt = tmp_dir / "outputs"
    output_dir_txt.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    max_new_tokens = config.get("qmsum_max_new_tokens", 200) if dataset_name == "qmsum" else 50
    
    for idx, example in enumerate(dataset["test"]):
        if dataset_name == "qmsum":
            context = example.get("context", "")
            input_text = example.get("input", "")
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            prompt = f"{prompt}\n\nSummary:"
        elif dataset_name == "truthfulqa":
            question = example.get("question", "")
            prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
        else:
            if "context" in example:
                context = example.get("context", "")
                input_text = example.get("input", "")
                prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
                prompt = f"{prompt}\n\nSummary:"
            else:
                question = example.get("question", "")
                prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        if dataset_name == "qmsum":
            output_filename = f"qmsum_test_{idx}.txt"
        elif dataset_name == "truthfulqa":
            output_filename = f"truthfulqa_test_{idx}.txt"
        else:
            if "context" in example:
                output_filename = f"qmsum_test_{idx}.txt"
            else:
                output_filename = f"truthfulqa_test_{idx}.txt"
        
        output_path = output_dir_txt / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Generated outputs for {idx + 1}/{len(dataset['test'])} samples")
    
    logger.info(f"Generated outputs saved to: {output_dir_txt}")
    
    param_info = {
        "parameter_counts": param_counts,
        "training_config": training_config,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "tmp_dir": str(tmp_dir),
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(param_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("✓ Fine-tuning complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Input/output dumps saved to: {tmp_dir}")
    logger.info(f"Generated outputs saved to: {output_dir_txt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
