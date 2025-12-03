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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, os.path.dirname(__file__))

from config import TRAINING_CONFIG, MODELS_DIR
from finetune import prepare_truthfulqa_dataset, prepare_qmsum_dataset
from bleurt_trainer import BLEURTTrainer
from datasets import concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def freeze_all_except_mlp(model):
    for param in model.parameters():
        param.requires_grad = False
    
    mlp_params_count = 0
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama, Qwen, etc.
        layers = model.model.layers
        for layer in layers:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
    elif hasattr(model, 'layers'):  # Some models have layers directly
        layers = model.layers
        for layer in layers:
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # GPT-2
        layers = model.transformer.h
        for layer in layers:
            if hasattr(layer, 'mlp'):  # Some GPT-2 variants
                for param in layer.mlp.parameters():
                    param.requires_grad = True
                    mlp_params_count += 1
            else:  # Standard GPT-2 uses c_fc and c_proj
                if hasattr(layer, 'c_fc'):
                    for param in layer.c_fc.parameters():
                        param.requires_grad = True
                        mlp_params_count += 1
                if hasattr(layer, 'c_proj'):
                    for param in layer.c_proj.parameters():
                        param.requires_grad = True
                        mlp_params_count += 1
    else:
        raise ValueError(f"Unsupported model architecture. Model type: {type(model).__name__}, attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    logger.info(f"Unfrozen {mlp_params_count} MLP parameters")
    return mlp_params_count


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percentage": 100 * trainable_params / total_params
    }


def load_model_and_tokenizer(model_path_or_id):
    local_path = Path(model_path_or_id)
    if local_path.exists() and local_path.is_dir():
        model_id = str(local_path)
        logger.info(f"Loading model from local path: {model_id}")
    else:
        model_id = model_path_or_id
        logger.info(f"Loading model from HuggingFace: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    
    return model, tokenizer


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
    
    parser.add_argument("--use_bleurt", action="store_true",
                       help="Use BLEURT evaluation during training")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                       choices=["truthfulqa", "qmsum", "both"],
                       help="Dataset to use for fine-tuning (default: truthfulqa)")
    parser.add_argument("--qmsum_num_samples", type=int, default=None,
                       help="Number of QMSum samples to load (default: all samples)")
    
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
    if args.use_bleurt:
        config["use_bleurt"] = True
    if args.dataset:
        config["dataset"] = args.dataset
    if args.qmsum_num_samples is not None:
        config["qmsum_num_samples"] = args.qmsum_num_samples
    
    return config, training_config


def get_output_dir(config, model_path):
    if config.get("output_dir"):
        return Path(config["output_dir"])
    
    if "/" in model_path and not Path(model_path).exists():
        model_name = model_path.split("/")[-1]
        return MODELS_DIR / f"{model_name}-ffn-finetuned"
    else:
        return Path(model_path).parent / f"{Path(model_path).name}-ffn-finetuned"


def main():
    args = parse_args()
    config, training_config = load_config(args)
    
    use_bleurt = config.get("use_bleurt", False)
    model_path = config["model_path"]
    output_dir = get_output_dir(config, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FFN-Only Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}\n")
    
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    logger.info("Freezing all parameters except MLP layers...")
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
    
    if dataset_name == "truthfulqa":
        dataset = prepare_truthfulqa_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=use_bleurt,
            model_type="causal"
        )
    elif dataset_name == "qmsum":
        num_samples = config.get("qmsum_num_samples")
        if num_samples is not None:
            logger.info(f"Loading {num_samples} QMSum samples...")
        dataset = prepare_qmsum_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=use_bleurt,
            model_type="causal",
            num_samples=num_samples
        )
    elif dataset_name == "both":
        logger.info("Loading TruthfulQA dataset...")
        truthfulqa_ds = prepare_truthfulqa_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=use_bleurt,
            model_type="causal"
        )
        logger.info("Loading QMSum dataset...")
        num_samples = config.get("qmsum_num_samples")
        if num_samples is not None:
            logger.info(f"Loading {num_samples} QMSum samples...")
        qmsum_ds = prepare_qmsum_dataset(
            tokenizer,
            max_length=training_config["max_length"],
            keep_metadata=use_bleurt,
            model_type="causal",
            num_samples=num_samples
        )
        logger.info(f"Combining datasets: {len(truthfulqa_ds)} TruthfulQA + {len(qmsum_ds)} QMSum samples")
        dataset = concatenate_datasets([truthfulqa_ds, qmsum_ds])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = dataset.train_test_split(
        test_size=training_config["eval_split"],
        seed=training_config["seed"]
    )
    
    logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['test'])}\n")
    
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
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
        metric_for_best_model="eval_bleurt_score" if use_bleurt else "eval_loss",
        greater_is_better=True if use_bleurt else False,
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to="none",
        seed=training_config["seed"],
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer_class = BLEURTTrainer if use_bleurt else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["test"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }
    
    if use_bleurt:
        trainer_kwargs["eval_dataset_with_answers"] = dataset["test"]
        trainer_kwargs["model_type"] = "causal"
    
    trainer = trainer_class(**trainer_kwargs)
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {training_config['num_epochs']}, "
                f"Batch size: {training_config['batch_size']}, "
                f"LR: {training_config['learning_rate']}, "
                f"BLEURT: {use_bleurt}\n")
    
    logger.info("Starting training...")
    logger.info("=" * 60)
    trainer.train()
    
    logger.info("=" * 60)
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    param_info = {
        "parameter_counts": param_counts,
        "training_config": training_config,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(param_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("✓ Fine-tuning complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
