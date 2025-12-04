#!/usr/bin/env python3

import os
import json
import argparse
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    MODELS_DIR,
    GGUF_OUTPUT_DIR,
    MODEL_CONFIGS,
    TRAINING_CONFIG,
    QUANTIZE_LEVEL,
)
from finetune import (
    download_model,
    prepare_truthfulqa_dataset,
)
from model_utils import load_model_and_tokenizer
from convert import convert_to_gguf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_finetune(config):
    """Run the fine-tuning step"""
    logger.info("=" * 60)
    logger.info("Step 1: Fine-tuning")
    logger.info("=" * 60)
    
    model_key = config["model"]
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_key]
    
    # Load model and tokenizer
    model_path = config.get("model_path")
    model, tokenizer, default_output_dir, model_type = load_model_and_tokenizer(
        model_key, model_path
    )
    
    # Use custom output dir if specified, otherwise use default
    output_dir = Path(config.get("output_dir", default_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge training config
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        k: v for k, v in config.items() 
        if k in ["max_length", "num_epochs", "batch_size", 
                 "gradient_accumulation_steps", "learning_rate", "weight_decay",
                 "eval_split", "seed"]
    })
    
    use_bleurt = config.get("use_bleurt", False)
    
    logger.info(f"Fine-tuning {model_config['display_name']} on TruthfulQA")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max length: {training_config['max_length']}, Epochs: {training_config['num_epochs']}")
    logger.info(f"Batch size: {training_config['batch_size']}, "
                f"Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}")
    logger.info(f"Effective batch size: {training_config['batch_size'] * training_config.get('gradient_accumulation_steps', 1)}")
    logger.info(f"LR: {training_config['learning_rate']}, Using BLEURT: {use_bleurt}")
    
    # Prepare dataset
    dataset = prepare_truthfulqa_dataset(
        tokenizer, 
        max_length=training_config["max_length"], 
        keep_metadata=use_bleurt, 
        model_type=model_type
    )
    dataset = dataset.train_test_split(
        test_size=training_config["eval_split"], 
        seed=training_config["seed"]
    )
    
    logger.info(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")
    
    # Import training dependencies
    import torch
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq,
    )
    from dataset_eval_trainer import DatasetEvalTrainer
    
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    # Setup training arguments
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
        metric_for_best_model="eval_bleurt_max_score" if use_bleurt else "eval_loss",
        greater_is_better=True if use_bleurt else False,
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to="none",
        seed=training_config["seed"],
    )
    
    # Setup trainer
    trainer_class = DatasetEvalTrainer if use_bleurt else Trainer
    data_collator = (
        DataCollatorForSeq2Seq(tokenizer=tokenizer) if model_type == "seq2seq" 
        else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
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
        trainer_kwargs["model_type"] = model_type
        trainer_kwargs["dataset_type"] = "truthfulqa"
    
    trainer = trainer_class(**trainer_kwargs)
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info("✓ Fine-tuning complete!")
    return output_dir


def run_convert(config, finetuned_model_path):
    """Run the conversion step"""
    logger.info("=" * 60)
    logger.info("Step 2: Converting to GGUF")
    logger.info("=" * 60)
    
    model_key = config["model"]
    model_config = MODEL_CONFIGS[model_key]
    
    if not model_config.get("supports_gguf", True):
        logger.warning(f"Model {model_config['display_name']} does not support GGUF conversion. Skipping conversion step.")
        return None
    
    # Determine output file
    if config.get("gguf_output"):
        output_file = Path(config["gguf_output"])
    else:
        quantize = config.get("quantize", QUANTIZE_LEVEL)
        output_file = GGUF_OUTPUT_DIR / f"{finetuned_model_path.name}-{quantize}.gguf"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    quantize_level = config.get("quantize", QUANTIZE_LEVEL)
    
    logger.info(f"Converting: {finetuned_model_path}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Quantization: {quantize_level}")
    
    convert_to_gguf(finetuned_model_path, output_file, quantize_level)
    
    logger.info("✓ Conversion complete!")
    return output_file


def run_push(config, gguf_file_path):
    """Run the ADB push step"""
    logger.info("=" * 60)
    logger.info("Step 3: Pushing to device")
    logger.info("=" * 60)
    
    if not config.get("push_to_device", False):
        logger.info("Skipping push step (push_to_device is False)")
        return
    
    if gguf_file_path is None or not gguf_file_path.exists():
        logger.error("No GGUF file to push. Skipping push step.")
        return
    
    push_script = Path(__file__).parent / "scripts" / "push-model.sh"
    if not push_script.exists():
        logger.error(f"Push script not found: {push_script}")
        return
    
    device_path = config.get("device_path", "/data/local/tmp/gguf/")
    adb_serial = config.get("adb_serial")
    
    logger.info(f"Pushing {gguf_file_path.name} to device")
    logger.info(f"Device path: {device_path}")
    if adb_serial:
        logger.info(f"ADB serial: {adb_serial}")
    
    # Build command
    cmd = ["bash", str(push_script), str(gguf_file_path), device_path]
    if adb_serial:
        cmd.extend(["--serial", adb_serial])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✓ Push complete!")
        logger.info(result.stdout)
    else:
        logger.error(f"Push failed with return code {result.returncode}")
        logger.error(result.stderr)
        raise RuntimeError(f"ADB push failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete training pipeline: fine-tune, convert, and push",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON config:
{
    "model": "qwen2-0.5b",
    "output_dir": "models/qwen2-0.5b-instruct-finetuned",
    "max_length": 512,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "use_bleurt": false,
    "quantize": "Q4_0",
    "gguf_output": "models/gguf/qwen2-0.5b-finetuned-Q4_0.gguf",
    "push_to_device": true,
    "device_path": "/data/local/tmp/gguf/",
    "adb_serial": null
}
        """
    )
    parser.add_argument(
        "config",
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning step"
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion step"
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip push step"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    logger.info(f"Starting training pipeline with config: {args.config}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    finetuned_model_path = None
    gguf_file_path = None
    
    try:
        # Step 1: Fine-tuning
        if not args.skip_finetune:
            finetuned_model_path = run_finetune(config)
        else:
            logger.info("Skipping fine-tuning step")
            # Try to infer path from config
            if config.get("output_dir"):
                finetuned_model_path = Path(config["output_dir"])
            else:
                model_key = config["model"]
                finetuned_model_path = MODELS_DIR / MODEL_CONFIGS[model_key]["finetuned_dir"]
        
        # Step 2: Conversion
        if not args.skip_convert and finetuned_model_path and finetuned_model_path.exists():
            gguf_file_path = run_convert(config, finetuned_model_path)
        else:
            logger.info("Skipping conversion step")
            # Try to infer path from config
            if config.get("gguf_output"):
                gguf_file_path = Path(config["gguf_output"])
        
        # Step 3: Push to device
        if not args.skip_push:
            run_push(config, gguf_file_path)
        else:
            logger.info("Skipping push step")
        
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info("=" * 60)
        if finetuned_model_path:
            logger.info(f"Fine-tuned model: {finetuned_model_path}")
        if gguf_file_path:
            logger.info(f"GGUF file: {gguf_file_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

