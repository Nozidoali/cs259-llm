#!/usr/bin/env python3

import os
import json
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import concatenate_datasets

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import WORK_DIR, QUANTIZE_LEVEL
from models import load_model_and_tokenizer, freeze_all_except_mlp
from data import prepare_truthfulqa_dataset, prepare_qmsum_dataset, download_model
from conversion import convert_to_gguf, merge_experts_to_standard_mlp
from rmoe.finetune import train_expert
from rmoe.gating import train_gating_network
from rmoe.merge import merge_experts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RMoE training pipeline")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert training")
    parser.add_argument("--skip-gating", action="store_true", help="Skip gating network training")
    parser.add_argument("--skip-merge", action="store_true", help="Skip model merging")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip full finetuning")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = json.load(f)
    if config.get("method") != "rmoe":
        logger.error("Config must specify method: 'rmoe'")
        sys.exit(1)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = WORK_DIR / "workspace" / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Work directory: {work_dir}")
    base_model = config.get("base_model", "qwen2-0.5b")
    from config import MODEL_CONFIGS
    if base_model in MODEL_CONFIGS:
        base_model_path = str(download_model(base_model))
    else:
        base_model_path = base_model
    if not base_model_path:
        logger.error("Config must specify base_model")
        sys.exit(1)
    try:
        if not args.skip_experts:
            datasets = config.get("datasets", ["truthfulqa", "longbench"])
            expert_paths = []
            expert_config = config.get("expert_training", {})
            for dataset_name in datasets:
                logger.info(f"=" * 60)
                logger.info(f"Training expert for dataset: {dataset_name}")
                logger.info(f"=" * 60)
                expert_output_dir = work_dir / "experts" / dataset_name
                expert_output_dir = train_expert(
                    base_model_path=base_model_path,
                    dataset_name=dataset_name,
                    output_dir=expert_output_dir,
                    all_datasets=datasets,
                    max_length=expert_config.get("max_length", 512),
                    num_epochs=expert_config.get("num_epochs", 3),
                    batch_size=expert_config.get("batch_size", 1),
                    gradient_accumulation_steps=expert_config.get("gradient_accumulation_steps", 4),
                    learning_rate=expert_config.get("learning_rate", 5e-5),
                    weight_decay=expert_config.get("weight_decay", 0.01),
                    eval_split=expert_config.get("eval_split", 0.2),
                    seed=config.get("seed", 42),
                    qmsum_max_new_tokens=expert_config.get("qmsum_max_new_tokens", 200),
                    temperature=expert_config.get("temperature", 0.0),
                )
                expert_paths.append(expert_output_dir)
        else:
            logger.info("Skipping expert training")
            expert_paths = [work_dir / "experts" / d for d in config.get("datasets", ["truthfulqa", "longbench"])]
        if not args.skip_gating:
            datasets = config.get("datasets", ["truthfulqa", "longbench"])
            logger.info(f"=" * 60)
            logger.info("Training gating network")
            logger.info(f"=" * 60)
            gating_config = config.get("gating", {})
            gating_output_dir = work_dir / "gating_network"
            gating_output_dir = train_gating_network(
                base_model=base_model_path,
                datasets=datasets,
                output_dir=gating_output_dir,
                hidden_dims=gating_config.get("hidden_dims", [512, 256]),
                dropout=gating_config.get("dropout", 0.1),
                learning_rate=gating_config.get("learning_rate", 1e-4),
                batch_size=gating_config.get("batch_size", 32),
                num_epochs=gating_config.get("num_epochs", 10),
                weight_decay=gating_config.get("weight_decay", 0.01),
                train_split=gating_config.get("train_split", 0.7),
                val_split=gating_config.get("val_split", 0.15),
                test_split=gating_config.get("test_split", 0.15),
                seed=config.get("seed", 42),
                prompt_dir=config.get("prompt_dir"),
            )
            gating_path = gating_output_dir
        else:
            logger.info("Skipping gating network training")
            gating_path = work_dir / "gating_network"
        if not args.skip_merge:
            logger.info(f"=" * 60)
            logger.info("Merging expert models into MoE")
            logger.info(f"=" * 60)
            merge_config = config.get("merge", {})
            rmoe_output_dir = work_dir / "rmoe_model"
            rmoe_output_dir = merge_experts(
                expert_paths=expert_paths,
                gating_model_path=gating_path,
                output_dir=rmoe_output_dir,
                base_model_path=config.get("base_model"),
                routing_mode=merge_config.get("routing_mode", "weighted_sum"),
            )
            rmoe_path = rmoe_output_dir
        else:
            logger.info("Skipping model merging")
            rmoe_path = work_dir / "rmoe_model"
        if not args.skip_finetune:
            finetune_config = config.get("full_finetune", {})
            if finetune_config.get("enabled", False):
                logger.info(f"=" * 60)
                logger.info("Full finetuning (all layers unfrozen)")
                logger.info(f"=" * 60)
                datasets = config.get("datasets", ["truthfulqa", "longbench"])
                model, tokenizer = load_model_and_tokenizer(model_path=rmoe_path, dtype=torch.float32)
                use_mps = torch.backends.mps.is_available()
                use_cuda = torch.cuda.is_available()
                device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
                
                # Clear CUDA cache before loading model
                if use_cuda:
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                
                model = model.to(device)
                for param in model.parameters():
                    param.requires_grad = True
                
                # Enable gradient checkpointing to reduce memory
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
                
                model.train()
                train_datasets = []
                for dataset_name in datasets:
                    if dataset_name == "truthfulqa":
                        ds = prepare_truthfulqa_dataset(tokenizer, max_length=finetune_config.get("max_length", 512), keep_metadata=False, model_type="causal")
                    elif dataset_name in ["longbench", "qmsum"]:
                        ds = prepare_qmsum_dataset(tokenizer, max_length=finetune_config.get("max_length", 512), keep_metadata=False, model_type="causal")
                    else:
                        continue
                    train_datasets.append(ds)
                combined_dataset = concatenate_datasets(train_datasets)
                combined_dataset = combined_dataset.train_test_split(test_size=finetune_config.get("eval_split", 0.2), seed=config.get("seed", 42))
                logger.info(f"Train samples: {len(combined_dataset['train'])}, Eval samples: {len(combined_dataset['test'])}")
                output_dir = work_dir / "rmoe_model_finetuned"
                output_dir.mkdir(parents=True, exist_ok=True)
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    overwrite_output_dir=True,
                    num_train_epochs=finetune_config.get("num_epochs", 3),
                    per_device_train_batch_size=finetune_config.get("batch_size", 1),
                    per_device_eval_batch_size=finetune_config.get("batch_size", 1),
                    gradient_accumulation_steps=finetune_config.get("gradient_accumulation_steps", 4),
                    learning_rate=finetune_config.get("learning_rate", 5e-5),
                    weight_decay=finetune_config.get("weight_decay", 0.01),
                    logging_dir=str(output_dir / "logs"),
                    logging_steps=1,  # Log every step to TensorBoard
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    fp16=use_cuda,
                    bf16=use_mps,
                    dataloader_num_workers=0 if use_mps else 2,
                    report_to=["tensorboard"],
                    seed=config.get("seed", 42),
                    gradient_checkpointing=True,  # Reduce memory usage
                    dataloader_pin_memory=False,  # Reduce memory usage
                )
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=combined_dataset["train"],
                    eval_dataset=combined_dataset["test"],
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )
                logger.info("Starting full finetuning...")
                trainer.train()
                logger.info(f"Saving finetuned model to: {output_dir}")
                trainer.save_model()
                tokenizer.save_pretrained(str(output_dir))
                final_model_path = output_dir
            else:
                logger.info("Skipping full finetuning (enabled=False)")
                final_model_path = rmoe_path
        else:
            logger.info("Skipping full finetuning")
            final_model_path = rmoe_path
        if not args.skip_convert:
            logger.info(f"=" * 60)
            logger.info("Converting to GGUF")
            logger.info(f"=" * 60)
            if config.get("gguf_output"):
                output_file = Path(config["gguf_output"])
            else:
                quantize = config.get("quantize", QUANTIZE_LEVEL)
                output_file = work_dir / f"rmoe_model_{quantize}.gguf"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            quantize_level = config.get("quantize", QUANTIZE_LEVEL)
            logger.info(f"Converting: {final_model_path}")
            logger.info(f"Output: {output_file}")
            logger.info(f"Quantization: {quantize_level}")
            standard_model_path = work_dir / "rmoe_standard"
            merge_experts_to_standard_mlp(Path(final_model_path), standard_model_path, merge_mode="average")
            convert_to_gguf(standard_model_path, output_file, quantize_level)
            logger.info(f"GGUF file saved to: {output_file}")
        else:
            logger.info("Skipping GGUF conversion")
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Work directory: {work_dir}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
