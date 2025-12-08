#!/usr/bin/env python3

import os
import json
import argparse
import sys
import logging
import gc
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
from rmoe.gating import train_gating_network, train_shared_expert_gating
from rmoe.merge import merge_experts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="RMoE training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conversion modes:
  preserve_moe - Keep all experts with routing (NEW, recommended)
  average      - Average experts into single model (OLD, simpler)

Examples:
  # Full pipeline with MoE preservation
  python train.py config.json --conversion-mode preserve_moe
  
  # Full pipeline with expert averaging (original behavior)
  python train.py config.json --conversion-mode average
  
  # Skip training, only convert existing model
  python train.py config.json --skip-experts --skip-gating --conversion-mode preserve_moe
        """
    )
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert training")
    parser.add_argument("--skip-gating", action="store_true", help="Skip gating network training")
    parser.add_argument("--skip-merge", action="store_true", help="Skip model merging")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip full finetuning")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    parser.add_argument(
        "--conversion-mode",
        choices=["preserve_moe", "average"],
        default="average",
        help="Conversion mode: 'preserve_moe' (keep all experts) or 'average' (average experts). Default: average (backward compatible)"
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = json.load(f)
    if config.get("method") != "rmoe":
        logger.error("Config must specify method: 'rmoe'")
        sys.exit(1)
    
    # Get timestamp from environment variable or generate one
    timestamp = os.getenv("WORKSPACE_TIMESTAMP")
    if not timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.warning(f"WORKSPACE_TIMESTAMP environment variable is not set. Auto-generated timestamp: {timestamp}")
    else:
        if len(timestamp) != 15 or timestamp[8] != '_':
            logger.warning(f"Timestamp format may be incorrect. Expected: YYYYMMDD_HHMMSS, got: {timestamp}")
        logger.info(f"Using timestamp from WORKSPACE_TIMESTAMP environment variable: {timestamp}")
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
        datasets = config.get("datasets", ["truthfulqa", "longbench"])
        expert_paths = []
        expert_config = config.get("expert_training", {})
        
        if not args.skip_experts:
            for dataset_name in datasets:
                expert_output_dir = work_dir / "experts" / dataset_name
                expert_exists = expert_output_dir.exists() and (expert_output_dir / "config.json").exists()
                
                if expert_exists:
                    logger.info(f"=" * 60)
                    logger.info(f"Expert for dataset '{dataset_name}' already exists at {expert_output_dir}")
                    logger.info(f"Skipping training and using existing expert")
                    logger.info(f"=" * 60)
                else:
                    logger.info(f"=" * 60)
                    logger.info(f"Training expert for dataset: {dataset_name}")
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
                        l2_regularization=expert_config.get("l2_regularization", 0.0),
                        max_grad_norm=expert_config.get("max_grad_norm", 1.0),
                        disable_eval_split=expert_config.get("disable_eval_split", False),
                        eval_split=expert_config.get("eval_split", 0.2),
                        seed=config.get("seed", 42),
                        qmsum_max_new_tokens=expert_config.get("qmsum_max_new_tokens", 200),
                        temperature=expert_config.get("temperature", 0.0),
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                
                expert_paths.append(expert_output_dir)
        else:
            logger.info("Skipping expert training (--skip-experts flag)")
            expert_paths = [work_dir / "experts" / d for d in datasets]
        
        # Convert individual expert models to GGUF before merging
        if not args.skip_convert:
            logger.info(f"=" * 60)
            logger.info("Converting individual expert models to GGUF")
            logger.info(f"=" * 60)
            quantize_level = config.get("quantize", QUANTIZE_LEVEL)
            for expert_path, dataset_name in zip(expert_paths, datasets):
                expert_gguf_path = work_dir / f"moe_{dataset_name}_{quantize_level}.gguf"
                logger.info(f"Converting expert '{dataset_name}' to GGUF: {expert_gguf_path}")
                try:
                    convert_to_gguf(expert_path, expert_gguf_path, quantize_level)
                    logger.info(f"✓ Expert '{dataset_name}' GGUF saved to: {expert_gguf_path}")
                except Exception as e:
                    logger.warning(f"Failed to convert expert '{dataset_name}' to GGUF: {e}")
            logger.info(f"=" * 60)
        
        gating_output_dir = work_dir / "gating_network"
        gating_exists = gating_output_dir.exists() and (gating_output_dir / "gating_network.pt").exists()
        
        if not args.skip_gating and not gating_exists:
            logger.info(f"=" * 60)
            logger.info("Training gating network")
            logger.info(f"=" * 60)
            gating_config = config.get("gating", {})
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
            if gating_exists:
                logger.info("Gating network already exists, skipping gating network training")
            else:
                logger.info("Skipping gating network training (--skip-gating flag)")
            gating_path = gating_output_dir
        
        merge_config = config.get("merge", {})
        if merge_config.get("use_shared_expert", False):
            shared_gating_exists = gating_output_dir.exists() and (gating_output_dir / "shared_expert_gating.pt").exists()
            
            if not args.skip_gating and not shared_gating_exists:
                logger.info(f"=" * 60)
                logger.info("Training shared expert gating network")
                logger.info(f"=" * 60)
                train_shared_expert_gating(
                    base_model=base_model_path,
                    datasets=datasets,
                    output_dir=gating_output_dir,
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
            else:
                if shared_gating_exists:
                    logger.info("Shared expert gating network already exists, skipping shared expert gating training")
                else:
                    logger.info("Skipping shared expert gating network training (--skip-gating flag)")
        
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
                base_model_path=base_model_path,
                routing_mode=merge_config.get("routing_mode", "weighted_sum"),
                target_architecture=merge_config.get("target_architecture", "auto"),
                use_shared_expert=merge_config.get("use_shared_expert", False),
                shared_expert_path=merge_config.get("shared_expert_path"),
            )
            rmoe_path = rmoe_output_dir
            logger.info(f"MoE model (rmoe_model) created at: {rmoe_path}")
        else:
            logger.info("Skipping model merging (--skip-merge flag)")
            rmoe_path = work_dir / "rmoe_model"
            if not rmoe_path.exists():
                logger.warning(f"rmoe_model directory does not exist at {rmoe_path}. Finetuning will fail if enabled.")
        
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
                
                if use_cuda:
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                model = model.to(device)
                for param in model.parameters():
                    param.requires_grad = True
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
                    logging_steps=1,
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
                    gradient_checkpointing=True,
                    dataloader_pin_memory=False,
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
                logger.info("Skipping full finetuning (enabled=False in config)")
                final_model_path = rmoe_path
        else:
            logger.info("Skipping full finetuning (--skip-finetune flag)")
            final_model_path = rmoe_path
        if not args.skip_convert:
            logger.info(f"=" * 60)
            logger.info("Converting to GGUF")
            logger.info(f"=" * 60)
            logger.info(f"Conversion mode: {args.conversion_mode}")
            quantize_level = config.get("quantize", QUANTIZE_LEVEL)
            if args.conversion_mode == "preserve_moe":
                logger.info("Using PRESERVE_MOE mode - keeping all experts!")
                from rmoe.moemodel import MoEModel
                logger.info("Loading MoE model...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                merge_cfg = config.get("merge", {})
                moe_model = MoEModel(
                    expert_paths=[str(p) for p in expert_paths],
                    gating_model_path=str(gating_path),
                    base_model_path=base_model_path,
                    routing_mode=merge_cfg.get("routing_mode", "weighted_sum"),
                    device=device,
                    target_architecture=merge_cfg.get("target_architecture", "auto"),
                    use_shared_expert=merge_cfg.get("use_shared_expert", False),
                    shared_expert_path=merge_cfg.get("shared_expert_path"),
                )
                qwen3_format_dir = work_dir / "rmoe_qwen3_format"
                arch_name = moe_model.target_architecture.upper()
                logger.info(f"Saving in {arch_name} format: {qwen3_format_dir}")
                moe_model.save_pretrained(qwen3_format_dir)
                if config.get("gguf_output"):
                    output_file = Path(config["gguf_output"])
                else:
                    quantize_map = {"Q4_0": "f16", "Q8_0": "q8_0"}
                    outtype = quantize_map.get(quantize_level, quantize_level)
                    output_file = work_dir / f"rmoe_model_moe_{outtype}.gguf"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Converting: {qwen3_format_dir}")
                logger.info(f"Output: {output_file}")
                logger.info(f"Quantization: {quantize_level}")
                convert_to_gguf(qwen3_format_dir, output_file, quantize_level)
                logger.info(f"✓ MoE-preserved GGUF saved to: {output_file}")
                logger.info(f"✓ All {len(expert_paths)} experts preserved with routing!")
                
            else:
                logger.info("Using AVERAGE mode - merging all experts to standard MLP")
                if config.get("gguf_output"):
                    output_file = Path(config["gguf_output"])
                else:
                    quantize = config.get("quantize", QUANTIZE_LEVEL)
                    output_file = work_dir / f"rmoe_model_{quantize}.gguf"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Converting: {final_model_path}")
                logger.info(f"Output: {output_file}")
                logger.info(f"Quantization: {quantize_level}")
                standard_model_path = work_dir / "rmoe_standard"
                logger.info(f"Merging experts to standard MLP: {standard_model_path}")
                merge_experts_to_standard_mlp(Path(final_model_path), standard_model_path, merge_mode="average")
                convert_to_gguf(standard_model_path, output_file, quantize_level)
                logger.info(f"✓ Standard GGUF saved to: {output_file}")
        else:
            logger.info("Skipping GGUF conversion")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Work directory: {work_dir}")
        if not args.skip_convert:
            logger.info(f"Conversion mode: {args.conversion_mode}")
            if args.conversion_mode == "preserve_moe":
                logger.info(f"  ✓ MoE structure preserved ({len(expert_paths)} experts with routing)")
                arch_display = config.get("merge", {}).get("target_architecture", "auto").upper()
                logger.info(f"  {arch_display} format: {work_dir / 'rmoe_qwen3_format'}")
            else:
                logger.info(f"  ✓ Experts averaged into single model")
                logger.info(f"  Standard format: {work_dir / 'rmoe_standard'}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
