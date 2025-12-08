import os

# Set environment variables before importing transformers
os.environ["USE_TF"] = "0"  # Disable TensorFlow in transformers
os.environ["USE_TORCH"] = "1"  # Use PyTorch only

import torch
import logging
from pathlib import Path
from transformers import TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from models import load_model_and_tokenizer, freeze_all_except_mlp
from data import prepare_truthfulqa_dataset, prepare_qmsum_dataset
from rmoe.trainer import MultiDatasetEvalTrainer, TruthfulQADataCollator
from rmoe.evaluate import evaluate_all_datasets

logger = logging.getLogger(__name__)

def prepare_dataset(dataset_name, tokenizer, max_length, keep_metadata=True):
    if dataset_name == "truthfulqa":
        return prepare_truthfulqa_dataset(tokenizer, max_length=max_length, keep_metadata=keep_metadata, model_type="causal")
    elif dataset_name in ["longbench", "qmsum"]:
        return prepare_qmsum_dataset(tokenizer, max_length=max_length, keep_metadata=keep_metadata, model_type="causal")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def train_expert(
    base_model_path,
    dataset_name,
    output_dir,
    all_datasets,
    max_length=512,
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    l2_regularization=0.0,
    max_grad_norm=1.0,
    disable_eval_split=False,
    eval_split=0.2,
    seed=42,
    qmsum_max_new_tokens=200,
    temperature=0.0,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=None,
):
    logger.info(f"Training expert for dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(model_path=base_model_path, dtype=torch.float32)
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    
    if use_cuda:
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = model.to(device)
    
    if use_lora:
        logger.info("Applying LoRA adapters for parameter-efficient fine-tuning...")
        
        # Default target modules for LoRA (typically attention and MLP layers)
        if lora_target_modules is None:
            # Auto-detect architecture and set appropriate target modules
            model_type = model.config.model_type if hasattr(model.config, 'model_type') else None
            
            if model_type in ["qwen2", "qwen"]:
                # Qwen2 architecture: target both attention and MLP
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                logger.info(f"Detected Qwen2 model, targeting attention + MLP: {lora_target_modules}")
            elif model_type == "llama":
                # Llama architecture
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                logger.info(f"Detected Llama model, targeting attention + MLP: {lora_target_modules}")
            else:
                # Generic fallback - target common module names
                lora_target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
                logger.info(f"Using generic target modules: {lora_target_modules}")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"LoRA Config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"LoRA target modules: {lora_target_modules}")
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("Freezing all parameters except MLP layers...")
        freeze_all_except_mlp(model)
    
    model.train()
    full_dataset = prepare_dataset(dataset_name, tokenizer, max_length, keep_metadata=True)
    
    # Limit qmsum to first 50 samples
    if dataset_name == "longbench":
        num_samples = min(50, len(full_dataset))
        full_dataset = full_dataset.select(range(num_samples))
        logger.info(f"Limited qmsum dataset to first {num_samples} samples")
    
    # Option to disable eval split for overfitting experts on full dataset
    if disable_eval_split:
        logger.info(f"Eval split disabled - training on full dataset to overfit expert")
        logger.info(f"Will evaluate on training set itself to select best checkpoint based on BLEURT score")
        train_dataset_split = full_dataset
        eval_dataset_split = full_dataset  # Evaluate on training set itself
        eval_datasets = {dataset_name: full_dataset}  # Use training set for evaluation
        logger.info(f"Train samples: {len(full_dataset)} (full dataset, no eval split)")
        logger.info(f"Eval samples: {len(full_dataset)} (same as training set)")
    else:
        train_dataset = full_dataset.train_test_split(test_size=eval_split, seed=seed)
        train_dataset_split = train_dataset["train"]
        eval_dataset_split = train_dataset["test"]
        eval_datasets = {}
        for ds_name in all_datasets:
            if ds_name == dataset_name:
                eval_datasets[ds_name] = train_dataset["test"]
            else:
                eval_ds = prepare_dataset(ds_name, tokenizer, max_length, keep_metadata=True)
                # Limit other datasets to first 50 samples for eval
                if ds_name == "longbench":
                    num_eval_samples = min(50, len(eval_ds))
                    eval_datasets[ds_name] = eval_ds.select(range(num_eval_samples))
                else:
                    eval_ds = eval_ds.train_test_split(test_size=0.1, seed=seed)
                    eval_datasets[ds_name] = eval_ds["test"].select(range(min(50, len(eval_ds["test"]))))
        logger.info(f"Train samples: {len(train_dataset['train'])}, Eval samples: {len(train_dataset['test'])}")
    logger.info(f"Training for {num_epochs} epochs with learning_rate={learning_rate}, l2_regularization={l2_regularization}, max_grad_norm={max_grad_norm}")
    logger.info(f"Eval split {'disabled (overfitting mode - evaluating on training set)' if disable_eval_split else f'enabled ({eval_split*100}% held out)'}")
    
    # Always enable evaluation and best model selection based on BLEURT score
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    metric_for_best_model = "eval_truthfulqa_bleurt_max_score" if dataset_name == "truthfulqa" else "eval_longbench_rougeL"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,  # Gradient clipping to prevent exploding gradients
        logging_dir=str(output_dir / "logs"),
        logging_steps=gradient_accumulation_steps,  # Only log after optimizer step to avoid loss spikes
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True if metric_for_best_model else None,
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to=["tensorboard"],
        seed=seed,
        gradient_checkpointing=False,  # Disabled: incompatible with frozen parameters
        dataloader_pin_memory=False,
    )
    
    # Use custom data collator for TruthfulQA to preserve metadata
    if dataset_name == "truthfulqa":
        data_collator = TruthfulQADataCollator(tokenizer=tokenizer, mlm=False)
        logger.info("Using TruthfulQADataCollator with custom loss components")
    else:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.0
    )
    
    trainer = MultiDatasetEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_split,
        eval_dataset=eval_dataset_split,
        eval_datasets=eval_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        model_type="causal",
        qmsum_max_new_tokens=qmsum_max_new_tokens,
        temperature=temperature,
        callbacks=[early_stopping_callback],
        l2_regularization=l2_regularization,
    )
    
    # Run initial evaluation before training to establish baseline
    logger.info("Running initial evaluation before training (Epoch 0)...")
    model.eval()
    initial_metrics = trainer.evaluate()
    logger.info(f"Initial evaluation results (before training): {initial_metrics}")
    
    logger.info("Starting training...")
    trainer.train()
    logger.info(f"Saving model to: {output_dir}")
    
    if use_lora:
        # Save LoRA adapter weights
        model.save_pretrained(str(output_dir))
        logger.info(f"LoRA adapter saved to: {output_dir}")
        
        # Also merge and save the full model for easier loading later
        logger.info("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        merged_output_dir = output_dir / "merged"
        merged_output_dir.mkdir(exist_ok=True)
        merged_model.save_pretrained(str(merged_output_dir))
        tokenizer.save_pretrained(str(merged_output_dir))
        logger.info(f"Merged model saved to: {merged_output_dir}")
    else:
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
    logger.info("Evaluating on all datasets...")
    model.eval()
    final_results = evaluate_all_datasets(model, tokenizer, eval_datasets, device, qmsum_max_new_tokens, temperature)
    logger.info(f"Final evaluation results: {final_results}")
    return output_dir
