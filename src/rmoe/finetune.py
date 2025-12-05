import os
import torch
import logging
from pathlib import Path
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from models import load_model_and_tokenizer, freeze_all_except_mlp
from data import prepare_truthfulqa_dataset, prepare_qmsum_dataset
from rmoe.trainer import MultiDatasetEvalTrainer
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
    eval_split=0.2,
    seed=42,
    qmsum_max_new_tokens=200,
):
    logger.info(f"Training expert for dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(model_path=base_model_path, dtype=torch.float32)
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    model = model.to(device)
    logger.info("Freezing all parameters except MLP layers...")
    freeze_all_except_mlp(model)
    model.train()
    train_dataset = prepare_dataset(dataset_name, tokenizer, max_length, keep_metadata=True)
    train_dataset = train_dataset.train_test_split(test_size=eval_split, seed=seed)
    eval_datasets = {}
    for ds_name in all_datasets:
        if ds_name == dataset_name:
            eval_datasets[ds_name] = train_dataset["test"]
        else:
            eval_ds = prepare_dataset(ds_name, tokenizer, max_length, keep_metadata=True)
            eval_ds = eval_ds.train_test_split(test_size=0.1, seed=seed)
            eval_datasets[ds_name] = eval_ds["test"].select(range(min(50, len(eval_ds["test"]))))
    logger.info(f"Train samples: {len(train_dataset['train'])}, Eval samples: {len(train_dataset['test'])}")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_truthfulqa_bleurt_max_score" if dataset_name == "truthfulqa" else "eval_qmsum_rougeL",
        greater_is_better=True,
        fp16=use_cuda,
        bf16=use_mps,
        dataloader_num_workers=0 if use_mps else 2,
        report_to="none",
        seed=seed,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = MultiDatasetEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        eval_datasets=eval_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        model_type="causal",
        qmsum_max_new_tokens=qmsum_max_new_tokens,
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Evaluating on all datasets...")
    model.eval()
    final_results = evaluate_all_datasets(model, tokenizer, eval_datasets, device, qmsum_max_new_tokens)
    logger.info(f"Final evaluation results: {final_results}")
    return output_dir

