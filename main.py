#!/usr/bin/env python3

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    TRAINING_CONFIG,
    LOGS_DIR,
    TRUTHFULQA_CACHE_DIR,
)


def prepare_truthfulqa_dataset(tokenizer, max_length=512):
    ds = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
        cache_dir=str(TRUTHFULQA_CACHE_DIR),
    )
    formatted = ds.map(
        lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['best_answer']}"},
        remove_columns=ds.column_names,
    )
    return formatted.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )


def main():
    config = TRAINING_CONFIG
    print("Fine-tuning GPT-2 Small on TruthfulQA")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    dataset = prepare_truthfulqa_dataset(tokenizer, max_length=config["max_length"])
    dataset = dataset.train_test_split(test_size=config["eval_split"], seed=config["seed"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_dir=str(LOGS_DIR),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()

