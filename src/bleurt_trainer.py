#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import Trainer
from config import BLEURT_CONFIG


class BLEURTTrainer(Trainer):
    def __init__(self, *args, eval_dataset_with_answers=None, model_type="causal", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_with_answers = eval_dataset_with_answers
        self.model_type = model_type
        self._bleurt = None
    
    @property
    def bleurt(self):
        if self._bleurt is None:
            import evaluate
            self._bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
        return self._bleurt
    
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
            
            if self.model_type == "seq2seq":
                inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
            else:
                prompt = BLEURT_CONFIG["prompt_template"].format(question=question)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=BLEURT_CONFIG["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else self.tokenizer.pad_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self.model_type == "seq2seq":
                answer = generated.strip()
            else:
                if "Answer:" in generated:
                    answer = generated.split("Answer:")[-1].strip()
                else:
                    prompt = BLEURT_CONFIG["prompt_template"].format(question=question)
                    answer = generated[len(prompt):].strip()
            
            if answer and best_answer:
                score = self.bleurt.compute(predictions=[answer], references=[best_answer])["scores"][0]
                bleurt_scores.append(score)
        
        avg_bleurt = np.mean(bleurt_scores) if bleurt_scores else 0.0
        
        metrics = {f"{metric_key_prefix}_bleurt_score": avg_bleurt}
        self.log(metrics)
        
        return metrics


