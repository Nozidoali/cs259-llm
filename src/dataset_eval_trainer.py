#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import Trainer
from config import BLEURT_CONFIG, DATASET_CONFIG

class DatasetEvalTrainer(Trainer):
    def __init__(self, *args, eval_dataset_with_answers=None, model_type="causal", dataset_type="truthfulqa", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_with_answers = eval_dataset_with_answers
        self.model_type = model_type
        self.dataset_type = dataset_type
        self._bleurt = None
        self._rouge = None
    
    @property
    def bleurt(self):
        if self._bleurt is None:
            import evaluate
            self._bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
        return self._bleurt
    
    @property
    def rouge(self):
        if self._rouge is None:
            import evaluate
            self._rouge = evaluate.load("rouge")
        return self._rouge
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if self.eval_dataset_with_answers is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        self.model.eval()
        
        if self.dataset_type == "truthfulqa":
            return self._evaluate_truthfulqa(eval_dataset, metric_key_prefix)
        elif self.dataset_type == "qmsum":
            return self._evaluate_qmsum(eval_dataset, metric_key_prefix)
        elif self.dataset_type == "both":
            return self._evaluate_both(eval_dataset, metric_key_prefix)
        else:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def _evaluate_both(self, eval_dataset, metric_key_prefix):
        truthfulqa_examples = []
        qmsum_examples = []
        
        for example in eval_dataset:
            if "correct_answers" in example and example.get("correct_answers"):
                truthfulqa_examples.append(example)
            elif "answer" in example and example.get("answer"):
                qmsum_examples.append(example)
        
        truthfulqa_metrics = self._evaluate_truthfulqa(truthfulqa_examples, metric_key_prefix) if truthfulqa_examples else {}
        qmsum_metrics = self._evaluate_qmsum(qmsum_examples, metric_key_prefix) if qmsum_examples else {}
        return {**truthfulqa_metrics, **qmsum_metrics}
    
    def _evaluate_truthfulqa(self, eval_dataset, metric_key_prefix):
        max_score_arr = []
        acc_score_arr = []
        
        for i, example in enumerate(eval_dataset):
            question = example.get("question", "")
            correct_answers = example.get("correct_answers", [])
            incorrect_answers = example.get("incorrect_answers", [])
            
            if not question or not correct_answers or not incorrect_answers:
                continue
            
            prompt = DATASET_CONFIG["format_template"].format(question=question).split("Answer:")[0] + "Answer:"
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
            if "Answer:" in generated:
                pred = generated.split("Answer:")[-1].strip()
            else:
                pred = generated[len(prompt):].strip()
            
            if not pred:
                continue
            
            predictions_true = [pred] * len(correct_answers)
            predictions_false = [pred] * len(incorrect_answers)
            score_true = self.bleurt.compute(predictions=predictions_true, references=correct_answers)["scores"]
            score_false = self.bleurt.compute(predictions=predictions_false, references=incorrect_answers)["scores"]
            max_score = max(score_true) if score_true else 0.0
            acc_score = int(max(score_true) > max(score_false)) if score_true and score_false else 0
            
            max_score_arr.append(max_score)
            acc_score_arr.append(acc_score)
        
        avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
        accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0
        
        metrics = {
            f"{metric_key_prefix}_bleurt_max_score": avg_max_score,
            f"{metric_key_prefix}_bleurt_accuracy": accuracy
        }
        self.log(metrics)
        return metrics
    
    def _evaluate_qmsum(self, eval_dataset, metric_key_prefix):
        predictions = []
        references = []
        
        for example in eval_dataset:
            context = example.get("context", "")
            input_text = example.get("input", "")
            answer = example.get("answer", "")
            
            if not answer:
                continue
            
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            prompt = f"{prompt}\n\nSummary:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else self.tokenizer.pad_token_id,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Summary:" in generated:
                pred = generated.split("Summary:")[-1].strip()
            else:
                pred = generated[len(prompt):].strip()
            
            if pred and answer:
                predictions.append(pred)
                references.append(answer)
        
        if not predictions:
            metrics = {
                f"{metric_key_prefix}_rouge2": 0.0
            }
            self.log(metrics)
            return metrics
        
        result = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        rouge2 = result.get("rouge2", 0.0)
        
        metrics = {
            f"{metric_key_prefix}_rouge2": rouge2
        }
        self.log(metrics)
        return metrics

