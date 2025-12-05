import torch
import numpy as np
import logging
from transformers import Trainer
from config import BLEURT_CONFIG, DATASET_CONFIG

logger = logging.getLogger(__name__)

class MultiDatasetEvalTrainer(Trainer):
    def __init__(self, *args, eval_datasets=None, model_type="causal", qmsum_max_new_tokens=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_datasets = eval_datasets or {}
        self.model_type = model_type
        self.qmsum_max_new_tokens = qmsum_max_new_tokens
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
        was_training = self.model.training
        gradient_checkpointing_enabled = False
        if hasattr(self.model, "gradient_checkpointing") and self.model.gradient_checkpointing:
            gradient_checkpointing_enabled = True
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        all_metrics = {}
        if eval_dataset is not None:
            base_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            all_metrics.update(base_metrics)
        for dataset_name, dataset in self.eval_datasets.items():
            if dataset_name == "truthfulqa":
                metrics = self._evaluate_truthfulqa(dataset, f"{metric_key_prefix}_{dataset_name}")
            elif dataset_name in ["longbench", "qmsum"]:
                metrics = self._evaluate_qmsum(dataset, f"{metric_key_prefix}_{dataset_name}")
            else:
                continue
            all_metrics.update(metrics)
        if was_training:
            self.model.train()
            if gradient_checkpointing_enabled:
                self.model.gradient_checkpointing_enable()
        return all_metrics
    
    def _evaluate_truthfulqa(self, eval_dataset, metric_key_prefix):
        self.model.eval()
        max_score_arr = []
        acc_score_arr = []
        for example in eval_dataset:
            question = example.get("question", "")
            correct_answers = example.get("correct_answers", [])
            incorrect_answers = example.get("incorrect_answers", [])
            if not question or not correct_answers or not incorrect_answers:
                continue
            prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                try:
                    outputs = self.model.generate(**inputs, max_new_tokens=BLEURT_CONFIG["max_new_tokens"], do_sample=False, pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else self.tokenizer.pad_token_id, use_cache=False)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}, skipping example")
                    continue
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
        self.model.eval()
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
            input_ids_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                try:
                    outputs = self.model.generate(**inputs, max_new_tokens=self.qmsum_max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else self.tokenizer.pad_token_id, use_cache=False)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}, skipping example")
                    continue
            generated_ids = outputs[0][input_ids_len:]
            pred = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if pred and answer:
                predictions.append(pred)
                references.append(answer)
        if not predictions:
            metrics = {f"{metric_key_prefix}_rougeL": 0.0}
            self.log(metrics)
            return metrics
        result = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        rougeL = result.get("rougeL", 0.0)
        metrics = {f"{metric_key_prefix}_rougeL": rougeL}
        self.log(metrics)
        return metrics

