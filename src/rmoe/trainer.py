import torch
import numpy as np
import logging
import gc
import os
from transformers import Trainer
from config import BLEURT_CONFIG, DATASET_CONFIG

logger = logging.getLogger(__name__)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
try:
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

class MultiDatasetEvalTrainer(Trainer):
    def __init__(self, *args, eval_datasets=None, model_type="causal", qmsum_max_new_tokens=200, temperature=0.0, output_head_params=None, l2_regularization=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_datasets = eval_datasets or {}
        self.model_type = model_type
        self.qmsum_max_new_tokens = qmsum_max_new_tokens
        self._bleurt = None
        self._rouge = None
        self.temperature = temperature
        self.output_head_params = set(output_head_params) if output_head_params else set()
        self.l2_regularization = l2_regularization
    
    @property
    def _get_tokenizer(self):
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            return self.processing_class
        return getattr(self, 'tokenizer', None)
    
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
    
    def create_optimizer(self):
        if self.optimizer is None:
            if self.output_head_params:
                decay_parameters = [n for n in self.get_decay_parameter_names(self.model) if "bias" not in n]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad and p not in self.output_head_params],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad and p not in self.output_head_params],
                        "weight_decay": 0.0,
                    },
                ]
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            else:
                super().create_optimizer()
        return self.optimizer
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.model.training
        gradient_checkpointing_enabled = False
        if hasattr(self.model, "gradient_checkpointing") and self.model.gradient_checkpointing:
            gradient_checkpointing_enabled = True
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        all_metrics = {}
        if eval_dataset is not None:
            all_metrics.update(super().evaluate(eval_dataset, ignore_keys, metric_key_prefix))
            self._clear_memory()
        for dataset_name, dataset in self.eval_datasets.items():
            if dataset_name == "truthfulqa":
                metrics = self._evaluate_truthfulqa(dataset, f"{metric_key_prefix}_{dataset_name}")
            elif dataset_name in ["longbench", "qmsum"]:
                metrics = self._evaluate_qmsum(dataset, f"{metric_key_prefix}_{dataset_name}")
            else:
                continue
            all_metrics.update(metrics)
            self._clear_memory()
        self._clear_memory()
        if was_training:
            self.model.train()
            if gradient_checkpointing_enabled:
                self.model.gradient_checkpointing_enable()
        return all_metrics
    
    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add L2 regularization term to the loss.
        This helps prevent overfitting during long training (100 epochs).
        """
        # Get standard loss from parent class
        # Pass num_items_in_batch if provided (newer transformers versions)
        if num_items_in_batch is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Add L2 regularization on trainable parameters
        # Check if model is in training mode (not eval mode)
        if self.l2_regularization > 0 and model.training:
            l2_reg = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    l2_reg = l2_reg + torch.sum(param ** 2)
            loss = loss + self.l2_regularization * l2_reg
        
        return (loss, outputs) if return_outputs else loss
    
    def _evaluate_truthfulqa(self, eval_dataset, metric_key_prefix):
        self.model.eval()
        all_predictions = []
        all_correct_refs = []
        all_incorrect_refs = []
        tokenizer = self._get_tokenizer
        
        # Store examples for readable logging
        example_logs = []
        
        for idx, example in enumerate(eval_dataset):
            question = example.get("question", "")
            correct_answers = example.get("correct_answers", [])
            incorrect_answers = example.get("incorrect_answers", [])
            if not question or not correct_answers or not incorrect_answers:
                continue
            prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                try:
                    do_sample = self.temperature > 0.0
                    pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id
                    outputs = self.model.generate(**inputs, max_new_tokens=BLEURT_CONFIG["max_new_tokens"], temperature=self.temperature if do_sample else None, do_sample=do_sample, pad_token_id=pad_token_id, use_cache=True)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}, skipping example")
                    continue
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated[len(prompt):].strip()
            if pred:
                all_predictions.append(pred)
                all_correct_refs.append(correct_answers)
                all_incorrect_refs.append(incorrect_answers)
                # Store for logging (sample first 3 examples)
                if len(example_logs) < 3:
                    example_logs.append({
                        "question": question,
                        "prediction": pred,
                        "correct_answers": correct_answers[:2],  # Show first 2 correct answers
                        "incorrect_answers": incorrect_answers[:2]  # Show first 2 incorrect answers
                    })
            del inputs, outputs, generated
            if (idx + 1) % 10 == 0:
                self._clear_memory()
        
        self._clear_memory()
        if not all_predictions:
            metrics = {f"{metric_key_prefix}_bleurt_max_score": 0.0, f"{metric_key_prefix}_bleurt_accuracy": 0.0}
            self.log(metrics)
            return metrics
        
        max_score_arr = []
        acc_score_arr = []
        batch_size = 32
        
        for i in range(0, len(all_predictions), batch_size):
            batch_preds = all_predictions[i:i+batch_size]
            batch_correct = all_correct_refs[i:i+batch_size]
            batch_incorrect = all_incorrect_refs[i:i+batch_size]
            expanded_preds_true = [p for p, refs in zip(batch_preds, batch_correct) for _ in refs]
            expanded_refs_true = [r for refs in batch_correct for r in refs]
            expanded_preds_false = [p for p, refs in zip(batch_preds, batch_incorrect) for _ in refs]
            expanded_refs_false = [r for refs in batch_incorrect for r in refs]
            scores_true = self.bleurt.compute(predictions=expanded_preds_true, references=expanded_refs_true)["scores"] if expanded_preds_true else []
            scores_false = self.bleurt.compute(predictions=expanded_preds_false, references=expanded_refs_false)["scores"] if expanded_preds_false else []
            true_idx = 0
            false_idx = 0
            for correct_refs, incorrect_refs in zip(batch_correct, batch_incorrect):
                example_scores_true = scores_true[true_idx:true_idx+len(correct_refs)]
                example_scores_false = scores_false[false_idx:false_idx+len(incorrect_refs)]
                true_idx += len(correct_refs)
                false_idx += len(incorrect_refs)
                max_score = max(example_scores_true) if example_scores_true else 0.0
                acc_score = int(max(example_scores_true) > max(example_scores_false)) if example_scores_true and example_scores_false else 0
                max_score_arr.append(max_score)
                acc_score_arr.append(acc_score)
            del batch_preds, batch_correct, batch_incorrect, expanded_preds_true, expanded_refs_true, expanded_preds_false, expanded_refs_false, scores_true, scores_false
            self._clear_memory()
        
        metrics = {
            f"{metric_key_prefix}_bleurt_max_score": np.mean(max_score_arr) if max_score_arr else 0.0,
            f"{metric_key_prefix}_bleurt_accuracy": np.mean(acc_score_arr) if acc_score_arr else 0.0
        }
        
        # Log example questions and answers in a readable format
        if example_logs:
            logger.info("=" * 80)
            logger.info(f"TruthfulQA Evaluation Examples ({metric_key_prefix}):")
            logger.info("=" * 80)
            for i, ex in enumerate(example_logs, 1):
                logger.info(f"\nExample {i}:")
                logger.info(f"  Question: {ex['question']}")
                logger.info(f"  Prediction: {ex['prediction']}")
                logger.info(f"  Correct Answers: {', '.join(ex['correct_answers'])}")
                logger.info(f"  Incorrect Answers: {', '.join(ex['incorrect_answers'])}")
            logger.info("=" * 80)
        
        self.log(metrics)
        return metrics
    
    def _evaluate_qmsum(self, eval_dataset, metric_key_prefix):
        self.model.eval()
        predictions = []
        references = []
        tokenizer = self._get_tokenizer
        
        # Store examples for readable logging
        example_logs = []
        
        for idx, example in enumerate(eval_dataset):
            answer = example.get("answer", "")
            if not answer:
                continue
            context = example.get("context", "")
            input_text = example.get("input", "")
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            prompt = f"{prompt}\n\nSummary:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_ids_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                try:
                    do_sample = self.temperature > 0.0
                    pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id
                    outputs = self.model.generate(**inputs, max_new_tokens=self.qmsum_max_new_tokens, temperature=self.temperature if do_sample else None, do_sample=do_sample, pad_token_id=pad_token_id, use_cache=True)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}, skipping example")
                    continue
            full_sequences = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
            generated_ids = full_sequences[input_ids_len:]
            full_text = tokenizer.decode(full_sequences, skip_special_tokens=True).strip()
            input_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True).strip()
            pred = full_text[len(input_decoded):].strip() if full_text.startswith(input_decoded) else tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if pred and answer:
                predictions.append(pred)
                references.append(answer)
                # Store for logging (sample first 3 examples)
                if len(example_logs) < 3:
                    # Truncate context/input for readability
                    display_context = context[:200] + "..." if len(context) > 200 else context
                    display_input = input_text[:200] + "..." if len(input_text) > 200 else input_text
                    example_logs.append({
                        "context": display_context,
                        "input": display_input,
                        "prediction": pred[:300] + "..." if len(pred) > 300 else pred,
                        "reference": answer[:300] + "..." if len(answer) > 300 else answer
                    })
            del inputs, outputs, full_sequences, generated_ids, full_text, input_decoded
            if (idx + 1) % 10 == 0:
                self._clear_memory()
        
        self._clear_memory()
        if not predictions:
            metrics = {f"{metric_key_prefix}_rougeL": 0.0}
            self.log(metrics)
            return metrics
        
        result = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        metrics = {f"{metric_key_prefix}_rougeL": result.get("rougeL", 0.0)}
        
        # Log example questions and answers in a readable format
        if example_logs:
            logger.info("=" * 80)
            logger.info(f"QMSum/LongBench Evaluation Examples ({metric_key_prefix}):")
            logger.info("=" * 80)
            for i, ex in enumerate(example_logs, 1):
                logger.info(f"\nExample {i}:")
                if ex.get("context"):
                    logger.info(f"  Context: {ex['context']}")
                if ex.get("input"):
                    logger.info(f"  Input: {ex['input']}")
                logger.info(f"  Prediction: {ex['prediction']}")
                logger.info(f"  Reference: {ex['reference']}")
            logger.info("=" * 80)
        
        self.log(metrics)
        del result
        self._clear_memory()
        return metrics
