#!/usr/bin/env python3

import torch
import evaluate
import numpy as np
from transformers import Trainer
from config import BLEURT_CONFIG


class BLEURTTrainer(Trainer):
    def __init__(self, *args, eval_dataset_with_answers=None, model_type="causal", bleurt_batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_with_answers = eval_dataset_with_answers
        self.model_type = model_type
        self.bleurt_batch_size = bleurt_batch_size
        self.bleurt = None
    
    def _get_bleurt(self):
        """Lazy load BLEURT to save memory when not evaluating."""
        if self.bleurt is None:
            print(f"Loading BLEURT model: {BLEURT_CONFIG['model_name']}")
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except (ImportError, AttributeError):
                pass
            self.bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
        return self.bleurt
    
    def _clear_bleurt(self):
        """Clear BLEURT model from memory to free up space."""
        if self.bleurt is not None:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except (ImportError, AttributeError):
                pass
            self.bleurt = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if self.eval_dataset_with_answers is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Get tokenizer using recommended approach (handles deprecation)
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer not found in trainer")
        
        self.model.eval()
        predictions = []
        references = []
        
        print("Computing BLEURT scores...")
        print("Step 1: Generating predictions...")
        
        # Step 1: Generate all predictions first
        for i, example in enumerate(eval_dataset):
            if i % 10 == 0:
                print(f"  Generating {i}/{len(eval_dataset)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            question = example.get("question", "")
            best_answer = example.get("best_answer", "")
            
            if not question or not best_answer:
                continue
            
            if self.model_type == "seq2seq":
                inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
            else:
                prompt = BLEURT_CONFIG["prompt_template"].format(question=question)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=BLEURT_CONFIG["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Explicitly delete tensors and clear GPU cache
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.model_type == "seq2seq":
                answer = generated.strip()
            else:
                if "Answer:" in generated:
                    answer = generated.split("Answer:")[-1].strip()
                else:
                    prompt = BLEURT_CONFIG["prompt_template"].format(question=question)
                    answer = generated[len(prompt):].strip()
            
            if answer and best_answer:
                predictions.append(answer)
                references.append(best_answer)
        
        # Step 2: Compute BLEURT scores in batches
        print(f"Step 2: Computing BLEURT scores for {len(predictions)} examples in batches of {self.bleurt_batch_size}...")
        bleurt = self._get_bleurt()
        bleurt_scores = []
        
        for i in range(0, len(predictions), self.bleurt_batch_size):
            batch_predictions = predictions[i:i + self.bleurt_batch_size]
            batch_references = references[i:i + self.bleurt_batch_size]
            
            batch_scores = bleurt.compute(
                predictions=batch_predictions,
                references=batch_references
            )["scores"]
            bleurt_scores.extend(batch_scores)
            
            if (i // self.bleurt_batch_size) % 10 == 0:
                print(f"  Processed {min(i + self.bleurt_batch_size, len(predictions))}/{len(predictions)}")
        
        avg_bleurt = np.mean(bleurt_scores) if bleurt_scores else 0.0
        
        metrics = {f"{metric_key_prefix}_bleurt_score": avg_bleurt}
        self.log(metrics)
        
        # Clear all intermediate variables
        del predictions, references, bleurt_scores
        
        # Clear BLEURT model to free memory (will be reloaded next time if needed)
        self._clear_bleurt()
        
        # Clear GPU cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return metrics

