#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import csv
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import evaluate

class TQAEval:
    def __init__(self, args):
        self.args = args
        root = Path(__file__).parent.parent
        self.models_dir = Path(args.models_dir)
        self.llama_cli = root / "external/llama.cpp/build/bin/llama-cli"
        self.csv = self.models_dir / "tqa_results.csv"
        self.existing = self._load()
        self.results = []
    
    def _load(self):
        if not self.csv.exists():
            return {}
        with open(self.csv, 'r') as f:
            return {row['model_name']: row for row in csv.DictReader(f)}
    
    def _save(self):
        all_results = {**self.existing, **{r['model_name']: r for r in self.results}}
        with open(self.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, ["model_name", "max_score", "accuracy", "avg_latency", 
                                   "num_samples", "timestamp"])
            w.writeheader()
            w.writerows(all_results.values())
    
    def _eval_model(self, model_path):
        print(f"Loading TruthfulQA dataset...")
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        
        total = len(ds)
        if self.args.samples and self.args.samples < total:
            random.seed(42)
            indices = random.sample(range(total), self.args.samples)
            ds = ds.select(sorted(indices))
            print(f"Using {len(ds)}/{total} samples (seed=42)")
        else:
            print(f"Using all {total} samples")
        
        print(f"Loading BLEURT model (bleurt-large-128)...")
        bleurt = evaluate.load('bleurt', 'bleurt-large-128')
        
        all_preds = []
        all_correct = []
        all_incorrect = []
        latencies = []
        
        print(f"Generating predictions for {len(ds)} samples...\n")
        
        import time
        for i, rec in enumerate(ds):
            question = rec['question'].replace("'", " ").replace('"', ' ')
            correct = rec['correct_answers']
            incorrect = rec['incorrect_answers']
            
            if not correct or not incorrect:
                continue
            
            cmd = [str(self.llama_cli), "-m", str(model_path), "-p", question, 
                   "-n", str(self.args.tokens), "-no-cnv", "-ngl", "0"]
            
            start = time.time()
            with open("tmp_output.txt", "w", encoding="utf-8", errors='replace') as fout:
                result = subprocess.run(cmd, stdout=fout, stderr=subprocess.DEVNULL, timeout=60)
            latency = time.time() - start
            
            if result.returncode != 0:
                print(f"[{i+1}/{len(ds)}] FAILED")
                continue
            
            with open("tmp_output.txt", "r", encoding="utf-8", errors='replace') as fin:
                pred = fin.read().strip()
            
            if pred:
                all_preds.append(pred)
                all_correct.append(correct)
                all_incorrect.append(incorrect)
                latencies.append(latency)
                
                if (i + 1) % 10 == 0 or i < 3:
                    print(f"[{i+1}/{len(ds)}] Generated (lat={latency:.1f}s)")
        
        if not all_preds:
            print("No valid predictions generated")
            return {"max_score": 0.0, "accuracy": 0.0, "avg_latency": 0.0, "num_samples": 0}
        
        print(f"\nEvaluating {len(all_preds)} predictions with BLEURT...")
        
        max_scores = []
        acc_scores = []
        batch_size = 32
        
        for i in range(0, len(all_preds), batch_size):
            batch_preds = all_preds[i:i+batch_size]
            batch_correct = all_correct[i:i+batch_size]
            batch_incorrect = all_incorrect[i:i+batch_size]
            
            expanded_preds_true = [p for p, refs in zip(batch_preds, batch_correct) for _ in refs]
            expanded_refs_true = [r for refs in batch_correct for r in refs]
            expanded_preds_false = [p for p, refs in zip(batch_preds, batch_incorrect) for _ in refs]
            expanded_refs_false = [r for refs in batch_incorrect for r in refs]
            
            scores_true = bleurt.compute(predictions=expanded_preds_true, references=expanded_refs_true)["scores"] if expanded_preds_true else []
            scores_false = bleurt.compute(predictions=expanded_preds_false, references=expanded_refs_false)["scores"] if expanded_preds_false else []
            
            true_idx = 0
            false_idx = 0
            for correct_refs, incorrect_refs in zip(batch_correct, batch_incorrect):
                example_scores_true = scores_true[true_idx:true_idx+len(correct_refs)]
                example_scores_false = scores_false[false_idx:false_idx+len(incorrect_refs)]
                true_idx += len(correct_refs)
                false_idx += len(incorrect_refs)
                
                max_score = max(example_scores_true) if example_scores_true else 0.0
                acc_score = int(max(example_scores_true) > max(example_scores_false)) if example_scores_true and example_scores_false else 0
                max_scores.append(max_score)
                acc_scores.append(acc_score)
        
        avg_max = np.mean(max_scores) if max_scores else 0.0
        avg_acc = np.mean(acc_scores) if acc_scores else 0.0
        avg_lat = np.mean(latencies) if latencies else 0.0
        
        print(f"\nResults: max_score={avg_max:.3f} accuracy={avg_acc:.3f} latency={avg_lat:.2f}s")
        
        if os.path.exists("tmp_output.txt"):
            os.remove("tmp_output.txt")
        
        return {
            "max_score": avg_max,
            "accuracy": avg_acc,
            "avg_latency": avg_lat,
            "num_samples": len(max_scores)
        }
    
    def run(self):
        print(f"Models: {self.models_dir}\n")
        
        if self.args.model:
            model_path = self.models_dir / self.args.model
            if not model_path.exists():
                sys.exit(f"Model not found: {self.args.model}")
            models = [model_path]
        else:
            models = list(self.models_dir.glob("*.gguf"))
            if not models:
                sys.exit(f"No models in {self.models_dir}")
        
        for i, path in enumerate(models, 1):
            if path.name in self.existing:
                print(f"[{i}/{len(models)}] SKIP {path.name} (already done)\n")
                continue
            
            print(f"[{i}/{len(models)}] {path.name}")
            
            metrics = self._eval_model(path)
            self.results.append({
                "model_name": path.name,
                "max_score": metrics["max_score"],
                "accuracy": metrics["accuracy"],
                "avg_latency": metrics["avg_latency"],
                "num_samples": metrics["num_samples"],
                "timestamp": datetime.now().isoformat()
            })
            self._save()
            print()
        
        print(f"Results saved → {self.csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="./models/rmoe-gguf")
    p.add_argument("-m", "--model", help="Specific model (e.g., model.gguf)")
    p.add_argument("--samples", type=int, default=None, help="Number of samples (default: all)")
    p.add_argument("--tokens", type=int, default=25, help="Tokens to generate (default: 25)")
    
    try:
        TQAEval(p.parse_args()).run()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()


