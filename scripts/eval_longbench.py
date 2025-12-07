#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import csv
import random
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import evaluate

class LongBenchEval:
    def __init__(self, args):
        self.args = args
        root = Path(__file__).parent.parent
        self.models_dir = Path(args.models_dir)
        self.llama_cli = root / "external/llama.cpp/build/bin/llama-cli"
        self.csv = self.models_dir / "longbench_results.csv"
        self.output_dir = Path("qmsum_outputs")
        self.output_dir.mkdir(exist_ok=True)
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
            w = csv.DictWriter(f, ["model_name", "rouge_l", "rouge_1", "rouge_2", 
                                   "avg_latency", "num_samples", "timestamp"])
            w.writeheader()
            w.writerows(all_results.values())
    
    def _load_references(self):
        print("Loading LongBench qmsum references...")
        ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
        ref_map = {}
        for i, rec in enumerate(ds):
            ans = rec["answers"]
            ref = ans[0] if isinstance(ans, list) and ans else ans
            ref_map[i] = ref.strip()
        return ref_map, ds
    
    def _eval_model(self, model_path):
        ref_map, ds = self._load_references()
        
        total = len(ds)
        if self.args.samples and self.args.samples < total:
            random.seed(42)
            indices = random.sample(range(total), self.args.samples)
            indices.sort()
            print(f"Using {len(indices)}/{total} samples (seed=42)")
        else:
            indices = list(range(total))
            print(f"Using all {total} samples")
        
        print(f"Generating summaries...\n")
        
        latencies = []
        import time
        
        for idx in indices:
            rec = ds[idx]
            context = rec.get("context", "")
            input_text = rec.get("input", "")
            
            prompt = f"{context}\n\nSummarize the meeting:\n{input_text}"
            
            prompt_file = Path("tmp_prompt.txt")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            
            output_file = self.output_dir / f"qmsum_test_{idx}.txt"
            
            cmd = [str(self.llama_cli), "-m", str(model_path), "-f", str(prompt_file),
                   "-n", str(self.args.tokens), "-no-cnv", "-ngl", "0"]
            
            start = time.time()
            with open("tmp_output.txt", "w", encoding="utf-8", errors='replace') as fout:
                result = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, 
                                      text=True, timeout=120)
            latency = time.time() - start
            
            if result.returncode != 0:
                if self.args.verbose and result.stderr:
                    print(f"[{idx}] FAILED: {result.stderr[:200]}")
                else:
                    print(f"[{idx}] FAILED")
                continue
            
            with open("tmp_output.txt", "r", encoding="utf-8", errors='replace') as fin:
                pred = fin.read().strip()
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(pred)
            
            latencies.append(latency)
            
            if len(latencies) % 10 == 0 or len(latencies) < 3:
                print(f"[{len(latencies)}/{len(indices)}] Generated (lat={latency:.1f}s)")
        
        if os.path.exists("tmp_output.txt"):
            os.remove("tmp_output.txt")
        if os.path.exists("tmp_prompt.txt"):
            os.remove("tmp_prompt.txt")
        
        print(f"\nEvaluating with ROUGE...")
        
        rouge = evaluate.load("rouge")
        pattern = re.compile(r"qmsum_test_(\d+)\.txt")
        predictions = []
        references = []
        sample_ids = []
        
        for fname in sorted(os.listdir(self.output_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            idx = int(m.group(1))
            if idx not in indices:
                continue
            
            with open(self.output_dir / fname, "r", encoding="utf-8", errors='replace') as f:
                pred = f.read().strip()
            
            if idx in ref_map:
                sample_ids.append(idx)
                predictions.append(pred)
                references.append(ref_map[idx])
        
        if not predictions:
            print("No valid predictions")
            return {"rouge_l": 0.0, "rouge_1": 0.0, "rouge_2": 0.0, 
                   "avg_latency": 0.0, "num_samples": 0}
        
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        avg_lat = np.mean(latencies) if latencies else 0.0
        
        print(f"\nResults: ROUGE-L={result['rougeL']:.3f} ROUGE-1={result['rouge1']:.3f} ROUGE-2={result['rouge2']:.3f} latency={avg_lat:.2f}s")
        
        return {
            "rouge_l": result['rougeL'],
            "rouge_1": result['rouge1'],
            "rouge_2": result['rouge2'],
            "avg_latency": avg_lat,
            "num_samples": len(predictions)
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
                "rouge_l": metrics["rouge_l"],
                "rouge_1": metrics["rouge_1"],
                "rouge_2": metrics["rouge_2"],
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
    p.add_argument("--samples", type=int, default=50, help="Number of samples (default: 50)")
    p.add_argument("--tokens", type=int, default=200, help="Tokens to generate (default: 200)")
    p.add_argument("-v", "--verbose", action="store_true", help="Show error messages")
    
    try:
        LongBenchEval(p.parse_args()).run()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()


