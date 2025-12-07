#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import csv
import re
from pathlib import Path
from datetime import datetime

class Eval:
    def __init__(self, args):
        self.args = args
        root = Path(__file__).parent.parent
        self.models_dir = Path(args.models_dir)
        self.llama_bench = root / "external/llama.cpp/build/bin/llama-bench"
        self.push_script = root / "scripts/push-model.sh"
        self.bench_script = root / "scripts/run-bench.sh"
        self.csv = self.models_dir / f"benchmark_{args.mode}.csv"
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
            w = csv.DictWriter(f, ["model_name", "model_size_mb", "prefill_tokens_per_sec", 
                                   "prefill_std_dev", "decode_tokens_per_sec", "decode_std_dev",
                                   "timestamp", "mode", "device"])
            w.writeheader()
            w.writerows(all_results.values())
    
    def _parse(self, out):
        r = {}
        for line in out.split('\n'):
            m = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
            if m and 'pp512' in line:
                r['pf'], r['pf_std'] = map(float, m.groups())
            elif m and 'tg128' in line:
                r['dc'], r['dc_std'] = map(float, m.groups())
        return r if 'pf' in r and 'dc' in r else None
    
    def _local(self, path):
        result = subprocess.run([str(self.llama_bench), "-m", str(path), "-p", "512", "-n", "128"],
                              capture_output=True, text=True, timeout=300)
        return self._parse(result.stdout + result.stderr)
    
    def _mobile(self, path):
        push = ["bash", str(self.push_script), str(path), "/data/local/tmp/gguf/"]
        if self.args.serial:
            push.extend(["--serial", self.args.serial])
        
        if subprocess.run(push, capture_output=True, timeout=600).returncode != 0:
            return None
        
        env = {**os.environ, "M": path.name, "D": "none"}
        if self.args.serial:
            env["S"] = self.args.serial
        
        result = subprocess.run(["bash", str(self.bench_script)], 
                              capture_output=True, text=True, timeout=300, env=env)
        return self._parse(result.stdout + result.stderr)
    
    def run(self):
        print(f"{self.args.mode.upper()} | {self.models_dir}")
        
        if self.args.model:
            model_path = self.models_dir / self.args.model
            if not model_path.exists():
                sys.exit(f"Model not found: {self.args.model}")
            models = [model_path]
        else:
            models = list(self.models_dir.glob("*.gguf"))
            if not models:
                sys.exit(f"No models in {self.models_dir}")
        
        done = sum(1 for m in models if m.name in self.existing)
        print(f"{len(models)} models ({done} done, {len(models)-done} remaining)\n")
        
        bench = self._mobile if self.args.mode == "mobile" else self._local
        
        for i, path in enumerate(models, 1):
            if path.name in self.existing:
                print(f"[{i}/{len(models)}] SKIP {path.name}")
                continue
            
            print(f"[{i}/{len(models)}] {path.name}... ", end="", flush=True)
            
            m = bench(path)
            if m:
                print(f"✓ {m['pf']:.1f}±{m['pf_std']:.1f} / {m['dc']:.1f}±{m['dc_std']:.1f} t/s")
                self.results.append({
                    "model_name": path.name,
                    "model_size_mb": path.stat().st_size / (1024 * 1024),
                    "prefill_tokens_per_sec": m['pf'],
                    "prefill_std_dev": m['pf_std'],
                    "decode_tokens_per_sec": m['dc'],
                    "decode_std_dev": m['dc_std'],
                    "timestamp": datetime.now().isoformat(),
                    "mode": self.args.mode,
                    "device": "none"
                })
                self._save()
            else:
                print("✗ FAILED")
        
        print(f"\n{len(self.results)} new benchmarks → {self.csv}")


def main():
    p = argparse.ArgumentParser()
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--local", action="store_const", const="local", dest="mode")
    m.add_argument("--mobile", action="store_const", const="mobile", dest="mode")
    p.add_argument("--models-dir", default="./models/rmoe-gguf")
    p.add_argument("-m", "--model", help="Specific model name (e.g., model.gguf)")
    p.add_argument("-s", "--serial")
    
    try:
        Eval(p.parse_args()).run()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()


