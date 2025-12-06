#!/usr/bin/env python3
"""
Multi-Model Benchmarking Script
Downloads GGUF models from HuggingFace, runs llama-bench on device, and saves results to CSV.
"""

import os
import sys
import subprocess
import argparse
import csv
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Setup logging
datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"logs/benchmark_{datetime_str}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Model definitions with HuggingFace repos and quantization levels
MODEL_CONFIGS = [
    # Llama 3.2 models (different sizes)
    {
        "name": "llama-3.2-1b",
        "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0", "f16"],
        "param_size": "1B",
        "layers": 16,
    },
    {
        "name": "llama-3.2-3b",
        "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "3B",
        "layers": 28,
    },
    
    # Llama 3.1 7B (Llama 3.2 doesn't have 7B variant)
    {
        "name": "llama-3.1-8b",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "8B",
        "layers": 32,
    },
    
    # Qwen models (different sizes)
    {
        "name": "qwen2.5-0.5b",
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "quantizations": ["q4_k_m", "q8_0", "f16"],
        "param_size": "0.5B",
        "layers": 24,
    },
    {
        "name": "qwen2.5-1.5b",
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "quantizations": ["q4_k_m", "q8_0"],
        "param_size": "1.5B",
        "layers": 28,
    },
    {
        "name": "qwen2.5-3b",
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "quantizations": ["q4_k_m", "q8_0"],
        "param_size": "3B",
        "layers": 36,
    },
    {
        "name": "qwen2.5-7b",
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "quantizations": ["q4_k_m", "q8_0"],
        "param_size": "7B",
        "layers": 28,
    },
    
    # Mistral 7B
    {
        "name": "mistral-7b-v0.3",
        "hf_repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "7B",
        "layers": 32,
    },
    
    # Phi models (different sizes)
    {
        "name": "phi-3.5-mini",
        "hf_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "3.8B",
        "layers": 32,
    },
    {
        "name": "phi-2",
        "hf_repo": "TheBloke/phi-2-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "2.7B",
        "layers": 32,
    },
    
    # GPT-2 models (different sizes)
    {
        "name": "gpt2",
        "hf_repo": "mradermacher/gpt2-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "124M",
        "layers": 12,
    },
    {
        "name": "gpt2-medium",
        "hf_repo": "mradermacher/gpt2-medium-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "355M",
        "layers": 24,
    },
    {
        "name": "gpt2-large",
        "hf_repo": "mradermacher/gpt2-large-GGUF",
        "quantizations": ["Q4_K_M", "Q8_0"],
        "param_size": "774M",
        "layers": 36,
    },
]


class ModelBenchmarker:
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).parent
        self.models_dir = Path(args.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device_path = args.device_path
        self.adb_serial = args.adb_serial
        
        # Scripts
        self.push_script = self.script_dir / "scripts" / "push-model.sh"
        self.bench_script = self.script_dir / "scripts" / "run-bench.sh"
        
        # Validate scripts exist
        if not self.push_script.exists():
            raise FileNotFoundError(f"Push script not found: {self.push_script}")
        if not self.bench_script.exists():
            raise FileNotFoundError(f"Bench script not found: {self.bench_script}")
        
        # Results - use fixed filename for resumable benchmarks
        self.results = []
        self.csv_file = "benchmarks.csv"
        
        # Load existing results if CSV exists
        self.existing_results = self.load_existing_results()
    
    def load_existing_results(self) -> dict:
        """Load existing benchmark results from CSV to enable resumable benchmarks."""
        existing = {}
        
        if not Path(self.csv_file).exists():
            logger.info("No existing benchmarks.csv found - starting fresh")
            return existing
        
        try:
            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create unique key for each model config
                    key = (row['model_name'], row['quantization'])
                    existing[key] = row
            
            logger.info(f"Loaded {len(existing)} existing benchmark results from {self.csv_file}")
            return existing
            
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")
            return {}
    
    def is_already_benchmarked(self, model_name: str, quantization: str) -> bool:
        """Check if a model+quantization combination has already been benchmarked."""
        key = (model_name, quantization)
        return key in self.existing_results
        
    def download_model(self, hf_repo: str, filename: str) -> Optional[Path]:
        """Download a GGUF model from HuggingFace."""
        local_path = self.models_dir / filename
        
        # Check if already downloaded
        if local_path.exists():
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"Model already exists: {filename} ({file_size_mb:.2f} MB)")
            return local_path
        
        logger.info(f"Downloading {filename} from {hf_repo}...")
        
        try:
            # Use huggingface-cli to download
            cmd = [
                "huggingface-cli",
                "download",
                hf_repo,
                filename,
                "--local-dir", str(self.models_dir),
                "--local-dir-use-symlinks", "False"
            ]
            
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return None
            
            if local_path.exists():
                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                logger.info(f"Downloaded successfully: {filename} ({file_size_mb:.2f} MB)")
                return local_path
            else:
                logger.error(f"Download completed but file not found: {local_path}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {filename}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
    
    def get_model_size_mb(self, file_path: Path) -> float:
        """Get model file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def push_to_device(self, local_path: Path) -> bool:
        """Push model to Android device using push-model.sh."""
        logger.info(f"Pushing {local_path.name} to device...")
        
        cmd = ["bash", str(self.push_script), str(local_path), self.device_path]
        if self.adb_serial:
            cmd.extend(["--serial", self.adb_serial])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully pushed {local_path.name}")
                return True
            else:
                # Log both stdout and stderr for better debugging
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                if not error_msg:
                    error_msg = f"Push script exited with code {result.returncode}"
                logger.error(f"Failed to push model: {error_msg}")
                if result.stdout:
                    logger.debug(f"Push stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Push stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Push timed out for {local_path.name}")
            return False
        except Exception as e:
            logger.error(f"Error pushing model: {e}")
            return False
    
    def run_benchmark(self, model_filename: str) -> Optional[Dict[str, float]]:
        """Run llama-bench on device and parse results."""
        logger.info(f"Running benchmark for {model_filename}...")
        
        env = os.environ.copy()
        env["M"] = model_filename
        
        if self.args.device:
            env["D"] = self.args.device
        if self.adb_serial:
            env["S"] = self.adb_serial
        if self.args.batch_size:
            env["BATCH_SIZE"] = str(self.args.batch_size)
        
        try:
            cmd = ["bash", str(self.bench_script)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.args.bench_timeout,
                env=env
            )
            
            if result.stdout:
                logger.debug(f"Benchmark stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"Benchmark stderr:\n{result.stderr}")
            
            # Parse output
            combined_output = result.stdout + "\n" + result.stderr
            metrics = self.parse_benchmark_output(combined_output)
            
            if metrics:
                logger.info(f"Benchmark results - Prefill: {metrics['prefill']:.2f} t/s, Decode: {metrics['decode']:.2f} t/s")
                return metrics
            else:
                logger.error("Failed to parse benchmark output")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {self.args.bench_timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return None
    
    def parse_benchmark_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse llama-bench output to extract prefill and decode speeds."""
        result = {"prefill": None, "decode": None}
        
        if not output:
            return None
        
        for line in output.strip().split('\n'):
            if not line.strip() or line.strip().startswith('|---'):
                continue
            
            # Look for pp512 (prefill) and tg128 (decode) lines
            if '|' in line and ('pp512' in line or 'tg128' in line):
                parts = [p.strip() for p in line.split('|')]
                
                test_type = None
                if 'pp512' in line:
                    test_type = 'prefill'
                elif 'tg128' in line:
                    test_type = 'decode'
                
                if test_type:
                    # Look for the token/s value (format: "123.45 ± 1.23")
                    for part in reversed(parts):
                        match = re.search(r'([\d.]+)\s*±', part)
                        if match:
                            try:
                                ts_value = float(match.group(1))
                                result[test_type] = ts_value
                                break
                            except ValueError:
                                continue
        
        if result["prefill"] is None or result["decode"] is None:
            logger.error("Could not find both prefill and decode metrics")
            return None
        
        return result
    
    def remove_from_device(self, model_filename: str) -> bool:
        """Remove model from device to save space."""
        logger.info(f"Removing {model_filename} from device...")
        
        device_file = f"{self.device_path}{model_filename}"
        cmd = ["adb"]
        if self.adb_serial:
            cmd.extend(["-s", self.adb_serial])
        cmd.extend(["shell", "rm", "-f", device_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Removed {model_filename} from device")
                return True
            else:
                logger.warning(f"Failed to remove model from device: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Error removing model: {e}")
            return False
    
    def get_gguf_filename(self, model_config: Dict, quant: str) -> str:
        """Generate expected GGUF filename based on model and quantization."""
        name = model_config["name"]
        
        # Different repos use different naming conventions
        if "llama-3.2" in name:
            # bartowski uses: Llama-3.2-1B-Instruct-Q4_K_M.gguf
            size = model_config["param_size"]
            return f"Llama-3.2-{size}-Instruct-{quant}.gguf"
        elif "llama-3.1" in name:
            # bartowski uses: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
            size = model_config["param_size"]
            return f"Meta-Llama-3.1-{size}-Instruct-{quant}.gguf"
        elif "mistral" in name:
            # bartowski uses: Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
            return f"Mistral-7B-Instruct-v0.3-{quant}.gguf"
        elif "qwen" in name:
            # Qwen uses: qwen2.5-0.5b-instruct-q4_k_m.gguf
            size = model_config["param_size"].lower()
            return f"qwen2.5-{size}-instruct-{quant.lower()}.gguf"
        elif "phi-3.5" in name:
            return f"Phi-3.5-mini-instruct-{quant}.gguf"
        elif "phi-2" in name:
            return f"phi-2.{quant}.gguf"
        elif "gpt2" in name:
            # mradermacher uses: gpt2.Q4_K_M.gguf
            if name == "gpt2":
                return f"gpt2.{quant}.gguf"
            elif name == "gpt2-medium":
                return f"gpt2-medium.{quant}.gguf"
            elif name == "gpt2-large":
                return f"gpt2-large.{quant}.gguf"
        
        # Fallback
        return f"{name}-{quant}.gguf"
    
    def benchmark_model(self, model_config: Dict, quant: str) -> Optional[Dict]:
        """Benchmark a single model with specific quantization."""
        model_name = model_config["name"]
        hf_repo = model_config["hf_repo"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Benchmarking: {model_name} ({quant})")
        logger.info(f"{'='*80}")
        
        # Generate filename
        filename = self.get_gguf_filename(model_config, quant)
        
        # Step 1: Download model
        local_path = self.download_model(hf_repo, filename)
        if not local_path:
            logger.error(f"Failed to download {filename}, skipping...")
            return None
        
        # Step 2: Push to device
        if not self.push_to_device(local_path):
            logger.error(f"Failed to push {filename}, skipping...")
            return None
        
        # Wait a bit for the device to settle
        time.sleep(2)
        
        # Step 3: Run benchmark
        metrics = self.run_benchmark(filename)
        if not metrics:
            logger.error(f"Failed to benchmark {filename}")
            # Still try to clean up
            self.remove_from_device(filename)
            return None
        
        # Step 4: Remove from device
        self.remove_from_device(filename)
        
        # Collect results
        result = {
            "model_name": model_name,
            "param_size": model_config["param_size"],
            "quantization": quant,
            "layers": model_config["layers"],
            "model_size_mb": self.get_model_size_mb(local_path),
            "hf_repo": hf_repo,
            "filename": filename,
            "prefill_tokens_per_sec": metrics["prefill"],
            "decode_tokens_per_sec": metrics["decode"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✓ Completed {model_name} ({quant})")
        return result
    
    def save_results_to_csv(self):
        """Save all results to CSV file, merging with existing results."""
        if not self.results and not self.existing_results:
            logger.warning("No results to save")
            return

        csv_path = self.csv_file
        fieldnames = [
            "model_name",
            "param_size",
            "quantization",
            "layers",
            "model_size_mb",
            "prefill_tokens_per_sec",
            "decode_tokens_per_sec",
            "hf_repo",
            "filename",
            "timestamp"
        ]

        # Combine existing results with new results
        all_results = list(self.existing_results.values()) + self.results

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {csv_path}")
        logger.info(f"Existing results: {len(self.existing_results)}")
        logger.info(f"New results this run: {len(self.results)}")
        logger.info(f"Total models in CSV: {len(all_results)}")
        logger.info(f"{'='*80}")
    
    def run(self):
        """Main execution loop."""
        # Filter models if in debug mode
        if self.args.debug:
            logger.info("DEBUG MODE: Testing smallest model only")
            # Find smallest model (gpt2 124M)
            test_models = [m for m in MODEL_CONFIGS if m["name"] == "gpt2"]
            if not test_models:
                test_models = [MODEL_CONFIGS[0]]  # Fallback to first model
            test_models[0]["quantizations"] = [test_models[0]["quantizations"][0]]  # Only first quant
        else:
            test_models = MODEL_CONFIGS
        
        total_tasks = sum(len(m["quantizations"]) for m in test_models)
        
        # Count how many are already done
        already_done = sum(
            1 for m in test_models 
            for q in m["quantizations"] 
            if self.is_already_benchmarked(m["name"], q)
        )
        remaining_tasks = total_tasks - already_done
        completed = 0
        
        logger.info(f"\nStarting benchmark run")
        logger.info(f"Total model configurations: {total_tasks}")
        logger.info(f"Already benchmarked: {already_done}")
        logger.info(f"Remaining to benchmark: {remaining_tasks}")
        logger.info(f"Results will be saved to: {self.csv_file}\n")
        
        for model_config in test_models:
            for quant in model_config["quantizations"]:
                completed += 1
                logger.info(f"\nProgress: {completed}/{total_tasks}")
                
                # Check if already benchmarked
                if self.is_already_benchmarked(model_config["name"], quant):
                    logger.info(f"✓ Skipping {model_config['name']} ({quant}) - already benchmarked")
                    continue
                
                result = self.benchmark_model(model_config, quant)
                if result:
                    self.results.append(result)
                    # Save incrementally (merges with existing results)
                    self.save_results_to_csv()
                
                # Brief pause between models
                if completed < total_tasks:
                    time.sleep(3)
        
        # Final save
        self.save_results_to_csv()
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Total configurations: {total_tasks}")
        logger.info(f"Skipped (already done): {already_done}")
        logger.info(f"New benchmarks this run: {len(self.results)}")
        logger.info(f"Results: {self.csv_file}")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple GGUF models using llama-bench on Android device"
    )
    
    # Required arguments
    parser.add_argument(
        "--models-dir",
        default="./models/benchmark",
        help="Directory to store downloaded models (default: ./models/benchmark)"
    )
    parser.add_argument(
        "--device-path",
        default="/data/local/tmp/gguf/",
        help="Device path for models (default: /data/local/tmp/gguf/)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--adb-serial",
        help="ADB device serial number (optional)"
    )
    parser.add_argument(
        "--device",
        default="none",
        help="Device type for llama-bench (default: none)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for benchmark (default: 128)"
    )
    parser.add_argument(
        "--bench-timeout",
        type=int,
        default=300,
        help="Timeout for each benchmark in seconds (default: 300)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only test the smallest model (gpt2)"
    )
    
    args = parser.parse_args()
    
    try:
        benchmarker = ModelBenchmarker(args)
        benchmarker.run()
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
