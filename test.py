#!/usr/bin/env python3

import os
import json
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation import get_truthfulqa_score, get_longbench_score, get_throughput
from config import EVALUATION_CONFIG, LOGS_DIR

datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOGS_DIR / f"test_{datetime_str}.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments with config")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results (default: ./results)")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting experiment with config: {args.config}")
    cli_args = []
    if config.get("threads"):
        cli_args.extend(["-t", str(config["threads"])])
    if config.get("ctx_size"):
        cli_args.extend(["--ctx-size", str(config["ctx_size"])])
    if config.get("batch_size"):
        cli_args.extend(["--batch-size", str(config["batch_size"])])
    if config.get("kv_cache_token_quant"):
        cli_args.extend(["-ctk", config["kv_cache_token_quant"]])
    if config.get("kv_cache_value_quant"):
        cli_args.extend(["-ctv", config["kv_cache_value_quant"]])
    if config.get("temperature"):
        cli_args.extend(["--temp", str(config["temperature"])])
    if config.get("seed"):
        cli_args.extend(["--seed", str(config["seed"])])
    if config.get("n_gpu_layers"):
        cli_args.extend(["-ngl", str(config["n_gpu_layers"])])
    if config.get("rope_scale"):
        cli_args.extend(["--rope-scale", str(config["rope_scale"])])
    if config.get("rope_freq_base"):
        cli_args.extend(["--rope-freq-base", str(config["rope_freq_base"])])
    if config.get("rope_freq_scale"):
        cli_args.extend(["--rope-freq-scale", str(config["rope_freq_scale"])])
    if config.get("no_mmap"):
        cli_args.append("--no-mmap")
    if config.get("no_kv_offload"):
        cli_args.append("--no-kv-offload")
    if config.get("kv_offload"):
        cli_args.append("--kv-offload")
    if config.get("flash_attn") is False:
        cli_args.append("--no-flash")
    elif config.get("flash_attn") is True:
        cli_args.extend(["-fa", "on"])
    env_vars = {}
    if config.get("verbose") is not None:
        env_vars["V"] = str(config["verbose"])
    if config.get("experimental") is not None:
        env_vars["E"] = str(config["experimental"])
    if config.get("sched") is not None:
        env_vars["SCHED"] = str(config["sched"])
    if config.get("profile") is not None:
        env_vars["PROF"] = str(config["profile"])
    if config.get("opmask") is not None:
        env_vars["OPMASK"] = str(config["opmask"])
    if config.get("nhvx") is not None:
        env_vars["NHVX"] = str(config["nhvx"])
    if config.get("ndev") is not None:
        env_vars["NDEV"] = str(config["ndev"])
    if config.get("adbserial") is not None:
        env_vars["S"] = str(config["adbserial"])
    if config.get("branch") is not None:
        env_vars["B"] = str(config["branch"])
    if config.get("model") is not None:
        env_vars["M"] = config["model"]
    if config.get("device") is not None:
        env_vars["D"] = str(config["device"])
    # Set LLAMA_LOG_LEVEL to debug for verbose llama.cpp logging
    env_vars["LLAMA_LOG_LEVEL"] = "debug"
    
    original_env = {}
    for key, value in env_vars.items():
        if value is not None:
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    logger.info("Running TruthfulQA evaluation...")
    try:
        max_bleurt, accuracy = get_truthfulqa_score(
            script_path="./scripts/run-cli.sh",
            num_samples=config.get("truthfulqa_num_samples", EVALUATION_CONFIG["truthfulqa_num_samples"]),
            num_tokens=config.get("truthfulqa_num_tokens", EVALUATION_CONFIG["truthfulqa_num_tokens"]),
            random_seed=config.get("seed", 42),
            extra_args=cli_args
        )
        if max_bleurt is None or accuracy is None:
            logger.error("TruthfulQA evaluation failed - CLI returned error")
            results["results"]["truthfulqa"] = {"error": "CLI execution failed"}
        else:
            results["results"]["truthfulqa"] = {
                "max_bleurt_score": max_bleurt,
                "accuracy": accuracy
            }
            logger.info(f"TruthfulQA - Max BLEURT: {max_bleurt:.4f}, Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error running TruthfulQA: {e}", exc_info=True)
        results["results"]["truthfulqa"] = {"error": str(e)}
    logger.info("Running throughput benchmark...")
    try:
        bench_metrics = get_throughput(script_path="./scripts/run-bench.sh", config=config)
        if bench_metrics:
            results["results"]["throughput"] = {
                "prefill_tokens_per_sec": bench_metrics.get("prefill"),
                "decode_tokens_per_sec": bench_metrics.get("decode")
            }
            logger.info(f"Throughput - Prefill: {bench_metrics.get('prefill'):.2f} tokens/s, Decode: {bench_metrics.get('decode'):.2f} tokens/s")
        else:
            results["results"]["throughput"] = {"error": "Failed to parse benchmark output"}
            logger.error("Failed to parse throughput benchmark output")
    except Exception as e:
        logger.error(f"Error running throughput: {e}", exc_info=True)
        results["results"]["throughput"] = {"error": str(e)}
    logger.info("Running LongBench evaluation...")
    try:
        rougeL = get_longbench_score(
            local_prompt_dir="./prompt_files",
            device_prompt_prefix="/data/local/tmp/prompt_files",
            output_dir="./qmsum_outputs",
            cli_path="./scripts/run-cli.sh",
            n_benchmarks=config.get("longbench_n_benchmarks", EVALUATION_CONFIG["longbench_n_benchmarks"]),
            num_tokens=config.get("longbench_num_tokens", EVALUATION_CONFIG["longbench_num_tokens"]),
            extra_args=cli_args,
            random_seed=config.get("seed", 42)
        )
        results["results"]["longbench"] = {"rougeL": rougeL}
        logger.info(f"LongBench - ROUGE-L: {rougeL:.4f}")
    except Exception as e:
        logger.error(f"Error running LongBench: {e}", exc_info=True)
        results["results"]["longbench"] = {"error": str(e)}
    output_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    for key in env_vars.keys():
        if key in original_env:
            if original_env[key] is not None:
                os.environ[key] = original_env[key]
            elif key in os.environ:
                del os.environ[key]
        elif key in os.environ:
            del os.environ[key]
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
