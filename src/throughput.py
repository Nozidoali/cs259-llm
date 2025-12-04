#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def parse_bench_output(output):
    result = {"prefill": None, "decode": None}
    if not output:
        return None
    
    for line in output.strip().split('\n'):
        if not line.strip() or line.strip().startswith('|---'):
            continue
        if '|' in line and ('pp512' in line or 'tg128' in line):
            parts = [p.strip() for p in line.split('|')]
            test_type = None
            if 'pp512' in line:
                test_type = 'prefill'
            elif 'tg128' in line:
                test_type = 'decode'
            
            if test_type:
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
        return None
    return result


def get_throughput(script_path="./scripts/run-bench.sh", model=None, config=None):
    if not os.path.exists(script_path):
        logger.error(f"Benchmark script not found at {script_path}")
        return None
    
    logger.info(f"Running throughput benchmark: {script_path}")
    try:
        cmd = ["bash", script_path]
        env = os.environ.copy()
        if config:
            if config.get("model"):
                env["M"] = config["model"]
            if config.get("device"):
                env["D"] = config["device"]
            if config.get("batch_size"):
                env["BATCH_SIZE"] = str(config["batch_size"])
        elif model:
            env["M"] = model
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        
        if result.stdout:
            logger.info(f"Benchmark stdout:\n{result.stdout}")
        if result.stderr:
            logger.info(f"Benchmark stderr:\n{result.stderr}")
        
        combined_output = result.stdout
        if result.stderr:
            combined_output += "\n" + result.stderr
        
        if not combined_output:
            logger.error("Benchmark output is empty")
            return None
        
        metrics = parse_bench_output(combined_output)
        
        if not metrics or metrics.get("prefill") is None or metrics.get("decode") is None:
            logger.error(f"Failed to parse throughput. Output length: {len(combined_output)}")
            logger.error(f"Last 1000 chars of output: {combined_output[-1000:]}")
            logger.error(f"Full output lines containing 'pp512' or 'tg128':")
            for line in combined_output.split('\n'):
                if 'pp512' in line or 'tg128' in line:
                    logger.error(f"  {line}")
            return None
        
        logger.info(f"Parsed throughput - Prefill: {metrics.get('prefill'):.2f} tokens/s, Decode: {metrics.get('decode'):.2f} tokens/s")
        return metrics
    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out after 300s")
        return None
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        return None
