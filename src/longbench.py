#!/usr/bin/env python3
import os
import re
import time
import subprocess
import random
import logging
from pathlib import Path
from datasets import load_dataset
import evaluate

logger = logging.getLogger(__name__)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_references():
    logger.info("Loading LongBench qmsum dataset...")
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
    
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    return ref_map


def run_one(cli_path: str, prompt_device_path: str, output_path: str, num_tokens=None, extra_args=None):
    if extra_args is None:
        extra_args = []
    cmd = [cli_path, "-no-cnv", "-f", prompt_device_path]
    if num_tokens is not None:
        cmd.extend(["-n", str(num_tokens)])
    cmd.extend(extra_args)
    if cli_path.endswith(".sh"):
        cmd = ["bash"] + cmd

    logger.debug(f"Running command: {' '.join(cmd)}")
    start = time.time()
    
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    
    with open(output_path, "w", encoding="utf-8", errors="replace") as fout:
        if proc.stdout:
            fout.write(proc.stdout)
    
    if proc.stdout:
        logger.debug(f"Output for {prompt_device_path} stdout:\n{proc.stdout}")
    if proc.stderr:
        logger.debug(f"Output for {prompt_device_path} stderr:\n{proc.stderr}")
    
    end = time.time()
    latency = end - start
    
    if proc.returncode != 0:
        logger.error(f"CLI failed for prompt {prompt_device_path} with return code {proc.returncode}")
        if proc.stderr:
            logger.error(f"Error output:\n{proc.stderr}")
    return latency


def run_benchmarks(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str,
                   cli_path: str, n_benchmarks=1, num_tokens=None, extra_args=None, random_seed=42):
    ensure_dir(output_dir)
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))
    
    if n_benchmarks < len(prompt_files):
        random.seed(random_seed)
        prompt_files = random.sample(prompt_files, n_benchmarks)
        prompt_files = sorted(prompt_files)
    
    latencies = []
    t0 = time.time()
    for pf in prompt_files:
        fname = pf.name
        prompt_dev_path = os.path.join(device_prompt_prefix, fname)
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)

        logger.info(f"Running prompt {fname} → {out_fname}")
        latency = run_one(cli_path, prompt_dev_path, out_path, num_tokens, extra_args)
        logger.info(f"  Completed in {latency:.3f}s")
        latencies.append((fname, latency))

    t1 = time.time()
    total = t1 - t0
    return latencies, total


def evaluate_outputs(output_dir: str, ref_map: dict):
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"qmsum_test_(\d+)\.txt")
    predictions = []
    references = []
    sample_ids = []

    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        outpath = os.path.join(output_dir, fname)
        with open(outpath, "r", encoding="utf-8", errors="replace") as f:
            pred = f.read().strip()

        if idx not in ref_map:
            logger.warning(f"No reference for sample {idx}, skipping")
            continue

        ref = ref_map[idx]
        sample_ids.append(idx)
        predictions.append(pred)
        references.append(ref)

    if not predictions:
        return None, None

    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)

    results = []
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        results.append((idx, rl))

    return results, result


def get_longbench_score(local_prompt_dir="./prompt_files", device_prompt_prefix="/data/local/tmp/prompt_files",
                        output_dir="./qmsum_outputs", cli_path="./scripts/run-cli.sh",
                        n_benchmarks=1, num_tokens=None, extra_args=None, random_seed=42):
    logger.info(f"Starting LongBench evaluation (n_benchmarks={n_benchmarks}, num_tokens={num_tokens}, seed={random_seed})...")
    ref_map = load_references()
    logger.info(f"Loaded {len(ref_map)} references")
    
    latencies, total_time = run_benchmarks(
        local_prompt_dir, device_prompt_prefix, output_dir, cli_path, n_benchmarks, num_tokens, extra_args, random_seed
    )
    logger.info(f"Completed {len(latencies)} benchmarks in {total_time:.2f}s (avg: {total_time/len(latencies):.2f}s)")
    
    logger.info("Evaluating outputs with ROUGE...")
    per_sample_scores, aggregated = evaluate_outputs(output_dir, ref_map)
    
    if aggregated is None:
        logger.error("No outputs to evaluate")
        return None
    
    rougeL = aggregated['rougeL']
    logger.info(f"LongBench ROUGE-L: {rougeL:.4f}")
    return rougeL

