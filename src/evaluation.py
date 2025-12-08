import os
import random
import re
import time
import subprocess
import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset as hf_load_dataset
import evaluate
from config import BLEURT_CONFIG, DATASET_CONFIG

logger = logging.getLogger(__name__)

def get_truthfulqa_score(script_path="./scripts/run-cli.sh", num_samples=100, num_tokens=25, random_seed=42, extra_args=None):
    extra_args = extra_args or []
    logger.info(f"Loading TruthfulQA dataset (samples={num_samples}, tokens={num_tokens}, seed={random_seed})...")
    try:
        ds = hf_load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except (TypeError, AttributeError) as e:
        logger.warning(f"Cache error: {e}, clearing cache...")
        import shutil
        cache_dirs = [os.path.expanduser("~/.cache/huggingface/datasets/truthful_qa"), os.path.expanduser("~/.cache/huggingface/datasets/truthfulqa___truthful_qa"), "./data/truthfulqa_cache"]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        ds = hf_load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir=None)
    total_samples = len(ds)
    if num_samples is not None and num_samples < total_samples:
        random.seed(random_seed)
        indices = random.sample(range(total_samples), num_samples)
        indices.sort()
        ds = ds.select(indices)
        n = num_samples
        logger.info(f"Using {n}/{total_samples} samples (seed={random_seed})")
    else:
        n = total_samples
        logger.info(f"Using all {n} TruthfulQA samples")
    
    # Collect all predictions first (matching rmoe/evaluate.py approach)
    all_predictions = []
    all_correct_refs = []
    all_incorrect_refs = []
    
    logger.info("Generating predictions...")
    for i, rec in enumerate(ds):
        question = rec.get("question", "")
        correct_answers = rec.get("correct_answers", [])
        incorrect_answers = rec.get("incorrect_answers", [])
        if not question or not correct_answers or not incorrect_answers:
            continue
        
        # Use format_template matching rmoe/evaluate.py
        prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
        # Clean question for CLI
        question_clean = question.replace('"', " ").replace("'", " ")
        
        cmd = ["bash", script_path, "-no-cnv", "-p", f"'{question_clean}'", "-n", str(num_tokens)] + extra_args
        logger.debug(f"Running command for sample {i}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.stdout:
            logger.debug(f"Sample {i} stdout:\n{proc.stdout}")
        if proc.stderr:
            logger.debug(f"Sample {i} stderr:\n{proc.stderr}")
        if proc.returncode != 0:
            logger.error(f"CLI failed for sample {i} with return code {proc.returncode}")
            if proc.stderr:
                logger.error(f"Error output: {proc.stderr}")
            continue
        
        pred = proc.stdout.strip()
        if pred:
            all_predictions.append(pred)
            all_correct_refs.append(correct_answers)
            all_incorrect_refs.append(incorrect_answers)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{n} predictions generated")
    
    if not all_predictions:
        logger.error("No valid predictions generated")
        return None, None
    
    logger.info(f"Generated {len(all_predictions)} predictions, evaluating with BLEURT...")
    
    # Configure TensorFlow to use CPU only before loading BLEURT
    # This prevents TensorFlow from interfering with PyTorch's CUDA usage
    import os
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force TensorFlow to use CPU only
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    
    bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
    
    # Restore CUDA_VISIBLE_DEVICES for PyTorch
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    max_score_arr = []
    acc_score_arr = []
    batch_size = 32
    
    # Process in batches matching rmoe/evaluate.py
    for i in range(0, len(all_predictions), batch_size):
        batch_preds = all_predictions[i:i+batch_size]
        batch_correct = all_correct_refs[i:i+batch_size]
        batch_incorrect = all_incorrect_refs[i:i+batch_size]
        
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
            max_score_arr.append(max_score)
            acc_score_arr.append(acc_score)
    
    avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
    accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0
    logger.info(f"TruthfulQA complete - Avg max_score: {avg_max_score:.3f}, Avg accuracy: {accuracy:.3f}")
    return avg_max_score, accuracy

def load_longbench_references():
    logger.info("Loading LongBench qmsum dataset...")
    ds = hf_load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    return ref_map

def run_longbench_one(cli_path: str, prompt_device_path: str, output_path: str, num_tokens=None, extra_args=None):
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

def run_longbench_benchmarks(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str, cli_path: str, n_benchmarks=1, num_tokens=None, extra_args=None, random_seed=42):
    os.makedirs(output_dir, exist_ok=True)
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
        latency = run_longbench_one(cli_path, prompt_dev_path, out_path, num_tokens, extra_args)
        logger.info(f"  Completed in {latency:.3f}s")
        latencies.append((fname, latency))
    t1 = time.time()
    total = t1 - t0
    return latencies, total

def evaluate_longbench_outputs(output_dir: str, ref_map: dict):
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
    rougeL = result.get("rougeL", 0.0)
    per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)
    results = []
    per_sample_rougeL = per_sample.get("rougeL", [])
    for i, idx in enumerate(sample_ids):
        rl = per_sample_rougeL[i] if i < len(per_sample_rougeL) else 0.0
        results.append((idx, rl))
    result_dict = {"rougeL": rougeL}
    return results, result_dict

def get_longbench_score(local_prompt_dir="./prompt_files", device_prompt_prefix="/data/local/tmp/prompt_files", output_dir="./qmsum_outputs", cli_path="./scripts/run-cli.sh", n_benchmarks=1, num_tokens=None, extra_args=None, random_seed=42):
    logger.info(f"Starting LongBench evaluation (n_benchmarks={n_benchmarks}, num_tokens={num_tokens}, seed={random_seed})...")
    ref_map = load_longbench_references()
    logger.info(f"Loaded {len(ref_map)} references")
    latencies, total_time = run_longbench_benchmarks(local_prompt_dir, device_prompt_prefix, output_dir, cli_path, n_benchmarks, num_tokens, extra_args, random_seed)
    logger.info(f"Completed {len(latencies)} benchmarks in {total_time:.2f}s (avg: {total_time/len(latencies):.2f}s)")
    logger.info("Evaluating outputs with ROUGE...")
    per_sample_scores, aggregated = evaluate_longbench_outputs(output_dir, ref_map)
    if aggregated is None:
        logger.error("No outputs to evaluate")
        return None
    rougeL = aggregated.get("rougeL", 0.0)
    logger.info(f"LongBench ROUGE-L: {rougeL:.4f}")
    return rougeL

def parse_throughput_output(output):
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
        metrics = parse_throughput_output(combined_output)
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

