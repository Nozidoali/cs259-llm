#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import numpy as np
import re
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
import evaluate

from moe_inference import MoEInference
from config import BLEURT_CONFIG, DATASET_CONFIG, TRUTHFULQA_CACHE_DIR

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def evaluate_truthfulqa(
    inference: MoEInference,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 50,
    random_seed: int = 42,
):
    logger.info("Loading TruthfulQA dataset...")
    try:
        ds = load_dataset(
            DATASET_CONFIG["name"],
            DATASET_CONFIG["config"],
            split=DATASET_CONFIG["split"],
            cache_dir=str(TRUTHFULQA_CACHE_DIR),
        )
    except Exception as e:
        logger.warning(f"Cache error: {e}, trying without cache...")
        ds = load_dataset(
            DATASET_CONFIG["name"],
            DATASET_CONFIG["config"],
            split=DATASET_CONFIG["split"],
        )
    
    if num_samples is not None and num_samples < len(ds):
        import random
        random.seed(random_seed)
        indices = random.sample(range(len(ds)), num_samples)
        indices.sort()
        ds = ds.select(indices)
        logger.info(f"Using {num_samples}/{len(ds)} samples (seed={random_seed})")
    else:
        logger.info(f"Using all {len(ds)} TruthfulQA samples")
    
    logger.info("Loading BLEURT evaluator...")
    bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
    
    max_score_arr = []
    acc_score_arr = []
    
    logger.info("Evaluating on TruthfulQA...")
    for i, example in enumerate(ds):
        question = example.get("question", "")
        correct_answers = example.get("correct_answers", [])
        incorrect_answers = example.get("incorrect_answers", [])
        
        if not question or not correct_answers or not incorrect_answers:
            continue
        
        prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
        
        result = inference.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            return_gating_probs=False,
        )
        
        pred = result["text"].strip()
        
        if not pred:
            continue
        
        predictions_true = [pred] * len(correct_answers)
        predictions_false = [pred] * len(incorrect_answers)
        score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)["scores"]
        score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)["scores"]
        max_score = max(score_true) if score_true else 0.0
        acc_score = int(max(score_true) > max(score_false)) if score_true and score_false else 0
        
        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(ds)} - max_score: {max_score:.3f}, acc: {acc_score}")
            clear_gpu_memory()
    
    avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
    accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0
    
    logger.info(f"TruthfulQA Evaluation Complete:")
    logger.info(f"  Average Max BLEURT Score: {avg_max_score:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    return {
        "bleurt_max_score": avg_max_score,
        "bleurt_accuracy": accuracy,
        "num_samples": len(max_score_arr),
    }


def evaluate_qmsum(
    inference: MoEInference,
    prompt_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 200,
    random_seed: int = 42,
):
    logger.info("Loading QMSum references from dataset...")
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
    
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    
    logger.info(f"Loaded {len(ref_map)} references")
    
    if prompt_dir is None:
        from config import WORK_DIR
        prompt_dir_path = WORK_DIR / "prompt_files"
    else:
        prompt_dir_path = Path(prompt_dir)
    
    logger.info(f"Loading QMSum prompts from: {prompt_dir_path}")
    prompt_files = sorted(prompt_dir_path.glob("qmsum_test_*.prompt.txt"))
    
    if num_samples is not None and num_samples < len(prompt_files):
        import random
        random.seed(random_seed)
        prompt_files = random.sample(prompt_files, num_samples)
        prompt_files = sorted(prompt_files)
        logger.info(f"Using {num_samples}/{len(prompt_files)} samples (seed={random_seed})")
    else:
        logger.info(f"Using all {len(prompt_files)} QMSum prompt files")
    
    logger.info("Loading ROUGE evaluator...")
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    sample_ids = []
    model = inference.model
    tokenizer = inference.tokenizer
    device = inference.device
    
    pattern = re.compile(r"qmsum_test_(\d+)\.prompt\.txt")
    
    logger.info("Evaluating on QMSum...")
    for i, pf in enumerate(prompt_files):
        fname = pf.name
        m = pattern.match(fname)
        if not m:
            logger.warning(f"Skipping unrecognized file name: {fname}")
            continue
        
        idx = int(m.group(1))
        if idx not in ref_map:
            logger.warning(f"No reference for sample {idx}, skipping")
            continue
        
        ref = ref_map[idx]
        
        try:
            with open(pf, "r", encoding="utf-8", errors="replace") as f:
                prompt_text = f.read().strip()
            
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.1
                    )
                except AttributeError as e:
                    if "'DynamicCache' object has no attribute 'seen_tokens'" in str(e) or "seen_tokens" in str(e):
                        logger.warning(f"  Cache error detected, retrying with use_cache=False")
                        outputs = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask', None),
                            max_new_tokens=max_new_tokens,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            use_cache=False
                        )
                    else:
                        raise
            
            if hasattr(outputs, 'sequences'):
                full_sequence = outputs.sequences[0]
            else:
                full_sequence = outputs[0]
            
            generated_ids = full_sequence[input_length:]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
        except Exception as e:
            logger.error(f"Sample {idx} ({fname}): Generation failed - {e}")
            clear_gpu_memory()
            continue
        
        if pred and ref:
            predictions.append(pred)
            references.append(ref)
            sample_ids.append(idx)
        else:
            logger.warning(f"Sample {idx}: Skipping - empty prediction or reference")
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(prompt_files)}")
            clear_gpu_memory()
    
    if not predictions:
        logger.warning("No valid predictions generated")
        return {
            "rougeL": 0.0,
            "num_samples": 0,
        }
    
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rougeL = result.get("rougeL", 0.0)
    
    logger.info(f"QMSum Evaluation Complete:")
    logger.info(f"  ROUGE-L Score: {rougeL:.4f}")
    
    return {
        "rougeL": rougeL,
        "rouge1": result.get("rouge1", 0.0),
        "rouge2": result.get("rouge2", 0.0),
        "num_samples": len(predictions),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MoE model on TruthfulQA and QMSum datasets")
    
    parser.add_argument("--saved_model_path", type=str, default=None,
                       help="Path to saved MoE model directory")
    parser.add_argument("--model1_path", type=str, default=None,
                       help="Path to first finetuned model")
    parser.add_argument("--model2_path", type=str, default=None,
                       help="Path to second finetuned model")
    parser.add_argument("--gating_model_path", type=str, default=None,
                       help="Path to gating network")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base model for embeddings")
    parser.add_argument("--routing_mode", type=str, default="weighted_sum",
                       choices=["weighted_sum", "select_one"],
                       help="Routing mode")
    
    parser.add_argument("--truthfulqa_num_samples", type=int, default=None,
                       help="Number of TruthfulQA samples to evaluate (default: all)")
    parser.add_argument("--truthfulqa_max_tokens", type=int, default=50,
                       help="Maximum tokens for TruthfulQA generation")
    
    parser.add_argument("--qmsum_prompt_dir", type=str, default=None,
                       help="Directory containing QMSum prompt files (default: WORK_DIR/prompt_files)")
    parser.add_argument("--qmsum_num_samples", type=int, default=None,
                       help="Number of QMSum samples to evaluate (default: all)")
    parser.add_argument("--qmsum_max_tokens", type=int, default=200,
                       help="Maximum tokens for QMSum generation")
    
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sampling")
    
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save results JSON file")
    
    parser.add_argument("--skip_truthfulqa", action="store_true",
                       help="Skip TruthfulQA evaluation")
    parser.add_argument("--skip_qmsum", action="store_true",
                       help="Skip QMSum evaluation")
    
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use for inference (default: auto-detect)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("MoE Model Evaluation")
    logger.info("=" * 60)
    
    logger.info("Initializing MoE inference...")
    logger.info("Clearing GPU cache before loading models...")
    clear_gpu_memory()
    
    device = None
    if args.device:
        device = torch.device(args.device)
        logger.info(f"Using specified device: {device}")
    
    inference = MoEInference(
        saved_model_path=args.saved_model_path,
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        gating_model_path=args.gating_model_path,
        base_model_path=args.base_model_path,
        routing_mode=args.routing_mode,
        device=device,
    )
    
    logger.info("Model loaded. Clearing GPU cache...")
    clear_gpu_memory()
    
    results = {}
    
    if not args.skip_truthfulqa:
        logger.info("=" * 60)
        logger.info("Evaluating on TruthfulQA")
        logger.info("=" * 60)
        truthfulqa_results = evaluate_truthfulqa(
            inference=inference,
            num_samples=args.truthfulqa_num_samples,
            max_new_tokens=args.truthfulqa_max_tokens,
            random_seed=args.random_seed,
        )
        results["truthfulqa"] = truthfulqa_results
    else:
        logger.info("Skipping TruthfulQA evaluation")
    
    if not args.skip_qmsum:
        logger.info("=" * 60)
        logger.info("Evaluating on QMSum")
        logger.info("=" * 60)
        qmsum_results = evaluate_qmsum(
            inference=inference,
            prompt_dir=args.qmsum_prompt_dir,
            num_samples=args.qmsum_num_samples,
            max_new_tokens=args.qmsum_max_tokens,
            random_seed=args.random_seed,
        )
        results["qmsum"] = qmsum_results
    else:
        logger.info("Skipping QMSum evaluation")
    
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    
    if "truthfulqa" in results:
        logger.info(f"TruthfulQA:")
        logger.info(f"  BLEURT Max Score: {results['truthfulqa']['bleurt_max_score']:.4f}")
        logger.info(f"  Accuracy: {results['truthfulqa']['bleurt_accuracy']:.4f}")
        logger.info(f"  Samples: {results['truthfulqa']['num_samples']}")
    
    if "qmsum" in results:
        logger.info(f"QMSum:")
        logger.info(f"  ROUGE-L: {results['qmsum']['rougeL']:.4f}")
        logger.info(f"  ROUGE-1: {results['qmsum']['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {results['qmsum']['rouge2']:.4f}")
        logger.info(f"  Samples: {results['qmsum']['num_samples']}")
    
    logger.info("=" * 60)
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()

