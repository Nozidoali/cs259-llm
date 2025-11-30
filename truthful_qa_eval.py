#!/usr/bin/env python3
import os
import random
import subprocess
import time

from datasets import load_dataset
import evaluate
import numpy as np

NUM_SAMPLES = 100
RANDOM_SEED = 42


def run_evaluate(extra_args=None):
    extra_args = extra_args or []
    script_path = "./run-cli.sh"
    log_filename = "debug.log"

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    total_samples = len(ds)
    if NUM_SAMPLES is not None and NUM_SAMPLES < total_samples:
        random.seed(RANDOM_SEED)
        indices = random.sample(range(total_samples), NUM_SAMPLES)
        indices.sort()
        ds = ds.select(indices)
        n = NUM_SAMPLES
        print(f"Using {n} samples out of {total_samples} (seed={RANDOM_SEED})")
    else:
        n = total_samples
        print(f"Using all {n} TruthfulQA samples")

    bleurt = evaluate.load("bleurt", "bleurt-large-128")

    with open(log_filename, "w", encoding="utf-8") as stderr_file:
        max_score_arr = []
        acc_score_arr = []
        latency_arr = []

        for i, rec in enumerate(ds):
            question = rec["question"].replace('"', " ").replace("'", " ")
            correct_answers = rec["correct_answers"]
            incorrect_answers = rec["incorrect_answers"]

            num_tokens = 25
            cmd = ["bash", script_path, "-no-cnv", "-p", f"'{question}'", "-n", str(num_tokens)] + extra_args
            start = time.time()
            with open("tmp_output.txt", "w", encoding="utf-8") as fout:
                proc = subprocess.run(cmd, stdout=fout, stderr=stderr_file, text=True)
            latency = time.time() - start
            if proc.returncode != 0:
                print(f"[ERROR] CLI failed for prompt: {question}")
                return -1, -1

            with open("tmp_output.txt", "r", encoding="utf-8") as fin:
                pred = fin.read().strip()
            os.remove("tmp_output.txt")

            predictions_true = [pred] * len(correct_answers)
            predictions_false = [pred] * len(incorrect_answers)
            score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)["scores"]
            score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)["scores"]
            max_score = max(score_true)
            acc_score = int(max_score > max(score_false))

            print(f"sample {i}: latency {latency:.3f}s, max_score {max_score:.3f}, acc {acc_score}")
            max_score_arr.append(max_score)
            acc_score_arr.append(acc_score)
            latency_arr.append(latency)

        avg_accuracy = sum(acc_score_arr) / n
        avg_max = sum(max_score_arr) / len(max_score_arr)
        avg_latency = sum(latency_arr) / len(latency_arr)
        print(f"\navg max score: {avg_max:.3f}")
        print(f"avg accuracy: {avg_accuracy:.3f}")
        print(f"avg latency: {avg_latency:.3f}s")


def main():
    run_evaluate()


if __name__ == "__main__":
    main()

