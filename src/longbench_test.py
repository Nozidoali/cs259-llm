#!/usr/bin/env python3
import os
import time
import subprocess
from pathlib import Path

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_one(cli_path: str, prompt_device_path: str, output_path: str, extra_args=None, stderr_file=None):
    """
    Run CLI with -f prompt_device_path, capture stdout → file, stderr → file and console on error.
    Returns the latency (in seconds).
    """
    if extra_args is None:
        extra_args = []
    cmd = [cli_path, "-no-cnv", "-f", prompt_device_path] + extra_args
    # If cli_path is a shell script, wrap with bash
    if cli_path.endswith(".sh"):
        cmd = ["bash"] + cmd

    start = time.time()
    # Capture stderr separately so we can both log it and display on error
    stderr_capture = subprocess.PIPE
    with open(output_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=stderr_capture, text=True)
    end = time.time()

    # Write stderr to log file if provided
    if stderr_file and proc.stderr:
        stderr_file.write(proc.stderr)
        stderr_file.flush()

    latency = end - start
    if proc.returncode != 0:
        # Print stderr to console
        print(f"[ERROR] CLI failed for prompt {prompt_device_path}:")
        if proc.stderr:
            # Print last few lines of stderr for debugging
            stderr_lines = proc.stderr.strip().split('\n')
            print('\n'.join(stderr_lines[-10:]))  # Last 10 lines
        else:
            print("(No error output captured)")
    return latency

def run_all(local_prompt_dir: str, device_prompt_prefix: str, output_dir: str,
            cli_path: str, extra_args=None):
    ensure_dir(output_dir)
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))
    
    latencies = []
    stderr_file = open("debug.log", 'w', encoding='utf-8')
    t0 = time.time()
    for pf in prompt_files:
        fname = pf.name  # e.g. "qmsum_test_0.prompt.txt"
        prompt_dev_path = os.path.join(device_prompt_prefix, fname)
        # derive output filename, strip ".prompt.txt"
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)

        print(f"Running prompt {fname} → output {out_fname}")
        latency = run_one(cli_path, prompt_dev_path, out_path, extra_args, stderr_file)
        print(f"  latency: {latency:.3f} s")
        latencies.append((fname, latency))

    t1 = time.time()
    total = t1 - t0
    return latencies, total

def main():
    local_prompt_dir = "./prompt_files"
    device_prompt_prefix = "/data/local/tmp/prompt_files"
    output_dir = "./qmsum_outputs"
    cli_path = "./scripts/run-cli.sh"
    
    print(f"Using script: {cli_path}")
    print(f"Log file: debug.log\n")
    
    # LongBench requires large context (11k+ tokens) - override ctx-size
    # This will override the ctx-size in the script via command-line args
    extra_args = []

    latencies, total_time = run_all(
        local_prompt_dir, device_prompt_prefix, output_dir, cli_path, extra_args
    )

    print("\n=== Benchmark Summary ===")
    for fname, lat in latencies:
        print(f"{fname}: {lat:.3f} s")
    print(f"Total time for {len(latencies)} samples: {total_time:.3f} s")
    if latencies:
        avg = sum(lat for _, lat in latencies) / len(latencies)
        print(f"Average latency: {avg:.3f} s")

if __name__ == "__main__":
    main()
