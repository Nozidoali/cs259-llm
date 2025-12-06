#!/usr/bin/env python3
"""
Helper script to list available GGUF files in HuggingFace repositories.
Useful for verifying model filenames before running benchmarks.
"""

import sys
import subprocess
import argparse
from typing import List

# Model repos from benchmark script
MODEL_REPOS = {
    "llama-3.2-1b": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "llama-3.2-3b": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "llama-3.1-8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "mistral-7b-v0.3": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "phi-3.5-mini": "bartowski/Phi-3.5-mini-instruct-GGUF",
    "phi-2": "TheBloke/phi-2-GGUF",
    "gpt2": "mradermacher/gpt2-GGUF",
    "gpt2-medium": "mradermacher/gpt2-medium-GGUF",
    "gpt2-large": "mradermacher/gpt2-large-GGUF",
}


def list_repo_files(repo_id: str) -> List[str]:
    """List all GGUF files in a HuggingFace repository."""
    try:
        cmd = ["huggingface-cli", "scan-cache", "--repo", repo_id]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Alternative: use the files API
        cmd = ["huggingface-cli", "download", repo_id, "--list-files"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            files = []
            for line in result.stdout.split('\n'):
                if line.strip().endswith('.gguf'):
                    files.append(line.strip())
            return files
        else:
            print(f"Error listing {repo_id}: {result.stderr}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="List available GGUF models in HuggingFace repositories"
    )
    parser.add_argument(
        "--model",
        help="Specific model to check (e.g., 'gpt2', 'llama-3.2-1b'). If not provided, checks all."
    )
    args = parser.parse_args()
    
    if args.model:
        if args.model not in MODEL_REPOS:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {', '.join(MODEL_REPOS.keys())}")
            sys.exit(1)
        repos_to_check = {args.model: MODEL_REPOS[args.model]}
    else:
        repos_to_check = MODEL_REPOS
    
    print("Checking HuggingFace repositories for available GGUF files...")
    print("=" * 80)
    
    for model_name, repo_id in repos_to_check.items():
        print(f"\n{model_name}: {repo_id}")
        print("-" * 80)
        
        # Use huggingface_hub API if available
        try:
            from huggingface_hub import list_repo_files as hf_list_files
            files = hf_list_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if gguf_files:
                for f in sorted(gguf_files):
                    print(f"  - {f}")
            else:
                print("  No GGUF files found")
                
        except ImportError:
            print("  (Install huggingface_hub for file listing: pip install huggingface_hub)")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("Note: Install 'huggingface-cli' to download models:")
    print("  pip install huggingface_hub[cli]")


if __name__ == "__main__":
    main()
