#!/usr/bin/env python3
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_model(model_name: str, device: str = "cuda"):
    """
    Load a Hugging Face model and tokenizer on the specified device.
    Returns (tokenizer, model).
    """
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return tokenizer, model

def run_one(tokenizer, model, prompt_path: str, output_path: str, 
            max_new_tokens: int = 512, temperature: float = 0.7, 
            device: str = "cuda"):
    """
    Run model inference on a prompt file, save output to file.
    Returns the latency (in seconds).
    """
    # Read prompt from file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    start = time.time()
    
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated part (remove the prompt)
    if len(generated_text) > len(prompt_text):
        response = generated_text[len(prompt_text):].strip()
    else:
        response = generated_text
    
    end = time.time()
    latency = end - start
    
    # Save output to file
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(response)
    
    return latency

def run_all(local_prompt_dir: str, output_dir: str, model_name: str,
            max_new_tokens: int = 512, temperature: float = 0.7, 
            device: str = "cuda"):
    """
    Run model inference on all prompt files in the directory.
    """
    ensure_dir(output_dir)
    
    # Load model once
    tokenizer, model = load_model(model_name, device)
    
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))
    
    latencies = []
    t0 = time.time()
    
    for pf in prompt_files:
        fname = pf.name  # e.g. "qmsum_test_0.prompt.txt"
        prompt_path = str(pf)  # Use local path directly
        
        # Derive output filename, strip ".prompt.txt"
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)
        
        print(f"Running prompt {fname} → output {out_fname}")
        try:
            latency = run_one(tokenizer, model, prompt_path, out_path, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature, device=device)
            print(f"  latency: {latency:.3f} s")
            latencies.append((fname, latency))
        except Exception as e:
            print(f"  [ERROR] Failed to process {fname}: {e}")
            latencies.append((fname, -1.0))
    
    t1 = time.time()
    total = t1 - t0
    return latencies, total

def main():
    # Configuration
    local_prompt_dir = "./prompt_files"
    output_dir = "./qmsum_outputs"
    
    # Model configuration - change this to your desired Hugging Face model
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # or "mistralai/Mistral-7B-Instruct-v0.2", etc.
    
    # Generation parameters
    max_new_tokens = 512
    temperature = 0.7
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU (will be slower)")
    
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    
    latencies, total_time = run_all(
        local_prompt_dir=local_prompt_dir,
        output_dir=output_dir,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device
    )
    
    print("\n=== Benchmark Summary ===")
    for fname, lat in latencies:
        if lat >= 0:
            print(f"{fname}: {lat:.3f} s")
        else:
            print(f"{fname}: FAILED")
    print(f"Total time for {len(latencies)} samples: {total_time:.3f} s")
    successful = [lat for _, lat in latencies if lat >= 0]
    if successful:
        avg = sum(successful) / len(successful)
        print(f"Average latency: {avg:.3f} s ({len(successful)} successful)")

if __name__ == "__main__":
    main()

