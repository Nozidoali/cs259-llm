#!/usr/bin/env python3
import os
import time
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_model_hf(model_name: str, device: str = "cuda"):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for HuggingFace models. Install with: pip install transformers")
    
    print(f"Loading HuggingFace model {model_name} on {device}...")

    if device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"PyTorch CUDA Version: {cuda_version}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return tokenizer, model

def load_model_gguf(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is required for GGUF models. Install with: pip install llama-cpp-python")
    
    if not os.path.isabs(model_path) and not os.path.dirname(model_path):
        model_path = os.path.join("./gguf", model_path)
    elif not os.path.exists(model_path) and os.path.exists(os.path.join("./gguf", os.path.basename(model_path))):
        model_path = os.path.join("./gguf", os.path.basename(model_path))
    
    print(f"Loading GGUF model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model file not found: {model_path}")
    
    if n_gpu_layers == -1:
        actual_n_gpu_layers = -1
        print("Attempting to use GPU acceleration (n_gpu_layers=-1)")
    elif n_gpu_layers == 0:
        actual_n_gpu_layers = 0
        print("Using CPU only (n_gpu_layers=0)")
    else:
        actual_n_gpu_layers = n_gpu_layers
        print(f"Using {n_gpu_layers} GPU layers")
    
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=actual_n_gpu_layers,
        verbose=False
    )
    
    print(f"GGUF model loaded successfully")
    if actual_n_gpu_layers == -1:
        print("Note: If GPU is not being used, ensure llama-cpp-python was compiled with CUDA support.")
        print("      Install with: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
    return None, model

def run_one_hf(tokenizer, model, prompt_path: str, output_path: str, 
               max_new_tokens: int = 512, temperature: float = 0.7, 
               device: str = "cuda", use_cache: bool = True):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    start = time.time()

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=use_cache,
                repetition_penalty=1.1
            )
        except AttributeError as e:
            if "'DynamicCache' object has no attribute 'seen_tokens'" in str(e) or "seen_tokens" in str(e):
                if use_cache:
                    print(f"  Warning: Cache error detected, retrying with use_cache=False")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False
                    )
                else:
                    raise
            else:
                raise
    
    if hasattr(outputs, 'sequences'):
        full_sequence = outputs.sequences[0]
    else:
        full_sequence = outputs[0]
    
    generated_ids = full_sequence[input_length:]
    response_token_level = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    full_text = tokenizer.decode(full_sequence, skip_special_tokens=True)
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True).strip()
    
    if full_text.startswith(input_text):
        response = full_text[len(input_text):].strip()
    else:
        response = response_token_level
    
    end = time.time()
    latency = end - start
    
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(response)
    
    return latency

def run_one_gguf(model, prompt_path: str, output_path: str, 
                max_new_tokens: int = 512, temperature: float = 0.7):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    start = time.time()
    
    output = model(
        prompt_text,
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=None,
        echo=False
    )
    
    response = output['choices'][0]['text'].strip()
    
    end = time.time()
    latency = end - start
    
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(response)
    
    return latency

def run_all(local_prompt_dir: str, output_dir: str, model_name: str,
            max_new_tokens: int = 512, temperature: float = 0.7, 
            device: str = "cuda", use_cache: bool = True, 
            model_type: str = "hf", n_gpu_layers: int = -1, n_ctx: int = 4096):
    ensure_dir(output_dir)
    
    if model_type == "gguf":
        tokenizer, model = load_model_gguf(model_name, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    else:
        tokenizer, model = load_model_hf(model_name, device)
    
    local = Path(local_prompt_dir)
    prompt_files = sorted(local.glob("*.prompt.txt"))
    
    latencies = []
    t0 = time.time()
    
    for pf in prompt_files:
        fname = pf.name
        prompt_path = str(pf)
        
        base = fname
        if base.endswith(".prompt.txt"):
            base = base[:-len(".prompt.txt")]
        out_fname = base + ".txt"
        out_path = os.path.join(output_dir, out_fname)
        
        print(f"Running prompt {fname} → output {out_fname}")
        try:
            if model_type == "gguf":
                latency = run_one_gguf(model, prompt_path, out_path, 
                                      max_new_tokens=max_new_tokens, 
                                      temperature=temperature)
            else:
                latency = run_one_hf(tokenizer, model, prompt_path, out_path, 
                                    max_new_tokens=max_new_tokens, 
                                    temperature=temperature, device=device,
                                    use_cache=use_cache)
            print(f"  latency: {latency:.3f} s")
            latencies.append((fname, latency))
        except Exception as e:
            print(f"  [ERROR] Failed to process {fname}: {e}")
            latencies.append((fname, -1.0))
    
    t1 = time.time()
    total = t1 - t0
    return latencies, total
