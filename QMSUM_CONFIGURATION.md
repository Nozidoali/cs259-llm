# QMSum Configuration Parameters

This document lists all configurable parameters for running QMSum evaluations.

## Generation Parameters

### Token Generation
- **`max_new_tokens`** (default: 200)
  - Maximum number of new tokens to generate
  - Used in: `evaluate_moe.py`, `ffn_finetune.py`, `dataset_eval_trainer.py`, `longbench_test_hf.py`, `main.py`
  - Command line: `--qmsum_max_new_tokens` (evaluate_moe.py, ffn_finetune.py), `--max_new_tokens` (main.py, moe_inference.py)

- **`min_new_tokens`** (default: 10)
  - Minimum number of new tokens to generate
  - Used in: `evaluate_moe.py` (line 205), `longbench_test_hf.py` (line 120)
  - Currently hardcoded, not exposed as a parameter

### Sampling Parameters
- **`temperature`** (default: 0.0 in evaluate_moe.py, 1.0 in main.py, 0.7 in moe_inference.py)
  - Sampling temperature (0.0 = greedy decoding, >0 = sampling)
  - Used in: `evaluate_moe.py`, `moe_inference.py`, `longbench_test_hf.py`, `main.py`
  - Command line: `--temperature` (main.py, moe_inference.py)

- **`do_sample`** (default: False when temperature=0.0, True when temperature>0)
  - Whether to use sampling or greedy decoding
  - Used in: `evaluate_moe.py`, `moe_inference.py`, `longbench_test_hf.py`
  - Automatically set based on temperature in most cases

- **`top_p`** (default: 0.9)
  - Nucleus sampling parameter (top-p sampling)
  - Used in: `moe_inference.py`
  - Command line: `--top_p` (moe_inference.py)
  - Not currently used in evaluate_moe.py (hardcoded to 0.0 temperature)

- **`repetition_penalty`** (default: 1.1)
  - Penalty for repetition (1.0 = no penalty, >1.0 = penalize repetition)
  - Used in: `evaluate_moe.py` (line 210), `longbench_test_hf.py` (line 125)
  - Currently hardcoded, not exposed as a parameter

## Input Processing Parameters

### Tokenization
- **`max_length`** (default: 512)
  - Maximum sequence length for tokenization (input + output)
  - Used in: `finetune.py`, `dataset_eval_trainer.py`, `ffn_finetune.py`
  - Command line: `--max_length` (ffn_finetune.py)
  - Note: In `evaluate_moe.py` and `longbench_test_hf.py`, `truncation=False` is used, so full prompts are processed

- **`truncation`** (default: True in training, False in inference)
  - Whether to truncate input sequences
  - Used in: `finetune.py`, `dataset_eval_trainer.py`, `evaluate_moe.py`, `longbench_test_hf.py`
  - In inference: `truncation=False` to preserve full context

- **`padding`** (default: "max_length" in training)
  - Padding strategy for tokenization
  - Used in: `finetune.py`
  - Not applicable for inference (single sequences)

## Model/Inference Parameters

### Device and Hardware
- **`device`** (default: auto-detect)
  - Device to use: "cuda", "cpu", or "mps"
  - Used in: `evaluate_moe.py`, `longbench_test_hf.py`
  - Command line: `--device` (evaluate_moe.py)

- **`use_cache`** (default: True)
  - Whether to use KV cache for faster generation
  - Used in: `evaluate_moe.py`, `longbench_test_hf.py`, `main.py`
  - Command line: `--disable_cache` (main.py, negates use_cache)

- **`n_gpu_layers`** (default: -1 for GGUF models)
  - Number of layers to offload to GPU for GGUF models (-1 = all, 0 = CPU only)
  - Used in: `longbench_test_hf.py`, `main.py`
  - Command line: `--n_gpu_layers` (main.py)

- **`n_ctx`** (default: 32768 for GGUF, 4096 in longbench_test_hf.py)
  - Context window size for GGUF models
  - Used in: `longbench_test_hf.py`, `main.py`
  - Command line: `--n_ctx` (main.py)

### MoE-Specific Parameters
- **`routing_mode`** (default: "weighted_sum")
  - Routing mode for MoE: "weighted_sum" or "select_one"
  - Used in: `evaluate_moe.py`, `moe_inference.py`
  - Command line: `--routing_mode` (evaluate_moe.py, moe_inference.py)

## Dataset/Evaluation Parameters

### Sample Selection
- **`qmsum_num_samples`** (default: None = all samples)
  - Number of QMSum samples to evaluate
  - Used in: `evaluate_moe.py`, `ffn_finetune.py`
  - Command line: `--qmsum_num_samples` (evaluate_moe.py, ffn_finetune.py)

- **`random_seed`** (default: 42)
  - Random seed for sample selection
  - Used in: `evaluate_moe.py`, `longbench.py`
  - Command line: `--random_seed` (evaluate_moe.py)

### Input/Output Paths
- **`qmsum_prompt_dir`** (default: WORK_DIR/prompt_files)
  - Directory containing QMSum prompt files
  - Used in: `evaluate_moe.py`
  - Command line: `--qmsum_prompt_dir` (evaluate_moe.py)

- **`prompt_dir`** (default: "./prompt_files")
  - Directory containing prompt files
  - Used in: `main.py`, `longbench.py`
  - Command line: `--prompt_dir` (main.py)

- **`output_dir`** (default: "./qmsum_outputs")
  - Directory to save output files
  - Used in: `main.py`, `longbench.py`
  - Command line: `--output_dir` (main.py)

## Model Path Parameters

### For MoE Evaluation
- **`saved_model_path`** (default: None)
  - Path to saved MoE model directory
  - Command line: `--saved_model_path` (evaluate_moe.py)

- **`model1_path`** (default: None)
  - Path to first finetuned model
  - Command line: `--model1_path` (evaluate_moe.py)

- **`model2_path`** (default: None)
  - Path to second finetuned model
  - Command line: `--model2_path` (evaluate_moe.py)

- **`gating_model_path`** (default: None)
  - Path to gating network
  - Command line: `--gating_model_path` (evaluate_moe.py)

### For Direct Model Evaluation
- **`model_name`** (default: "meta-llama/Llama-3.2-1B-Instruct")
  - HuggingFace model name or path to GGUF file
  - Command line: `--model_name` (main.py)

- **`model_type`** (default: auto-detect)
  - Model type: "hf" for HuggingFace, "gguf" for GGUF
  - Command line: `--model_type` (main.py)

## Evaluation Parameters

- **`skip_eval`** (default: False)
  - Skip evaluation after generation
  - Command line: `--skip_eval` (main.py)

- **`eval_only`** (default: False)
  - Skip generation and only run evaluation on existing outputs
  - Command line: `--eval_only` (main.py)

- **`results_file`** (default: auto-generated timestamp)
  - Output JSON file path for evaluation results
  - Command line: `--results_file` (main.py)

- **`output_file`** (default: None)
  - Path to save results JSON file
  - Command line: `--output_file` (evaluate_moe.py)

## Parameters Currently Hardcoded (Not Configurable)

These parameters are used in the code but are not currently exposed as command-line arguments:

1. **`min_new_tokens`** = 10 (in evaluate_moe.py, longbench_test_hf.py)
2. **`repetition_penalty`** = 1.1 (in evaluate_moe.py, longbench_test_hf.py)
3. **`pad_token_id`** = tokenizer.pad_token_id or tokenizer.eos_token_id (auto-set)
4. **`use_stemmer`** = True (for ROUGE evaluation)
5. **`truncation`** = False (in inference, to preserve full context)

## Summary by Script

### `evaluate_moe.py`
- `--qmsum_max_new_tokens` (default: 200)
- `--truthfulqa_max_new_tokens` (default: 50)
- `--qmsum_num_samples`
- `--qmsum_prompt_dir`
- `--random_seed`
- `--device`
- `--routing_mode`
- `--saved_model_path` / `--model1_path` / `--model2_path` / `--gating_model_path`
- `--output_file`

### `main.py` (cs259-longbench)
- `--max_new_tokens`
- `--temperature`
- `--model_name`
- `--model_type`
- `--prompt_dir`
- `--output_dir`
- `--disable_cache`
- `--n_gpu_layers`
- `--n_ctx`
- `--skip_eval`
- `--eval_only`
- `--results_file`

### `ffn_finetune.py`
- `--qmsum_max_new_tokens`
- `--qmsum_num_samples`
- `--max_length`
- `--dataset` (can be "qmsum")

### `moe_inference.py`
- `--max_new_tokens`
- `--temperature`
- `--top_p`
- `--no_sample` (sets do_sample=False)

