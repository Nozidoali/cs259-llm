from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
work_dir_env = os.getenv("WORK_DIR", "").strip()
WORK_DIR = Path(work_dir_env) if work_dir_env else Path.cwd()
llama_cpp_env = os.getenv("LLAMA_CPP_DIR", "").strip()
LLAMA_CPP_DIR = Path(llama_cpp_env) if llama_cpp_env else Path("/Users/hanyu/Documents/llama.cpp")

if not WORK_DIR.exists() or not WORK_DIR.is_dir():
    raise ValueError(f"WORK_DIR does not exist or is not a directory: {WORK_DIR}")

if not LLAMA_CPP_DIR.exists() or not LLAMA_CPP_DIR.is_dir():
    raise ValueError(f"LLAMA_CPP_DIR does not exist or is not a directory: {LLAMA_CPP_DIR}")

convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
if not convert_script.exists():
    raise ValueError(f"convert_hf_to_gguf.py not found in LLAMA_CPP_DIR: {LLAMA_CPP_DIR}")

MODELS_DIR = WORK_DIR / "models"
GGUF_OUTPUT_DIR = MODELS_DIR / "gguf"
QUANTIZE_LEVEL = "Q4_0"
DATA_DIR = WORK_DIR / "data"
TRUTHFULQA_CACHE_DIR = DATA_DIR / "truthfulqa_cache"
LOGS_DIR = WORK_DIR / "logs"

MODEL_CONFIGS = {
    "qwen2-0.5b": {
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "base_dir": "qwen2-0.5b-instruct",
        "finetuned_dir": "qwen2-0.5b-instruct-finetuned",
        "display_name": "Qwen2 0.5B Instruct",
        "model_type": "causal",
        "supports_gguf": True,
    },
    "qwen2-1.5b": {
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
        "base_dir": "qwen2-1.5b-instruct",
        "finetuned_dir": "qwen2-1.5b-instruct-finetuned",
        "display_name": "Qwen2 1.5B Instruct",
        "model_type": "causal",
        "supports_gguf": True,
    },
    "switch-base-8": {
        "model_id": "google/switch-base-8",
        "base_dir": "switch-base-8",
        "finetuned_dir": "switch-base-8-finetuned",
        "display_name": "Switch-Base-8",
        "model_type": "seq2seq",
        "supports_gguf": False,
    },
    "qwen1.5-moe-a2.7b": {
        "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "base_dir": "qwen1.5-moe-a2.7b-chat",
        "finetuned_dir": "qwen1.5-moe-a2.7b-chat-finetuned",
        "display_name": "Qwen1.5-MoE A2.7B Chat",
        "model_type": "causal",
        "supports_gguf": True,
    },
}

TRAINING_CONFIG = {
    "max_length": 512,
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "eval_split": 0.2,
    "seed": 42,
}

BLEURT_CONFIG = {
    "model_name": "bleurt-large-128",
    "max_new_tokens": 50,
    "prompt_template": "Question: {question}\nAnswer:",
}

DATASET_CONFIG = {
    "name": "truthfulqa/truthful_qa",
    "config": "generation",
    "split": "validation",
    "format_template": "Question: {question}\nAnswer: {best_answer}",
}

EVALUATION_CONFIG = {
    "truthfulqa_num_tokens": 25,
    "truthfulqa_num_samples": 10,
    "longbench_n_benchmarks": 1,
    "longbench_num_tokens": 200,
}

for dir_path in [MODELS_DIR, GGUF_OUTPUT_DIR, DATA_DIR, TRUTHFULQA_CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
