from pathlib import Path

BASE_DIR = Path(__file__).parent
WORK_DIR = Path.cwd()
MODELS_DIR = WORK_DIR / "models"
GGUF_OUTPUT_DIR = MODELS_DIR / "gguf"
LLAMA_CPP_DIR = Path("/Users/hanyu/Documents/llama.cpp")
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

for dir_path in [MODELS_DIR, GGUF_OUTPUT_DIR, DATA_DIR, TRUTHFULQA_CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
