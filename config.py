from pathlib import Path

BASE_DIR = Path(__file__).parent
SERVER_USER = "hanyu"
SERVER_HOST = "vastlab"
SCRATCH_ROOT = Path("/scratch/hanyu")
MODEL_NAME = "gpt2"
MODELS_DIR = SCRATCH_ROOT / "models"
OUTPUT_DIR = MODELS_DIR / "gpt2-truthfulqa-finetuned"
GGUF_OUTPUT_DIR = MODELS_DIR / "gguf"
DATA_DIR = SCRATCH_ROOT / "data"
TRUTHFULQA_CACHE_DIR = DATA_DIR / "truthfulqa_cache"
LOGS_DIR = SCRATCH_ROOT / "logs"
TRAINING_CONFIG = {
    "max_length": 512,
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "eval_split": 0.2,
    "seed": 42,
}
for dir_path in [MODELS_DIR, GGUF_OUTPUT_DIR, DATA_DIR, TRUTHFULQA_CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

