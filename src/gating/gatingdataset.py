import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

_file_dir = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.dirname(_file_dir)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from config import TRUTHFULQA_CACHE_DIR, DATA_DIR, DATASET_CONFIG, MODEL_CONFIGS, MODELS_DIR, WORK_DIR
from data import prepare_truthfulqa_dataset, prepare_qmsum_dataset

logger = logging.getLogger(__name__)

def _get_datasets_hash(datasets: List[str]) -> str:
    """Generate a short hash from the datasets list to avoid long filenames."""
    datasets_tuple = tuple(sorted(datasets))
    hash_obj = hashlib.md5(str(datasets_tuple).encode())
    return hash_obj.hexdigest()[:12]

def load_base_model_for_embeddings(base_model: str):
    if base_model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[base_model]
        model_id = config["model_id"]
        model_path = MODELS_DIR / config["base_dir"]
        if not model_path.exists() or not (model_path / "config.json").exists():
            logger.info(f"Downloading {model_id}...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=str(model_path), local_dir_use_symlinks=False, resume_download=True, max_workers=1)
        model_path = str(model_path)
    else:
        model_path = base_model
        logger.info(f"Using HuggingFace model: {base_model}")
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embedding_dim = model.model.embed_tokens.embedding_dim
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embedding_dim = model.transformer.wte.embedding_dim
    else:
        embedding_dim = model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")
    return model, tokenizer, embedding_dim

def extract_embeddings(model, tokenizer, texts, max_length=512, batch_size=8, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embeddings = model.model.embed_tokens(input_ids)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                embeddings = model.transformer.wte(input_ids)
            else:
                raise ValueError("Could not find embedding layer in model")
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9).to(embeddings.dtype)
            mean_embeddings = sum_embeddings / sum_mask
        all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def load_dataset_texts(dataset_name: str, prompt_dir: Optional[Path] = None):
    if dataset_name == "truthfulqa":
        logger.info("Loading TruthfulQA dataset...")
        ds = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["config"], split=DATASET_CONFIG["split"], cache_dir=str(TRUTHFULQA_CACHE_DIR))
        texts = [f"Question: {ex['question']}" for ex in ds]
        logger.info(f"Loaded {len(texts)} TruthfulQA samples")
        return texts
    elif dataset_name == "longbench" or dataset_name == "qmsum":
        if prompt_dir is None:
            prompt_dir = WORK_DIR / "prompt_files"
        else:
            prompt_dir = Path(prompt_dir)
        logger.info(f"Loading QMSum prompts from: {prompt_dir}")
        prompt_files = sorted(prompt_dir.glob("qmsum_test_*.prompt.txt"))
        texts = [open(f, "r", encoding="utf-8", errors="replace").read().strip() for f in prompt_files]
        logger.info(f"Loaded {len(texts)} QMSum samples")
        return texts
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def prepare_gating_dataset_multi(
    base_model: str,
    datasets: List[str],
    max_length: int = 512,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
    prompt_dir: Optional[Path] = None,
    seed: int = 42,
):
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must equal 1.0")
    if cache_dir is None:
        model_name = base_model.replace("/", "_").replace("-", "_")
        cache_key = _get_datasets_hash(datasets)
        cache_dir = DATA_DIR / "gating_cache" / f"{model_name}_{cache_key}"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gating_dataset.arrow"
    if cache_file.exists():
        logger.info(f"Loading cached dataset from: {cache_dir}")
        try:
            return DatasetDict.load_from_disk(str(cache_dir))
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, regenerating...")
    logger.info("Preparing gating dataset from scratch...")
    model, tokenizer, embedding_dim = load_base_model_for_embeddings(base_model)
    all_texts = []
    all_labels = []
    for idx, dataset_name in enumerate(datasets):
        texts = load_dataset_texts(dataset_name, prompt_dir)
        all_texts.extend(texts)
        all_labels.extend([idx] * len(texts))
        logger.info(f"Dataset {dataset_name}: {len(texts)} samples (label={idx})")
    logger.info(f"Total samples: {len(all_texts)} across {len(datasets)} datasets")
    logger.info("Extracting embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    embeddings = extract_embeddings(model, tokenizer, all_texts, max_length=max_length, batch_size=batch_size, device=device)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    dataset = Dataset.from_dict({
        "embedding": embeddings.tolist(),
        "label": all_labels,
        "text": all_texts,
    })
    dataset = dataset.train_test_split(test_size=test_split, seed=seed)
    test_dataset = dataset["test"]
    train_val_split = val_split / (train_split + val_split)
    train_dataset = dataset["train"].train_test_split(test_size=train_val_split, seed=seed)
    dataset_dict = DatasetDict({
        "train": train_dataset["train"],
        "validation": train_dataset["test"],
        "test": test_dataset,
    })
    logger.info(f"Dataset splits - Train: {len(dataset_dict['train'])}, Validation: {len(dataset_dict['validation'])}, Test: {len(dataset_dict['test'])}")
    logger.info(f"Saving dataset to cache: {cache_dir}")
    dataset_dict.save_to_disk(str(cache_dir))
    return dataset_dict


def prepare_shared_expert_gating_dataset(
    base_model: str,
    datasets: List[str],
    max_length: int = 512,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
    prompt_dir: Optional[Path] = None,
    seed: int = 42,
    never_use_shared_expert: bool = False,
):
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must equal 1.0")
    
    if cache_dir is None:
        model_name = base_model.replace("/", "_").replace("-", "_")
        cache_key = _get_datasets_hash(datasets)
        suffix = "never_shared" if never_use_shared_expert else "shared"
        cache_dir = DATA_DIR / "gating_cache" / f"{model_name}_{cache_key}_{suffix}"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "shared_gating_dataset.arrow"
    
    if cache_file.exists():
        logger.info(f"Loading cached shared expert gating dataset from: {cache_dir}")
        try:
            return DatasetDict.load_from_disk(str(cache_dir))
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, regenerating...")
    
    logger.info("Preparing shared expert gating dataset from scratch...")
    model, tokenizer, embedding_dim = load_base_model_for_embeddings(base_model)
    
    all_texts = []
    all_labels = []
    
    label_value = 0.0 if never_use_shared_expert else 1.0
    label_desc = "never use shared expert" if never_use_shared_expert else "use shared expert"
    
    for dataset_name in datasets:
        texts = load_dataset_texts(dataset_name, prompt_dir)
        all_texts.extend(texts)
        all_labels.extend([label_value] * len(texts))
        logger.info(f"Dataset {dataset_name}: {len(texts)} samples (label={label_value} - {label_desc})")
    
    logger.info(f"Total samples: {len(all_texts)} for shared expert gating")
    logger.info("Extracting embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    embeddings = extract_embeddings(model, tokenizer, all_texts, max_length=max_length, batch_size=batch_size, device=device)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    
    dataset = Dataset.from_dict({
        "embedding": embeddings.tolist(),
        "label": all_labels,  # Float labels for binary classification
        "text": all_texts,
    })
    
    dataset = dataset.train_test_split(test_size=test_split, seed=seed)
    test_dataset = dataset["test"]
    train_val_split = val_split / (train_split + val_split)
    train_dataset = dataset["train"].train_test_split(test_size=train_val_split, seed=seed)
    
    dataset_dict = DatasetDict({
        "train": train_dataset["train"],
        "validation": train_dataset["test"],
        "test": test_dataset,
    })
    
    logger.info(f"Shared expert gating dataset splits - Train: {len(dataset_dict['train'])}, Validation: {len(dataset_dict['validation'])}, Test: {len(dataset_dict['test'])}")
    logger.info(f"Saving shared expert gating dataset to cache: {cache_dir}")
    dataset_dict.save_to_disk(str(cache_dir))
    
    return dataset_dict

