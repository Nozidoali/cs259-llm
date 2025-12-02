#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))

from config import GATING_CONFIG, GATING_MODEL_DIR
from gating_dataset import prepare_gating_dataset
from gating_model import GatingNetwork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    embeddings = torch.tensor([item["embedding"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1)
    return {"embedding": embeddings, "label": labels}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs >= 0.5).long().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train gating network for task classification")
    
    parser.add_argument("--base_model", type=str, default=None,
                       help=f"Base model for embeddings (default: {GATING_CONFIG['base_model']})")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: models/gating-network/{base_model})")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    
    for key in ["hidden_dims", "dropout", "learning_rate", "batch_size", "num_epochs",
                "weight_decay", "train_split", "val_split", "test_split", "seed"]:
        default = GATING_CONFIG.get(key)
        if default is not None:
            parser.add_argument(f"--{key}", type=type(default) if not isinstance(default, list) else str,
                               default=None, help=f"Default: {default}")
    
    return parser.parse_args()


def load_config(args):
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            config = json.load(f)
    
    gating_config = GATING_CONFIG.copy()
    gating_config.update(config)
    
    for key in ["hidden_dims", "dropout", "learning_rate", "batch_size", "num_epochs",
                "weight_decay", "train_split", "val_split", "test_split", "seed"]:
        value = getattr(args, key, None)
        if value is not None:
            if key == "hidden_dims" and isinstance(value, str):
                gating_config[key] = [int(x.strip()) for x in value.split(",")]
            else:
                gating_config[key] = value
    
    if args.base_model:
        gating_config["base_model"] = args.base_model
    if args.output_dir:
        gating_config["output_dir"] = args.output_dir
    
    return gating_config


def get_output_dir(config, base_model):
    if config.get("output_dir"):
        return Path(config["output_dir"])
    model_name = base_model.replace("/", "_").replace("-", "_")
    return GATING_MODEL_DIR / model_name


def main():
    args = parse_args()
    config = load_config(args)
    
    base_model = config["base_model"]
    output_dir = get_output_dir(config, base_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Gating Network Training")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")
    
    logger.info("Preparing dataset...")
    dataset = prepare_gating_dataset(
        base_model=base_model,
        train_split=config["train_split"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        seed=config["seed"],
    )
    
    embedding_dim = len(dataset["train"][0]["embedding"])
    logger.info(f"Embedding dimension: {embedding_dim}\n")
    
    train_loader = DataLoader(dataset["train"], batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    logger.info("Initializing gating network...")
    model = GatingNetwork(
        input_dim=embedding_dim,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}\n")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {config['num_epochs']}, Batch size: {config['batch_size']}, "
                f"LR: {config['learning_rate']}, Hidden dims: {config['hidden_dims']}\n")
    
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_f1 = -1.0
    best_epoch = -1
    
    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, "
                   f"Rec: {val_metrics['recall']:.4f}")
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
        
        logger.info("")
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, "
               f"F1: {test_metrics['f1']:.4f}, Prec: {test_metrics['precision']:.4f}, "
               f"Rec: {test_metrics['recall']:.4f}")
    
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    
    training_info = {
        "base_model": base_model,
        "embedding_dim": embedding_dim,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "total_parameters": total_params,
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"✓ Model saved to: {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
