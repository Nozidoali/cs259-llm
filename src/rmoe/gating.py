import os
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from gating.gatingmodel import GatingNetwork
from gating.gatingdataset import prepare_gating_dataset_multi

logger = logging.getLogger(__name__)

def collate_fn(batch):
    embeddings = torch.tensor([item["embedding"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
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
        logits = model.get_logits(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = model(embeddings).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
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
            logits = model.get_logits(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = model(embeddings).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader)
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
    }

def train_gating_network(
    base_model,
    datasets,
    output_dir,
    hidden_dims=[512, 256],
    dropout=0.1,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10,
    weight_decay=0.01,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    prompt_dir=None,
):
    logger.info(f"Training gating network for {len(datasets)} datasets")
    logger.info(f"Output directory: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("Preparing gating dataset...")
    dataset = prepare_gating_dataset_multi(
        base_model=base_model,
        datasets=datasets,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        prompt_dir=prompt_dir,
    )
    embedding_dim = len(dataset["train"][0]["embedding"])
    num_classes = len(datasets)
    logger.info(f"Embedding dimension: {embedding_dim}, Number of classes: {num_classes}")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("Initializing gating network...")
    model = GatingNetwork(
        input_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    train_labels = [item["label"] for item in dataset["train"]]
    class_counts = [train_labels.count(i) for i in range(num_classes)]
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * count) if count > 0 else 1.0 for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights: {class_weights.tolist()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(f"Training configuration: Epochs={num_epochs}, Batch size={batch_size}, LR={learning_rate}")
    logger.info("Starting training...")
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        is_best = False
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            is_best = True
        elif val_metrics["f1"] == best_val_f1 == 0.0 and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            is_best = True
        if is_best:
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"  Saved best model (F1: {val_metrics['f1']:.4f}, Loss: {val_metrics['loss']:.4f})")
    logger.info(f"Training complete! Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    training_info = {
        "base_model": base_model,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "datasets": datasets,
        "config": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
        },
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "total_parameters": total_params,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Model saved to: {output_dir}")
    return output_dir

