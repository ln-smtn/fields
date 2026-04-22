"""
Эксперимент 3. Fine-tune: полное дообучение PRESTO-энкодера + головы.

Идея: размораживаем PRESTO-энкодер и обучаем его вместе с линейной
головой классификатора обычным SGD на ВСЕХ размеченных данных.

Отличия от linear probe (эксп. 2):
  - Энкодер дообучается → эмбеддинги адаптируются под культуры РФ
  - Лучше качество, но дольше обучение и риск переобучения на малых классах

Протокол:
  - Stratified 5-fold CV
  - AdamW, lr=1e-4 (маленький — чтобы не «забыть» предобучение)
  - 30 эпох, early stopping по validation loss
  - Метрики: F1 macro, F1 weighted, accuracy

Запуск:
  uv run python crop_classification/experiment_finetune.py --year 2021 --tile 38TLR
"""

import argparse
import json
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATASET_DIR,
    ID_TO_SHORT,
    NUM_CLASSES,
    OUTPUT_DIR,
    PILOT_TILE,
    PILOT_YEAR,
    PRESTO_EMBEDDING_DIM,
)


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class PrestoClassifier(nn.Module):
    """PRESTO-энкодер + линейная голова для классификации культур."""

    def __init__(self, num_classes: int):
        super().__init__()
        from presto import Presto
        pretrained = Presto.load_pretrained()
        self.encoder = pretrained.encoder
        self.head = nn.Linear(PRESTO_EMBEDDING_DIM, num_classes)

    def forward(self, x, mask, latlons):
        emb = self.encoder(x, mask=mask, latlons=latlons)
        return self.head(emb)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, mask, latlons, y in loader:
        x, mask, latlons, y = (x.to(device), mask.to(device),
                                latlons.to(device), y.to(device))
        logits = model(x, mask, latlons)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    for x, mask, latlons, y in loader:
        x, mask, latlons, y = (x.to(device), mask.to(device),
                                latlons.to(device), y.to(device))
        logits = model(x, mask, latlons)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return np.array(all_preds), np.array(all_true), avg_loss


def main():
    parser = argparse.ArgumentParser(description="Эксп. 3: Fine-tune PRESTO")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    args = parser.parse_args()

    ds_path = DATASET_DIR / f"{args.year}_{args.tile}.npz"
    if not ds_path.exists():
        sys.exit(f"Нет датасета: {ds_path}")

    ds = np.load(ds_path)
    X = torch.tensor(ds["x"], dtype=torch.float32)
    M = torch.tensor(ds["mask"], dtype=torch.float32)
    L = torch.tensor(ds["latlons"], dtype=torch.float32)
    y = torch.tensor(ds["labels"], dtype=torch.long)

    device = detect_device()

    print(f"\n{'='*60}")
    print(f"Эксперимент 3: Fine-tune PRESTO")
    print(f"{'='*60}")
    print(f"  Полей: {len(y)}, классов: {len(y.unique())}")
    print(f"  Устройство: {device}")
    print(f"  Эпох: {args.epochs}, lr: {args.lr}, patience: {args.patience}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X.numpy(), y.numpy())):
        print(f"\n  --- Fold {fold+1}/{args.folds} ---")

        train_ds = TensorDataset(X[train_idx], M[train_idx],
                                  L[train_idx], y[train_idx])
        test_ds = TensorDataset(X[test_idx], M[test_idx],
                                 L[test_idx], y[test_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        model = PrestoClassifier(NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            preds, true, val_loss = evaluate(model, test_loader, device)
            f1 = f1_score(true, preds, average="macro", zero_division=0)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(f"    Epoch {epoch+1:2d}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, F1={f1:.3f}"
                      f"{'  *best*' if patience_counter == 0 else ''}")

            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Лучшая модель
        model.load_state_dict(best_state)
        preds, true, _ = evaluate(model, test_loader, device)
        all_preds.extend(preds)
        all_true.extend(true)

        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        print(f"    Best: accuracy={acc:.3f}, F1_macro={f1:.3f}")

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    print(f"\n  ИТОГО ({args.folds}-fold CV):")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 macro:    {f1_macro:.4f}")
    print(f"    F1 weighted: {f1_weighted:.4f}")

    labels_present = sorted(np.unique(np.concatenate([all_true, all_preds])))
    names = [ID_TO_SHORT.get(i, f"id={i}") for i in labels_present]
    print(f"\n{classification_report(all_true, all_preds, labels=labels_present, target_names=names, zero_division=0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "finetune",
        "year": args.year, "tile": args.tile,
        "epochs": args.epochs, "lr": args.lr,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }
    out_path = OUTPUT_DIR / f"exp3_finetune_{args.year}_{args.tile}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Результаты: {out_path}")


if __name__ == "__main__":
    main()
