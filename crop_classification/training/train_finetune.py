"""
Fine-tune PRESTO + классификационная голова под наши культуры.

Метод:
  - Берём предобученную PRESTO (NASA Harvest).
  - Замораживаем нижнюю часть энкодера (большинство параметров).
  - Размораживаем верхние 2 блока трансформера + новую голову Linear(128, 15).
  - Учим оба компонента совместно с маленьким learning rate (1e-4).

Стратегия "частичная разморозка" (по умолчанию):
  - Защищает основные веса PRESTO от разрушения.
  - Достаточно мощно адаптирует к РФ-культурам.
  - Меньше риск переобучения.

Альтернативы (через --strategy):
  - "full"        — разморозить всё (риск переобучения, но потенциально лучше);
  - "head-only"   — заморозить всю PRESTO (эквивалент Linear probe, но через PyTorch).

Сохраняет:
  - models/finetuned_model.pt   — словарь {encoder_state, head_state}
  - models/finetuned_metrics.json
  - models/confusion_matrix_finetune.png
  - models/training_curves.png

Запуск:
  uv run python crop_classification/train_finetune.py
  uv run python crop_classification/train_finetune.py --strategy full
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# Конфиг лежит на уровень выше (crop_classification/config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    ID_TO_SHORT,
    NUM_CLASSES,
    PRESTO_DW_MISSING,
    PRESTO_EMBEDDING_DIM,
)

MASTER_DIR = DATA_DIR / "training_master"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_unfreeze_strategy(presto, strategy: str):
    """Заморозить/разморозить части PRESTO согласно стратегии."""
    # Сначала всё заморозить
    for p in presto.parameters():
        p.requires_grad = False

    if strategy == "head-only":
        # Ничего не размораживаем — будет учиться только наша голова
        n_trainable = 0
    elif strategy == "partial":
        # Размораживаем последние 2 блока трансформера (verify имя у твоей PRESTO)
        # У PRESTO encoder: blocks = nn.ModuleList(...)
        try:
            blocks = presto.encoder.blocks
            n_blocks = len(blocks)
            for block in list(blocks)[-2:]:
                for p in block.parameters():
                    p.requires_grad = True
            print(f"  Разморожены последние 2 блока из {n_blocks}")
        except AttributeError:
            print("  ⚠ Не нашёл presto.encoder.blocks, размораживаю всё (fallback)")
            for p in presto.parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in presto.parameters() if p.requires_grad)
    elif strategy == "full":
        for p in presto.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in presto.parameters())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total = sum(p.numel() for p in presto.parameters())
    print(f"  PRESTO trainable: {n_trainable}/{total} параметров "
          f"({100*n_trainable/total:.1f}%)")


class FinetuneModel(nn.Module):
    """PRESTO encoder + linear classification head."""

    def __init__(self, presto, num_classes: int):
        super().__init__()
        self.presto = presto
        self.head = nn.Linear(PRESTO_EMBEDDING_DIM, num_classes)
        # Голова инициализируется случайно — Xavier по умолчанию

    def forward(self, x, mask, dynamic_world, latlons, month):
        emb = self.presto.encoder(
            x, dynamic_world=dynamic_world, latlons=latlons,
            mask=mask, month=month,
        )
        return self.head(emb)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses, preds, trues = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, mask, dw, latlons, month, y = [b.to(device) for b in batch]
            logits = model(x, mask, dw, latlons, month)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    return (np.mean(losses),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            np.array(preds), np.array(trues))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PRESTO под классификацию культур")
    parser.add_argument("--master", default=str(MASTER_DIR / "all_raw.npz"))
    parser.add_argument("--split", default=str(MASTER_DIR / "split.json"))
    parser.add_argument("--strategy", default="partial",
                        choices=["partial", "full", "head-only"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5,
                        help="Сколько эпох ждать улучшения val F1 macro")
    args = parser.parse_args()

    device = detect_device()
    print(f"\n{'='*60}")
    print(f"Fine-tune PRESTO ({args.strategy})")
    print(f"  Устройство: {device}")
    print(f"{'='*60}")

    # 1) Загрузить данные
    d = np.load(args.master, allow_pickle=True)
    with open(args.split) as f:
        split = json.load(f)

    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])
    test_idx = np.array(split["test"])

    def to_dataset(idx):
        return TensorDataset(
            torch.tensor(d["x"][idx], dtype=torch.float32),
            torch.tensor(d["mask"][idx], dtype=torch.float32),
            torch.tensor(d["dynamic_world"][idx], dtype=torch.long),
            torch.tensor(d["latlons"][idx], dtype=torch.float32),
            torch.tensor(d["months"][idx][:, 0] if "months" in d.files
                         else np.zeros(len(idx)), dtype=torch.long),
            torch.tensor(d["labels"][idx], dtype=torch.long),
        )

    train_loader = DataLoader(to_dataset(train_idx), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(to_dataset(val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(to_dataset(test_idx), batch_size=args.batch_size)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 2) Веса классов
    y_train = d["labels"][train_idx]
    classes_present = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes_present, y=y_train,
    )
    weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
    for cid, w in zip(classes_present, class_weights):
        weight_tensor[cid] = w
    weight_tensor = weight_tensor.to(device)
    print(f"\nКласс-веса:")
    for cid in classes_present:
        print(f"  {ID_TO_SHORT.get(cid, f'id={cid}'):>15s}: {weight_tensor[cid].item():.3f}")

    # 3) Модель
    from presto import Presto
    presto = Presto.load_pretrained()
    apply_unfreeze_strategy(presto, args.strategy)

    model = FinetuneModel(presto, NUM_CLASSES).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nМодель: trainable {n_train}/{n_total} параметров")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # 4) Цикл обучения с early stopping
    history = {"train_loss": [], "train_f1": [],
               "val_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
        )
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
        )

        history["train_loss"].append(tr_loss)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_f1"].append(vl_f1)

        improved = vl_f1 > best_val_f1
        marker = "  *новый лучший" if improved else ""
        print(f"  Эпоха {epoch:3d}: train F1={tr_f1:.4f}  val F1={vl_f1:.4f}"
              f"  train loss={tr_loss:.3f}  val loss={vl_loss:.3f}{marker}")

        if improved:
            best_val_f1 = vl_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stop: val F1 не растёт {args.patience} эпох")
                break

    # 5) Загрузить лучшую модель и оценить на test
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n{'='*60}")
    print("Финальная оценка на test")
    print(f"{'='*60}")
    te_loss, te_acc, te_f1, te_pred, te_true = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False,
    )

    classes_in_test = sorted(np.unique(te_true).tolist())
    target_names = [ID_TO_SHORT.get(c, f"id={c}") for c in classes_in_test]
    print(f"  Accuracy: {te_acc:.4f}")
    print(f"  F1 macro: {te_f1:.4f}")
    print("\n  Classification report:")
    print(classification_report(
        te_true, te_pred, labels=classes_in_test,
        target_names=target_names, zero_division=0,
    ))

    # 6) Сохранение
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": model.presto.encoder.state_dict(),
        "head_state": model.head.state_dict(),
        "strategy": args.strategy,
    }, MODELS_DIR / "finetuned_model.pt")
    print(f"\n  Сохранено: {MODELS_DIR / 'finetuned_model.pt'}")

    metrics = {
        "method": "finetune",
        "strategy": args.strategy,
        "test": {"accuracy": float(te_acc), "f1_macro": float(te_f1)},
        "best_val_f1_macro": float(best_val_f1),
        "epochs_trained": len(history["train_loss"]),
        "lr": args.lr, "batch_size": args.batch_size,
    }
    with open(MODELS_DIR / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        ax1.plot(history["train_loss"], label="train")
        ax1.plot(history["val_loss"], label="val")
        ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")
        ax2.plot(history["train_f1"], label="train")
        ax2.plot(history["val_f1"], label="val")
        ax2.set_xlabel("Эпоха"); ax2.set_ylabel("F1 macro"); ax2.legend()
        ax2.set_title("F1 macro")
        plt.tight_layout()
        fig.savefig(MODELS_DIR / "training_curves.png", dpi=150)
        plt.close()

        cm = confusion_matrix(te_true, te_pred, labels=classes_in_test)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
        ax.set_title(f"Fine-tune ({args.strategy}) — F1 macro = {te_f1:.3f}")
        plt.tight_layout()
        fig.savefig(MODELS_DIR / "confusion_matrix_finetune.png", dpi=150)
        plt.close()
        print(f"  Графики: training_curves.png, confusion_matrix_finetune.png")
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print("Fine-tune готов. Следующий шаг (опционально):")
    print("  uv run python crop_classification/train_maml.py")


if __name__ == "__main__":
    main()
