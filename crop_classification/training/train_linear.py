"""
Обучение Linear probe поверх замороженных эмбеддингов PRESTO.

Метод:
  - PRESTO уже отработала на этапе run_presto_embed.py, эмбеддинги сохранены.
  - Здесь обучаем линейный классификатор (LogisticRegression) на этих
    эмбеддингах. PRESTO не меняется.

Сильные стороны:
  - Очень быстро (секунды-минуты на CPU);
  - Не переобучается даже на 1000 примеров;
  - Понятный, объяснимый (1920 + 15 чисел = веса 15 классов × 128 + bias).

Слабые стороны:
  - Ограничен качеством исходных эмбеддингов PRESTO;
  - Не «подкручивает» PRESTO под РФ-специфику.

Сохраняет:
  - models/linear_head.pt          — веса логистической регрессии в формате PyTorch;
  - models/class_centers.npy       — центры классов в пространстве эмбеддингов (для OOD);
  - models/class_std.npy           — общий разброс эмбеддингов (для OOD);
  - models/class_mapping.json       — соответствие ID ↔ название культуры;
  - models/linear_metrics.json      — метрики на test;
  - models/confusion_matrix_linear.png — матрица путаницы.

Запуск:
  uv run python crop_classification/train_linear.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Конфиг лежит на уровень выше (crop_classification/config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    ID_TO_CULTURE,
    ID_TO_SHORT,
    NUM_CLASSES,
    PRESTO_EMBEDDING_DIM,
)

MASTER_DIR = DATA_DIR / "training_master"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Linear probe поверх PRESTO эмбеддингов")
    parser.add_argument("--master", type=str,
                        default=str(MASTER_DIR / "all_embeddings.npz"))
    parser.add_argument("--split", type=str,
                        default=str(MASTER_DIR / "split.json"))
    parser.add_argument("--C", type=float, default=1.0,
                        help="Регуляризация LogisticRegression")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Linear probe — обучение")
    print(f"{'='*60}")

    # 1) Загрузить master + split
    d = np.load(args.master, allow_pickle=True)
    X = d["embeddings"]
    y = d["labels"]
    print(f"Загружено: {X.shape[0]} полей, эмбеддинги {X.shape[1]}-мерные")

    with open(args.split) as f:
        split = json.load(f)
    train_idx = np.array(split["train"])
    val_idx = np.array(split["val"])
    test_idx = np.array(split["test"])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2) Нормализация
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 3) Веса классов (для борьбы с дисбалансом)
    classes_present = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes_present, y=y_train
    )
    weight_dict = dict(zip(classes_present.tolist(), class_weights.tolist()))
    print(f"\nКласс-веса (для несбалансированных данных):")
    for cid, w in sorted(weight_dict.items()):
        name = ID_TO_SHORT.get(cid, f"id={cid}")
        print(f"  {name:>15s}: weight={w:.3f}")

    # 4) Обучение
    print(f"\nОбучаю LogisticRegression (C={args.C})...")
    clf = LogisticRegression(
        max_iter=2000, C=args.C, solver="lbfgs",
        class_weight=weight_dict, random_state=42,
    )
    clf.fit(X_train_s, y_train)

    # 5) Оценка
    def evaluate(name, X_s, y_true):
        y_pred = clf.predict(X_s)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"\n  {name}:")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    F1 macro:    {f1m:.4f}")
        print(f"    F1 weighted: {f1w:.4f}")
        return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}, y_pred

    val_metrics, val_pred = evaluate("Validation", X_val_s, y_val)
    test_metrics, test_pred = evaluate("Test", X_test_s, y_test)

    # Подробный отчёт на test
    classes_in_test = sorted(np.unique(y_test).tolist())
    target_names = [ID_TO_SHORT.get(c, f"id={c}") for c in classes_in_test]
    print("\n  Classification report (test):")
    print(classification_report(
        y_test, test_pred, labels=classes_in_test,
        target_names=target_names, zero_division=0,
    ))

    # 6) Сохранить веса как PyTorch-модуль (для совместимости с inference)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    head = torch.nn.Linear(PRESTO_EMBEDDING_DIM, NUM_CLASSES)
    # sklearn возвращает coef_ shape (n_classes, n_features), переводим в torch
    full_weight = np.zeros((NUM_CLASSES, PRESTO_EMBEDDING_DIM), dtype=np.float32)
    full_bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, cid in enumerate(clf.classes_):
        full_weight[cid] = clf.coef_[i]
        full_bias[cid] = clf.intercept_[i]
    head.weight.data = torch.tensor(full_weight)
    head.bias.data = torch.tensor(full_bias)

    torch.save({
        "head_state": head.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "classes": clf.classes_.tolist(),
    }, MODELS_DIR / "linear_head.pt")
    print(f"\n  Сохранено: {MODELS_DIR / 'linear_head.pt'}")

    # 7) Центры классов (для OOD-детекции)
    print(f"\n  Считаю центры классов для OOD-фильтра...")
    centers = np.zeros((NUM_CLASSES, PRESTO_EMBEDDING_DIM), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for cid in np.unique(y_train):
        mask = y_train == cid
        centers[cid] = X_train[mask].mean(axis=0)
        counts[cid] = mask.sum()
    # Общий разброс внутри классов
    intra_distances = []
    for cid in np.unique(y_train):
        mask = y_train == cid
        if mask.sum() > 1:
            d = np.linalg.norm(X_train[mask] - centers[cid], axis=1)
            intra_distances.extend(d.tolist())
    global_std = float(np.std(intra_distances)) if intra_distances else 1.0

    np.save(MODELS_DIR / "class_centers.npy", centers)
    np.save(MODELS_DIR / "class_std.npy", np.array([global_std], dtype=np.float32))
    print(f"  Центры классов: {MODELS_DIR / 'class_centers.npy'}")
    print(f"  Глобальный std: {global_std:.4f}")

    # 8) Mapping классов
    mapping = {str(cid): {"name": name, "short": ID_TO_SHORT[cid]}
               for cid, name in ID_TO_CULTURE.items()}
    with open(MODELS_DIR / "class_mapping.json", "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  Class mapping: {MODELS_DIR / 'class_mapping.json'}")

    # 9) Метрики
    metrics = {
        "method": "linear_probe",
        "val": val_metrics, "test": test_metrics,
        "C": args.C, "n_train": int(len(X_train)),
        "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "class_distribution_train": {
            int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))
        },
    }
    with open(MODELS_DIR / "linear_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Метрики: {MODELS_DIR / 'linear_metrics.json'}")

    # 10) Confusion matrix (PNG)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, test_pred, labels=classes_in_test)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Предсказание")
        ax.set_ylabel("Истина")
        ax.set_title(f"Linear probe (test) — F1 macro = {test_metrics['f1_macro']:.3f}")
        plt.tight_layout()
        cm_path = MODELS_DIR / "confusion_matrix_linear.png"
        fig.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix: {cm_path}")
    except ImportError:
        print("  (для confusion matrix установи: uv add matplotlib seaborn)")

    print(f"\n{'='*60}")
    print("Linear probe готов. Следующий шаг:")
    print("  uv run python crop_classification/train_finetune.py")


if __name__ == "__main__":
    main()
