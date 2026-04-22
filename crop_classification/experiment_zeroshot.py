"""
Эксперимент 1. Zero-shot: kNN по эмбеддингам PRESTO (без обучения).

Идея: если PRESTO хорошо предобучен, то поля одной культуры уже
расположены близко в пространстве эмбеддингов. Проверяем это через kNN:
для каждого поля ищем K ближайших соседей → голосуем за культуру.

Это НИЖНЯЯ ГРАНИЦА качества — показывает, насколько хорошо PRESTO
обобщается на данные РФ без какого-либо дообучения.

Протокол оценки:
  - 5-fold cross-validation (стратифицированное по культурам)
  - В каждом фолде: 80% полей = «размеченная база», 20% = тест
  - kNN с K=5 (можно варьировать)
  - Метрики: F1 macro, F1 weighted, accuracy, confusion matrix

Запуск:
  uv run python crop_classification/experiment_zeroshot.py --year 2021 --tile 38TLR
"""

import argparse
import json
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from config import DATASET_DIR, ID_TO_SHORT, OUTPUT_DIR, PILOT_TILE, PILOT_YEAR


def main():
    parser = argparse.ArgumentParser(description="Эксп. 1: Zero-shot kNN")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--k", type=int, default=5, help="Число соседей kNN")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    # Загрузить эмбеддинги
    emb_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    if not emb_path.exists():
        sys.exit(f"Нет эмбеддингов: {emb_path}\n"
                 f"Сначала: run_presto_embed.py")

    data = np.load(emb_path)
    X = data["embeddings"]   # [N, 128]
    y = data["labels"]       # [N]

    print(f"\n{'='*60}")
    print(f"Эксперимент 1: Zero-shot kNN (K={args.k})")
    print(f"{'='*60}")
    print(f"  Полей: {len(y)}, классов: {len(np.unique(y))}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        knn = KNeighborsClassifier(n_neighbors=args.k, metric="cosine")
        knn.fit(X[train_idx], y[train_idx])
        preds = knn.predict(X[test_idx])
        all_preds.extend(preds)
        all_true.extend(y[test_idx])

        acc = accuracy_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds, average="macro", zero_division=0)
        print(f"  Fold {fold+1}/{args.folds}: accuracy={acc:.3f}, F1_macro={f1:.3f}")

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Итоговые метрики
    acc = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    print(f"\n  ИТОГО ({args.folds}-fold CV):")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 macro:    {f1_macro:.4f}")
    print(f"    F1 weighted: {f1_weighted:.4f}")

    # Отчёт по классам
    labels_present = sorted(np.unique(np.concatenate([all_true, all_preds])))
    names = [ID_TO_SHORT.get(i, f"id={i}") for i in labels_present]
    print(f"\n{classification_report(all_true, all_preds, labels=labels_present, target_names=names, zero_division=0)}")

    # Сохранить результаты
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "zeroshot_knn",
        "year": args.year,
        "tile": args.tile,
        "k": args.k,
        "folds": args.folds,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }
    out_path = OUTPUT_DIR / f"exp1_zeroshot_{args.year}_{args.tile}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Результаты: {out_path}")

    cm = confusion_matrix(all_true, all_preds, labels=labels_present)
    cm_path = OUTPUT_DIR / f"exp1_zeroshot_{args.year}_{args.tile}_cm.npy"
    np.save(cm_path, cm)
    print(f"  Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
