"""
Эксперимент 2. Linear probe: линейный классификатор поверх PRESTO.

Идея: замороженный PRESTO-энкодер + LogisticRegression.
Если линейный классификатор работает хорошо → эмбеддинги PRESTO
уже линейно разделимы по культурам (хороший признак для few-shot).

Это ПРОСТОЙ BASELINE — между zero-shot (kNN) и дообучением.

Протокол:
  - Stratified 5-fold CV
  - LogisticRegression(max_iter=1000, C=1.0)
  - Метрики: F1 macro, F1 weighted, accuracy

Запуск:
  uv run python crop_classification/experiment_linear.py --year 2021 --tile 38TLR
"""

import argparse
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import DATASET_DIR, ID_TO_SHORT, OUTPUT_DIR, PILOT_TILE, PILOT_YEAR


def main():
    parser = argparse.ArgumentParser(description="Эксп. 2: Linear probe")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    emb_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    if not emb_path.exists():
        sys.exit(f"Нет эмбеддингов: {emb_path}")

    data = np.load(emb_path)
    X = data["embeddings"]
    y = data["labels"]

    print(f"\n{'='*60}")
    print(f"Эксперимент 2: Linear probe (LogisticRegression)")
    print(f"{'='*60}")
    print(f"  Полей: {len(y)}, классов: {len(np.unique(y))}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_true = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Нормализация (стандартная для линейных моделей)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        # multi_class удалён в sklearn>=1.7 — multinomial теперь по умолчанию
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)

        all_preds.extend(preds)
        all_true.extend(y[test_idx])

        acc = accuracy_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds, average="macro", zero_division=0)
        print(f"  Fold {fold+1}/{args.folds}: accuracy={acc:.3f}, F1_macro={f1:.3f}")

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
        "experiment": "linear_probe",
        "year": args.year, "tile": args.tile,
        "folds": args.folds,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }
    out_path = OUTPUT_DIR / f"exp2_linear_{args.year}_{args.tile}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Результаты: {out_path}")


if __name__ == "__main__":
    main()
