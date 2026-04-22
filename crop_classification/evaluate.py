"""
Сравнение результатов всех экспериментов.

Загружает JSON-файлы из outputs/crop_cls/ и выводит сводную таблицу:
  Эксперимент         Accuracy   F1 macro   F1 weighted
  ─────────────────────────────────────────────────────
  1. Zero-shot kNN     0.XXX      0.XXX      0.XXX
  2. Linear probe      0.XXX      0.XXX      0.XXX
  3. Fine-tune         0.XXX      0.XXX      0.XXX
  4. MAML/ANIL         0.XXX      0.XXX      0.XXX

Также строит confusion matrix для лучшего эксперимента.

Запуск:
  uv run python crop_classification/evaluate.py --year 2021 --tile 38TLR
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import ID_TO_SHORT, OUTPUT_DIR, PILOT_TILE, PILOT_YEAR


def load_results(year: int, tile: str) -> list[dict]:
    """Загрузить все JSON-результаты экспериментов."""
    results = []
    patterns = [
        (f"exp1_zeroshot_{year}_{tile}.json", "1. Zero-shot kNN"),
        (f"exp2_linear_{year}_{tile}.json", "2. Linear probe"),
        (f"exp3_finetune_{year}_{tile}.json", "3. Fine-tune"),
        (f"exp4_maml_{year}_{tile}.json", "4. MAML/ANIL"),
    ]
    for filename, label in patterns:
        path = OUTPUT_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["_label"] = label
            results.append(data)
        else:
            print(f"  Нет результатов: {filename}")
    return results


def print_comparison_table(results: list[dict]):
    """Напечатать сводную таблицу сравнения экспериментов."""
    print(f"\n{'='*65}")
    print(f"  {'Эксперимент':<25s} {'Accuracy':>10s} {'F1 macro':>10s} {'F1 weighted':>12s}")
    print(f"  {'-'*57}")

    best_f1 = 0
    best_exp = ""

    for r in results:
        label = r["_label"]

        # Для MAML берём full_test метрики (все классы)
        if "full_test" in r:
            acc = r["full_test"]["accuracy_mean"]
            f1m = r["full_test"]["f1_macro_mean"]
            f1w = f1m  # approximation
            label += f" ({r.get('k_shot', '?')}-shot)"
        else:
            acc = r.get("accuracy", 0)
            f1m = r.get("f1_macro", 0)
            f1w = r.get("f1_weighted", 0)

        print(f"  {label:<25s} {acc:>10.4f} {f1m:>10.4f} {f1w:>12.4f}")

        if f1m > best_f1:
            best_f1 = f1m
            best_exp = label

    print(f"  {'-'*57}")
    print(f"  Лучший по F1 macro: {best_exp} ({best_f1:.4f})")
    print(f"{'='*65}")


def plot_confusion_matrix(year: int, tile: str):
    """Построить confusion matrix для zero-shot (если есть)."""
    cm_path = OUTPUT_DIR / f"exp1_zeroshot_{year}_{tile}_cm.npy"
    if not cm_path.exists():
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  (для визуализации установите: pip install matplotlib seaborn)")
        return

    cm = np.load(cm_path)
    n = cm.shape[0]
    names = [ID_TO_SHORT.get(i, f"{i}") for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title(f"Confusion Matrix — Zero-shot kNN ({year}, {tile})")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"confusion_matrix_{year}_{tile}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Confusion matrix: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Сравнение экспериментов")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"Сравнение экспериментов: {args.year}, тайл {args.tile}")
    print(f"{'='*65}")

    results = load_results(args.year, args.tile)
    if not results:
        print("  Нет результатов. Сначала запустите эксперименты.")
        return

    print_comparison_table(results)
    plot_confusion_matrix(args.year, args.tile)


if __name__ == "__main__":
    main()
