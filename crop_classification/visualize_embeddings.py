"""
Визуализация эмбеддингов PRESTO в 2D (t-SNE).

Что делает:
  1. Загружает эмбеддинги [N, 128] и метки [N]
  2. Сжимает 128 измерений в 2 через t-SNE
  3. Рисует точки, раскрашивая по культуре
  4. Хорошая картинка: точки одного цвета кучкуются вместе
     Плохая картинка: цвета вперемешку → модель не различает культуры

Запуск:
  uv run python crop_classification/visualize_embeddings.py --year 2021 --tile 38TLR
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from config import DATASET_DIR, ID_TO_SHORT, OUTPUT_DIR, PILOT_TILE, PILOT_YEAR


def main():
    parser = argparse.ArgumentParser(description="t-SNE визуализация PRESTO эмбеддингов")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="Параметр t-SNE (меньше=локальная структура, больше=глобальная)")
    args = parser.parse_args()

    # Загрузить эмбеддинги
    emb_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    d = np.load(emb_path)
    X = d["embeddings"]  # [N, 128]
    y = d["labels"]      # [N]
    print(f"Загружено: {X.shape[0]} полей, {X.shape[1]}-мерные эмбеддинги, "
          f"{len(np.unique(y))} классов")

    # t-SNE: сжатие 128 -> 2
    print(f"Считаю t-SNE (perplexity={args.perplexity}, ~20 сек)...")
    X2 = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=42, init="pca", max_iter=1000,
    ).fit_transform(X)

    # Рисуем
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9))

    # Цветовая палитра: отдельный цвет на каждый класс
    unique_classes = np.unique(y)
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(unique_classes):
        mask = y == cid
        label = f"{ID_TO_SHORT.get(cid, f'id={cid}')} (n={mask.sum()})"
        ax.scatter(X2[mask, 0], X2[mask, 1], s=28, alpha=0.7,
                   color=cmap(i / max(1, len(unique_classes) - 1)),
                   label=label, edgecolors="white", linewidths=0.3)

    ax.set_title(
        f"t-SNE эмбеддингов PRESTO — {args.year}, тайл {args.tile}\n"
        f"Цвет = культура. Если кластеры чистые → модель различает культуры."
    )
    ax.set_xlabel("t-SNE компонента 1")
    ax.set_ylabel("t-SNE компонента 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"tsne_{args.year}_{args.tile}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nСохранено: {out_path}")
    print(f"Открой в Preview/QuickLook и посмотри — кластеры должны быть раздельными.")


if __name__ == "__main__":
    main()
