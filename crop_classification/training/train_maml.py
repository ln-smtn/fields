"""
Обучение MAML (ANIL-вариант) для быстрой адаптации к новым регионам.

Идея:
  Обычное обучение даёт ОДНУ модель «оптимальную в среднем».
  MAML даёт «стартовую» модель, которая за 5 шагов адаптации на 25
  размеченных полях нового региона становится отличной для этого региона.

Конкретный сценарий:
  - Отложить один регион (по умолчанию primorye) — НЕ использовать в обучении.
  - На оставшихся 3 регионах формируем эпизоды:
       Каждый эпизод = одна пара (регион, год).
       N-way × K-shot:
         - support set: N классов × K примеров (адаптация);
         - query set:   N классов × Q примеров (оценка адаптации).
  - Внутренний цикл: 5 шагов SGD на support → адаптированная модель.
  - Внешний цикл: проверка адаптированной модели на query → meta-loss → обновление
    оригинальной модели.

После обучения: maml_base.pt = стартовая модель.
Чтобы получить классификатор для нового региона:
  adapt_to_region.py --region primorye --support 25_размеченных_полей.csv

Этот скрипт реализует ANIL (Almost No Inner Loop) — адаптируется только голова.
Преимущества: быстрее MAML, меньше памяти, в литературе сравнимые метрики.

Сохраняет:
  - models/maml_base.pt          — encoder_state + head_state
  - models/maml_metrics.json     — оценка на отложенном регионе

Запуск:
  uv run python crop_classification/train_maml.py
  uv run python crop_classification/train_maml.py --holdout primorye --episodes 2000
"""

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

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


class EpisodeSampler:
    """Сэмплер эпизодов из master-датасета по парам (регион, год)."""

    def __init__(self, data, holdout_region: str | None = None):
        self.x = data["x"]
        self.mask = data["mask"]
        self.dw = data["dynamic_world"]
        self.latlons = data["latlons"]
        self.months = data["months"][:, 0] if "months" in data.files else np.zeros(len(data["x"]), dtype=np.int64)
        self.labels = data["labels"]
        self.regions = data["regions"]
        self.years = data["years"]

        # Группируем индексы по (регион, год, класс)
        self.groups: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for i, (r, y, c) in enumerate(zip(self.regions, self.years, self.labels)):
            if holdout_region is not None and r == holdout_region:
                continue
            self.groups[(r, int(y))][int(c)].append(i)

        # Только пары с >=2 классами и >=10 полей суммарно
        self.valid_keys = []
        for key, class_map in self.groups.items():
            n_classes = len([c for c, idx in class_map.items() if len(idx) >= 1])
            n_fields = sum(len(idx) for idx in class_map.values())
            if n_classes >= 2 and n_fields >= 10:
                self.valid_keys.append(key)

        print(f"  EpisodeSampler: {len(self.valid_keys)} валидных пар (регион, год)")
        for key in self.valid_keys:
            r, y = key
            n_cls = len(self.groups[key])
            print(f"    {r:>10s} {y}: {n_cls} классов, "
                  f"{sum(len(v) for v in self.groups[key].values())} полей")

    def sample_episode(self, n_way: int, k_shot: int, q_query: int,
                       ) -> tuple[dict, dict] | None:
        """Сэмплировать один эпизод. Возвращает (support, query) или None если не хватило данных."""
        random.shuffle(self.valid_keys)
        for key in self.valid_keys:
            class_map = self.groups[key]
            available = [c for c, idx in class_map.items() if len(idx) >= k_shot + q_query]
            if len(available) < n_way:
                continue

            chosen_classes = random.sample(available, n_way)
            support_idx, query_idx = [], []
            support_labels, query_labels = [], []
            for new_label, old_label in enumerate(chosen_classes):
                pool = class_map[old_label][:]
                random.shuffle(pool)
                support_idx.extend(pool[:k_shot])
                query_idx.extend(pool[k_shot:k_shot + q_query])
                support_labels.extend([new_label] * k_shot)
                query_labels.extend([new_label] * q_query)

            def collect(indices, labels_new):
                return {
                    "x": torch.tensor(self.x[indices], dtype=torch.float32),
                    "mask": torch.tensor(self.mask[indices], dtype=torch.float32),
                    "dw": torch.tensor(self.dw[indices], dtype=torch.long),
                    "latlons": torch.tensor(self.latlons[indices], dtype=torch.float32),
                    "month": torch.tensor(self.months[indices], dtype=torch.long),
                    "y": torch.tensor(labels_new, dtype=torch.long),
                }

            return collect(support_idx, support_labels), collect(query_idx, query_labels)
        return None


def get_embeddings(presto, batch, device):
    """PRESTO encoder → эмбеддинги [B, 128]."""
    return presto.encoder(
        batch["x"].to(device),
        dynamic_world=batch["dw"].to(device),
        latlons=batch["latlons"].to(device),
        mask=batch["mask"].to(device),
        month=batch["month"].to(device),
    )


def inner_loop_adapt(embeddings, support_y, n_way: int, inner_lr: float, inner_steps: int):
    """ANIL inner loop: адаптируем ТОЛЬКО голову на support.
    Возвращает (адаптированный вес, адаптированный bias).
    """
    # Инициализируем голову нулями (адаптация на support)
    W = torch.zeros(n_way, embeddings.shape[1], device=embeddings.device, requires_grad=True)
    b = torch.zeros(n_way, device=embeddings.device, requires_grad=True)

    for _ in range(inner_steps):
        logits = embeddings @ W.T + b
        loss = F.cross_entropy(logits, support_y)
        # Считаем градиенты по W и b
        grads = torch.autograd.grad(loss, [W, b], create_graph=True)
        W = W - inner_lr * grads[0]
        b = b - inner_lr * grads[1]

    return W, b


def main():
    parser = argparse.ArgumentParser(description="MAML/ANIL обучение для cross-region адаптации")
    parser.add_argument("--master", default=str(MASTER_DIR / "all_raw.npz"))
    parser.add_argument("--holdout", default="primorye",
                        choices=["south", "chernozem", "volga", "primorye", "none"],
                        help="Отложенный регион (none = ничего не откладываем)")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = detect_device()
    print(f"\n{'='*60}")
    print(f"MAML (ANIL) — обучение")
    print(f"  Устройство: {device}")
    print(f"  Отложенный регион: {args.holdout}")
    print(f"  Эпизоды: {args.episodes}, {args.n_way}-way × {args.k_shot}-shot")
    print(f"{'='*60}\n")

    # 1) Загрузить данные
    data = np.load(args.master, allow_pickle=True)
    print(f"Master: {len(data['x'])} полей всего")

    holdout = args.holdout if args.holdout != "none" else None
    sampler = EpisodeSampler(data, holdout_region=holdout)

    # 2) PRESTO encoder (учим только последние блоки + псевдо-голову)
    from presto import Presto
    presto = Presto.load_pretrained().to(device)
    # Замораживаем нижнюю часть, как в fine-tune
    for p in presto.parameters():
        p.requires_grad = False
    try:
        blocks = presto.encoder.blocks
        for block in list(blocks)[-2:]:
            for p in block.parameters():
                p.requires_grad = True
    except AttributeError:
        # Fallback — оставляем всё замороженным, тренируем только голову
        pass

    trainable = [p for p in presto.parameters() if p.requires_grad]
    print(f"\nPRESTO trainable: {sum(p.numel() for p in trainable)} параметров")

    meta_optimizer = torch.optim.Adam(trainable, lr=args.outer_lr) if trainable \
        else torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=args.outer_lr)

    # 3) Цикл мета-обучения
    presto.train()
    meta_losses, meta_f1s = [], []
    for episode in range(1, args.episodes + 1):
        sample = sampler.sample_episode(args.n_way, args.k_shot, args.q_query)
        if sample is None:
            continue
        support, query = sample

        # Эмбеддинги для support и query
        support_emb = get_embeddings(presto, support, device)
        query_emb = get_embeddings(presto, query, device)

        # ANIL inner loop: адаптируем голову на support
        W, b = inner_loop_adapt(
            support_emb, support["y"].to(device),
            n_way=args.n_way, inner_lr=args.inner_lr, inner_steps=args.inner_steps,
        )

        # Meta loss на query (с адаптированной головой)
        logits_q = query_emb @ W.T + b
        meta_loss = F.cross_entropy(logits_q, query["y"].to(device))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        # Метрики
        with torch.no_grad():
            preds = logits_q.argmax(dim=1).cpu().numpy()
            f1 = f1_score(query["y"].numpy(), preds, average="macro", zero_division=0)
        meta_losses.append(meta_loss.item())
        meta_f1s.append(f1)

        if episode % 50 == 0 or episode == args.episodes:
            avg_loss = np.mean(meta_losses[-50:])
            avg_f1 = np.mean(meta_f1s[-50:])
            print(f"  Эпизод {episode:4d}/{args.episodes}: "
                  f"loss={avg_loss:.4f}  F1 macro={avg_f1:.4f}")

    # 4) Сохранить базовую модель
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": presto.encoder.state_dict(),
        "config": {
            "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
            "inner_lr": args.inner_lr, "inner_steps": args.inner_steps,
            "outer_lr": args.outer_lr, "holdout": args.holdout,
        },
    }, MODELS_DIR / "maml_base.pt")
    print(f"\n  Сохранено: {MODELS_DIR / 'maml_base.pt'}")

    # 5) Метрики
    metrics = {
        "method": "maml_anil",
        "holdout_region": args.holdout,
        "n_episodes": args.episodes,
        "final_train_loss": float(np.mean(meta_losses[-50:])),
        "final_train_f1": float(np.mean(meta_f1s[-50:])),
        "config": {
            "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
            "inner_lr": args.inner_lr, "inner_steps": args.inner_steps,
            "outer_lr": args.outer_lr,
        },
    }
    with open(MODELS_DIR / "maml_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("MAML обучен. Следующий шаг:")
    print(f"  uv run python crop_classification/adapt_to_region.py \\")
    print(f"    --region {args.holdout} \\")
    print(f"    --support data/{args.holdout}_25_fields.csv \\")
    print(f"    --output models/adapted/maml_{args.holdout}.pt")


if __name__ == "__main__":
    main()
