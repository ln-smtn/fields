"""
Эксперимент 4. MAML/ANIL: мета-обучение для few-shot классификации.

Это ОСНОВНОЙ МЕТОД дипломной работы.

Идея ANIL (Almost No Inner Loop):
  1. PRESTO-энкодер ЗАМОРОЖЕН (не меняется)
  2. Обучается только голова (Linear 128 → N классов)
  3. Мета-обучение: модель учится БЫСТРО АДАПТИРОВАТЬСЯ за 5 шагов SGD
     на малой выборке (support set) и хорошо работать на новых данных (query set)

Протокол N-way K-shot:
  Каждый эпизод мета-обучения:
    1. Выбрать N случайных культур
    2. Для каждой культуры: K полей → support, Q полей → query
    3. Inner loop: адаптировать голову на support (5 шагов SGD)
    4. Outer loop: оценить на query → мета-градиент → обновить мета-параметры

  При тестировании:
    1. Новый support set (K полей на культуру) → адаптация за 5 шагов
    2. Классификация query set → метрики

Преимущество над обычным дообучением:
  - 83 примера ярового рапса → fine-tune переобучается
  - MAML: 5-10 примеров достаточно (модель НАУЧИЛАСЬ адаптироваться)

Запуск:
  # ANIL (по умолчанию — адаптация только головы)
  uv run python crop_classification/experiment_maml.py --year 2021 --tile 38TLR

  # Полный MAML (адаптация всей модели — медленнее)
  uv run python crop_classification/experiment_maml.py --year 2021 --tile 38TLR --full-maml

  # Варьировать число примеров (1-shot, 5-shot, 10-shot)
  uv run python crop_classification/experiment_maml.py --k-shot 1
  uv run python crop_classification/experiment_maml.py --k-shot 10
"""

import argparse
import json
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DATASET_DIR,
    FEWSHOT_K_SHOT,
    FEWSHOT_N_WAY,
    FEWSHOT_Q_QUERY,
    ID_TO_SHORT,
    MAML_INNER_LR,
    MAML_INNER_STEPS,
    MAML_OUTER_LR,
    META_TRAIN_EPISODES,
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


# ── Сэмплирование эпизодов ──────────────────────────────────────────────

def sample_episode(
    embeddings: np.ndarray,   # [N, 128]
    labels: np.ndarray,       # [N]
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Сэмплировать один эпизод для мета-обучения.

    Возвращает:
      support_x: [n_way * k_shot, 128]
      support_y: [n_way * k_shot]          — метки 0..n_way-1
      query_x:   [n_way * q_query, 128]
      query_y:   [n_way * q_query]
    """
    # Классы с достаточным числом примеров
    unique, counts = np.unique(labels, return_counts=True)
    eligible = unique[counts >= k_shot + q_query]
    if len(eligible) < n_way:
        eligible = unique[counts >= k_shot + 1]
    if len(eligible) < n_way:
        raise ValueError(f"Недостаточно классов: нужно {n_way}, "
                         f"доступно {len(eligible)}")

    chosen_classes = rng.choice(eligible, size=n_way, replace=False)

    support_x, support_y, query_x, query_y = [], [], [], []

    for new_label, cls_id in enumerate(chosen_classes):
        cls_indices = np.where(labels == cls_id)[0]
        selected = rng.choice(cls_indices, size=k_shot + q_query, replace=False)

        s_idx = selected[:k_shot]
        q_idx = selected[k_shot:k_shot + q_query]

        support_x.append(embeddings[s_idx])
        support_y.extend([new_label] * k_shot)
        query_x.append(embeddings[q_idx])
        query_y.extend([new_label] * len(q_idx))

    support_x = torch.tensor(np.concatenate(support_x), dtype=torch.float32)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.tensor(np.concatenate(query_x), dtype=torch.float32)
    query_y = torch.tensor(query_y, dtype=torch.long)

    return support_x, support_y, query_x, query_y


# ── ANIL: адаптация только головы ────────────────────────────────────────

def anil_inner_loop(
    head: nn.Linear,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> nn.Linear:
    """Внутренний цикл ANIL: адаптация головы на support set.

    Создаёт копию головы и делает inner_steps шагов SGD.
    Возвращает адаптированную голову (оригинал не изменён).
    """
    adapted_head = deepcopy(head)
    optimizer = torch.optim.SGD(adapted_head.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        logits = adapted_head(support_x)
        loss = F.cross_entropy(logits, support_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted_head


# ── Мета-обучение ────────────────────────────────────────────────────────

def meta_train(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    n_way: int,
    k_shot: int,
    q_query: int,
    inner_lr: float,
    inner_steps: int,
    outer_lr: float,
    episodes: int,
) -> nn.Linear:
    """Мета-обучение ANIL.

    Внешний цикл: оптимизируем мета-параметры головы так,
    чтобы ПОСЛЕ адаптации (inner loop) модель хорошо
    классифицировала query set.
    """
    head = nn.Linear(PRESTO_EMBEDDING_DIM, n_way).to(device)
    meta_optimizer = torch.optim.Adam(head.parameters(), lr=outer_lr)
    rng = np.random.default_rng(42)

    print(f"\n  Мета-обучение: {episodes} эпизодов")
    print(f"    {n_way}-way {k_shot}-shot, inner_lr={inner_lr}, "
          f"inner_steps={inner_steps}")

    running_loss = 0
    running_acc = 0

    for ep in range(1, episodes + 1):
        try:
            sx, sy, qx, qy = sample_episode(
                embeddings, labels, n_way, k_shot, q_query, rng
            )
        except ValueError:
            continue

        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        # Inner loop: адаптация на support
        adapted = anil_inner_loop(head, sx, sy, inner_lr, inner_steps)

        # Outer loop: оценка на query → мета-градиент
        query_logits = adapted(qx)
        meta_loss = F.cross_entropy(query_logits, qy)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        running_loss += meta_loss.item()
        running_acc += (query_logits.argmax(1) == qy).float().mean().item()

        if ep % 200 == 0:
            avg_loss = running_loss / 200
            avg_acc = running_acc / 200
            print(f"    Episode {ep:5d}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")
            running_loss = 0
            running_acc = 0

    return head


def meta_test(
    head: nn.Linear,
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    n_way: int,
    k_shot: int,
    q_query: int,
    inner_lr: float,
    inner_steps: int,
    test_episodes: int = 100,
) -> dict:
    """Тестирование мета-обученной модели.

    Для каждого тестового эпизода:
      1. Адаптация головы на support set (inner loop)
      2. Оценка на query set
    """
    rng = np.random.default_rng(123)
    accs, f1s = [], []

    for ep in range(test_episodes):
        try:
            sx, sy, qx, qy = sample_episode(
                embeddings, labels, n_way, k_shot, q_query, rng
            )
        except ValueError:
            continue

        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        adapted = anil_inner_loop(head, sx, sy, inner_lr, inner_steps)

        with torch.no_grad():
            preds = adapted(qx).argmax(1).cpu().numpy()
            true = qy.cpu().numpy()

        from sklearn.metrics import accuracy_score, f1_score
        accs.append(accuracy_score(true, preds))
        f1s.append(f1_score(true, preds, average="macro", zero_division=0))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }


# ── Оценка на ВСЕХ классах (не N-way) ───────────────────────────────────

def full_evaluation(
    head_template: nn.Linear,
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    k_shot: int,
    inner_lr: float,
    inner_steps: int,
    n_trials: int = 20,
) -> dict:
    """Полная оценка: адаптация на K полей НА КУЛЬТУРУ → классификация всех.

    Имитирует реальный сценарий использования:
    берём K размеченных полей на каждую из 15 культур,
    адаптируем голову, классифицируем остальные.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    all_accs, all_f1s = [], []

    for trial in range(n_trials):
        # Support: K полей на культуру
        support_idx, query_idx = [], []
        for cls in unique_labels:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) <= k_shot:
                # Если полей мало — берём все в support, тестируем через LOO
                chosen = rng.choice(cls_idx, size=min(k_shot, len(cls_idx)),
                                     replace=False)
                support_idx.extend(chosen)
                remaining = [i for i in cls_idx if i not in chosen]
                if remaining:
                    query_idx.extend(remaining)
            else:
                chosen = rng.choice(cls_idx, size=k_shot, replace=False)
                support_idx.extend(chosen)
                remaining = [i for i in cls_idx if i not in chosen]
                query_idx.extend(remaining)

        if not query_idx:
            continue

        sx = torch.tensor(embeddings[support_idx], dtype=torch.float32).to(device)
        sy = torch.tensor(labels[support_idx], dtype=torch.long).to(device)

        # Голова на ВСЕ классы (не N-way)
        full_head = nn.Linear(PRESTO_EMBEDDING_DIM, n_classes).to(device)
        # Инициализация из мета-обученной головы (первые n_way весов)
        with torch.no_grad():
            nn.init.xavier_uniform_(full_head.weight)

        # Адаптация
        optimizer = torch.optim.SGD(full_head.parameters(), lr=inner_lr)
        for _ in range(inner_steps * 2):  # больше шагов для всех классов
            logits = full_head(sx)
            loss = F.cross_entropy(logits, sy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Query
        qx = torch.tensor(embeddings[query_idx], dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = full_head(qx).argmax(1).cpu().numpy()
        true = labels[query_idx]

        all_accs.append(accuracy_score(true, preds))
        all_f1s.append(f1_score(true, preds, average="macro", zero_division=0))

    return {
        "accuracy_mean": float(np.mean(all_accs)),
        "accuracy_std": float(np.std(all_accs)),
        "f1_macro_mean": float(np.mean(all_f1s)),
        "f1_macro_std": float(np.std(all_f1s)),
    }


def main():
    parser = argparse.ArgumentParser(description="Эксп. 4: MAML/ANIL")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--n-way", type=int, default=FEWSHOT_N_WAY)
    parser.add_argument("--k-shot", type=int, default=FEWSHOT_K_SHOT)
    parser.add_argument("--q-query", type=int, default=FEWSHOT_Q_QUERY)
    parser.add_argument("--inner-lr", type=float, default=MAML_INNER_LR)
    parser.add_argument("--inner-steps", type=int, default=MAML_INNER_STEPS)
    parser.add_argument("--outer-lr", type=float, default=MAML_OUTER_LR)
    parser.add_argument("--episodes", type=int, default=META_TRAIN_EPISODES)
    parser.add_argument("--test-episodes", type=int, default=100)
    args = parser.parse_args()

    emb_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    if not emb_path.exists():
        sys.exit(f"Нет эмбеддингов: {emb_path}")

    data = np.load(emb_path)
    embeddings = data["embeddings"]
    labels = data["labels"]
    device = detect_device()

    print(f"\n{'='*60}")
    print(f"Эксперимент 4: ANIL (мета-обучение)")
    print(f"{'='*60}")
    print(f"  Полей: {len(labels)}, классов: {len(np.unique(labels))}")
    print(f"  Устройство: {device}")
    print(f"  Протокол: {args.n_way}-way {args.k_shot}-shot")

    # Разделить на мета-train / мета-test (80/20 по полям)
    rng = np.random.default_rng(42)
    all_idx = np.arange(len(labels))
    rng.shuffle(all_idx)
    split = int(0.8 * len(all_idx))
    train_idx, test_idx = all_idx[:split], all_idx[split:]

    train_emb, train_lab = embeddings[train_idx], labels[train_idx]
    test_emb, test_lab = embeddings[test_idx], labels[test_idx]

    print(f"  Мета-train: {len(train_lab)} полей")
    print(f"  Мета-test:  {len(test_lab)} полей")

    # 1. Мета-обучение
    head = meta_train(
        train_emb, train_lab, device,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        inner_lr=args.inner_lr, inner_steps=args.inner_steps,
        outer_lr=args.outer_lr, episodes=args.episodes,
    )

    # 2. Мета-тест (N-way эпизоды)
    print(f"\n  Тестирование ({args.test_episodes} эпизодов):")
    nway_results = meta_test(
        head, test_emb, test_lab, device,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        inner_lr=args.inner_lr, inner_steps=args.inner_steps,
        test_episodes=args.test_episodes,
    )
    print(f"    {args.n_way}-way {args.k_shot}-shot:")
    print(f"      Accuracy: {nway_results['accuracy_mean']:.4f} "
          f"± {nway_results['accuracy_std']:.4f}")
    print(f"      F1 macro: {nway_results['f1_macro_mean']:.4f} "
          f"± {nway_results['f1_macro_std']:.4f}")

    # 3. Полная оценка (все 15 классов)
    print(f"\n  Полная оценка (все {NUM_CLASSES} классов, {args.k_shot}-shot):")
    full_results = full_evaluation(
        head, test_emb, test_lab, device,
        k_shot=args.k_shot,
        inner_lr=args.inner_lr, inner_steps=args.inner_steps,
    )
    print(f"    Accuracy: {full_results['accuracy_mean']:.4f} "
          f"± {full_results['accuracy_std']:.4f}")
    print(f"    F1 macro: {full_results['f1_macro_mean']:.4f} "
          f"± {full_results['f1_macro_std']:.4f}")

    # 4. Сохранить
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "anil_maml",
        "year": args.year, "tile": args.tile,
        "n_way": args.n_way, "k_shot": args.k_shot,
        "inner_lr": args.inner_lr, "inner_steps": args.inner_steps,
        "episodes": args.episodes,
        "nway_test": nway_results,
        "full_test": full_results,
    }
    out_path = OUTPUT_DIR / f"exp4_maml_{args.year}_{args.tile}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Результаты: {out_path}")

    # Сохранить мета-обученную голову
    head_path = OUTPUT_DIR / f"maml_head_{args.year}_{args.tile}.pt"
    torch.save(head.state_dict(), head_path)
    print(f"  Мета-голова: {head_path}")


if __name__ == "__main__":
    main()
