"""
Шаг 6. Получение эмбеддингов PRESTO для всех полей.

Прогоняет собранный датасет (*.npz) через предобученный PRESTO-энкодер
и сохраняет 128-мерные эмбеддинги для каждого поля.

PRESTO (Pre-trained Remote Sensing Transformer):
  - Вход: x [N, 24, 18] + mask + latlons + months
  - Выход: embeddings [N, 128]  — вектор, кодирующий
    «фенологический портрет» поля за сезон

Эмбеддинг — компактное представление поля, в котором:
  - Поля одной культуры расположены БЛИЗКО друг к другу
  - Поля разных культур — ДАЛЕКО
  (если PRESTO хорошо обобщается на РФ)

Эмбеддинги используются во ВСЕХ экспериментах:
  - Zero-shot (kNN)   — ищем ближайших соседей в пространстве эмбеддингов
  - Linear probe      — обучаем линейный классификатор поверх эмбеддингов
  - Fine-tune         — дообучаем энкодер + голову (пересчёт эмбеддингов)
  - MAML/ANIL         — мета-обучение головы поверх эмбеддингов

Результат: data/presto_dataset/<year>_<tile>_embeddings.npz
  embeddings: [N, 128]
  labels:     [N]
  row_ids:    [N]
  latlons:    [N, 2]

Запуск:
  uv run python crop_classification/run_presto_embed.py --year 2021 --tile 38TLR

Предусловие:
  pip install presto-ml  (или: pip install -e ./presto)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from config import (
    DATASET_DIR,
    ID_TO_SHORT,
    PILOT_TILE,
    PILOT_YEAR,
    PRESTO_DW_MISSING,
    PRESTO_EMBEDDING_DIM,
)


def detect_device() -> torch.device:
    """Определить устройство: MPS (Apple Silicon) → CUDA → CPU."""
    if torch.backends.mps.is_available():
        print(f"  Устройство: MPS (Apple Silicon GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"  Устройство: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda:0")
    print(f"  Устройство: CPU (будет медленнее)")
    return torch.device("cpu")


def load_dataset(year: int, tile: str) -> dict:
    """Загрузить собранный датасет из .npz."""
    path = DATASET_DIR / f"{year}_{tile}.npz"
    if not path.exists():
        sys.exit(f"Датасет не найден: {path}\n"
                 f"Сначала запустите build_dataset.py")
    data = dict(np.load(path))
    print(f"  Датасет: {path.name}")
    print(f"    x:       {data['x'].shape}")
    print(f"    labels:  {data['labels'].shape}  "
          f"({len(np.unique(data['labels']))} классов)")
    return data


def compute_embeddings(data: dict, device: torch.device, batch_size: int = 64
                       ) -> np.ndarray:
    """Прогнать датасет через PRESTO → эмбеддинги [N, 128].

    PRESTO.encoder.forward принимает:
      x:             [B, T, 17]  — S1+S2+ERA5+SRTM+NDVI
      dynamic_world: [B, T]      — int класс DW (9 = пропуск)
      latlons:       [B, 2]      — координаты
      mask:          [B, T, 17]  — маска пропусков (опционально)
      month:         int | [B]   — стартовый месяц (для позиц. эмбеддингов)

    Мы передаём данные батчами (batch_size=64) для экономии памяти.
    """
    try:
        from presto import Presto
    except ImportError:
        sys.exit(
            "Не установлен presto.\n\n"
            "Установите:\n"
            "  uv pip install -e ./presto_repo --no-deps"
        )

    print(f"\n  Загрузка предобученного PRESTO...")
    model = Presto.load_pretrained()
    model = model.to(device)
    model.eval()

    x = torch.tensor(data["x"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    latlons = torch.tensor(data["latlons"], dtype=torch.float32)

    # Dynamic World: загружаем из датасета (должен быть заполнен 9 = «пропуск»).
    # Если его нет (старый датасет) — создаём на лету.
    if "dynamic_world" in data:
        dw = torch.tensor(data["dynamic_world"], dtype=torch.long)
    else:
        print(f"  (в датасете нет dynamic_world — создаю sentinel {PRESTO_DW_MISSING})")
        dw = torch.full(x.shape[:2], PRESTO_DW_MISSING, dtype=torch.long)

    # Стартовый месяц для каждого поля — первый реальный timestep из months.
    # Нужен PRESTO'у для корректных позиционных эмбеддингов (фенология).
    months_arr = data.get("months")
    if months_arr is not None:
        start_month = torch.tensor(months_arr[:, 0], dtype=torch.long)
    else:
        start_month = torch.zeros(x.shape[0], dtype=torch.long)

    N = x.shape[0]
    embeddings = np.zeros((N, PRESTO_EMBEDDING_DIM), dtype=np.float32)

    print(f"  Вычисление эмбеддингов: {N} полей, batch_size={batch_size}")

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_x = x[start:end].to(device)
            batch_mask = mask[start:end].to(device)
            batch_latlons = latlons[start:end].to(device)
            batch_dw = dw[start:end].to(device)
            batch_month = start_month[start:end].to(device)

            # PRESTO encoder → [B, 128]
            emb = model.encoder(
                batch_x,
                dynamic_world=batch_dw,
                latlons=batch_latlons,
                mask=batch_mask,
                month=batch_month,
            )
            embeddings[start:end] = emb.cpu().numpy()

            pct = 100 * end / N
            print(f"\r  Прогресс: {end}/{N} ({pct:.0f}%)", end="", flush=True)

    print()
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Получение эмбеддингов PRESTO для полей"
    )
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Эмбеддинги PRESTO: {args.year}, тайл {args.tile}")
    print(f"{'='*60}")

    device = detect_device()
    data = load_dataset(args.year, args.tile)
    embeddings = compute_embeddings(data, device, args.batch_size)

    # Сохранить
    out_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=data["labels"],
        row_ids=data["row_ids"],
        latlons=data["latlons"],
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nСохранено: {out_path} ({size_mb:.1f} МБ)")
    print(f"  embeddings: {embeddings.shape}")

    # Быстрая проверка: косинусное сходство между культурами
    labels = data["labels"]
    unique = np.unique(labels)
    print(f"\nСредние эмбеддинги по культурам (норма L2):")
    for cid in unique:
        emb_cls = embeddings[labels == cid]
        mean_norm = np.linalg.norm(emb_cls.mean(axis=0))
        std_norm = np.mean(np.linalg.norm(emb_cls - emb_cls.mean(axis=0), axis=1))
        name = ID_TO_SHORT.get(cid, f"id={cid}")
        print(f"  {name:>15s}: ||mean||={mean_norm:.3f}, "
              f"std_intra={std_norm:.3f}  (n={len(emb_cls)})")

    print(f"\n{'='*60}")
    print(f"Следующий шаг — эксперименты:")
    print(f"  uv run python crop_classification/experiment_zeroshot.py "
          f"--year {args.year} --tile {args.tile}")


if __name__ == "__main__":
    main()
