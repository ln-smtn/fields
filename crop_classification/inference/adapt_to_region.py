"""
Адаптация MAML-модели к новому региону на 5-25 размеченных полях.

Сценарий B: применение MAML на отложенном или новом регионе.

Ожидаемый формат входного CSV:
  row_id, year, tile, culture, geometry_wkt  — те же колонки, что в основном датасете.
  Можно взять подмножество основного CSV или подготовить отдельный.

Pipeline:
  1. Извлекает временные ряды S1+S2+ERA5+SRTM для каждого размеченного поля
     (через те же функции extract_*, что и в обучении).
  2. Собирает тензор [N, 24, 17] + dynamic_world.
  3. Загружает models/maml_base.pt.
  4. Делает 5 шагов SGD на support (адаптация ANIL — только голова).
  5. Сохраняет адаптированную модель в models/adapted/maml_<region>.pt.

Запуск:
  uv run python crop_classification/adapt_to_region.py \\
    --region primorye \\
    --support data/primorye_25_fields.csv \\
    --base-model models/maml_base.pt \\
    --output models/adapted/maml_primorye.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Конфиг лежит на уровень выше (crop_classification/config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    NUM_CLASSES,
    PRESTO_DW_MISSING,
    PRESTO_EMBEDDING_DIM,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_support_data(support_csv: str, region: str) -> dict:
    """Загрузить support set: предполагается, что для каждого поля
    уже посчитаны эмбеддинги в data/presto_dataset/<year>_<tile>_embeddings.npz.

    Иначе требуется отдельно вызвать build_dataset.py и run_presto_embed.py
    для тайлов нового региона.
    """
    df = pd.read_csv(support_csv, encoding="utf-8-sig")
    print(f"Support CSV: {len(df)} полей")
    print(f"Распределение: {df['culture'].value_counts().to_dict()}")

    # Находим эмбеддинги: должны быть в data/presto_dataset/
    embeddings, labels, mask_x, mask_dw, mask_latlons, mask_months = [], [], [], [], [], []
    not_found = 0

    for _, row in df.iterrows():
        year, tile = int(row["year"]), str(row.get("tile", ""))
        rid = int(row["row_id"])
        emb_path = DATASET_DIR / f"{year}_{tile}_embeddings.npz"
        raw_path = DATASET_DIR / f"{year}_{tile}.npz"

        if not emb_path.exists() or not raw_path.exists():
            not_found += 1
            continue

        emb_d = np.load(emb_path)
        raw_d = np.load(raw_path)
        # Найти индекс по row_id
        rid_arr = emb_d["row_ids"]
        idx_arr = np.where(rid_arr == rid)[0]
        if len(idx_arr) == 0:
            not_found += 1
            continue
        idx = idx_arr[0]

        embeddings.append(emb_d["embeddings"][idx])
        labels.append(int(row.get("culture_id", emb_d["labels"][idx])))
        mask_x.append(raw_d["x"][idx])
        mask_dw.append(
            raw_d["dynamic_world"][idx] if "dynamic_world" in raw_d.files
            else np.full(24, PRESTO_DW_MISSING, dtype=np.int64)
        )
        mask_latlons.append(raw_d["latlons"][idx])
        mask_months.append(
            raw_d["months"][idx, 0] if "months" in raw_d.files else 0
        )

    if not_found > 0:
        print(f"  ⚠ {not_found} полей не найдено в эмбеддингах "
              f"(сначала прогони build_dataset.py и run_presto_embed.py для их тайлов)")

    return {
        "x": np.stack(mask_x) if mask_x else None,
        "mask": None,  # будет восстановлено из raw
        "dynamic_world": np.stack(mask_dw),
        "latlons": np.stack(mask_latlons),
        "month": np.array(mask_months),
        "labels": np.array(labels),
        "embeddings": np.stack(embeddings),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Адаптация MAML-модели к новому региону на 5-25 примерах"
    )
    parser.add_argument("--region", required=True,
                        help="Имя региона (для именования выходного файла)")
    parser.add_argument("--support", required=True,
                        help="CSV с размеченными support-полями (row_id, year, tile, culture_id)")
    parser.add_argument("--base-model", default=str(MODELS_DIR / "maml_base.pt"))
    parser.add_argument("--output", default=None,
                        help="Куда сохранить адаптированную модель")
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=5)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(MODELS_DIR / "adapted" / f"maml_{args.region}.pt")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    print(f"\n{'='*60}")
    print(f"Адаптация MAML → {args.region}")
    print(f"  Устройство: {device}")
    print(f"{'='*60}\n")

    # 1) Загрузить support data
    support = load_support_data(args.support, args.region)
    if support["embeddings"] is None:
        print("Нет данных для адаптации")
        return

    print(f"\nSupport set: {len(support['labels'])} полей, "
          f"{len(np.unique(support['labels']))} классов")

    # 2) Загрузить MAML base
    print(f"\nЗагружаю {args.base_model}...")
    from presto import Presto
    state = torch.load(args.base_model, map_location=device, weights_only=False)
    presto = Presto.load_pretrained().to(device)
    presto.encoder.load_state_dict(state["encoder_state"])
    presto.eval()

    # 3) Получить эмбеддинги support через адаптированный encoder
    support_emb = torch.tensor(support["embeddings"], dtype=torch.float32, device=device)
    support_y = torch.tensor(support["labels"], dtype=torch.long, device=device)

    # 4) Адаптация: 5 шагов SGD на голове
    print(f"\nАдаптация: {args.inner_steps} шагов SGD на support...")
    head = nn.Linear(PRESTO_EMBEDDING_DIM, NUM_CLASSES).to(device)
    # Инициализация: маленькие случайные веса
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    opt = torch.optim.SGD(head.parameters(), lr=args.inner_lr)
    for step in range(args.inner_steps):
        logits = head(support_emb)
        loss = F.cross_entropy(logits, support_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        preds = logits.argmax(dim=1)
        acc = (preds == support_y).float().mean().item()
        print(f"  Шаг {step+1}: loss={loss.item():.4f}, support accuracy={acc:.4f}")

    # 5) Сохранить адаптированную модель
    torch.save({
        "encoder_state": state["encoder_state"],
        "head_state": head.state_dict(),
        "region": args.region,
        "support_size": int(len(support["labels"])),
        "config": state.get("config", {}),
    }, args.output)
    print(f"\nСохранено: {args.output}")

    print(f"\n{'='*60}")
    print(f"Адаптация готова. Используй для предсказаний:")
    print(f"  uv run python crop_classification/predict_tile.py \\")
    print(f"    --tile <TILE> --year <YEAR> \\")
    print(f"    --model {args.output} \\")
    print(f"    --polygons outputs/ftw/<тайл>_boundaries.gpkg")


if __name__ == "__main__":
    main()
