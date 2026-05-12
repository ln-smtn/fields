"""
Сборка master-датасета: объединение всех (регион, год, тайл) в один файл.

Этот скрипт сливает результаты всех run_presto_embed.py запусков в два файла:
  - all_embeddings.npz — только эмбеддинги PRESTO [N, 128] + метки + мета.
                        Используется в train_linear.py.
  - all_raw.npz        — сырые тензоры [N, 24, 17] + dynamic_world + мета.
                        Используется в train_finetune.py и train_maml.py.

Дополнительно вычисляет stratified split (train/val/test) с учётом
культура × регион × год — чтобы каждое из трёх множеств содержало
все классы из всех регионов и годов.

Запуск:
  uv run python crop_classification/build_master_dataset.py

  # С отложенным регионом для cross-region теста:
  uv run python crop_classification/build_master_dataset.py --holdout-region primorye
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt
from sklearn.model_selection import train_test_split

# Конфиг лежит на уровень выше (crop_classification/config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CULTURES_CSV,
    DATASET_DIR,
    DATA_DIR,
    NUM_CLASSES,
    PRESTO_DW_MISSING,
    PRESTO_EMBEDDING_DIM,
    PRESTO_MAX_TIMESTEPS,
    PRESTO_NUM_CHANNELS,
)


MASTER_DIR = DATA_DIR / "training_master"


def region_for(lat: float, lon: float) -> str:
    """Грубая привязка по координатам центроида (как в tools/check_regions.py)."""
    if 42 <= lat <= 48 and 35 <= lon <= 46:
        return "south"
    if 50 <= lat <= 55 and 34 <= lon <= 41:
        return "chernozem"
    if 48 <= lat <= 55 and 40 <= lon <= 50:
        return "volga"
    if 42 <= lat <= 46 and 130 <= lon <= 135:
        return "primorye"
    return "other"


def parse_year_tile_from_filename(stem: str) -> tuple[int, str] | None:
    """Извлечь (year, tile) из имени файла '2021_38TLR' или '2021_38TLR_embeddings'."""
    m = re.match(r"^(\d{4})_([0-9A-Z]+)(?:_embeddings)?$", stem)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def collect_npz_files(emb: bool) -> list[tuple[int, str, Path]]:
    """Найти все нужные .npz файлы в DATASET_DIR."""
    suffix = "_embeddings.npz" if emb else ".npz"
    files = []
    for path in sorted(DATASET_DIR.glob(f"*{suffix}")):
        # Пропускаем эмбеддинги при поиске сырых и наоборот
        if emb and not path.stem.endswith("_embeddings"):
            continue
        if not emb and path.stem.endswith("_embeddings"):
            continue
        parsed = parse_year_tile_from_filename(path.stem)
        if parsed is None:
            print(f"  Пропуск (не распознано): {path.name}")
            continue
        year, tile = parsed
        files.append((year, tile, path))
    return files


def load_polygon_centroids() -> dict[int, tuple[float, float]]:
    """Загрузить центроиды из CSV для определения региона каждого поля."""
    df = pd.read_csv(CULTURES_CSV, encoding="utf-8-sig")
    df["geom"] = df["geometry_wkt"].apply(wkt.loads)
    df["lat"] = df["geom"].apply(lambda g: g.centroid.y)
    df["lon"] = df["geom"].apply(lambda g: g.centroid.x)
    return dict(zip(df["row_id"], zip(df["lat"], df["lon"])))


def build_master(emb: bool, holdout_region: str | None = None) -> dict:
    """Собрать master-датасет (эмбеддинги ИЛИ сырые тензоры)."""
    print(f"\n{'='*60}")
    print(f"Сборка master ({'эмбеддинги' if emb else 'сырые тензоры'})")
    print(f"{'='*60}")

    files = collect_npz_files(emb=emb)
    if not files:
        raise SystemExit(f"Не найдено файлов *{'_embeddings.npz' if emb else '.npz'} в {DATASET_DIR}")

    print(f"Найдено {len(files)} файлов:")
    for year, tile, path in files:
        print(f"  {year} {tile}: {path.name}")

    centroids = load_polygon_centroids()

    all_arrays = {}  # имя массива → список
    all_meta = {"years": [], "tiles": [], "regions": [], "row_ids": []}

    for year, tile, path in files:
        d = np.load(path)
        n = len(d["labels"])

        # Главные массивы
        if emb:
            all_arrays.setdefault("embeddings", []).append(d["embeddings"])
        else:
            all_arrays.setdefault("x", []).append(d["x"])
            all_arrays.setdefault("mask", []).append(d["mask"])
            # dynamic_world может отсутствовать в старых файлах — генерим заглушку
            if "dynamic_world" in d.files:
                all_arrays.setdefault("dynamic_world", []).append(d["dynamic_world"])
            else:
                dw = np.full((n, PRESTO_MAX_TIMESTEPS),
                             PRESTO_DW_MISSING, dtype=np.int64)
                all_arrays.setdefault("dynamic_world", []).append(dw)
            if "months" in d.files:
                all_arrays.setdefault("months", []).append(d["months"])

        all_arrays.setdefault("labels", []).append(d["labels"])
        if "latlons" in d.files:
            all_arrays.setdefault("latlons", []).append(d["latlons"])

        # Мета: год, тайл, регион (по центроиду из CSV)
        all_meta["years"].extend([year] * n)
        all_meta["tiles"].extend([tile] * n)
        all_meta["row_ids"].extend(d["row_ids"].tolist())

        for rid in d["row_ids"]:
            if rid in centroids:
                lat, lon = centroids[int(rid)]
                all_meta["regions"].append(region_for(lat, lon))
            else:
                all_meta["regions"].append("other")

    # Конкатенация
    master = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    master["years"] = np.array(all_meta["years"])
    master["tiles"] = np.array(all_meta["tiles"])
    master["regions"] = np.array(all_meta["regions"])
    master["row_ids"] = np.array(all_meta["row_ids"])

    n_total = len(master["labels"])
    print(f"\nИтого {'эмбеддингов' if emb else 'тензоров'}: {n_total}")
    print(f"  Регионы: {Counter(master['regions'].tolist())}")
    print(f"  Годы:    {Counter(master['years'].tolist())}")
    print(f"  Классы:  {Counter(master['labels'].tolist())}")

    # Отложить регион
    if holdout_region:
        mask_train = master["regions"] != holdout_region
        mask_holdout = master["regions"] == holdout_region
        print(f"\nОтложенный регион '{holdout_region}': {mask_holdout.sum()} полей")
        print(f"Остаётся для обучения: {mask_train.sum()} полей")

        train_master = {k: v[mask_train] for k, v in master.items()}
        holdout_master = {k: v[mask_holdout] for k, v in master.items()}
        return {"train": train_master, "holdout": holdout_master}

    return {"train": master, "holdout": None}


def stratified_split(labels: np.ndarray, regions: np.ndarray,
                     years: np.ndarray,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     seed: int = 42,
                     ) -> dict[str, np.ndarray]:
    """Stratified split по комбинации (label, region, year).

    Гарантирует, что каждое сочетание класс×регион×год представлено
    в train, val и test (если в нём ≥3 элементов).
    """
    n = len(labels)
    indices = np.arange(n)

    # Составной ключ для стратификации
    strat_key = np.array([f"{l}_{r}_{y}" for l, r, y in zip(labels, regions, years)])

    # Слишком редкие комбинации (1-2 примера) объединяем в "rare"
    counts = Counter(strat_key.tolist())
    strat_key = np.array([k if counts[k] >= 3 else "rare" for k in strat_key])

    # Сначала отделяем test
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=strat_key, random_state=seed,
    )

    # Из trainval отделяем val
    rel_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val,
        stratify=strat_key[trainval_idx], random_state=seed,
    )

    print(f"\nStratified split:")
    print(f"  Train: {len(train_idx)} ({100*len(train_idx)/n:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({100*len(val_idx)/n:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({100*len(test_idx)/n:.1f}%)")
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main():
    parser = argparse.ArgumentParser(
        description="Сборка master-датасета для обучения PRESTO-классификатора"
    )
    parser.add_argument(
        "--holdout-region", type=str, default=None,
        choices=["south", "chernozem", "volga", "primorye"],
        help="Регион, который НЕ войдёт в обучение (для cross-region теста MAML).",
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Не собирать сырые тензоры (только эмбеддинги — быстрее, меньше места).",
    )
    args = parser.parse_args()

    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Эмбеддинги
    emb_data = build_master(emb=True, holdout_region=args.holdout_region)

    # 2) Split на train/val/test (на полном train без holdout)
    train = emb_data["train"]
    split = stratified_split(
        train["labels"], train["regions"], train["years"],
    )

    # Сохраняем эмбеддинги
    emb_path = MASTER_DIR / "all_embeddings.npz"
    np.savez_compressed(emb_path, **train)
    size_mb = emb_path.stat().st_size / 1024 / 1024
    print(f"\nСохранено: {emb_path} ({size_mb:.1f} МБ)")

    if emb_data["holdout"] is not None:
        holdout_path = MASTER_DIR / f"holdout_{args.holdout_region}_embeddings.npz"
        np.savez_compressed(holdout_path, **emb_data["holdout"])
        size_mb = holdout_path.stat().st_size / 1024 / 1024
        print(f"Сохранено: {holdout_path} ({size_mb:.1f} МБ)")

    # 3) Split-индексы
    split_path = MASTER_DIR / "split.json"
    with open(split_path, "w") as f:
        json.dump({k: v.tolist() for k, v in split.items()}, f)
    print(f"Сохранено: {split_path}")

    # 4) Сырые тензоры (если не --skip-raw)
    if not args.skip_raw:
        raw_data = build_master(emb=False, holdout_region=args.holdout_region)
        raw_path = MASTER_DIR / "all_raw.npz"
        np.savez_compressed(raw_path, **raw_data["train"])
        size_mb = raw_path.stat().st_size / 1024 / 1024
        print(f"\nСохранено: {raw_path} ({size_mb:.1f} МБ)")

        if raw_data["holdout"] is not None:
            holdout_path = MASTER_DIR / f"holdout_{args.holdout_region}_raw.npz"
            np.savez_compressed(holdout_path, **raw_data["holdout"])
            size_mb = holdout_path.stat().st_size / 1024 / 1024
            print(f"Сохранено: {holdout_path} ({size_mb:.1f} МБ)")

    print(f"\n{'='*60}")
    print("Готово! Следующий шаг:")
    print("  uv run python crop_classification/train_linear.py")
    print("  uv run python crop_classification/train_finetune.py")
    if args.holdout_region:
        print(f"  uv run python crop_classification/train_maml.py "
              f"--holdout {args.holdout_region}")


if __name__ == "__main__":
    main()
