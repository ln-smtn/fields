"""
Inference: классификация культур на полигонах FTW.

Этот скрипт запускается ПОСЛЕ экспериментов (experiment_*.py).
К этому моменту ты уже знаешь, какой метод лучший, и используешь
соответствующую обученную модель.

Поддерживает 3 режима (в зависимости от результатов экспериментов):
  --method knn       → Zero-shot kNN (без обучения, эксп. 1)
  --method linear    → Linear probe (sklearn, эксп. 2)
  --method maml      → MAML/ANIL head (эксп. 4, по умолчанию)

Полный pipeline:
  1. Загрузить полигоны FTW (*_ftw_boundaries.gpkg)
  2. Для каждого — извлечь временной ряд S2+S1+ERA5+SRTM
  3. PRESTO → эмбеддинг 128-d
  4. Классификатор → культура + уверенность
  5. Фильтрация «не-полей»:
     - Уверенность < порога → «неизвестно»
     - Площадь < 0.5 га → артефакт FTW
  6. Сохранить GeoPackage с предсказаниями

Вход:
  - *_ftw_boundaries.gpkg (от ftw_oneshot.py, Этап 1)
  - Обученная модель (зависит от --method)
  - Эмбеддинги обучающей выборки (для kNN)

Выход:
  - classified_fields_<year>_<tile>.gpkg:
      geometry, predicted_culture, confidence, area_ha, is_valid_field

Запуск:
  # Лучший метод (MAML)
  uv run python crop_classification/predict_ftw_fields.py \
    --gpkg outputs/ftw/2021_tile_38TLR/*_ftw_boundaries.gpkg \
    --year 2021 --tile 38TLR --method maml

  # Без дообучения (kNN)
  uv run python crop_classification/predict_ftw_fields.py \
    --gpkg outputs/ftw/2021_tile_38TLR/*_ftw_boundaries.gpkg \
    --year 2021 --tile 38TLR --method knn
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn

from config import (
    DATASET_DIR,
    ID_TO_SHORT,
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


def load_ftw_polygons(gpkg_path: str) -> gpd.GeoDataFrame:
    """Загрузить полигоны FTW из GeoPackage."""
    path = Path(gpkg_path)
    if not path.exists():
        sys.exit(f"Файл не найден: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf_m = gdf.to_crs("EPSG:3857")
    gdf["area_ha"] = gdf_m.geometry.area / 10000.0
    print(f"  Загружено {len(gdf)} полигонов FTW из {path.name}")
    return gdf


def get_ftw_embeddings(gdf: gpd.GeoDataFrame, year: int, tile: str,
                       device: torch.device) -> np.ndarray:
    """Получить PRESTO-эмбеддинги для полигонов FTW.

    Полный pipeline:
      1. Для каждого полигона — извлечь временной ряд S2 + S1
      2. Подгрузить ERA5 + SRTM
      3. Собрать тензор [N, 24, 18]
      4. PRESTO encoder → [N, 128]

    ВАЖНО: этот процесс требует скачивания данных из Planetary Computer
    и занимает значительное время. Для ускорения можно сначала
    запустить extract_s2_timeseries.py / extract_s1_timeseries.py
    для полигонов FTW отдельно.
    """
    # TODO: В полной версии здесь будут вызовы:
    #   from extract_s2_timeseries import extract_polygon_values, search_scenes
    #   from extract_s1_timeseries import extract_s1_polygon_values
    #   from build_dataset import build_field_sample, resample_to_fixed_length
    #
    # Сейчас проверяем, есть ли готовые эмбеддинги для FTW полигонов
    ftw_emb_path = DATASET_DIR / f"{year}_{tile}_ftw_embeddings.npz"
    if ftw_emb_path.exists():
        data = np.load(ftw_emb_path)
        print(f"  Загружены готовые FTW-эмбеддинги: {ftw_emb_path.name}")
        return data["embeddings"]

    print(f"\n  FTW-эмбеддинги не найдены: {ftw_emb_path}")
    print(f"  Для их создания нужно:")
    print(f"    1. Извлечь временные ряды S2/S1 для полигонов FTW")
    print(f"    2. Собрать датасет и прогнать через PRESTO")
    print(f"  Это можно сделать адаптировав extract_s2_timeseries.py")
    print(f"  для входа из GeoPackage вместо CSV.")
    sys.exit(1)


# ── Классификаторы ───────────────────────────────────────────────────────

def predict_knn(ftw_embeddings: np.ndarray, train_embeddings: np.ndarray,
                train_labels: np.ndarray, k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
    """Метод kNN: ближайшие соседи в пространстве эмбеддингов.
    Без обучения — используем обучающую выборку как «базу».
    """
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings, train_labels)
    predictions = knn.predict(ftw_embeddings)
    # Уверенность = доля голосов за победивший класс
    proba = knn.predict_proba(ftw_embeddings)
    confidences = proba.max(axis=1)
    return predictions, confidences


def predict_linear(ftw_embeddings: np.ndarray, train_embeddings: np.ndarray,
                   train_labels: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Метод Linear probe: LogisticRegression поверх эмбеддингов."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embeddings)
    X_test = scaler.transform(ftw_embeddings)
    # multi_class удалён в sklearn>=1.7 — multinomial теперь по умолчанию
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    confidences = proba.max(axis=1)
    return predictions, confidences


def predict_maml(ftw_embeddings: np.ndarray, head_path: Path,
                 device: torch.device
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Метод MAML/ANIL: обученная мета-голова."""
    head = nn.Linear(PRESTO_EMBEDDING_DIM, NUM_CLASSES).to(device)
    head.load_state_dict(
        torch.load(head_path, map_location=device, weights_only=True)
    )
    head.eval()
    x = torch.tensor(ftw_embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = head(x)
        probs = torch.softmax(logits, dim=1)
        confidences = probs.max(dim=1).values.cpu().numpy()
        predictions = probs.argmax(dim=1).cpu().numpy()
    return predictions, confidences


# ── Фильтрация ──────────────────────────────────────────────────────────

def filter_non_fields(
    gdf: gpd.GeoDataFrame,
    confidences: np.ndarray,
    confidence_threshold: float = 0.5,
    min_area_ha: float = 0.5,
) -> gpd.GeoDataFrame:
    """Пометить «не-поля» и ненадёжные предсказания."""
    gdf = gdf.copy()

    low_conf = confidences < confidence_threshold
    small = gdf["area_ha"] < min_area_ha
    gdf["is_valid_field"] = True

    gdf.loc[low_conf, "predicted_culture"] = "неизвестно"
    gdf.loc[low_conf, "is_valid_field"] = False

    gdf.loc[small, "predicted_culture"] = "артефакт FTW"
    gdf.loc[small, "is_valid_field"] = False

    n_valid = gdf["is_valid_field"].sum()
    print(f"\n  Фильтрация:")
    print(f"    Всего полигонов:     {len(gdf)}")
    print(f"    Валидных:            {n_valid}")
    print(f"    Низкая уверенность:  {low_conf.sum()}")
    print(f"    Мелкие (< {min_area_ha} га): {small.sum()}")
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Классификация культур на полигонах FTW"
    )
    parser.add_argument("--gpkg", type=str, required=True,
                        help="GeoPackage с полигонами FTW")
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    parser.add_argument("--method", choices=["knn", "linear", "maml"],
                        default="maml",
                        help="Метод классификации (по умолчанию: maml)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Порог уверенности (по умолчанию 0.5)")
    parser.add_argument("--min-area", type=float, default=0.5,
                        help="Мин. площадь поля в га (по умолчанию 0.5)")
    args = parser.parse_args()

    device = detect_device()

    print(f"\n{'='*60}")
    print(f"Классификация культур на полигонах FTW")
    print(f"  Метод: {args.method}")
    print(f"{'='*60}")

    # 1. Загрузить полигоны FTW
    gdf = load_ftw_polygons(args.gpkg)

    # 2. Получить эмбеддинги FTW-полигонов
    ftw_emb = get_ftw_embeddings(gdf, args.year, args.tile, device)

    # 3. Загрузить обучающую выборку (для kNN и linear)
    train_emb_path = DATASET_DIR / f"{args.year}_{args.tile}_embeddings.npz"
    train_data = None
    if args.method in ("knn", "linear"):
        if not train_emb_path.exists():
            sys.exit(f"Нет обучающих эмбеддингов: {train_emb_path}")
        train_data = np.load(train_emb_path)

    # 4. Классификация
    print(f"\n  Классификация ({args.method})...")

    if args.method == "knn":
        predictions, confidences = predict_knn(
            ftw_emb, train_data["embeddings"], train_data["labels"]
        )
    elif args.method == "linear":
        predictions, confidences = predict_linear(
            ftw_emb, train_data["embeddings"], train_data["labels"]
        )
    elif args.method == "maml":
        head_path = OUTPUT_DIR / f"maml_head_{args.year}_{args.tile}.pt"
        if not head_path.exists():
            sys.exit(f"MAML head не найден: {head_path}\n"
                     f"Сначала запустите experiment_maml.py")
        predictions, confidences = predict_maml(ftw_emb, head_path, device)

    # 5. Записать результаты
    gdf["predicted_culture_id"] = predictions
    gdf["predicted_culture"] = [ID_TO_SHORT.get(p, f"id={p}") for p in predictions]
    gdf["confidence"] = np.round(confidences, 3)

    # 6. Фильтрация
    gdf = filter_non_fields(gdf, confidences,
                             confidence_threshold=args.confidence,
                             min_area_ha=args.min_area)

    # 7. Статистика
    valid = gdf[gdf["is_valid_field"]]
    if not valid.empty:
        print(f"\n  Предсказанные культуры:")
        for culture, count in valid["predicted_culture"].value_counts().items():
            pct = 100 * count / len(valid)
            print(f"    {culture}: {count} ({pct:.1f}%)")

    # 8. Сохранить GeoPackage
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"classified_fields_{args.year}_{args.tile}.gpkg"
    save_cols = [c for c in [
        "geometry", "predicted_culture", "predicted_culture_id",
        "confidence", "area_ha", "is_valid_field",
        "centroid_lat", "centroid_lon",
    ] if c in gdf.columns]
    gdf[save_cols].to_file(out_path, driver="GPKG")
    print(f"\n  Сохранено: {out_path}")
    print(f"\n  Открой в QGIS:")
    print(f"    1. Загрузить {out_path}")
    print(f"    2. Стиль → Категоризированный → поле 'predicted_culture'")
    print(f"    3. Прозрачность по 'confidence' для визуализации уверенности")


if __name__ == "__main__":
    main()
