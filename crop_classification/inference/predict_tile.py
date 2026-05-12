"""
predict_tile.py — единый пайплайн применения модели на тайл Sentinel-2.

Принимает на вход полигоны (от FTW) + год + тайл и выдаёт GeoPackage с
классифицированными полями и метками "не поле" для подозрительных.

Pipeline:
  1. Загрузить полигоны из GeoPackage (результат FTW).
  2. Извлечь временные ряды S1+S2+ERA5+SRTM для каждого полигона.
  3. Собрать тензор [N, 24, 17] + dynamic_world (как в build_dataset.py).
  4. PRESTO encoder → эмбеддинги [N, 128].
  5. Загрузить выбранный classifier head и предсказать.
  6. Фильтр "не-полей": confidence threshold, OOD distance, площадь.
  7. Сохранить data/inference_cache/<year>_<tile>/classified.gpkg.

Запуск:
  uv run python crop_classification/predict_tile.py \\
    --tile 38TLR --year 2025 \\
    --model models/finetuned_model.pt \\
    --polygons outputs/ftw/2025_38TLR_boundaries.gpkg

ВАЖНО: extract_s2_timeseries.py и др. рассчитаны на полигоны из CSV.
Здесь временные ряды для FTW-полигонов извлекаются ОТ ЭТОГО скрипта.
Если для тайла уже есть готовый _embeddings.npz — используется он.
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Конфиг лежит на уровень выше (crop_classification/config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    ID_TO_SHORT,
    NUM_CLASSES,
    PRESTO_DW_MISSING,
    PRESTO_EMBEDDING_DIM,
)

INFERENCE_CACHE = DATA_DIR / "inference_cache"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_polygons(gpkg_path: str) -> gpd.GeoDataFrame:
    """Загрузить полигоны из FTW и добавить мета-колонки."""
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    # Площадь в проекции метров
    gdf_m = gdf.to_crs("EPSG:3857")
    gdf["area_ha"] = gdf_m.geometry.area / 10000.0
    # Центроиды (для информации в выходе)
    gdf_m["c"] = gdf_m.geometry.centroid
    cent_4326 = gpd.GeoSeries(gdf_m["c"], crs="EPSG:3857").to_crs("EPSG:4326")
    gdf["centroid_lat"] = cent_4326.y
    gdf["centroid_lon"] = cent_4326.x
    print(f"Загружено {len(gdf)} полигонов из {gpkg_path}")
    return gdf


def extract_or_load_embeddings(gdf, year: int, tile: str, device: torch.device,
                                ) -> np.ndarray:
    """Получить эмбеддинги для полигонов.

    Стратегия:
      1) Если в data/presto_dataset/<year>_<tile>_embeddings.npz уже есть
         эмбеддинги для этих полигонов (по row_id) — берём оттуда.
      2) Иначе извлекаем временные ряды (S1+S2+ERA5+SRTM) → строим тензоры →
         прогоняем через PRESTO.

    Полная реализация (2) требует адаптации extract_s2_timeseries.py
    для работы с GeoPackage-входом, а не CSV. Пока (TODO) пайплайн
    использует только готовые эмбеддинги — для демонстрации.
    """
    from config import DATASET_DIR
    emb_path = DATASET_DIR / f"{year}_{tile}_embeddings.npz"

    if emb_path.exists():
        d = np.load(emb_path)
        print(f"Используются готовые эмбеддинги: {emb_path}")
        return d["embeddings"]

    raise NotImplementedError(
        f"\nЭмбеддинги для {year}_{tile} не найдены: {emb_path}\n"
        f"\nДля новых тайлов нужно:\n"
        f"  1. Сначала запустить extract_s2_timeseries.py для FTW-полигонов\n"
        f"     (адаптировать его, чтобы он принимал .gpkg на входе);\n"
        f"  2. extract_s1_timeseries.py, download_era5.py, download_srtm.py\n"
        f"  3. build_dataset.py;\n"
        f"  4. run_presto_embed.py.\n"
        f"\nПока что demo работает с тайлами, для которых эмбеддинги уже посчитаны."
    )


def load_classifier(model_path: str, device: torch.device):
    """Загрузить голову классификатора (linear / finetune / maml-adapted).

    Возвращает функцию predict(embeddings) → (probs, classes).
    """
    state = torch.load(model_path, map_location=device, weights_only=False)
    head = nn.Linear(PRESTO_EMBEDDING_DIM, NUM_CLASSES).to(device)

    if "head_state" in state:
        head.load_state_dict(state["head_state"])
    else:
        raise ValueError(f"В {model_path} нет 'head_state'")

    head.eval()

    # Linear probe требует StandardScaler нормализации эмбеддингов
    scaler_mean = state.get("scaler_mean")
    scaler_scale = state.get("scaler_scale")

    @torch.no_grad()
    def predict(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = emb.astype(np.float32)
        if scaler_mean is not None:
            x = (x - scaler_mean) / scaler_scale
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        logits = head(x_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        return preds, confidence

    return predict


def compute_ood_scores(embeddings: np.ndarray,
                       centers: np.ndarray, std: float) -> np.ndarray:
    """OOD score: расстояние до ближайшего центра класса в std-единицах.
    Чем больше — тем дальше от обучающего распределения.
    """
    dists = np.linalg.norm(
        embeddings[:, None, :] - centers[None, :, :], axis=2,
    )
    min_dist = dists.min(axis=1)
    return min_dist / std


def filter_non_fields(gdf, predictions, confidences, ood_scores,
                       min_conf: float, max_ood: float, min_area: float):
    """Помечать полигоны как "не поле" по трём порогам."""
    gdf = gdf.copy()

    is_low_conf = confidences < min_conf
    is_ood = ood_scores > max_ood
    is_small = gdf["area_ha"] < min_area

    gdf["is_field"] = ~(is_low_conf | is_ood | is_small)

    # Подписи
    gdf["predicted_culture_id"] = predictions.astype(int)
    gdf["predicted_culture"] = [ID_TO_SHORT.get(int(p), f"id={p}") for p in predictions]
    gdf.loc[is_small, "predicted_culture"] = "артефакт FTW"
    gdf.loc[is_low_conf & ~is_small, "predicted_culture"] = "не поле (низкая уверенность)"
    gdf.loc[is_ood & ~is_low_conf & ~is_small, "predicted_culture"] = "не поле (OOD)"

    gdf["confidence"] = np.round(confidences, 3)
    gdf["ood_score"] = np.round(ood_scores, 3)
    return gdf


def main():
    parser = argparse.ArgumentParser(description="Предсказание культур на FTW-полигонах")
    parser.add_argument("--tile", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--polygons", required=True,
                        help="GeoPackage с полигонами из FTW")
    parser.add_argument("--model", required=True,
                        help="Путь к модели: linear_head.pt / finetuned_model.pt / adapted/...pt")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-ood", type=float, default=3.0)
    parser.add_argument("--min-area", type=float, default=0.5)
    args = parser.parse_args()

    device = detect_device()
    print(f"\n{'='*60}")
    print(f"Предсказание культур: {args.year}, тайл {args.tile}")
    print(f"  Модель: {args.model}")
    print(f"  Устройство: {device}")
    print(f"{'='*60}\n")

    # 1) Полигоны
    gdf = load_polygons(args.polygons)

    # 2) Эмбеддинги
    embeddings = extract_or_load_embeddings(gdf, args.year, args.tile, device)
    if len(embeddings) != len(gdf):
        print(f"  ⚠ Размеры не совпадают: эмбеддингов {len(embeddings)}, полигонов {len(gdf)}")
        gdf = gdf.iloc[:len(embeddings)].copy()

    # 3) Классификатор
    predict_fn = load_classifier(args.model, device)
    predictions, confidences = predict_fn(embeddings)

    # 4) OOD
    centers = np.load(MODELS_DIR / "class_centers.npy")
    std = float(np.load(MODELS_DIR / "class_std.npy")[0])
    ood_scores = compute_ood_scores(embeddings, centers, std)

    # 5) Фильтр
    gdf = filter_non_fields(gdf, predictions, confidences, ood_scores,
                            min_conf=args.min_confidence,
                            max_ood=args.max_ood, min_area=args.min_area)

    gdf["method"] = Path(args.model).stem
    gdf["year"] = args.year
    gdf["tile"] = args.tile

    n_valid = gdf["is_field"].sum()
    print(f"\nИтоги фильтрации:")
    print(f"  Всего полигонов:     {len(gdf)}")
    print(f"  Реальных полей:      {n_valid}")
    print(f"  Не-полей/артефактов: {len(gdf) - n_valid}")

    if n_valid > 0:
        valid = gdf[gdf["is_field"]]
        print(f"\n  Распределение культур (среди валидных полей):")
        for cult, cnt in valid["predicted_culture"].value_counts().items():
            print(f"    {cult}: {cnt} ({100*cnt/len(valid):.1f}%)")

    # 6) Сохранение
    out_dir = INFERENCE_CACHE / f"{args.year}_{args.tile}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg = out_dir / "classified.gpkg"

    save_cols = [c for c in [
        "geometry", "predicted_culture", "predicted_culture_id",
        "confidence", "ood_score", "area_ha", "is_field",
        "centroid_lat", "centroid_lon", "method", "year", "tile",
    ] if c in gdf.columns]
    gdf[save_cols].to_file(out_gpkg, driver="GPKG")

    # Метаданные
    metadata = {
        "tile": args.tile, "year": args.year,
        "model": args.model,
        "total_polygons": len(gdf),
        "valid_fields": int(n_valid),
        "non_fields": int(len(gdf) - n_valid),
        "total_field_area_ha": float(gdf[gdf["is_field"]]["area_ha"].sum()),
        "culture_distribution": gdf[gdf["is_field"]]["predicted_culture"]
            .value_counts().to_dict(),
        "thresholds": {
            "min_confidence": args.min_confidence,
            "max_ood": args.max_ood, "min_area_ha": args.min_area,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n  Сохранено:")
    print(f"    {out_gpkg}")
    print(f"    {out_dir / 'metadata.json'}")
    print(f"\n  Открой в QGIS:")
    print(f"    1. Загрузить {out_gpkg.name}")
    print(f"    2. Стиль → Категоризированный → поле 'predicted_culture'")
    print(f"    3. Прозрачность по 'confidence' (опционально)")


if __name__ == "__main__":
    main()
