"""
Шаг 4. Скачивание данных рельефа SRTM (высота + уклон) для полигонов.

SRTM (Shuttle Radar Topography Mission) — глобальная модель высот:
  - Разрешение: 30 м (Copernicus DEM GLO-30)
  - Покрытие: весь мир (56°S — 60°N)
  - Данные статические — скачиваются ОДИН РАЗ для региона

PRESTO использует 2 канала рельефа:
  elevation — высота над уровнем моря (м)
              Зачем: влияет на температуру (–6°C на каждые 1000 м),
              длину сезона, набор культур. В предгорьях Кавказа
              (~500-800 м) другие культуры, чем на равнине (~100 м).

  slope     — уклон поверхности (°)
              Зачем: южные склоны теплее, низины влажнее.
              Вычисляется из elevation через np.gradient.

Источник: Planetary Computer, коллекция "cop-dem-glo-30"
  Бесплатно, без регистрации (тот же API, что для S2/S1).

Результат: data/srtm/<year>_<tile>.parquet
  Колонки: row_id, elevation, slope, lat, lon
  (одна строка на полигон — рельеф не меняется во времени)

Запуск:
  # Пилот — полигоны одного тайла
  uv run python crop_classification/download_srtm.py --year 2021 --tile 38TLR

  # Все полигоны за год
  uv run python crop_classification/download_srtm.py --year 2021
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.geometry import mapping

from config import (
    CULTURES_CSV,
    CULTURE_TO_ID,
    SRTM_COLLECTION,
    SRTM_DIR,
    STAC_API_URL,
)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def load_polygons(year: int) -> gpd.GeoDataFrame:
    """Загрузить полигоны из CSV за год → GeoDataFrame."""
    df = pd.read_csv(CULTURES_CSV, encoding="utf-8-sig")
    df = df[df["year"] == year].copy()
    if df.empty:
        sys.exit(f"Нет полигонов за {year} год")
    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf["culture_id"] = gdf["culture"].map(CULTURE_TO_ID)
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf = gdf.dropna(subset=["culture_id"])
    gdf["culture_id"] = gdf["culture_id"].astype(int)
    print(f"  Загружено {len(gdf)} полигонов за {year} год")
    return gdf


def filter_polygons_by_tile(gdf: gpd.GeoDataFrame, tile_id: str) -> gpd.GeoDataFrame:
    """Оставить полигоны, попадающие в тайл MGRS."""
    import mgrs
    m = mgrs.MGRS()
    mask = []
    for _, row in gdf.iterrows():
        try:
            t = m.toMGRS(row["centroid_lat"], row["centroid_lon"],
                         MGRSPrecision=0)[:5]
            mask.append(t == tile_id)
        except Exception:
            mask.append(False)
    subset = gdf[mask].copy()
    print(f"  Тайл {tile_id}: {len(subset)} полигонов")
    return subset


def search_dem_items(bbox: list[float]) -> list:
    """Найти тайлы DEM (Copernicus GLO-30) для bbox.

    Copernicus DEM разбит на тайлы 1°×1°. Для региона 4°×4° нужно ~16 тайлов.
    Каждый тайл — один GeoTIFF, ~20-50 МБ.
    """
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SRTM_COLLECTION],
        bbox=bbox,
        max_items=200,
    )
    items = list(search.items())
    print(f"  Найдено {len(items)} тайлов DEM для bbox {bbox}")
    return items


def compute_slope(elevation: np.ndarray, resolution_m: float = 30.0) -> np.ndarray:
    """Вычислить уклон (slope) из матрицы высот.

    Slope = arctan(sqrt(dz/dx² + dz/dy²)) — угол наклона в градусах.
    resolution_m: размер пикселя в метрах (30 м для GLO-30).
    """
    dy, dx = np.gradient(elevation, resolution_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def extract_dem_for_polygon(items: list, polygon_geom) -> dict | None:
    """Извлечь медианную высоту и уклон внутри полигона.

    Алгоритм:
    1. Найти тайл DEM, покрывающий центроид полигона
    2. Прочитать высоты внутри bbox полигона (COG windowed read)
    3. Замаскировать по контуру полигона
    4. Вычислить slope из elevation
    5. Вернуть медианы elevation и slope

    Если полигон на стыке тайлов — берём тайл по центроиду.
    """
    centroid = polygon_geom.centroid
    geom_bounds = polygon_geom.bounds
    geom_geojson = mapping(polygon_geom)

    # Найти тайл, покрывающий центроид
    target_item = None
    for item in items:
        item_bbox = item.bbox  # [W, S, E, N]
        if (item_bbox[0] <= centroid.x <= item_bbox[2] and
                item_bbox[1] <= centroid.y <= item_bbox[3]):
            target_item = item
            break

    if target_item is None:
        return None

    # Читаем высоты
    try:
        href = target_item.assets["data"].href
        with rasterio.open(href) as src:
            window = from_bounds(*geom_bounds, src.transform)
            h = max(1, int(window.height))
            w = max(1, int(window.width))
            if h < 2 or w < 2:
                return None

            elev_data = src.read(1, window=window, out_shape=(h, w)).astype(float)
            win_transform = src.window_transform(window)

            # Маска полигона
            poly_mask = geometry_mask(
                [geom_geojson], out_shape=elev_data.shape,
                transform=win_transform, invert=True,
            )

            # Убрать nodata
            nodata = src.nodata or -32768
            valid_mask = poly_mask & (elev_data != nodata) & np.isfinite(elev_data)

            if valid_mask.sum() == 0:
                return None

            # Вычислить slope
            res_m = abs(src.res[0]) * 111320  # грубый пересчёт ° → м
            slope_data = compute_slope(elev_data, resolution_m=res_m)

            return {
                "elevation": float(np.median(elev_data[valid_mask])),
                "slope": float(np.median(slope_data[valid_mask])),
            }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение данных рельефа SRTM (высота, уклон) "
                    "для полигонов полей"
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Год (для выбора полигонов из CSV)")
    parser.add_argument("--tile", type=str, default=None,
                        help="Тайл MGRS (напр. 38TLR)")
    args = parser.parse_args()

    SRTM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Извлечение рельефа SRTM (elevation + slope)")
    print(f"{'='*60}")

    gdf = load_polygons(args.year)

    if args.tile:
        gdf = filter_polygons_by_tile(gdf, args.tile)
        if gdf.empty:
            sys.exit(f"Нет полигонов на тайле {args.tile}")

    # Bbox всех полигонов для поиска DEM-тайлов
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    print(f"  Bbox полигонов: {bbox}")

    # Найти тайлы DEM
    dem_items = search_dem_items(bbox)
    if not dem_items:
        sys.exit("Не найдены тайлы DEM для данного bbox")

    # Извлечь elevation + slope для каждого полигона
    records = []
    total = len(gdf)

    for i, (_, row) in enumerate(gdf.iterrows()):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"\r  Прогресс: {i+1}/{total} ({100*(i+1)/total:.0f}%)",
                  end="", flush=True)

        result = extract_dem_for_polygon(dem_items, row.geometry)
        if result is None:
            result = {"elevation": np.nan, "slope": np.nan}

        records.append({
            "row_id": row["row_id"],
            **result,
            "lat": row["centroid_lat"],
            "lon": row["centroid_lon"],
        })

    print()

    df = pd.DataFrame(records)
    valid = df.dropna(subset=["elevation"])
    print(f"  Полигонов с данными: {len(valid)} из {total}")

    if not valid.empty:
        print(f"  Elevation: min={valid['elevation'].min():.0f} м, "
              f"max={valid['elevation'].max():.0f} м, "
              f"median={valid['elevation'].median():.0f} м")
        print(f"  Slope:     min={valid['slope'].min():.1f}°, "
              f"max={valid['slope'].max():.1f}°, "
              f"median={valid['slope'].median():.1f}°")

    # Сохранить
    suffix = f"_{args.tile}" if args.tile else ""
    out_path = SRTM_DIR / f"{args.year}{suffix}.parquet"
    df.to_parquet(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Сохранено: {out_path} ({size_kb:.0f} КБ)")

    print(f"\n{'='*60}")
    print(f"Готово! Файлы в {SRTM_DIR}/")
    print(f"\nСледующий шаг: объединить все данные в датасет PRESTO:")
    print(f"  uv run python crop_classification/build_dataset.py "
          f"--year {args.year}" + (f" --tile {args.tile}" if args.tile else ""))


if __name__ == "__main__":
    main()
