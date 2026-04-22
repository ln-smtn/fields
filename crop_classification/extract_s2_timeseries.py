"""
Шаг 1. Извлечение временных рядов Sentinel-2 для каждого полигона поля.

Для каждого полигона из cultures_polygons_dataset.csv:
  1. Находит все снимки Sentinel-2 L2A за вегетационный сезон (март–октябрь)
  2. Для каждого снимка читает 10 каналов + SCL (маска облаков) через COG
  3. Маскирует облачные/некачественные пиксели по SCL
  4. Вычисляет медианные значения каналов внутри полигона
  5. Сохраняет временной ряд в Parquet

Результат: data/timeseries/<year>_<tile>.parquet
  Колонки: row_id, date, scene_id, B02..B12, culture, culture_id, lat, lon

Как работает чтение COG:
  COG (Cloud-Optimized GeoTIFF) позволяет читать ТОЛЬКО нужный bbox
  из огромного файла (~10000×10000 px) через HTTP range requests.
  Для полигона размером 500×500 м скачивается ~50×50 px = ~5 КБ на канал,
  а не весь тайл (~1 ГБ). Поэтому скрипт не скачивает файлы целиком.

Запуск:
  # Пилот — один тайл, ~600 полигонов
  uv run python crop_classification/extract_s2_timeseries.py --year 2021 --tile 38TLR

  # Все тайлы за год
  uv run python crop_classification/extract_s2_timeseries.py --year 2021

  # Отладка — первые 3 сцены
  uv run python crop_classification/extract_s2_timeseries.py --year 2021 --tile 38TLR --max-scenes 3
"""

import argparse
import sys
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pyproj
import pystac_client
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform

from config import (
    CULTURES_CSV,
    CULTURE_TO_ID,
    MAX_CLOUD_COVER,
    S2_BANDS,
    S2_COLLECTION,
    SCL_VALID_CLASSES,
    SEASON_END,
    SEASON_START,
    STAC_API_URL,
    S2_TIMESERIES_DIR,
)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ── Загрузка полигонов из CSV ────────────────────────────────────────────

def load_polygons(year: int) -> gpd.GeoDataFrame:
    """Загрузить полигоны из CSV за указанный год → GeoDataFrame.

    Каждая строка CSV содержит WKT-геометрию полигона в EPSG:4326.
    Добавляем числовой ID культуры и координаты центра (для привязки ERA5).
    """
    df = pd.read_csv(CULTURES_CSV, encoding="utf-8-sig")
    df = df[df["year"] == year].copy()
    if df.empty:
        sys.exit(f"Нет полигонов за {year} год в {CULTURES_CSV}")

    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Нормализация Unicode: CSV может содержать «й» как и+бревис (NFD),
    # а config.py — как единый символ (NFC). Без этого 4 культуры не маппятся.
    gdf["culture"] = gdf["culture"].apply(lambda s: unicodedata.normalize("NFC", s))
    gdf["culture_id"] = gdf["culture"].map(CULTURE_TO_ID)
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["centroid_lon"] = gdf.geometry.centroid.x

    unknown = gdf[gdf["culture_id"].isna()]["culture"].unique()
    if len(unknown) > 0:
        print(f"  Внимание: культуры без маппинга (пропущены): {unknown}")
        gdf = gdf.dropna(subset=["culture_id"])
    gdf["culture_id"] = gdf["culture_id"].astype(int)

    print(f"  Загружено {len(gdf)} полигонов за {year} год")
    return gdf


# ── Определение тайлов MGRS по координатам полигонов ─────────────────────

def find_mgrs_tiles(gdf: gpd.GeoDataFrame) -> list[str]:
    """Определить тайлы MGRS (5 символов, напр. '38TLR') для всех полигонов.

    Используем библиотеку mgrs — конвертирует (lat, lon) → MGRS-код.
    Берём только первые 5 символов: зона + пояс + квадрат (100×100 км).
    """
    import mgrs
    m = mgrs.MGRS()
    tiles = set()
    for _, row in gdf.iterrows():
        try:
            tile_id = m.toMGRS(row["centroid_lat"], row["centroid_lon"],
                               MGRSPrecision=0)[:5]
            tiles.add(tile_id)
        except Exception:
            pass
    tiles = sorted(tiles)
    print(f"  Найдено {len(tiles)} тайлов MGRS: "
          f"{tiles[:10]}{'...' if len(tiles) > 10 else ''}")
    return tiles


def filter_polygons_by_tile(gdf: gpd.GeoDataFrame, tile_id: str) -> gpd.GeoDataFrame:
    """Оставить только полигоны, центроид которых попадает в тайл MGRS."""
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


# ── Поиск сцен Sentinel-2 за сезон ──────────────────────────────────────

def search_scenes(year: int, tile_id: str) -> list:
    """Найти все сцены Sentinel-2 L2A за вегетационный сезон для тайла.

    Запрос к STAC-каталогу Planetary Computer:
    - Коллекция: sentinel-2-l2a (атмосферно-скорректированные)
    - Фильтр по mgrs_tile (точное совпадение — без соседних тайлов)
    - Фильтр по облачности < MAX_CLOUD_COVER (30%)
    - Сортировка по дате
    """
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    date_range = f"{year}-{SEASON_START}/{year}-{SEASON_END}"
    search = catalog.search(
        collections=[S2_COLLECTION],
        datetime=date_range,
        query={
            "eo:cloud_cover": {"lt": MAX_CLOUD_COVER},
            "s2:mgrs_tile": {"eq": tile_id},
        },
        max_items=300,
    )
    items = sorted(search.items(), key=lambda x: x.datetime)
    print(f"  Найдено {len(items)} сцен для тайла {tile_id} за {year} "
          f"({SEASON_START}–{SEASON_END}, облачность <{MAX_CLOUD_COVER}%)")
    return items


# ── Извлечение значений каналов для одного полигона из одной сцены ───────

def _reproject_geom(geom, src_crs: str, dst_crs):
    """Перепроецировать shapely-геометрию из src_crs в dst_crs."""
    project = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform
    return shapely_transform(project, geom)


def extract_polygon_values(item, polygon_geom, bands: list[str]) -> dict | None:
    """Извлечь медианные значения каналов внутри полигона из одной сцены.

    Алгоритм:
    1. Перепроецируем полигон из EPSG:4326 в CRS растра (обычно UTM)
    2. Читаем SCL-канал (маска качества) только для bbox полигона (COG windowed read)
    3. Строим маску: пиксели ВНУТРИ полигона И с допустимым SCL-классом
    4. Если чистых пикселей нет (всё в облаках) → возвращаем None
    5. Для каждого из 10 каналов S2: читаем bbox → медиана по чистым пикселям

    Возвращает: {"B02": 456.0, "B03": 612.0, ...} или None.
    """
    # ── 1. Читаем SCL (Scene Classification Layer) ──
    if "SCL" not in item.assets:
        return None

    try:
        scl_href = item.assets["SCL"].href
        with rasterio.open(scl_href) as scl_src:
            # Перепроецируем полигон в CRS растра (EPSG:4326 → UTM)
            raster_crs = scl_src.crs
            projected_geom = _reproject_geom(polygon_geom, "EPSG:4326", raster_crs)
            geom_bounds = projected_geom.bounds  # теперь в метрах UTM
            geom_geojson = mapping(projected_geom)

            scl_window = from_bounds(*geom_bounds, scl_src.transform)
            h = max(1, int(scl_window.height))
            w = max(1, int(scl_window.width))
            if h < 1 or w < 1:
                return None

            scl_data = scl_src.read(1, window=scl_window, out_shape=(h, w))
            scl_transform = scl_src.window_transform(scl_window)

            # Маска полигона: True внутри полигона
            poly_mask = geometry_mask(
                [geom_geojson], out_shape=scl_data.shape,
                transform=scl_transform, invert=True,
            )
            # Пересечение: внутри полигона И допустимый SCL-класс
            valid_mask = poly_mask & np.isin(scl_data, list(SCL_VALID_CLASSES))
            n_valid = valid_mask.sum()
            if n_valid == 0:
                return None
    except Exception:
        return None

    # ── 2. Читаем каждый канал S2 ──
    result = {}
    for band_name in bands:
        if band_name not in item.assets:
            result[band_name] = np.nan
            continue
        try:
            href = item.assets[band_name].href
            with rasterio.open(href) as src:
                band_window = from_bounds(*geom_bounds, src.transform)
                data = src.read(
                    1, window=band_window,
                    out_shape=scl_data.shape,  # ресемпл к разрешению SCL (20 м)
                )
                clean_pixels = data[valid_mask]
                clean_pixels = clean_pixels[clean_pixels > 0]  # убрать nodata
                result[band_name] = (
                    float(np.median(clean_pixels)) if len(clean_pixels) > 0
                    else np.nan
                )
        except Exception:
            result[band_name] = np.nan

    if all(np.isnan(v) for v in result.values()):
        return None
    return result


# ── Основной цикл: все полигоны × все сцены ─────────────────────────────

def extract_timeseries_for_tile(
    gdf: gpd.GeoDataFrame,
    items: list,
    bands: list[str],
) -> pd.DataFrame:
    """Извлечь временные ряды для всех полигонов из всех сцен тайла.

    Порядок обхода: сцены → полигоны (а не наоборот), потому что
    для одной сцены rasterio кеширует HTTP-соединение → меньше overhead.

    Возвращает DataFrame с колонками:
      row_id, date, scene_id, B02..B12, culture, culture_id, lat, lon
    """
    records = []
    total = len(gdf) * len(items)
    done = 0

    for scene_idx, item in enumerate(items):
        scene_date = item.datetime.strftime("%Y-%m-%d")
        scene_id = item.id

        for poly_idx, (_, row) in enumerate(gdf.iterrows()):
            done += 1
            if done % 200 == 0 or done == total:
                pct = 100 * done / total
                print(f"\r  Прогресс: {done}/{total} ({pct:.0f}%)"
                      f"  сцена {scene_idx+1}/{len(items)} [{scene_date}]"
                      f"  поле {poly_idx+1}/{len(gdf)}", end="", flush=True)

            values = extract_polygon_values(item, row.geometry, bands)
            if values is None:
                continue

            records.append({
                "row_id": row["row_id"],
                "date": scene_date,
                "scene_id": scene_id,
                **values,
                "culture": row["culture"],
                "culture_id": row["culture_id"],
                "lat": row["centroid_lat"],
                "lon": row["centroid_lon"],
            })

    print()  # перенос строки после прогресс-бара

    df = pd.DataFrame(records)
    if df.empty:
        return df

    print(f"  Итого: {len(df)} наблюдений, "
          f"{df['row_id'].nunique()} полигонов, "
          f"{df['date'].nunique()} дат")
    return df


# ── Точка входа ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение временных рядов Sentinel-2 для полигонов полей"
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Год (2016-2021)")
    parser.add_argument("--tile", type=str, default=None,
                        help="Тайл MGRS (напр. 38TLR). "
                             "Если не указан — все тайлы за год.")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Макс. число сцен (для отладки)")
    args = parser.parse_args()

    S2_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Извлечение временных рядов Sentinel-2 за {args.year} год")
    print(f"{'='*60}")

    # 1. Загрузить полигоны
    gdf = load_polygons(args.year)

    # 2. Определить тайлы
    if args.tile:
        tiles = [args.tile]
        gdf_filtered = filter_polygons_by_tile(gdf, args.tile)
        if gdf_filtered.empty:
            sys.exit(f"Нет полигонов на тайле {args.tile}")
    else:
        tiles = find_mgrs_tiles(gdf)
        gdf_filtered = None  # будем фильтровать в цикле

    # 3. Для каждого тайла: поиск сцен → извлечение → сохранение
    for tile_id in tiles:
        print(f"\n--- Тайл {tile_id} ---")

        tile_gdf = (gdf_filtered if gdf_filtered is not None
                    else filter_polygons_by_tile(gdf, tile_id))
        if tile_gdf.empty:
            print(f"  Пропуск: нет полигонов")
            continue

        items = search_scenes(args.year, tile_id)
        if not items:
            print(f"  Пропуск: нет сцен Sentinel-2")
            continue

        if args.max_scenes:
            items = items[:args.max_scenes]
            print(f"  Ограничено до {args.max_scenes} сцен (--max-scenes)")

        ts_df = extract_timeseries_for_tile(tile_gdf, items, S2_BANDS)
        if ts_df.empty:
            print(f"  Пропуск: нет чистых наблюдений")
            continue

        # 4. Сохранить Parquet
        out_path = S2_TIMESERIES_DIR / f"{args.year}_{tile_id}.parquet"
        ts_df.to_parquet(out_path, index=False)
        size_kb = out_path.stat().st_size / 1024
        print(f"  Сохранено: {out_path} ({size_kb:.0f} КБ)")

        # Краткая статистика
        print(f"\n  Статистика:")
        print(f"    Полигонов с данными: "
              f"{ts_df['row_id'].nunique()} из {len(tile_gdf)}")
        print(f"    Дат (сцен):          {ts_df['date'].nunique()}")
        print(f"    Наблюдений всего:    {len(ts_df)}")
        cult_counts = ts_df.groupby("culture")["row_id"].nunique()
        print(f"    Культуры (полигонов):")
        for cult, cnt in sorted(cult_counts.items(), key=lambda x: -x[1]):
            print(f"      {cult}: {cnt}")

    print(f"\n{'='*60}")
    print(f"Готово! Файлы в {S2_TIMESERIES_DIR}/")


if __name__ == "__main__":
    main()
