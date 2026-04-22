"""
Шаг 2. Извлечение временных рядов Sentinel-1 (радар) для каждого полигона.

Sentinel-1 — радар C-диапазона (5.4 ГГц), активный сенсор.
Ключевые отличия от Sentinel-2:
  - Видит сквозь облака → данные есть ВСЕГДА, без пропусков
  - Два канала поляризации (VV, VH) вместо 10 спектральных каналов
  - Не нужна SCL-маска (нет облачности)
  - Повторяемость ~12 дней (vs ~5 дней у S2)

Каналы:
  VV (вертикально-вертикальная) — чувствителен к высоте/структуре растений
  VH (вертикально-горизонтальная) — чувствителен к биомассе, объёму

Коллекция Planetary Computer: sentinel-1-rtc
  RTC = Radiometric Terrain Corrected — коррекция за рельеф.
  Значения: дБ (децибелы обратного рассеяния, gamma0).
  Типичные значения: VV = -10..-5 дБ (поле), VH = -20..-12 дБ.

Результат: data/timeseries_s1/<year>_<tile>.parquet
  Колонки: row_id, date, scene_id, vv, vh, culture, culture_id, lat, lon

Запуск:
  uv run python crop_classification/extract_s1_timeseries.py --year 2021 --tile 38TLR
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
    S1_BANDS,
    S1_COLLECTION,
    S1_TIMESERIES_DIR,
    SEASON_END,
    SEASON_START,
    STAC_API_URL,
)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ── Загрузка полигонов (общая с S2) ──────────────────────────────────────

def load_polygons(year: int) -> gpd.GeoDataFrame:
    """Загрузить полигоны из CSV за указанный год → GeoDataFrame."""
    df = pd.read_csv(CULTURES_CSV, encoding="utf-8-sig")
    df = df[df["year"] == year].copy()
    if df.empty:
        sys.exit(f"Нет полигонов за {year} год в {CULTURES_CSV}")

    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # Нормализация Unicode: «й» в CSV как и+бревис (NFD), в config.py — единый символ (NFC).
    # Без этого 4 культуры (811 полигонов) теряются при .map().
    gdf["culture"] = gdf["culture"].apply(lambda s: unicodedata.normalize("NFC", s))
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


# ── Поиск сцен Sentinel-1 ───────────────────────────────────────────────

def search_s1_scenes(year: int, bbox: list[float]) -> list:
    """Найти все сцены Sentinel-1 RTC за сезон для bbox.

    Sentinel-1 в Planetary Computer индексируется по bbox (не по MGRS).
    Фильтры:
      - sat:orbit_state = "ascending" — восходящая орбита
        (более стабильная геометрия для агро, стандарт в литературе)
      - platform = "SENTINEL-1A" или "SENTINEL-1B"
    """
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    date_range = f"{year}-{SEASON_START}/{year}-{SEASON_END}"
    search = catalog.search(
        collections=[S1_COLLECTION],
        bbox=bbox,
        datetime=date_range,
        max_items=300,
    )
    items = sorted(search.items(), key=lambda x: x.datetime)
    print(f"  Найдено {len(items)} сцен Sentinel-1 за {year} "
          f"({SEASON_START}–{SEASON_END})")
    return items


# ── Извлечение VV/VH для одного полигона из одной сцены ──────────────────

def _reproject_geom(geom, src_crs: str, dst_crs):
    """Перепроецировать shapely-геометрию из src_crs в dst_crs."""
    project = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform
    return shapely_transform(project, geom)


def extract_s1_polygon_values(item, polygon_geom, bands: list[str]) -> dict | None:
    """Извлечь медианные значения VV и VH внутри полигона.

    В отличие от S2: НЕТ SCL-маски (радар не зависит от облаков).
    Маскируем только по геометрии полигона.
    Значения в дБ (gamma0 backscatter).

    Полигон в EPSG:4326, растр S1 RTC обычно в UTM — нужна перепроекция.
    """
    result = {}
    ref_shape = None

    for band_name in bands:
        if band_name not in item.assets:
            result[band_name] = np.nan
            continue
        try:
            href = item.assets[band_name].href
            with rasterio.open(href) as src:
                # Перепроецируем полигон в CRS растра
                proj_geom = _reproject_geom(polygon_geom, "EPSG:4326", src.crs)
                geom_bounds = proj_geom.bounds
                geom_geojson = mapping(proj_geom)

                band_window = from_bounds(*geom_bounds, src.transform)
                h = max(1, int(band_window.height))
                w = max(1, int(band_window.width))
                if h < 1 or w < 1:
                    result[band_name] = np.nan
                    continue

                data = src.read(1, window=band_window, out_shape=(h, w))
                band_transform = src.window_transform(band_window)

                if ref_shape is None:
                    ref_shape = data.shape

                poly_mask = geometry_mask(
                    [geom_geojson], out_shape=data.shape,
                    transform=band_transform, invert=True,
                )

                clean_pixels = data[poly_mask]
                # Убрать nodata (обычно 0 или NaN для S1 RTC)
                clean_pixels = clean_pixels[np.isfinite(clean_pixels)]
                clean_pixels = clean_pixels[clean_pixels != 0]

                result[band_name] = (
                    float(np.median(clean_pixels)) if len(clean_pixels) > 0
                    else np.nan
                )
        except Exception:
            result[band_name] = np.nan

    if all(np.isnan(v) for v in result.values()):
        return None
    return result


# ── Основной цикл ───────────────────────────────────────────────────────

def extract_s1_timeseries(
    gdf: gpd.GeoDataFrame,
    items: list,
    bands: list[str],
) -> pd.DataFrame:
    """Извлечь временные ряды S1 для всех полигонов из всех сцен."""
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

            values = extract_s1_polygon_values(item, row.geometry, bands)
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

    print()

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
        description="Извлечение временных рядов Sentinel-1 (VV, VH) "
                    "для полигонов полей"
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Год (2016-2021)")
    parser.add_argument("--tile", type=str, default=None,
                        help="Тайл MGRS (напр. 38TLR)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Макс. число сцен (для отладки)")
    args = parser.parse_args()

    S1_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Извлечение временных рядов Sentinel-1 за {args.year} год")
    print(f"{'='*60}")

    gdf = load_polygons(args.year)

    if args.tile:
        tile_gdf = filter_polygons_by_tile(gdf, args.tile)
        if tile_gdf.empty:
            sys.exit(f"Нет полигонов на тайле {args.tile}")

        # Для S1 нужен bbox (не MGRS-тайл), берём bbox всех полигонов
        bounds = tile_gdf.total_bounds  # (minx, miny, maxx, maxy)
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
        print(f"  Bbox для STAC-запроса: {bbox}")

        items = search_s1_scenes(args.year, bbox)
        if not items:
            sys.exit("Нет сцен Sentinel-1")
        if args.max_scenes:
            items = items[:args.max_scenes]

        ts_df = extract_s1_timeseries(tile_gdf, items, S1_BANDS)
        if ts_df.empty:
            sys.exit("Нет данных")

        out_path = S1_TIMESERIES_DIR / f"{args.year}_{args.tile}.parquet"
        ts_df.to_parquet(out_path, index=False)
        size_kb = out_path.stat().st_size / 1024
        print(f"  Сохранено: {out_path} ({size_kb:.0f} КБ)")

    else:
        # Все тайлы
        import mgrs
        m = mgrs.MGRS()
        tiles = set()
        for _, row in gdf.iterrows():
            try:
                t = m.toMGRS(row["centroid_lat"], row["centroid_lon"],
                             MGRSPrecision=0)[:5]
                tiles.add(t)
            except Exception:
                pass

        for tile_id in sorted(tiles):
            print(f"\n--- Тайл {tile_id} ---")
            tile_gdf = filter_polygons_by_tile(gdf, tile_id)
            if tile_gdf.empty:
                continue

            bounds = tile_gdf.total_bounds
            bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
            items = search_s1_scenes(args.year, bbox)
            if not items:
                continue
            if args.max_scenes:
                items = items[:args.max_scenes]

            ts_df = extract_s1_timeseries(tile_gdf, items, S1_BANDS)
            if ts_df.empty:
                continue

            out_path = S1_TIMESERIES_DIR / f"{args.year}_{tile_id}.parquet"
            ts_df.to_parquet(out_path, index=False)
            size_kb = out_path.stat().st_size / 1024
            print(f"  Сохранено: {out_path} ({size_kb:.0f} КБ)")

    print(f"\n{'='*60}")
    print(f"Готово! Файлы в {S1_TIMESERIES_DIR}/")


if __name__ == "__main__":
    main()
