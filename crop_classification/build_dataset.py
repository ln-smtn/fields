"""
Шаг 5. Сборка датасета PRESTO из всех источников данных.

Объединяет промежуточные данные предыдущих шагов в единый датасет:
  - S2 временные ряды  (timeseries/*.parquet)     → 10 каналов, ~5 дней
  - S1 временные ряды  (timeseries_s1/*.parquet)   → 2 канала, ~12 дней
  - ERA5 климат        (era5/*.nc)                 → 2 канала, ежедневно
  - SRTM рельеф        (srtm/*.parquet)            → 2 канала, статические

Выравнивание по времени:
  Опорные даты = даты S2 (самые частые, ~20 дат за сезон).
  - S1 → для каждой даты S2 ищем ближайшую дату S1 (±7 дней)
  - ERA5 → для каждой даты S2 берём значение за тот же день (±1 день)
  - SRTM → статический, одинаковый для всех timesteps
  - Если дат > 24 → равномерная подвыборка
  - Если дат < 24 → дополняем нулями + mask=1 (PRESTO проигнорирует паддинг)

Нормализация:
  S2:    /10000           (стандарт L2A Surface Reflectance)
  S1:    10^(дБ/10)       (из дБ в линейные значения)
  ERA5:  t2m → (K-273.15)/30, tp → *1000 мм/сут
  NDVI:  (B8-B4)/(B8+B4)  (уже [-1, 1])
  SRTM:  elevation/1000, slope/45

Результат: data/presto_dataset/<year>_<tile>.npz
  x:              [N, 24, 17]  — входные данные (S1+S2+ERA5+SRTM+NDVI)
  mask:           [N, 24, 17]  — маска пропусков (1 = нет данных, 0 = есть)
  dynamic_world:  [N, 24]      — ID класса DW (9 = пропуск)
  months:         [N, 24]      — месяц каждого timestep (0-11)
  latlons:        [N, 2]       — координаты центра поля (lat, lon)
  labels:         [N]          — ID культуры (0-14)
  row_ids:        [N]          — row_id из CSV (для обратной привязки)

Запуск:
  uv run python crop_classification/build_dataset.py --year 2021 --tile 38TLR
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATASET_DIR,
    ERA5_DIR,
    ERA5_VARIABLES_SHORT,
    ID_TO_SHORT,
    NDVI_NIR_BAND,
    NDVI_RED_BAND,
    PILOT_TILE,
    PILOT_YEAR,
    PRESTO_CHANNEL_ORDER,
    PRESTO_DW_MISSING,
    PRESTO_MAX_TIMESTEPS,
    PRESTO_NUM_CHANNELS,
    REGIONS,
    S1_BANDS,
    S1_TIMESERIES_DIR,
    S2_BANDS,
    SRTM_DIR,
    S2_TIMESERIES_DIR,
    region_for_point,
)


# ── Загрузка источников ─────────────────────────────────────────────────

def load_s2(year: int, tile: str) -> pd.DataFrame | None:
    """Загрузить временные ряды Sentinel-2 из Parquet."""
    path = S2_TIMESERIES_DIR / f"{year}_{tile}.parquet"
    if not path.exists():
        print(f"  ОШИБКА: нет S2 данных: {path}")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  S2:   {len(df)} наблюдений, "
          f"{df['row_id'].nunique()} полигонов, "
          f"{df['date'].nunique()} дат")
    return df


def load_s1(year: int, tile: str) -> pd.DataFrame | None:
    """Загрузить временные ряды Sentinel-1 из Parquet."""
    path = S1_TIMESERIES_DIR / f"{year}_{tile}.parquet"
    if not path.exists():
        print(f"  Нет S1 данных: {path} — каналы VV/VH будут замаскированы")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  S1:   {len(df)} наблюдений, "
          f"{df['row_id'].nunique()} полигонов, "
          f"{df['date'].nunique()} дат")
    return df


def load_era5_for_point(year: int, lat: float, lon: float) -> pd.DataFrame | None:
    """Загрузить ERA5 для ближайшей точки к координатам поля.

    ERA5 — сетка 0.25°. Выбираем ближайший узел по lat/lon (nearest neighbor).
    Возвращаем DataFrame: (date, t2m, tp).
    """
    region = region_for_point(lat, lon)
    if region is None:
        return None
    path = ERA5_DIR / f"{year}_{region}.nc"
    if not path.exists():
        return None
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        # CDS переименовал координату: valid_time (новый API) / time (старый).
        time_name = "valid_time" if "valid_time" in point.coords else "time"
        time_coord = point[time_name]
        records = []
        for t in range(len(time_coord)):
            date = pd.Timestamp(time_coord.values[t])
            row = {"date": date}
            # CDS может назвать переменные "t2m"/"tp" или "2t"/"tp"
            for var in ERA5_VARIABLES_SHORT:
                if var in point:
                    row[var] = float(point[var].values[t])
            if "2t" in point and "t2m" not in row:
                row["t2m"] = float(point["2t"].values[t])
            records.append(row)
        ds.close()
        return pd.DataFrame(records)
    except Exception:
        return None


def load_srtm(year: int, tile: str | None) -> pd.DataFrame | None:
    """Загрузить данные рельефа SRTM из Parquet."""
    suffix = f"_{tile}" if tile else ""
    path = SRTM_DIR / f"{year}{suffix}.parquet"
    if not path.exists():
        print(f"  Нет SRTM данных: {path} — каналы будут замаскированы")
        return None
    df = pd.read_parquet(path)
    print(f"  SRTM: {len(df)} полигонов")
    return df


# ── Нормализация ─────────────────────────────────────────────────────────

def normalize_s2(values: np.ndarray) -> np.ndarray:
    """S2 Surface Reflectance / 10000 → [0, ~1]."""
    return values / 10000.0


def normalize_s1(values: float) -> float:
    """S1 из дБ в линейные: 10^(дБ/10). Типичные значения: 0.001..0.3."""
    return 10.0 ** (values / 10.0)


def compute_ndvi(b04: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), диапазон [-1, 1]."""
    denom = b08 + b04
    ndvi = np.where(denom > 0, (b08 - b04) / denom, 0.0)
    return np.clip(ndvi, -1.0, 1.0)


# ── Ресемплинг до фиксированной длины ───────────────────────────────────

def resample_to_fixed_length(
    data: np.ndarray,     # [T_current, 18]
    mask: np.ndarray,     # [T_current, 18]
    months: np.ndarray,   # [T_current]
    target_len: int = PRESTO_MAX_TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Привести к фиксированной длине T=24.

    > 24 дат: равномерная подвыборка (каждая N-я дата).
    < 24 дат: паддинг нулями + mask=1 (PRESTO проигнорирует).
    = 24 дат: без изменений.
    """
    T = data.shape[0]
    if T == target_len:
        return data, mask, months
    if T > target_len:
        idx = np.linspace(0, T - 1, target_len, dtype=int)
        return data[idx], mask[idx], months[idx]
    # Паддинг
    C = data.shape[1]
    pad_data = np.zeros((target_len, C), dtype=data.dtype)
    pad_mask = np.ones((target_len, C), dtype=mask.dtype)
    pad_months = np.zeros(target_len, dtype=months.dtype)
    pad_data[:T] = data
    pad_mask[:T] = mask
    pad_months[:T] = months
    return pad_data, pad_mask, pad_months


# ── Сборка одного образца ────────────────────────────────────────────────

def build_field_sample(
    row_id: int,
    s2_df: pd.DataFrame,
    s1_df: pd.DataFrame | None,
    era5_df: pd.DataFrame | None,
    srtm_row: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Собрать один образец [T, 17] из всех источников.

    Порядок 17 каналов (PRESTO_CHANNEL_ORDER, совпадает с NORMED_BANDS PRESTO):
      0-1:   S1 (VV, VH)              линейные из дБ
      2-11:  S2 (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)  нормализовано /10000
      12-13: ERA5 (t2m, tp)           t2m: (K-273.15)/30, tp: мм/сут
      14-15: SRTM (elevation, slope)  elev/1000, slope/45
      16:    NDVI                     [-1, 1]

    Dynamic World возвращается не здесь — его формирует caller как
    тензор [T] со значением PRESTO_DW_MISSING (=9), т.к. данных нет.
    """
    # ── S2 (опорные даты) ──
    field_s2 = s2_df[s2_df["row_id"] == row_id].sort_values("date")
    if field_s2.empty:
        return None

    dates = field_s2["date"].values
    T = len(dates)
    C = PRESTO_NUM_CHANNELS  # 17

    data = np.zeros((T, C), dtype=np.float32)
    mask = np.ones((T, C), dtype=np.float32)  # 1 = пропуск (нет данных)

    # ── Каналы 2-11: Sentinel-2 (10 каналов) ──
    s2_vals = field_s2[S2_BANDS].values.astype(np.float32)
    s2_valid = ~np.isnan(s2_vals)
    s2_vals = np.nan_to_num(s2_vals, nan=0.0)
    data[:, 2:12] = normalize_s2(s2_vals)
    mask[:, 2:12] = (~s2_valid).astype(np.float32)

    # ── Канал 16: NDVI (из B04, B08) ──
    b04_idx = S2_BANDS.index(NDVI_RED_BAND)
    b08_idx = S2_BANDS.index(NDVI_NIR_BAND)
    data[:, 16] = compute_ndvi(s2_vals[:, b04_idx], s2_vals[:, b08_idx])
    ndvi_ok = s2_valid[:, b04_idx] & s2_valid[:, b08_idx]
    mask[:, 16] = (~ndvi_ok).astype(np.float32)

    # ── Каналы 0-1: Sentinel-1 (VV, VH) ──
    if s1_df is not None:
        field_s1 = s1_df[s1_df["row_id"] == row_id].sort_values("date")
        if not field_s1.empty:
            s1_dates = field_s1["date"].values
            for t, s2_date in enumerate(dates):
                diffs = np.abs(s1_dates - s2_date)
                best = np.argmin(diffs)
                if diffs[best] <= np.timedelta64(7, "D"):
                    for j, band in enumerate(S1_BANDS):
                        val = field_s1.iloc[best].get(band, np.nan)
                        if not np.isnan(val):
                            data[t, 0 + j] = normalize_s1(val)
                            mask[t, 0 + j] = 0.0

    # ── Каналы 12-13: ERA5 (t2m, tp) ──
    if era5_df is not None and not era5_df.empty:
        era5_dates = era5_df["date"].values
        for t, s2_date in enumerate(dates):
            diffs = np.abs(era5_dates - s2_date)
            best = np.argmin(diffs)
            if diffs[best] <= np.timedelta64(1, "D"):
                row = era5_df.iloc[best]
                if "t2m" in row and not np.isnan(row["t2m"]):
                    data[t, 12] = (row["t2m"] - 273.15) / 30.0
                    mask[t, 12] = 0.0
                if "tp" in row and not np.isnan(row["tp"]):
                    data[t, 13] = row["tp"] * 1000.0
                    mask[t, 13] = 0.0

    # ── Каналы 14-15: SRTM (статические) ──
    if srtm_row is not None:
        elev = srtm_row.get("elevation", np.nan)
        slope = srtm_row.get("slope", np.nan)
        if not np.isnan(elev):
            data[:, 14] = elev / 1000.0
            mask[:, 14] = 0.0
        if not np.isnan(slope):
            data[:, 15] = slope / 45.0
            mask[:, 15] = 0.0

    # Месяцы (0-11)
    months = np.array([pd.Timestamp(d).month - 1 for d in dates], dtype=np.int32)

    return data, mask, months


# ── Основная сборка ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Сборка датасета PRESTO из S2 + S1 + ERA5 + SRTM"
    )
    parser.add_argument("--year", type=int, default=PILOT_YEAR)
    parser.add_argument("--tile", type=str, default=PILOT_TILE)
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Сборка датасета PRESTO: {args.year}, тайл {args.tile}")
    print(f"{'='*60}")

    # 1. Загрузить все источники
    print(f"\nЗагрузка данных:")
    s2_df = load_s2(args.year, args.tile)
    if s2_df is None:
        sys.exit("Нет S2 данных — невозможно собрать датасет.\n"
                 "Сначала запустите extract_s2_timeseries.py")

    s1_df = load_s1(args.year, args.tile)
    srtm_df = load_srtm(args.year, args.tile)

    era5_available = any(
        (ERA5_DIR / f"{args.year}_{r}.nc").exists() for r in REGIONS
    )
    if not era5_available:
        print(f"  Нет ERA5 данных — каналы t2m/tp будут замаскированы")

    # 2. Уникальные полигоны и мета-данные
    row_ids = s2_df["row_id"].unique()
    print(f"\nПолигонов: {len(row_ids)}")

    meta = (s2_df.drop_duplicates("row_id")
            [["row_id", "culture_id", "lat", "lon"]].set_index("row_id"))

    # 3. Сборка каждого образца
    all_data, all_mask, all_months = [], [], []
    all_latlons, all_labels, all_rids = [], [], []
    skipped = 0

    for i, rid in enumerate(row_ids):
        if (i + 1) % 100 == 0 or (i + 1) == len(row_ids):
            print(f"\r  Сборка: {i+1}/{len(row_ids)} "
                  f"({100*(i+1)/len(row_ids):.0f}%)", end="", flush=True)

        m = meta.loc[rid]
        lat, lon = float(m["lat"]), float(m["lon"])

        # ERA5 для координат этого поля
        era5_df = load_era5_for_point(args.year, lat, lon) if era5_available else None

        # SRTM для этого поля
        srtm_row = None
        if srtm_df is not None:
            match = srtm_df[srtm_df["row_id"] == rid]
            if not match.empty:
                srtm_row = match.iloc[0]

        result = build_field_sample(rid, s2_df, s1_df, era5_df, srtm_row)
        if result is None:
            skipped += 1
            continue

        d, msk, mon = resample_to_fixed_length(*result)
        all_data.append(d)
        all_mask.append(msk)
        all_months.append(mon)
        all_latlons.append([lat, lon])
        all_labels.append(int(m["culture_id"]))
        all_rids.append(rid)

    print()

    if not all_data:
        sys.exit("Не удалось собрать ни одного образца")

    # 4. Стек в numpy
    X = np.stack(all_data)           # [N, 24, 17]
    M = np.stack(all_mask)           # [N, 24, 17]
    months = np.stack(all_months)    # [N, 24]
    latlons = np.array(all_latlons)  # [N, 2]
    labels = np.array(all_labels)    # [N]
    rids = np.array(all_rids)        # [N]

    # Dynamic World: у нас данных нет → заполняем sentinel'ом 9 (missing).
    # PRESTO различает «есть DW-класс 0-8» и «нет данных = 9».
    N, T = X.shape[0], X.shape[1]
    dynamic_world = np.full((N, T), PRESTO_DW_MISSING, dtype=np.int64)

    # 5. Статистика
    print(f"\nДатасет собран:")
    print(f"  x:              {X.shape}  (полей x timesteps x каналов)")
    print(f"  mask:           {M.shape}")
    print(f"  dynamic_world:  {dynamic_world.shape}  (всё = {PRESTO_DW_MISSING}, пропуск)")
    print(f"  months:         {months.shape}")
    print(f"  latlons:        {latlons.shape}")
    print(f"  labels:         {labels.shape}")
    print(f"  Пропущено: {skipped} полей")

    # Заполненность каналов
    fill = 1.0 - M.mean(axis=(0, 1))
    print(f"\nЗаполненность каналов:")
    for idx, name in enumerate(PRESTO_CHANNEL_ORDER):
        bar = "#" * int(fill[idx] * 30)
        print(f"  {name:>15s}: {fill[idx]*100:5.1f}%  |{bar}")

    # Распределение культур
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nКультуры:")
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {ID_TO_SHORT.get(cid, f'id={cid}'):>15s}: "
              f"{cnt:4d} ({100*cnt/len(labels):5.1f}%)")

    # 6. Сохранить
    out_path = DATASET_DIR / f"{args.year}_{args.tile}.npz"
    np.savez_compressed(
        out_path,
        x=X, mask=M, dynamic_world=dynamic_world,
        months=months, latlons=latlons,
        labels=labels, row_ids=rids,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nСохранено: {out_path} ({size_mb:.1f} МБ)")

    print(f"\n{'='*60}")
    print(f"Следующий шаг — получить эмбеддинги PRESTO:")
    print(f"  uv run python crop_classification/run_presto_embed.py "
          f"--year {args.year} --tile {args.tile}")


if __name__ == "__main__":
    main()
