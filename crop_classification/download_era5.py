"""
Шаг 3. Скачивание климатических данных ERA5 для региона и года.

ERA5 — глобальный реанализ погоды от ECMWF (Европейский центр прогнозов):
  - Разрешение: ~31 км (0.25° × 0.25°)
  - Покрытие: весь мир, 1979 — настоящее время
  - Доступ: CDS API (бесплатно, нужна регистрация)

PRESTO использует 2 климатические переменные:
  t2m (2m_temperature)      — температура воздуха на 2 м (Кельвины)
                               Зачем: определяет фенофазы — при 5°C всходы
                               озимых, при 25°C цветение подсолнечника.
                               Одинаковый NDVI при разной температуре = разные культуры.

  tp (total_precipitation)  — суммарные осадки (м/сутки)
                               Зачем: водный баланс, рост биомассы.
                               Засуха в июле → пшеница желтеет раньше.

Разрешение 31 км — грубое, но достаточно для климатического контекста:
температура и осадки не меняются на масштабе 30 км. Для каждого полигона
берём значение ближайшего узла сетки ERA5 (nearest neighbor).

Результат: data/era5/<year>_<region>.nc  (NetCDF файл)
  Переменные: t2m (K), tp (м/сут)
  Размерности: time × latitude × longitude
  Размер файла: ~5-20 МБ на регион×год

Предусловие (одноразовая настройка):
  1. Зарегистрироваться: https://cds.climate.copernicus.eu/
  2. Принять лицензию на ERA5: в каталоге найти "ERA5 hourly data on single levels"
     → нажать "Download" → принять Terms of Use
  3. Получить API-ключ: Profile → API Key (или Personal Access Token)
  4. Создать файл ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api
       key: <ваш-UID>:<ваш-API-KEY>
  5. Установить пакеты:
       pip install cdsapi xarray netcdf4

Запуск:
  # Пилот — один регион, один год (~5-10 мин ожидания очереди CDS)
  uv run python crop_classification/download_era5.py --year 2021 --region south

  # Все регионы за все годы
  uv run python crop_classification/download_era5.py --all

  # Просмотреть содержимое уже скачанного файла
  uv run python crop_classification/download_era5.py --inspect data/era5/2021_south.nc
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

from config import (
    ALL_REGIONS,
    ALL_YEARS,
    ERA5_DIR,
    ERA5_VARIABLES_CDS,
    PILOT_REGION,
    PILOT_YEAR,
    REGIONS,
    SEASON_END,
    SEASON_START,
)


def check_cdsapi():
    """Проверить, что cdsapi установлен и настроен."""
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        sys.exit(
            "Не установлен cdsapi.\n\n"
            "Установите:\n"
            "  pip install cdsapi\n\n"
            "Затем настройте доступ:\n"
            "  1. Регистрация: https://cds.climate.copernicus.eu/\n"
            "  2. API-ключ: Profile → API Key\n"
            "  3. Создать ~/.cdsapirc:\n"
            "       url: https://cds.climate.copernicus.eu/api\n"
            "       key: <UID>:<API-KEY>"
        )

    cdsrc = Path.home() / ".cdsapirc"
    if not cdsrc.exists():
        sys.exit(
            f"Не найден файл {cdsrc}\n\n"
            "Создайте его:\n"
            "  echo 'url: https://cds.climate.copernicus.eu/api' > ~/.cdsapirc\n"
            "  echo 'key: <UID>:<API-KEY>' >> ~/.cdsapirc\n\n"
            "API-ключ: https://cds.climate.copernicus.eu/ → Profile → API Key"
        )


def download_era5(year: int, region: str) -> Path:
    """Скачать ERA5 данные для региона и года через CDS API.

    Алгоритм:
      1. Проверяем, не скачан ли файл ранее (skip если есть)
      2. Формируем запрос: переменные, даты сезона, bbox региона
      3. Отправляем в CDS (запрос встаёт в очередь — обычно 1-10 мин)
      4. CDS обрабатывает и отдаёт NetCDF файл
      5. Проверяем содержимое

    Время скачивания:
      - Очередь CDS: 1-10 мин (зависит от загрузки сервера)
      - Сам файл: ~5-20 МБ, скачивается за секунды
    """
    import cdsapi

    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    out_path = ERA5_DIR / f"{year}_{region}.nc"
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Уже скачано: {out_path} ({size_mb:.1f} МБ) — пропуск")
        return out_path

    reg = REGIONS[region]
    area = reg["era5_area"]  # [N, W, S, E] — формат CDS API

    # Месяцы вегетационного сезона: SEASON_START="03-01" → 3, SEASON_END="10-31" → 10
    start_month = int(SEASON_START.split("-")[0])
    end_month = int(SEASON_END.split("-")[0])
    months = [f"{m:02d}" for m in range(start_month, end_month + 1)]

    print(f"  Запрос к CDS API:")
    print(f"    Год:         {year}")
    print(f"    Регион:      {region} — {reg['name']}")
    print(f"    Переменные:  {ERA5_VARIABLES_CDS}")
    print(f"    Месяцы:      {months[0]}–{months[-1]} ({len(months)} мес.)")
    print(f"    Bbox [N,W,S,E]: {area}")
    print(f"    Время суток: 12:00 (полдень)")
    print(f"  Отправляю запрос... (ожидание очереди CDS: обычно 1-10 мин)")

    c = cdsapi.Client()
    # Скачиваем во временный файл — CDS теперь может вернуть ZIP-архив,
    # когда переменные имеют разный stepType (t2m=instant, tp=accum).
    tmp_path = out_path.with_suffix(".download")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES_CDS,
            "year": str(year),
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": "12:00",
            "area": area,              # [N, W, S, E]
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(tmp_path),
    )

    _post_process_download(tmp_path, out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Скачано: {out_path} ({size_mb:.1f} МБ)")
    return out_path


def _post_process_download(tmp_path: Path, out_path: Path):
    """Распаковать ZIP (если это ZIP) и смёржить .nc файлы в один.

    CDS может вернуть либо:
      - Один NetCDF (всё ок, просто переименовываем)
      - ZIP с несколькими .nc внутри (например, data_stream-oper_stepType-instant.nc
        для t2m и data_stream-oper_stepType-accum.nc для tp) — распаковываем и мёржим.
    """
    if not zipfile.is_zipfile(tmp_path):
        # Обычный NetCDF — просто переименовываем
        tmp_path.rename(out_path)
        return

    print(f"  CDS вернул ZIP — распаковываю и объединяю...")
    import xarray as xr

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(tmpdir)
        nc_files = sorted(tmpdir.glob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"В ZIP нет .nc файлов: {tmp_path}")

        print(f"    Найдено {len(nc_files)} NetCDF: {[f.name for f in nc_files]}")
        datasets = [xr.open_dataset(f) for f in nc_files]
        merged = xr.merge(datasets, compat="override")
        merged.to_netcdf(out_path)
        for ds in datasets:
            ds.close()

    tmp_path.unlink()


def inspect_era5(path: Path):
    """Показать содержимое скачанного NetCDF файла.

    Выводит:
      - Список переменных и их размерности
      - Диапазоны значений (для проверки корректности)
      - Временной диапазон и пространственный охват
    """
    try:
        import xarray as xr
    except ImportError:
        print("  (для просмотра установите: pip install xarray netcdf4)")
        return

    ds = xr.open_dataset(path)

    print(f"\n  Содержимое {path.name}:")
    print(f"    Переменные:  {list(ds.data_vars)}")
    print(f"    Размерности: {dict(ds.sizes)}")

    for var in ds.data_vars:
        arr = ds[var]
        vmin, vmax = float(arr.min()), float(arr.max())
        print(f"    {var}: shape={arr.shape}, min={vmin:.2f}, max={vmax:.2f}", end="")
        # Подсказка по единицам
        if var in ("t2m", "2t"):
            print(f"  (в Кельвинах; {vmin - 273.15:.1f}..{vmax - 273.15:.1f} °C)")
        elif var in ("tp",):
            print(f"  (м/сут; {vmin*1000:.2f}..{vmax*1000:.2f} мм/сут)")
        else:
            print()

    # CDS переименовал координату времени: в новом API — valid_time, раньше — time.
    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    time_vals = ds[time_name].values
    print(f"    Период:   {str(time_vals[0])[:10]} — {str(time_vals[-1])[:10]}"
          f" ({len(time_vals)} дат)")
    print(f"    Широта:   {float(ds.latitude.min()):.2f}° — "
          f"{float(ds.latitude.max()):.2f}°")
    print(f"    Долгота:  {float(ds.longitude.min()):.2f}° — "
          f"{float(ds.longitude.max()):.2f}°")

    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Скачивание климатических данных ERA5 через CDS API"
    )
    parser.add_argument("--year", type=int, default=None,
                        help=f"Год (по умолчанию пилот: {PILOT_YEAR})")
    parser.add_argument("--region", type=str, default=None,
                        choices=ALL_REGIONS,
                        help=f"Регион (по умолчанию пилот: {PILOT_REGION})")
    parser.add_argument("--all", action="store_true",
                        help="Скачать все годы × все регионы")
    parser.add_argument("--inspect", type=str, default=None,
                        metavar="FILE.nc",
                        help="Только показать содержимое существующего файла")
    args = parser.parse_args()

    # Режим просмотра
    if args.inspect:
        p = Path(args.inspect)
        if not p.exists():
            sys.exit(f"Файл не найден: {p}")
        inspect_era5(p)
        return

    # Проверяем наличие cdsapi и ~/.cdsapirc
    check_cdsapi()

    print(f"\n{'='*60}")
    print(f"Скачивание ERA5 климатических данных")
    print(f"{'='*60}")

    if args.all:
        for year in ALL_YEARS:
            for region in ALL_REGIONS:
                print(f"\n--- {year}, {region} ---")
                path = download_era5(year, region)
                inspect_era5(path)
    else:
        year = args.year or PILOT_YEAR
        region = args.region or PILOT_REGION
        print(f"\n--- {year}, {region} ---")
        path = download_era5(year, region)
        inspect_era5(path)

    print(f"\n{'='*60}")
    print(f"Готово! Файлы в {ERA5_DIR}/")
    print(f"\nСледующий шаг:")
    print(f"  uv run python crop_classification/download_srtm.py --region {args.region or PILOT_REGION}")


if __name__ == "__main__":
    main()
