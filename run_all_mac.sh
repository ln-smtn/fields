#!/bin/bash
set -e
mkdir -p logs

# Годы можно переопределить переменной YEARS:
#   YEARS="2019 2020 2021" ./run_all_mac.sh
YEARS="${YEARS:-2016 2017 2018 2019 2020 2021}"
echo "Будут обработаны годы: $YEARS"

# === Этап 1: Скачивание для всех лет (тайлы определяются автоматически) ===
for year in $YEARS; do
  echo "=== $(date '+%H:%M') Скачивание $year ==="

  echo "  S2..."
  uv run python crop_classification/data_collection/extract_s2_timeseries.py --year $year > logs/s2_$year.log 2>&1

  echo "  S1..."
  uv run python crop_classification/data_collection/extract_s1_timeseries.py --year $year > logs/s1_$year.log 2>&1

  echo "  SRTM..."
  uv run python crop_classification/data_collection/download_srtm.py --year $year > logs/srtm_$year.log 2>&1
done

# === Этап 2: ERA5 для всех регионов и лет (одной командой) ===
echo "=== $(date '+%H:%M') ERA5 для всех (год × регион) ==="
uv run python crop_classification/data_collection/download_era5.py --all > logs/era5_all.log 2>&1

# === Этап 3: Сборка тензоров + эмбеддинги PRESTO для каждого (год, тайл) ===
echo "=== $(date '+%H:%M') Сборка датасетов + эмбеддинги ==="
for f in data/timeseries_s2/*.parquet; do
  name=$(basename "$f" .parquet)        # например "2021_38TLR"
  year=${name%_*}                       # "2021"
  tile=${name#*_}                       # "38TLR"
  echo "  build+embed $year $tile"
  uv run python crop_classification/data_collection/build_dataset.py --year $year --tile $tile >> logs/build.log 2>&1
  uv run python crop_classification/data_collection/run_presto_embed.py --year $year --tile $tile >> logs/embed.log 2>&1
done

echo "=== $(date '+%H:%M') ГОТОВО ==="
