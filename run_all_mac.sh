#!/bin/bash
set -e
mkdir -p logs

# Годы можно переопределить переменной YEARS:
#   YEARS="2019 2020 2021" ./run_all_mac.sh
YEARS="${YEARS:-2016 2017 2018 2019 2020 2021}"

# Этапы можно отключить переменными окружения (true/false):
#   STAGE2=false STAGE3=false ./run_all_mac.sh   ← только Этап 1 (S2/S1/SRTM)
STAGE1="${STAGE1:-true}"      # S2 + S1 + SRTM по годам
STAGE2="${STAGE2:-true}"      # ERA5 для всех регионов и лет
STAGE3="${STAGE3:-true}"      # build_dataset + run_presto_embed

echo "=== Конфиг ==="
echo "  YEARS:  $YEARS"
echo "  STAGE1 (S2/S1/SRTM): $STAGE1"
echo "  STAGE2 (ERA5):       $STAGE2"
echo "  STAGE3 (PRESTO):     $STAGE3"
echo

# === Этап 1: Скачивание для всех лет (тайлы определяются автоматически) ===
if [ "$STAGE1" = "true" ]; then
  for year in $YEARS; do
    echo "=== $(date '+%H:%M') Скачивание $year ==="

    echo "  S2..."
    uv run python crop_classification/data_collection/extract_s2_timeseries.py --year $year > logs/s2_$year.log 2>&1

    echo "  S1..."
    uv run python crop_classification/data_collection/extract_s1_timeseries.py --year $year > logs/s1_$year.log 2>&1

    echo "  SRTM..."
    uv run python crop_classification/data_collection/download_srtm.py --year $year > logs/srtm_$year.log 2>&1
  done
fi

# === Этап 2: ERA5 для всех регионов и лет (одной командой) ===
if [ "$STAGE2" = "true" ]; then
  echo "=== $(date '+%H:%M') ERA5 для всех (год × регион) ==="
  uv run python crop_classification/data_collection/download_era5.py --all > logs/era5_all.log 2>&1
fi

# === Этап 3: Сборка тензоров + эмбеддинги PRESTO для каждого (год, тайл) ===
if [ "$STAGE3" = "true" ]; then
  echo "=== $(date '+%H:%M') Сборка датасетов + эмбеддинги ==="
  for f in data/timeseries_s2/*.parquet; do
    [ -e "$f" ] || continue                 # на случай если папка пуста
    name=$(basename "$f" .parquet)          # например "2021_38TLR"
    year=${name%_*}                         # "2021"
    tile=${name#*_}                         # "38TLR"
    echo "  build+embed $year $tile"
    uv run python crop_classification/data_collection/build_dataset.py --year $year --tile $tile >> logs/build.log 2>&1
    uv run python crop_classification/data_collection/run_presto_embed.py --year $year --tile $tile >> logs/embed.log 2>&1
  done
fi

echo "=== $(date '+%H:%M') ГОТОВО ==="
