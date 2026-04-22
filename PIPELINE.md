# Полная инструкция: как запускать проект

## Структура проекта (что создано)

```
crop_classification/
├── __init__.py
├── config.py                   ← конфигурация (пути, каналы, культуры)
│
│  ── ФАЗА 1: ДАННЫЕ ──
├── extract_s2_timeseries.py    ← Sentinel-2 временные ряды (10 каналов)
├── extract_s1_timeseries.py    ← Sentinel-1 временные ряды (VV, VH)
├── download_era5.py            ← ERA5 климат (температура, осадки)
├── download_srtm.py            ← SRTM рельеф (высота, уклон)
├── build_dataset.py            ← объединение всех данных → тензоры PRESTO
│
│  ── ФАЗА 2: МОДЕЛЬ ──
├── run_presto_embed.py         ← PRESTO → эмбеддинги 128-d
│
│  ── ФАЗА 3: ЭКСПЕРИМЕНТЫ (от простого к сложному) ──
├── experiment_zeroshot.py      ← Эксп. 1: kNN без обучения
├── experiment_linear.py        ← Эксп. 2: линейный классификатор
├── experiment_finetune.py      ← Эксп. 3: полное дообучение
├── experiment_maml.py          ← Эксп. 4: MAML/ANIL (основной метод)
│
│  ── ФАЗА 4: ОЦЕНКА И ПРИМЕНЕНИЕ ──
├── evaluate.py                 ← сравнение всех экспериментов
└── predict_ftw_fields.py       ← inference на полях FTW
```

---

## Шаг 0. Установка зависимостей

```bash
cd ~/Desktop/Sbertech/Diplom/fields

# Добавить зависимости в pyproject.toml (вручную):
#   "presto-ml", "learn2learn", "cdsapi", "xarray", "netcdf4",
#   "scikit-learn", "matplotlib", "seaborn", "pyarrow"

# Затем:
uv sync

# Или поставить отдельно:
uv pip install presto-ml learn2learn cdsapi xarray netcdf4 scikit-learn matplotlib seaborn pyarrow
```

Проверка:

```bash
uv run python -c "from presto import Presto; m = Presto.load_pretrained(); print('PRESTO OK:', m)"
uv run python -c "import learn2learn; print('learn2learn OK')"
uv run python -c "import planetary_computer; print('planetary_computer OK')"
```

---

## Шаг 1. Настройка ERA5 (одноразово)

```bash
# 1. Зарегистрироваться: https://cds.climate.copernicus.eu/
# 2. В профиле → API Key → скопировать
# 3. Создать файл:
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: ВАШ_UID:ВАШ_API_KEY
EOF

# 4. Проверить:
uv run python -c "import cdsapi; c = cdsapi.Client(); print('CDS API OK')"
```

---

## Шаг 2. Скачивание данных (ФАЗА 1)

Все 4 команды можно запускать параллельно — они не зависят друг от друга.

Стратегия: сначала **пилот** (3 сцены локально на MacBook, ~10-20 мин на каждый скрипт), проверяем что весь пайплайн до конца работает, потом **полный прогон** на GPU.

### 2.1. Пилотный прогон (локально, небольшие данные)

```bash
cd ~/Desktop/Sbertech/Diplom/fields

# ── 2a. Sentinel-2 временные ряды (3 сцены, ~10-20 мин) ──
uv run python crop_classification/extract_s2_timeseries.py --year 2021 --tile 38TLR --max-scenes 3

# ── 2b. Sentinel-1 временные ряды (3 сцены, ~5-10 мин) ──
uv run python crop_classification/extract_s1_timeseries.py --year 2021 --tile 38TLR --max-scenes 3

# ── 2c. ERA5 климат (5-15 мин, зависит от очереди CDS) ──
# ERA5 не использует --max-scenes (скачивается сразу весь сезон — небольшой файл)
uv run python crop_classification/download_era5.py --year 2021 --region south

# ── 2d. SRTM рельеф (5-10 мин) ──
# SRTM статический, скачивается один раз
uv run python crop_classification/download_srtm.py --year 2021 --tile 38TLR
```

**Важно**: на 3 сценах качество классификации будет низким (слишком мало временных точек).
Это нужно только чтобы проверить что весь пайплайн (S2→S1→ERA5→SRTM→build_dataset→PRESTO→эксперименты) работает end-to-end без ошибок.

### 2.2. Полный прогон (GPU, все сцены)

```bash
# Все сцены за сезон (~30-40 сцен S2, ~1-3 часа)
uv run python crop_classification/extract_s2_timeseries.py --year 2021 --tile 38TLR
uv run python crop_classification/extract_s1_timeseries.py --year 2021 --tile 38TLR
uv run python crop_classification/download_era5.py --year 2021 --region south
uv run python crop_classification/download_srtm.py --year 2021 --tile 38TLR
```

Что получится:

```
data/
├── timeseries_s2/2021_38TLR.parquet    ← S2 (~1-5 МБ)
├── timeseries_s1/2021_38TLR.parquet    ← S1 (~0.5-2 МБ)
├── era5/2021_south.nc                  ← ERA5 (~5-20 МБ)
└── srtm/2021_38TLR.parquet             ← SRTM (~50 КБ)
```

---

## Шаг 3. Сборка датасета PRESTO

```bash
# Объединяет S2+S1+ERA5+SRTM+NDVI → тензоры [N, 24, 18]
uv run python crop_classification/build_dataset.py --year 2021 --tile 38TLR
```

Что получится:

```
data/presto_dataset/2021_38TLR.npz     ← ~5-20 МБ
  x:       [600, 24, 18]   — 600 полей × 24 timesteps × 18 каналов
  mask:    [600, 24, 18]   — маска пропусков
  months:  [600, 24]       — месяц каждого timestep
  latlons: [600, 2]        — координаты
  labels:  [600]           — ID культуры
```

---

## Шаг 4. Получение эмбеддингов PRESTO

```bash
# Прогнать через предобученный PRESTO → эмбеддинги 128-d
uv run python crop_classification/run_presto_embed.py --year 2021 --tile 38TLR
```

Что получится:

```
data/presto_dataset/2021_38TLR_embeddings.npz   ← ~1 МБ
  embeddings: [600, 128]
  labels:     [600]
```

---

## Шаг 5. Эксперименты (ФАЗА 3)

Запускай последовательно — каждый следующий сложнее предыдущего:

```bash
# ── Эксп. 1: Zero-shot kNN (без обучения, ~10 сек) ──
# Проверяем: что PRESTO знает о культурах РФ без обучения?
uv run python crop_classification/experiment_zeroshot.py --year 2021 --tile 38TLR

# ── Эксп. 2: Linear probe (линейный классификатор, ~30 сек) ──
# Проверяем: достаточно ли линейного разделения?
uv run python crop_classification/experiment_linear.py --year 2021 --tile 38TLR

# ── Эксп. 3: Fine-tune (полное дообучение, ~5-30 мин) ──
# Проверяем: помогает ли адаптация энкодера?
uv run python crop_classification/experiment_finetune.py --year 2021 --tile 38TLR

# ── Эксп. 4: MAML/ANIL (мета-обучение, ~10-30 мин) ──
# Основной метод дипломной работы
uv run python crop_classification/experiment_maml.py --year 2021 --tile 38TLR

# Варианты few-shot (1-shot, 5-shot, 10-shot):
uv run python crop_classification/experiment_maml.py --year 2021 --tile 38TLR --k-shot 1
uv run python crop_classification/experiment_maml.py --year 2021 --tile 38TLR --k-shot 10
```

---

## Шаг 6. Сравнение результатов

```bash
uv run python crop_classification/evaluate.py --year 2021 --tile 38TLR
```

Пример вывода:

```
  Эксперимент              Accuracy   F1 macro   F1 weighted
  ─────────────────────────────────────────────────────────
  1. Zero-shot kNN          0.3200     0.2800      0.3100
  2. Linear probe           0.5500     0.4900      0.5300
  3. Fine-tune              0.6500     0.6000      0.6400
  4. MAML/ANIL (5-shot)     0.7000     0.6500      0.6900
  ─────────────────────────────────────────────────────────
  Лучший по F1 macro: 4. MAML/ANIL (0.6500)
```

---

## Шаг 7. Применение на полях FTW (inference)

```bash
# Используем лучший метод (допустим MAML):
uv run python crop_classification/predict_ftw_fields.py \
  --gpkg outputs/ftw/2021_tile_38TLR/*_ftw_boundaries.gpkg \
  --year 2021 --tile 38TLR --method maml

# Или без дообучения (kNN):
uv run python crop_classification/predict_ftw_fields.py \
  --gpkg outputs/ftw/2021_tile_38TLR/*_ftw_boundaries.gpkg \
  --year 2021 --tile 38TLR --method knn
```

Результат: `outputs/crop_cls/classified_fields_2021_38TLR.gpkg` — открываешь в QGIS.

---

## Масштабирование (все годы, все регионы)

После успешного пилота на одном тайле:

```bash
# Все тайлы за все годы
for year in 2016 2017 2018 2019 2020 2021; do
  uv run python crop_classification/extract_s2_timeseries.py --year $year
  uv run python crop_classification/extract_s1_timeseries.py --year $year
  uv run python crop_classification/download_era5.py --year $year --all
done
```

---

## Как тестировать / отлаживать

```bash
# 1. Быстрая проверка (3 сцены вместо 20):
uv run python crop_classification/extract_s2_timeseries.py \
  --year 2021 --tile 38TLR --max-scenes 3

# 2. Проверить конфигурацию:
uv run python -c "from crop_classification.config import *; print(f'Культур: {NUM_CLASSES}, каналов: {PRESTO_NUM_CHANNELS}')"

# 3. Посмотреть содержимое датасета:
uv run python -c "
import numpy as np
d = np.load('data/presto_dataset/2021_38TLR.npz')
for k in d: print(f'{k}: {d[k].shape} {d[k].dtype}')
"

# 4. Посмотреть ERA5:
uv run python crop_classification/download_era5.py --inspect data/era5/2021_south.nc

# 5. Проверить эмбеддинги:
uv run python -c "
import numpy as np
d = np.load('data/presto_dataset/2021_38TLR_embeddings.npz')
print(f'Эмбеддинги: {d[\"embeddings\"].shape}')
print(f'Классов: {len(np.unique(d[\"labels\"]))}')
"
```

---

## Схема зависимостей между скриптами

```
extract_s2_timeseries.py ──┐
extract_s1_timeseries.py ──┤
download_era5.py ──────────┼──→ build_dataset.py ──→ run_presto_embed.py
download_srtm.py ──────────┘          │                      │
                                      │                      ├→ experiment_zeroshot.py ─┐
                                      │                      ├→ experiment_linear.py ───┤
                                      ├→ experiment_finetune.py ────────────────────────┤
                                      │                      └→ experiment_maml.py ─────┤
                                      │                                                 │
                                      │                                          evaluate.py
                                      │                                                 │
                                      └─────────────────────→ predict_ftw_fields.py ←───┘
```

Стрелки = зависимости. Скрипты на одном уровне можно запускать параллельно.
