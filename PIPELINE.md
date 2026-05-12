# Production pipeline: классификация культур через PRESTO

## Что делает этот пайплайн

На вход — спутниковый тайл Sentinel-2 (например `38TLR`) и год.
На выход — карта полей с подписанной культурой и уверенностью.

```
Снимок S2 → FTW (контуры полей) → PRESTO (эмбеддинги) → Классификатор → Карта культур
```

## Структура проекта

```
crop_classification/
├── __init__.py
├── config.py                            ← конфигурация
│
├── data_collection/                     ── СБОР ДАННЫХ ──
│   ├── extract_s2_timeseries.py        Sentinel-2 (10 каналов)
│   ├── extract_s1_timeseries.py        Sentinel-1 (VV, VH)
│   ├── download_era5.py                ERA5 климат (t2m, tp)
│   ├── download_srtm.py                SRTM рельеф (elev, slope)
│   ├── build_dataset.py                S2+S1+ERA5+SRTM+NDVI → тензор [N,24,17]
│   └── run_presto_embed.py             PRESTO → эмбеддинги [N,128]
│
├── training/                            ── ОБУЧЕНИЕ ──
│   ├── build_master_dataset.py         объединение всех тайлов в один master
│   ├── train_linear.py                 Linear probe
│   ├── train_finetune.py               Fine-tune PRESTO (частичная разморозка)
│   └── train_maml.py                   MAML/ANIL для cross-region адаптации
│
└── inference/                           ── ПРИМЕНЕНИЕ ──
    ├── adapt_to_region.py              адаптация MAML к новому региону
    └── predict_tile.py                 единый inference на тайле
```

## Регионы в обучающей выборке

| Регион | Тайлы MGRS | Полигонов | Доля | Годы доступных данных |
|--------|-----------|-----------|------|----------------------|
| **South** (Краснодар, Ставрополье, Ростов) | 38TLR, 37TGM, 38TLS, 38TLQ | 11 863 | 45% | 2017-2021 |
| **Chernozem** (Курская, Белгородская, Воронежская обл.) | 36UXV, 37UDB, 37UEB | 10 923 | 41% | 2016-2019 |
| **Volga** (Среднее Поволжье) | 38UQB, 38UPC, 38UQC | 1 951 | 7% | 2016-2021 |
| _(Primorye)_ — **отложен** для cross-region теста | _53TLR, 53TPR_ | _1 656_ | _6%_ | _2016-2019_ |

Модель устойчива к **небольшим географическим сдвигам**: соседние тайлы в радиусе ~200 км от обучавшихся дают качественные предсказания без переобучения.

---

## Три сценария применения

### Сценарий A: классификация в обучавшемся регионе (основной)

**Когда применять**: новый тайл или год **в одном из 3 обученных регионов** (South, Chernozem, Volga). Например, обучала на 2017-2021, а сейчас 2025.

```bash
python crop_classification/inference/predict_tile.py \
  --tile 38TLR --year 2025 \
  --model models/finetuned_model.pt \
  --polygons outputs/ftw/2025_38TLR_boundaries.gpkg
```

**Что произойдёт**:
1. Скрипт берёт полигоны из готового FTW GeoPackage (этап 1 пайплайна);
2. Для каждого полигона скачивает временные ряды S1+S2+ERA5+SRTM за 2025 год;
3. Прогоняет через PRESTO → эмбеддинги [N, 128];
4. `finetuned_model.pt` классифицирует → культура + уверенность;
5. OOD-фильтр определяет, является ли это полем (площадь, уверенность, расстояние до обучения);
6. Сохраняет `data/inference_cache/2025_38TLR/classified.gpkg`.

**Что в `classified.gpkg`** — таблица с колонками:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `geometry` | Polygon | геометрия полигона (EPSG:4326) |
| `predicted_culture` | str | название культуры (или "не поле" / "артефакт FTW") |
| `predicted_culture_id` | int | ID культуры (0-14) |
| `confidence` | float [0,1] | уверенность модели (probs.max()) |
| `ood_score` | float | расстояние до центра ближайшего класса (для OOD-детекции) |
| `area_ha` | float | площадь поля в гектарах |
| `is_field` | bool | True если поле, False если артефакт/не-поле |
| `centroid_lat` | float | широта центра |
| `centroid_lon` | float | долгота центра |
| `method` | str | "linear" / "finetune" / "maml" / "maml_adapted" |
| `year` | int | год снимков |
| `tile` | str | MGRS тайл |

Этот `.gpkg` открывается в QGIS, ArcGIS, веб-картах. Это **главный продукт** пайплайна.

---

### Сценарий B: классификация в новом (отложенном) регионе через адаптацию MAML

**Когда применять**: новый регион, **не обучавшийся** (например, Приморье или Алтай). У тебя есть **5-25 размеченных полей** этого региона.

**Шаг 1**: одноразовая адаптация модели к региону

```bash
python crop_classification/inference/adapt_to_region.py \
  --region primorye \
  --support data/primorye_25_fields.csv \
  --base-model models/maml_base.pt \
  --output models/adapted/maml_primorye.pt
```

`support` — CSV с 25 размеченными полями (5 культур × 5 примеров). Модель делает 5 шагов градиентного спуска и сохраняется в `models/adapted/maml_primorye.pt`.

**Шаг 2**: предсказание (так же, как сценарий A, но с адаптированной моделью)

```bash
python crop_classification/inference/predict_tile.py \
  --tile 53TLR --year 2025 \
  --model models/adapted/maml_primorye.pt \
  --polygons outputs/ftw/2025_53TLR_boundaries.gpkg
```

**Когда переадаптировать**: если регион тот же, но год сильно отличается (например, засуха 2025 vs дождливый 2021) — берёшь 5 свежих размеченных полей из 2025 и заново вызываешь `adapt_to_region.py`.

---

### Сценарий C: демо в Streamlit

**Идея**: предсказания заранее посчитаны и закэшированы. Пользователь демо ничего не считает — только смотрит на готовые результаты.

```bash
# Заранее (один раз) — посчитать предсказания для демо-тайлов:
python crop_classification/inference/predict_tile.py --tile 38TLR --year 2021 --model models/finetuned_model.pt --polygons outputs/ftw/2021_38TLR_boundaries.gpkg
python crop_classification/inference/predict_tile.py --tile 53TLR --year 2021 --model models/adapted/maml_primorye.pt --polygons outputs/ftw/2021_53TLR_boundaries.gpkg
# и т.д.

# Запустить демо:
streamlit run demo_app.py
```

Пользователь в браузере:
- Выбирает регион из списка (Краснодар, Приморье и т.д.);
- Видит интерактивную карту с полигонами и культурами;
- Кликает на поле → видит культуру и уверенность;
- Фильтрует по уверенности, площади, конкретным культурам;
- Скачивает `classified.gpkg` или CSV.

`demo_app.py` пока **не написан** — будет на следующем этапе после обучения моделей.

---

## Фазы работы

### Фаза 1: сбор данных (~10-24 часа, на GPU/сервере)

Для каждой пары `(регион, год)` где есть данные:

```bash
# Sentinel-2: 10 каналов B02-B12 (без B1, B9, B10), 30-40 сцен за сезон
python crop_classification/data_collection/extract_s2_timeseries.py --year 2021 --tile 38TLR

# Sentinel-1: VV, VH, 40-80 сцен
python crop_classification/data_collection/extract_s1_timeseries.py --year 2021 --tile 38TLR

# ERA5: t2m, tp, по дням
python crop_classification/data_collection/download_era5.py --year 2021 --region south

# SRTM: elevation, slope (статика, один раз на тайл)
python crop_classification/data_collection/download_srtm.py --year 2021 --tile 38TLR
```

Параллелить можно по тайлам (несколько терминалов одновременно).

### Фаза 2: сборка тензоров и эмбеддингов (~2-4 часа на CPU)

```bash
for year_tile in <все пары>; do
  python crop_classification/data_collection/build_dataset.py --year $year --tile $tile
  python crop_classification/data_collection/run_presto_embed.py --year $year --tile $tile
done
```

Результат: `data/presto_dataset/*.npz` и `data/presto_dataset/*_embeddings.npz`.

### Фаза 3: объединение в master-датасет (~10 минут)

```bash
# С отложенным регионом для MAML cross-region теста:
python crop_classification/training/build_master_dataset.py --holdout-region primorye

# Если MAML не используется:
python crop_classification/training/build_master_dataset.py
```

Результат:
- `data/training_master/all_embeddings.npz` — для linear probe (только эмбеддинги);
- `data/training_master/all_raw.npz` — для fine-tune и MAML (сырые тензоры [N,24,17]);
- `data/training_master/split.json` — индексы train/val/test со стратификацией;
- `data/training_master/holdout_primorye_*.npz` — данные отложенного региона (для проверки).

### Фаза 4: обучение моделей (~6-9 часов на GPU)

```bash
# Linear probe — на эмбеддингах, sklearn (5 минут)
python crop_classification/training/train_linear.py

# Fine-tune — PRESTO + голова, частичная разморозка (30-60 минут на GPU)
python crop_classification/training/train_finetune.py

# MAML — на 3 регионах с эпизодами (3-5 часов на GPU)
python crop_classification/training/train_maml.py --holdout primorye
```

Результат: всё в `models/` (см. `models/README.md`).

### Фаза 5: предсказания (~30-60 минут на тайл, на маке)

```bash
# Сценарий A
python crop_classification/inference/predict_tile.py --tile 38TLR --year 2025 \
  --model models/finetuned_model.pt \
  --polygons outputs/ftw/2025_38TLR_boundaries.gpkg

# Сценарий B (после адаптации)
python crop_classification/inference/adapt_to_region.py --region primorye \
  --support data/primorye_25_fields.csv \
  --output models/adapted/maml_primorye.pt
python crop_classification/inference/predict_tile.py --tile 53TLR --year 2025 \
  --model models/adapted/maml_primorye.pt \
  --polygons outputs/ftw/2025_53TLR_boundaries.gpkg
```

Результат: `data/inference_cache/<year>_<tile>/classified.gpkg`.

### Фаза 6: демо (после всего)

```bash
uv add streamlit folium streamlit-folium
streamlit run demo_app.py
```

---

## Что хранится после обучения (артефакты)

После выполнения Фазы 4 у тебя в `models/` будут следующие файлы:

```
models/
├── README.md                            ← карточка моделей (заполнена скриптами)
├── class_mapping.json                   ← {0: "оз.пшеница", 1: ...}  <1 КБ
├── class_centers.npy                    ← средние эмбеддинги классов  ~30 КБ
├── class_std.npy                        ← глобальный разброс для OOD  <1 КБ
│
│  ── ОСНОВНЫЕ МОДЕЛИ ──
├── linear_head.pt                       ← Linear probe веса         ~50 КБ
├── linear_metrics.json                  ← метрики linear probe       <1 КБ
├── finetuned_model.pt                   ← Fine-tune модель           ~2 МБ
├── finetuned_metrics.json               ← метрики fine-tune          <1 КБ
├── maml_base.pt                         ← MAML базовая модель        ~2 МБ
├── maml_metrics.json                    ← метрики MAML обучения      <1 КБ
│
│  ── ВИЗУАЛИЗАЦИИ ──
├── confusion_matrix_linear.png          ← матрица путаницы linear
├── confusion_matrix_finetune.png        ← матрица путаницы fine-tune
├── training_curves.png                  ← кривые обучения fine-tune
│
│  ── АДАПТИРОВАННЫЕ ПОД РЕГИОН (опционально) ──
└── adapted/
    ├── maml_primorye.pt                 ← после adapt_to_region.py   ~2 МБ
    └── maml_<region>.pt                  ← для других регионов
```

**Суммарный размер**: **~10 МБ**. Это **главный артефакт обучения**, который ты:

1. **Скачиваешь** с GPU обратно на мак:
   ```bash
   scp -r user@gpu-server:~/fields/models/ ./
   ```
2. **Коммитишь** в git (10 МБ — не страшно):
   ```bash
   git add models/
   git commit -m "Production models: F1 macro linear=0.78, finetune=0.83"
   git push
   ```
3. **Используешь** для предсказаний на любых новых тайлах через `predict_tile.py`.

**Что НЕ нужно тащить с GPU обратно**:
- `data/timeseries_s2/`, `data/timeseries_s1/`, `data/era5/`, `data/srtm/` — сырые источники, тяжёлые (десятки ГБ);
- `data/presto_dataset/` — промежуточные тензоры (несколько ГБ);
- `data/training_master/` — можно скачать, если планируешь переобучать (несколько ГБ).

---

## Где хранится что — полная карта

```
fields/
├── crop_classification/                    ← КОД (в git)
│   ├── config.py
│   ├── data_collection/    (6 скриптов)
│   ├── training/           (4 скрипта)
│   └── inference/          (2 скрипта)
│
├── data/                                    ← ДАННЫЕ (.gitignore — большие)
│   ├── cultures_polygons_dataset.csv       ← исходный датасет с метками
│   ├── timeseries_s2/                       ← S2 временные ряды
│   ├── timeseries_s1/                       ← S1 временные ряды
│   ├── era5/                                ← ERA5 файлы
│   ├── srtm/                                ← SRTM файлы
│   ├── presto_dataset/                      ← собранные тензоры + эмбеддинги
│   ├── training_master/                     ← объединённый master-датасет
│   │   ├── all_embeddings.npz
│   │   ├── all_raw.npz
│   │   ├── split.json
│   │   └── holdout_primorye_*.npz
│   └── inference_cache/                     ← результаты применения
│       └── <year>_<tile>/
│           ├── classified.gpkg
│           ├── metadata.json
│           └── thumbnail.png
│
├── models/                                  ← ОБУЧЕННЫЕ МОДЕЛИ (в git, ~10 МБ)
│   ├── README.md
│   ├── class_mapping.json
│   ├── linear_head.pt
│   ├── finetuned_model.pt
│   ├── maml_base.pt
│   └── adapted/
│
├── presto_repo/                             ← PRESTO с патчами (в git)
├── PIPELINE.md                              ← этот файл
└── pyproject.toml
```

---

## Что коммитить в git, что не коммитить

**В git идёт**:
- Весь код `crop_classification/`;
- `models/` после обучения (это 10 МБ, не страшно);
- `presto_repo/` (наш форк с патчами);
- `PIPELINE.md`, `pyproject.toml`, `uv.lock`.

**В git НЕ идёт** (через `.gitignore`):
- `data/` полностью — слишком тяжёлое (40+ ГБ);
- `__pycache__/`, `.venv/`, тд.

После обучения на GPU:
```bash
git add models/
git commit -m "Production models trained: F1 macro = 0.83 finetune, 0.85 maml"
git push
```

Затем на маке:
```bash
git pull
# models/ обновлены, можно делать предсказания
```

---

## Что такое tmux и зачем

`tmux` (terminal multiplexer) — программа для **запуска долгих процессов на сервере**, которые **не оборвутся**, если у тебя пропадёт SSH-соединение.

### Проблема без tmux

Запускаешь на сервере `python extract_s2_timeseries.py` — это 8 часов работы. Ты закрываешь крышку ноутбука. SSH-сессия обрывается. **Процесс на сервере умирает**. Утром приходишь — ничего не посчитано.

### Решение с tmux

```bash
# 1. Подключилась к серверу
ssh user@gpu-server

# 2. Создала именованную сессию tmux
tmux new -s training

# 3. Внутри сессии запустила долгий скрипт
python crop_classification/data_collection/extract_s2_timeseries.py --year 2021 --tile 38TLR

# 4. Сворачиваешь сессию (она продолжит работать): Ctrl+B, потом D
# Теперь можно закрыть SSH — скрипт продолжит работать на сервере.

# 5. Возвращаешься позже:
ssh user@gpu-server
tmux attach -t training
# Видишь как прогресс продолжается с того места.

# 6. Когда всё посчитано — выйти из tmux:
exit  # внутри сессии (закроет её)
# или Ctrl+B, потом D — если хочешь оставить ещё одну сессию.
```

### Полезные команды tmux

| Команда | Что делает |
|---------|-----------|
| `tmux new -s <имя>` | Создать сессию с именем |
| `tmux ls` | Список всех сессий |
| `tmux attach -t <имя>` | Подключиться к сессии |
| `Ctrl+B` затем `D` | Отключиться от сессии (она продолжит работать) |
| `Ctrl+B` затем `C` | Создать новое окно внутри сессии |
| `Ctrl+B` затем `0..9` | Переключиться между окнами |
| `Ctrl+B` затем `[` | Войти в режим прокрутки (выйти — `q`) |
| `tmux kill-session -t <имя>` | Убить сессию |

### Когда использовать

- **Сбор данных** (Фаза 1) — 10-24 часа;
- **Обучение моделей** (Фаза 4) — 6-9 часов;
- Любая операция, которая занимает больше 5 минут на сервере по SSH.

Альтернатива — `screen` (то же самое, чуть проще, но менее популярна). На большинстве серверов установлены оба.

---

## Запуск пайплайна на Jupyter-сервере (полный сценарий)

### Подготовка

```bash
# Локально, на маке
git add -A
git commit -m "Reorganize into data_collection/training/inference subfolders"
git push origin feature/production-training
```

### На сервере

```bash
ssh user@jupyter-server

# Клонирование
git clone <repo-url> fields
cd fields
git checkout feature/production-training

# Установка
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync
uv pip install -e ./presto_repo --no-deps
uv add scikit-learn matplotlib seaborn pyarrow xarray netcdf4

# CDS API ключ для ERA5
nano ~/.cdsapirc  # вставить url и ключ

# Создать tmux-сессию для долгой работы
tmux new -s pipeline

# Запустить все 4 фазы по очереди (~20 часов суммарно)
# Фаза 1+2: сбор данных и эмбеддинги
for year in 2017 2018 2019 2020 2021; do
  for tile in 38TLR 37TGM 38TLS; do
    python crop_classification/data_collection/extract_s2_timeseries.py --year $year --tile $tile
    python crop_classification/data_collection/extract_s1_timeseries.py --year $year --tile $tile
    python crop_classification/data_collection/build_dataset.py --year $year --tile $tile
    python crop_classification/data_collection/run_presto_embed.py --year $year --tile $tile
  done
done
# (повторить для chernozem и volga тайлов)

# Фаза 3: master-датасет
python crop_classification/training/build_master_dataset.py --holdout-region primorye

# Фаза 4: обучение
python crop_classification/training/train_linear.py
python crop_classification/training/train_finetune.py
python crop_classification/training/train_maml.py --holdout primorye

# Отключиться от tmux: Ctrl+B, D
```

### Скачать модели

```bash
# Локально, на маке
scp -r user@jupyter-server:~/fields/models/ ./
git add models/
git commit -m "Production models trained"
git push
```

Готово — на маке у тебя 10 МБ обученных моделей, можно делать предсказания и демо.
