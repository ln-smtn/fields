# Fields v0.0.1

Автоматическая сегментация сельскохозяйственных полей на спутниковых снимках Sentinel-2
с использованием модели [Fields of The World (FTW)](https://github.com/fieldsoftheworld/ftw-baselines)
и валидация результатов по эталонным полигонам.

---

## Что делает проект

1. **Сбор датасета** — из KML-файлов с контурами полей (культура + год) собирается единый CSV с WKT-геометриями.
2. **Поиск тайлов Sentinel-2** — по эталонным полигонам находится тайл MGRS (100×100 км), на котором больше всего полей.
3. **Скачивание снимков** — два снимка Sentinel-2 (весна + лето) вырезаются по области пересечения их footprints (зона без nodata).
4. **Инференс модели FTW** — модель сегментации обрабатывает 8-канальный стек (4 канала × 2 даты) и находит границы полей.
5. **Валидация** — сравнение полигонов модели с эталонными по метрике IoU (Intersection over Union).

---

## Структура проекта

```
fields/
├── pyproject.toml              # Зависимости (uv)
├── .python-version             # Python ≥ 3.12
├── ftw-baselines/              # Клон FTW (git submodule / копия)
│   └── prue_efnetb7_ccby_checkpoint.ckpt  # Веса модели (скачать отдельно)
├── build_cultures_dataset.py   # Шаг 1: KML → CSV
├── ftw_oneshot.py              # Шаг 2-4: поиск тайлов, скачивание, инференс
├── validate_iou.py             # Шаг 5: валидация IoU
├── site_geo.py                 # Вспомогательные геоопераци
├── 
├── data/
│   ├── .gitkeep
│   └── .csv  #   данные 
└── outputs/ftw/                # Результаты прогонов (не в git)
    └── <год>_tile_<MGRS>/
        ├── manifest.json
        ├── REPORT.md
        ├── *_stack_8band.tif
        ├── *_inference.tif
        ├── *_ftw_boundaries.gpkg
        └── *_validation_iou_*.csv/json
```
---

## Установка

### Требования

- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** — менеджер пакетов
- **Git** — для клонирования FTW
- **Интернет** — для STAC-каталога Planetary Computer и скачивания снимков
- **macOS Apple Silicon** → инференс через MPS (Metal); **Linux + NVIDIA GPU** → CUDA

### Шаг 1. Клонировать проект

```bash
git clone <url-репозитория> fields
cd fields
```

### Шаг 2. Клонировать FTW (если ещё нет)

```bash
git clone https://github.com/fieldsoftheworld/ftw-baselines.git
```

### Шаг 3. Скачать веса модели

Файл `prue_efnetb7_ccby_checkpoint.ckpt` (~258 МБ) положить в `ftw-baselines/`:

```bash
# Вариант 1: скачать через ftw после установки окружения (шаг 4)
uv run ftw model download --model FTW_PRUE_EFNET_B7

# Вариант 2: скопировать вручную, если файл уже есть
cp /путь/к/prue_efnetb7_ccby_checkpoint.ckpt ftw-baselines/
```

Если файла нет, скрипт использует встроенную модель `FTW_PRUE_EFNET_B5` (скачивается автоматически, но менее точная).

### Шаг 4. Установить окружение

```bash
uv sync
```

Это создаст `.venv/` и установит все зависимости: `ftw-tools` (из `ftw-baselines/`), `geopandas`, `shapely`, `mgrs`, `torch` и др.

### Проверка

```bash
uv run ftw --help
uv run python -c "import geopandas; print('ok')"
```

---

## Использование

### Шаг 1. Подготовка датасета (KML → CSV)

Положите файлы в папку `cultures_/`. Имя каждого файла: `<культура>_<год>.`
с `Polygon` в WGS84.

```bash
uv run python build_cultures_dataset.py
```

Результат: `data/cultures_polygons_dataset.csv` — таблица, где каждая строка = один полигон:

| Колонка | Пример |
|---------|--------|
| `row_id` | `1` |
| `source_file` | `культура 1.kml` |
| `culture` | `культура 1` |
| `year` | `2021` |
| `geometry_wkt` | `POLYGON((43.1 45.6, ...))` |

### Шаг 2. Посмотреть доступные тайлы

```bash
uv run python ftw_oneshot.py --list-cells --years 2021
```

Покажет таблицу тайлов MGRS Sentinel-2, на которых есть эталонные полигоны, отсортированную по заполненности (`fill`). Пример вывода:

```
  --- Таблица: топ-20 тайлов по fill, затем по числу полигонов на тайле ---
    #  MGRS      fill   полиг.  примечание          bbox
  ---  --------  ------  ------  ------------------  ------------------------------------------------
    1  38TLR       6.3%    677  densest             42.506190,45.056580,43.515251,45.945652
    2  37TGM       4.8%    578  сосед тайла         41.598802,46.332930,42.937857,46.917659
    3  38TLS       3.2%    448  сосед тайла         42.372999,46.406304,43.330913,46.939875
    ...
```

Из этой таблицы выбирается тайл для прогона. Код `38TLR` — это MGRS-идентификатор квадрата 100×100 км, которым маркированы снимки Sentinel-2.

### Шаг 3. Запуск пайплайна (поиск + скачивание + инференс)

**Автоматический выбор** (плотнейший тайл):

```bash
uv run python ftw_oneshot.py --years 2021 --run
```

**Конкретный тайл** (например `38TLR` из таблицы выше):

```bash
uv run python ftw_oneshot.py --years 2021 --tile 38TLR --run
```

Что произойдёт:

1. Из CSV загрузятся полигоны за 2021 год.
2. Через STAC (Planetary Computer) найдутся тайлы MGRS, пересекающие эталонные полигоны.
3. Выберется указанный тайл (`--tile`) или **плотнейший** (максимум fill).
4. Для двух сезонных окон (весна `04-15/06-20`, лето `07-01/09-15`) подберётся пара снимков **с одной орбиты**, чьи footprints пересекаются.
5. `ftw inference download` — скачает 4 канала × 2 даты в `*_stack_8band.tif` (bbox = пересечение footprints, без nodata).
6. `ftw inference run` — модель обработает стек → `*_inference.tif`.
7. `ftw inference polygonize` — маска → `*_ftw_boundaries.gpkg`.

Результат: папка `outputs/ftw/2021_tile_38TLR/` (MGRS-код зависит от тайла).

### Шаг 4. Валидация (сравнение с эталоном)

```bash
uv run python validate_iou.py --year 2021
```

Или с явным manifest:

```bash
uv run python validate_iou.py --year 2021 --manifest outputs/ftw/2021_tile_38TLR/manifest.json
```

Результат:

```
Валидация IoU — год 2021, AOI из manifest
  Эталон в AOI: 650 полигонов | FTW в AOI: 580
  Пар с IoU ≥ 0.05: 320
  Без пары (эталон): 330 | Без пары (FTW): 260
  Средний IoU по найденным парам: 0.4500
```

Файлы: `*_validation_iou_per_field.csv`, `*_validation_iou_summary.json`, `*_reference_aoi.gpkg`.

### Шаг 5. Просмотр в QGIS

Откройте в QGIS:
- `*_reference_aoi.gpkg` — эталонные полигоны в AOI
- `*_ftw_boundaries.gpkg` — полигоны модели
- `*_stack_8band.tif` — снимок (RGB: каналы 3,2,1)

---

## Что лежит в папке прогона

| Файл | Описание |
|------|----------|
| `manifest.json` | Все параметры прогона: bbox, сцены, модель, пути |
| `REPORT.md` | Краткий отчёт |
| `*_stack_8band.tif` | 8-канальный стек (вход модели) |
| `*_inference.tif` | Маска сегментации (выход модели) |
| `*_ftw_boundaries.gpkg` | Полигоны границ полей (GeoPackage) |
| `*_ftw_boundaries_labeled.gpkg` | То же + колонки с метаданными |
| `*_validation_iou_per_field.csv` | IoU по каждой паре эталон↔модель |
| `*_validation_iou_summary.json` | Сводная статистика IoU |
| `*_reference_aoi.gpkg` | Эталонные полигоны в AOI (для QGIS) |

---

## Ключевые параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--years` | все годы из CSV | Год(ы) через запятую: `2020,2021` |
| `--run` | нет | Выполнить download + инференс + полигонизацию |
| `--tile MGRS` | автовыбор | Конкретный тайл (например `38TLR`) |
| `--cell gi,gj` | — | Маленькая ячейка сетки вместо тайла MGRS |
| `--window-a` | `04-15/06-20` | Весеннее окно дат |
| `--window-b` | `07-01/09-15` | Летнее окно дат |
| `--cloud-lt` | `25.0` | Максимальная облачность сцены (%) |
| `--max-items` | `100` | Кандидатов на окно дат при выборе сцены |
| `--model` | `prue_efnetb7_ccby_checkpoint.ckpt` | Модель (путь к .ckpt или имя из `ftw model list`) |
| `--list-cells` | — | Показать таблицу тайлов и соседей |
| `--run-tag` | — | Суффикс папки для повторного эксперимента |



---

## Зависимости

Определены в `pyproject.toml`, устанавливаются через `uv sync`:

- **ftw-tools** — из `ftw-baselines/` (editable install)
- **geopandas** ≥ 1.0
- **shapely** ≥ 2.0
- **mgrs** ≥ 1.5.4
- **PyTorch** — устанавливается как зависимость ftw-tools

Внешние сервисы:
- **Planetary Computer STAC** — каталог снимков Sentinel-2 (бесплатный, без ключей)

---

## Ссылки

- [Fields of The World (FTW)](https://github.com/fieldsoftheworld/ftw-baselines) — модель и инструменты
- [Planetary Computer STAC](https://planetarycomputer.microsoft.com/api/stac/v1) — каталог Sentinel-2
- [MGRS](https://en.wikipedia.org/wiki/Military_Grid_Reference_System) — система тайлов Sentinel-2
