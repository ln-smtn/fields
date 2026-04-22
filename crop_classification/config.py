"""
Конфигурация пайплайна темпоральной классификации культур.

Все константы, пути, маппинги и гиперпараметры собраны здесь,
чтобы остальные скрипты импортировали из одного места.

Пайплайн поддерживает два режима:
  - Пилотный: один тайл, один год (для отладки)
  - Полный:   все регионы, все годы (для финальной модели)

Каждый скрипт принимает --year и --tile / --region через CLI.

Полный вход PRESTO (18 каналов на timestep):
  Sentinel-2:     10 каналов  (B2-B8A, B11, B12) — оптический спектр
  Sentinel-1:      2 канала   (VV, VH)           — радар (сквозь облака)
  ERA5:            2 канала   (t2m, tp)           — климат
  NDVI:            1 канал    (B8-B4)/(B8+B4)     — вегетационный индекс
  SRTM:            2 канала   (elevation, slope)  — рельеф (статический)
  Dynamic World:   1 канал    (land cover 0-9)    — покров (опционально)
"""

from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CULTURES_CSV = DATA_DIR / "cultures_polygons_dataset.csv"

# Промежуточные данные — каждый источник в своей папке
S2_TIMESERIES_DIR = DATA_DIR / "timeseries_s2"    # S2 временные ряды (parquet)
S1_TIMESERIES_DIR = DATA_DIR / "timeseries_s1"    # S1 временные ряды (parquet)
ERA5_DIR = DATA_DIR / "era5"                       # климат ERA5 (netcdf)
SRTM_DIR = DATA_DIR / "srtm"                      # рельеф SRTM (geotiff)
DW_DIR = DATA_DIR / "dynamic_world"                # Dynamic World (parquet)

# Собранный датасет и результаты
DATASET_DIR = DATA_DIR / "presto_dataset"          # объединённый датасет
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "crop_cls"  # результаты классификации

# ── Sentinel-2: 10 оптических каналов ────────────────────────────────────
# Sentinel-2 имеет 13 каналов. PRESTO использует 10, исключая:
#   B1 (Aerosol, 60 м)  — атмосферная коррекция, не для культур
#   B9 (Water Vapor, 60 м) — водяной пар, не для культур
#   B10 (Cirrus, 60 м)  — перистые облака, только для маскировки
#
# Оставшиеся 10 каналов покрывают весь полезный спектр:
#   Видимый (B2-B4): цвет почвы/листьев, хлорофилл
#   Red Edge (B5-B7): биомасса, стадия развития — ключевое для агро
#   NIR (B8, B8A): структура листьев, NDVI
#   SWIR (B11, B12): влажность, сухая биомасса

S2_BANDS = [
    "B02",   # Blue,       490 нм, 10 м — отражение воды, почвы
    "B03",   # Green,      560 нм, 10 м — пик отражения хлорофилла
    "B04",   # Red,        665 нм, 10 м — поглощение хлорофиллом
    "B05",   # Red Edge 1, 705 нм, 20 м — переход к NIR, чувствителен к биомассе
    "B06",   # Red Edge 2, 740 нм, 20 м — структура кроны, LAI
    "B07",   # Red Edge 3, 783 нм, 20 м — хлорофилл, отличает пшеницу от ячменя
    "B08",   # NIR,        842 нм, 10 м — структура листьев → NDVI
    "B8A",   # NIR narrow, 865 нм, 20 м — уточнённый NIR, менее шумный
    "B11",   # SWIR 1,     1610 нм, 20 м — влажность растений/почвы
    "B12",   # SWIR 2,     2190 нм, 20 м — минералы, сухая биомасса
]

# SCL (Scene Classification Layer) — маска качества пикселей Sentinel-2.
# Оставляем только «чистые» пиксели:
#   4 = vegetation — растительность
#   5 = bare soil  — голая почва (убранное поле, пар)
#   7 = unclassified (low cloud probability) — скорее чисто
SCL_VALID_CLASSES = {4, 5, 7}

# ── Sentinel-1: 2 радарных канала ────────────────────────────────────────
# Sentinel-1 — радар C-диапазона (5.4 ГГц), видит сквозь облака.
# Два канала поляризации:
#   VV — вертикально-вертикальная: чувствительна к высоте растений
#   VH — вертикально-горизонтальная: чувствительна к биомассе/структуре
#
# Sentinel-1 особенно полезен для:
#   - облачных дат (когда S2 нет)
#   - различения культур по текстуре (кукуруза vs пшеница — разная высота)
#   - определения даты уборки (резкое падение VH)
#
# Planetary Computer: коллекция "sentinel-1-rtc" (Radiometric Terrain Corrected)
# Разрешение: 10 м, повторяемость: 12 дней (одна орбита)

S1_BANDS = ["vv", "vh"]
S1_COLLECTION = "sentinel-1-rtc"

# ── ERA5: 2 климатических переменных ─────────────────────────────────────
# ERA5 — глобальный реанализ ECMWF, разрешение ~31 км, почасовые данные.
# PRESTO использует 2 переменные:
#   t2m — температура на 2 м (K): определяет фенофазы (всходы, цветение, созревание)
#   tp  — осадки (м/сутки): водный баланс, рост биомассы
#
# Одна точка ERA5 покрывает десятки полей — для каждого поля берём
# ближайший узел сетки (nearest neighbor interpolation).
# Источник: CDS API (Climate Data Store, бесплатная регистрация)

ERA5_VARIABLES_CDS = [
    "2m_temperature",       # температура воздуха на 2 м (K)
    "total_precipitation",  # суммарные осадки (м/сутки)
]
ERA5_VARIABLES_SHORT = ["t2m", "tp"]

# ── NDVI: вычисляемый индекс ─────────────────────────────────────────────
# NDVI = (B8 - B4) / (B8 + B4)
# Normalized Difference Vegetation Index — главный индикатор зелёной биомассы.
# Значения: -1..1, типичные для культур: 0.2 (голая почва) — 0.9 (густая растительность).
# Вычисляется из уже скачанных B4 и B8, отдельного скачивания не требует.
# PRESTO принимает NDVI как отдельный 15-й канал.

NDVI_RED_BAND = "B04"
NDVI_NIR_BAND = "B08"

# ── SRTM: 2 статических канала рельефа ───────────────────────────────────
# SRTM (Shuttle Radar Topography Mission) — глобальная модель высот, 30 м.
# PRESTO использует:
#   elevation — высота над уровнем моря (м)
#   slope     — уклон поверхности (°)
#
# Статические данные: скачиваются ОДИН РАЗ для региона.
# Рельеф влияет на микроклимат поля (южный/северный склон, водосбор).
# Planetary Computer: коллекция "cop-dem-glo-30" (Copernicus DEM 30m)

SRTM_COLLECTION = "cop-dem-glo-30"

# ── Dynamic World: 1 канал земного покрова (опционально) ──────────────────
# Dynamic World (Google) — глобальная классификация покрова, ~10 м.
# 9 классов: 0=water, 1=trees, 2=grass, 3=flooded_vegetation,
#            4=crops, 5=shrub_and_scrub, 6=built, 7=bare, 8=snow_and_ice
#
# PRESTO использует как категориальный признак (не числовой).
# Источник: Google Earth Engine (нужен аккаунт GEE).
# Опционально: если нет GEE — передаём маску, PRESTO проигнорирует.

DW_CLASSES = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare",
    8: "snow_and_ice",
}

# ── Сводка каналов PRESTO ────────────────────────────────────────────────
# Порядок каналов во входном тензоре x: [batch, T, 17]
# ВАЖНО: порядок жёстко соответствует NORMED_BANDS в PRESTO
# (presto/dataops/pipelines/s1_s2_era5_srtm.py, BANDS_GROUPS_IDX).
# Менять порядок нельзя — PRESTO по нему разбивает каналы на группы.
#
#  Индекс    Канал           Источник         Тип
#  ──────    ──────          ────────         ─────────
#   0- 1     VV, VH          Sentinel-1       динамический
#   2- 3     B2, B3          Sentinel-2       динамический (S2_RGB part)
#      4     B4              Sentinel-2       динамический
#   5- 7     B5, B6, B7      Sentinel-2       динамический (S2_Red_Edge)
#      8     B8              Sentinel-2       динамический (S2_NIR_10m)
#      9     B8A             Sentinel-2       динамический (S2_NIR_20m)
#  10-11     B11, B12        Sentinel-2       динамический (S2_SWIR)
#  12-13     t2m, tp         ERA5             динамический
#  14-15     elev, slope     SRTM             статический
#     16     NDVI            вычисляемый      динамический
#
# Dynamic World передаётся ОТДЕЛЬНЫМ тензором shape=[batch, T] с целыми
# числами 0-9 (9 = DynamicWorld2020_2021.class_amount = «пропуск»).

PRESTO_CHANNEL_ORDER = (
    S1_BANDS                # 0-1:   VV, VH
    + S2_BANDS              # 2-11:  B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    + ERA5_VARIABLES_SHORT  # 12-13: t2m, tp
    + ["elevation", "slope"]  # 14-15: SRTM
    + ["ndvi"]              # 16:    NDVI
)

PRESTO_NUM_CHANNELS = len(PRESTO_CHANNEL_ORDER)  # 17
PRESTO_DW_MISSING = 9  # sentinel для «нет данных Dynamic World»

# ── Вегетационный сезон ──────────────────────────────────────────────────
SEASON_START = "03-01"
SEASON_END = "10-31"
MAX_CLOUD_COVER = 30  # % макс. облачность сцены при STAC-поиске

# ── Географические регионы ───────────────────────────────────────────────
# Данные покрывают 4 региона РФ. Для каждого задан:
#   era5_area:  [N, W, S, E] — bbox для скачивания ERA5 (формат CDS API)
#   stac_bbox:  [W, S, E, N] — bbox для STAC-запросов S1/S2 (GeoJSON order)
#   mgrs_tiles_approx: примерные тайлы MGRS в регионе
#
# При запуске с --tile 38TLR скрипт определяет регион автоматически
# через region_for_point() и берёт нужный ERA5 bbox.

REGIONS = {
    "chernozem": {
        "name": "Центральное Черноземье (Курская, Белгородская, Воронежская обл.)",
        "era5_area": [55, 34, 50, 41],
        "stac_bbox": [34, 50, 41, 55],
        "mgrs_tiles_approx": ["36UXV", "37UDB", "37UEB"],
        "polygons_approx": 12500,
    },
    "south": {
        "name": "Юг РФ (Краснодарский край, Ставрополье, Ростовская обл.)",
        "era5_area": [48, 35, 42, 46],
        "stac_bbox": [35, 42, 46, 48],
        "mgrs_tiles_approx": ["38TLR", "37TGM", "38TLS", "38TLQ"],
        "polygons_approx": 10300,
    },
    "volga": {
        "name": "Среднее Поволжье",
        "era5_area": [55, 40, 48, 50],
        "stac_bbox": [40, 48, 50, 55],
        "mgrs_tiles_approx": [],
        "polygons_approx": 1700,
    },
    "primorye": {
        "name": "Приморский край",
        "era5_area": [46, 130, 42, 135],
        "stac_bbox": [130, 42, 135, 46],
        "mgrs_tiles_approx": [],
        "polygons_approx": 1650,
    },
}

# ── Пилотный режим ───────────────────────────────────────────────────────
# Для первого запуска и отладки: один тайл, один год.
# Пилот: тайл 38TLR (Краснодарский край), 2021 год, ~600 полигонов.
#
# Запуск пилота:
#   uv run python crop_classification/extract_s2_timeseries.py --year 2021 --tile 38TLR
#   uv run python crop_classification/extract_s1_timeseries.py --year 2021 --tile 38TLR
#   uv run python crop_classification/download_era5.py --year 2021 --region south
#   uv run python crop_classification/download_srtm.py --region south
#
# Полный запуск (все годы, все регионы):
#   for year in 2016..2021; for region in chernozem south volga primorye; ...

PILOT_TILE = "38TLR"
PILOT_YEAR = 2021
PILOT_REGION = "south"

ALL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
ALL_REGIONS = list(REGIONS.keys())

# ── Маппинг культур ──────────────────────────────────────────────────────

CULTURE_TO_ID = {
    "озимая пшеница":  0,
    "яровая пшеница":  1,
    "озимый ячмень":   2,
    "яровой ячмень":   3,
    "кукуруза":        4,
    "подсолнечник":    5,
    "соя":             6,
    "свекла сахарная": 7,
    "горох":           8,
    "нут":             9,
    "озимый рапс":     10,
    "яровой рапс":     11,
    "лен":             12,
    "пар  сидеральный пар  смешанный пар  эспарцет": 13,
    "прочее":          14,
}

ID_TO_CULTURE = {v: k for k, v in CULTURE_TO_ID.items()}

ID_TO_SHORT = {
    0: "оз.пшеница",   1: "яр.пшеница",  2: "оз.ячмень",
    3: "яр.ячмень",    4: "кукуруза",     5: "подсолнечник",
    6: "соя",          7: "свёкла",        8: "горох",
    9: "нут",          10: "оз.рапс",     11: "яр.рапс",
    12: "лён",         13: "пар",          14: "прочее",
}

NUM_CLASSES = len(CULTURE_TO_ID)

# ── PRESTO модель ────────────────────────────────────────────────────────
PRESTO_EMBEDDING_DIM = 128    # размерность выходного эмбеддинга
PRESTO_MAX_TIMESTEPS = 24     # максимальное число дат (паддинг/субсэмплинг)

# ── Few-shot гиперпараметры ──────────────────────────────────────────────
FEWSHOT_N_WAY = 5              # число классов в одном эпизоде мета-обучения
FEWSHOT_K_SHOT = 5             # число примеров на класс (support set)
FEWSHOT_Q_QUERY = 15           # число запросов на класс (query set)
MAML_INNER_LR = 0.01           # learning rate внутреннего цикла MAML
MAML_OUTER_LR = 0.001          # learning rate внешнего цикла (Adam)
MAML_INNER_STEPS = 5           # число шагов адаптации на support set
META_TRAIN_EPISODES = 2000     # число эпизодов мета-обучения

# ── STAC API (Planetary Computer) ────────────────────────────────────────
# Бесплатный, без регистрации. Используется для S2, S1, SRTM.
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"


# ── Утилиты ──────────────────────────────────────────────────────────────

def region_for_point(lat: float, lon: float) -> str | None:
    """Определяет регион по координатам точки.
    Используется для автоматического выбора ERA5 bbox по координатам тайла/поля.
    """
    for key, reg in REGIONS.items():
        n, w, s, e = reg["era5_area"]
        if s <= lat <= n and w <= lon <= e:
            return key
    return None


def get_era5_area(region: str) -> list[int]:
    """Возвращает ERA5 bbox [N, W, S, E] для указанного региона."""
    return REGIONS[region]["era5_area"]


def get_stac_bbox(region: str) -> list[int]:
    """Возвращает STAC bbox [W, S, E, N] для указанного региона."""
    return REGIONS[region]["stac_bbox"]
