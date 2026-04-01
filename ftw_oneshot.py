#!/usr/bin/env python3
"""
Oneshot FTW: эталон из CSV → тайл MGRS с полигонами → два снимка Sentinel-2 на  тайле
→ outputs/ftw/<год>_tile_<MGRS>/ с manifest.json, REPORT.md; с --run — ftw download / run / polygonize.

  uv run python ftw_oneshot.py --years 2021 --run
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pystac_client

from site_geo import (
    bbox_for_grid_cell,
    bbox_from_percentiles,
    count_polygons_intersecting_grid_cells,
    default_polygons_csv,
    densest_grid_cell_bbox,
    filter_reference_gdf_to_main_bbox,
    load_polygon_boxes_by_year,
    load_reference_geodataframe_year,
    mgrs_100km_cell_polygon_wgs84,
    normalize_mgrs_tile,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _rel_to_project(p: Path) -> str:
    """Путь от корня проекта (для вывода и manifest, без /Users/...)."""
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.name


def _ftw_path_arg(p: Path) -> str:
    """Аргумент пути для вызова ftw с cwd=PROJECT_ROOT — относительный, чтобы не засорять лог абсолютами."""
    return _rel_to_project(p.resolve())
S2_COLLECTION = "sentinel-2-l2a"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "ftw"


def resolve_ftw_executable() -> Path | None:
    bindir = Path(sys.executable).resolve().parent
    cand = bindir / ("ftw.exe" if os.name == "nt" else "ftw")
    if cand.is_file():
        return cand
    w = shutil.which("ftw")
    return Path(w) if w else None


def _mgrs(item) -> str | None:
    p = item.properties or {}
    return p.get("s2:mgrs_tile") or p.get("mgrs_tile")


def _cloud(item) -> float:
    c = (item.properties or {}).get("eo:cloud_cover")
    return float(c) if c is not None else 999.0


def _orbit(item) -> int | None:
    """Относительная орбита сцены (sat:relative_orbit или из id: ...R035...)."""
    p = item.properties or {}
    ro = p.get("sat:relative_orbit")
    if ro is not None:
        return int(ro)
    m = re.search(r"_R(\d{3})_", item.id or "")
    return int(m.group(1)) if m else None


def search_best_scene(
    catalog,
    bbox: list[float],
    date_start: str,
    date_end: str,
    cloud_lt: float,
    max_items: int,
    prefer_mgrs: str | None,
    *,
    require_mgrs_match: bool = False,
    prefer_orbit: int | None = None,
):
    """Поиск лучшей сцены.

    prefer_orbit: если задан, сначала берём сцены с этой орбитой (чтобы win_a и win_b
    гарантированно перекрывались в bbox). Если на этой орбите нет — берём любую.
    """
    dt = f"{date_start}T00:00:00Z/{date_end}T23:59:59Z"
    res = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=dt,
        query={"eo:cloud_cover": {"lt": cloud_lt}},
        max_items=max_items,
    )
    items = list(res.items())
    if not items:
        return None
    if prefer_mgrs:
        pk = normalize_mgrs_tile(prefer_mgrs)
        matching = [
            it
            for it in items
            if _mgrs(it) and normalize_mgrs_tile(_mgrs(it)) == pk
        ]
        if matching:
            if prefer_orbit is not None:
                same_orb = [it for it in matching if _orbit(it) == prefer_orbit]
                if same_orb:
                    same_orb.sort(key=_cloud)
                    return same_orb[0]
            matching.sort(key=_cloud)
            return matching[0]
        if require_mgrs_match:
            return None
    if prefer_orbit is not None:
        same_orb = [it for it in items if _orbit(it) == prefer_orbit]
        if same_orb:
            same_orb.sort(key=_cloud)
            return same_orb[0]
    items.sort(key=_cloud)
    return items[0]


def _footprints_intersect(item_a, item_b) -> bool:
    """Проверка, что footprints двух сцен пересекаются (как делает ftw inference download)."""
    from shapely.geometry import shape as shp_shape

    try:
        g0 = shp_shape(item_a.geometry)
        g1 = shp_shape(item_b.geometry)
        return g0.intersects(g1)
    except Exception:
        return True


def footprint_intersection_bbox(
    item_a, item_b
) -> tuple[float, float, float, float] | None:
    """Bbox области, где оба снимка имеют реальные данные (пересечение footprints).

    Возвращает (minx, miny, maxx, maxy) в WGS84 или None если нет пересечения.
    """
    from shapely.geometry import shape as shp_shape

    try:
        g0 = shp_shape(item_a.geometry)
        g1 = shp_shape(item_b.geometry)
        inter = g0.intersection(g1)
        if inter.is_empty:
            return None
        return tuple(float(x) for x in inter.bounds)
    except Exception:
        return None


def count_reference_in_footprint(
    gdf, item_a, item_b
) -> tuple[int, object | None]:
    """Сколько эталонных полигонов попадают в область пересечения footprints.

    Возвращает (count, intersection_geom).
    """
    from shapely.geometry import shape as shp_shape

    try:
        g0 = shp_shape(item_a.geometry)
        g1 = shp_shape(item_b.geometry)
        inter = g0.intersection(g1)
        if inter.is_empty:
            return 0, None
        mask = gdf.geometry.intersects(inter)
        return int(mask.sum()), inter
    except Exception:
        return 0, None


def search_scene_pair(
    catalog,
    bbox: list[float],
    date_a: tuple[str, str],
    date_b: tuple[str, str],
    cloud_lt: float,
    max_items: int,
    prefer_mgrs: str | None,
    *,
    require_mgrs_match: bool = False,
):
    """Найти пару сцен (win_a, win_b) с одной орбиты, чьи footprints пересекаются.

    Стратегия:
    1. Найти win_a (лучшая по облачности).
    2. Найти win_b на той же орбите.
    3. Проверить, что footprints пересекаются.
    4. Если нет — перебрать другие орбиты win_a и повторить.
    """
    a0, a1 = date_a
    b0, b1 = date_b
    dt_a = f"{a0}T00:00:00Z/{a1}T23:59:59Z"
    dt_b = f"{b0}T00:00:00Z/{b1}T23:59:59Z"

    def _fetch(dt: str):
        res = catalog.search(
            collections=[S2_COLLECTION],
            bbox=bbox,
            datetime=dt,
            query={"eo:cloud_cover": {"lt": cloud_lt}},
            max_items=max_items,
        )
        return list(res.items())

    items_a = _fetch(dt_a)
    items_b = _fetch(dt_b)
    if not items_a or not items_b:
        return None, None

    def _filter_mgrs(items):
        if not prefer_mgrs:
            return items
        pk = normalize_mgrs_tile(prefer_mgrs)
        m = [it for it in items if _mgrs(it) and normalize_mgrs_tile(_mgrs(it)) == pk]
        if m:
            return m
        return [] if require_mgrs_match else items

    cands_a = _filter_mgrs(items_a)
    cands_b = _filter_mgrs(items_b)
    if not cands_a or not cands_b:
        return None, None

    cands_a.sort(key=_cloud)
    cands_b.sort(key=_cloud)

    orbits_a: dict[int | None, list] = {}
    for it in cands_a:
        orbits_a.setdefault(_orbit(it), []).append(it)

    for best_a in cands_a:
        orb = _orbit(best_a)
        same_orb_b = [it for it in cands_b if _orbit(it) == orb] if orb is not None else []
        pool_b = same_orb_b if same_orb_b else cands_b
        pool_b.sort(key=_cloud)
        for best_b in pool_b:
            if _footprints_intersect(best_a, best_b):
                return best_a, best_b

    cands_a.sort(key=_cloud)
    cands_b.sort(key=_cloud)
    return cands_a[0], cands_b[0]


def fmt_bbox(b: tuple[float, float, float, float]) -> str:
    w, s, e, n = b
    return f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}"


def output_file_stem(
    year: int,
    run_tag: str,
    *,
    grid_cell: tuple[int, int] | None = None,
    mgrs_tile: str | None = None,
) -> str:
    """Префикс имён: год + тайл MGRS или ячейка (gi,gj); при необходимости суффикс run_tag."""
    if mgrs_tile:
        safe = re.sub(r"[^\w]+", "_", mgrs_tile.strip())
        base = f"{year}_tile_{safe}"
    elif grid_cell is not None:
        gi, gj = grid_cell
        base = f"{year}_cell_{gi}_{gj}"
    else:
        raise ValueError("нужен mgrs_tile или grid_cell")
    if not run_tag:
        return base
    if grid_cell is not None:
        gi, gj = grid_cell
        if run_tag.strip() == f"cell_{gi}_{gj}":
            return base
    return f"{base}_{run_tag}"


def parse_cell_arg(s: str) -> tuple[int, int]:
    parts = re.split(r"[,;/\s]+", s.strip())
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("ожидалось два целых: gi,gj (например 5,12)")
    return int(parts[0]), int(parts[1])


def _unify_footprint(geom):
    from shapely import make_valid
    from shapely.ops import unary_union

    g = make_valid(geom)
    if g.geom_type == "GeometryCollection":
        polys = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            g = unary_union(polys)
    return g


def _tiles_are_neighbors(g0, g1) -> bool:
    if g0 is None or g1 is None:
        return False
    if g0.equals(g1):
        return False
    try:
        if g0.touches(g1):
            return True
    except Exception:
        pass
    try:
        if g0.distance(g1) < 1e-7:
            return True
    except Exception:
        pass
    try:
        if g0.intersects(g1):
            a0, a1 = g0.area, g1.area
            if a0 <= 0 or a1 <= 0:
                return False
            inter = g0.intersection(g1)
            if inter.area / min(a0, a1) < 0.02:
                return True
    except Exception:
        pass
    return False


def nominal_tile_geoms_from_stac_keys(
    tile_geoms_stac: dict[str, object],
) -> dict[str, object]:
    """Номинальная сетка MGRS 100×100 km по id тайлов (для пересечения с эталоном, как в QGIS)."""
    return {tid: mgrs_100km_cell_polygon_wgs84(tid) for tid in tile_geoms_stac}


def discover_mgrs_tile_geometries(
    catalog,
    bbox: list[float],
    date_start: str,
    date_end: str,
    cloud_lt: float,
    max_items: int,
) -> dict[str, object]:
    """STAC: s2:mgrs_tile → union footprint сцен (WGS84). Нужен только чтобы узнать, какие id тайлов есть в выборке.

    Пересечение с эталоном считается по номинальной ячейке MGRS (см. mgrs_100km_cell_polygon_wgs84), не по footprint.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    dt = f"{date_start}T00:00:00Z/{date_end}T23:59:59Z"
    res = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=dt,
        query={"eo:cloud_cover": {"lt": cloud_lt}},
        max_items=max_items,
    )
    parts: dict[str, list] = defaultdict(list)
    for item in res.items():
        t = _mgrs(item)
        if not t:
            continue
        key = normalize_mgrs_tile(t)
        raw = shape(item.geometry)
        parts[key].append(_unify_footprint(raw))
    tiles: dict[str, object] = {}
    for key, geoms in parts.items():
        if not geoms:
            continue
        tiles[key] = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    return tiles


def _merge_item_tile_geom(tile_geoms: dict[str, object], item) -> dict[str, object]:
    from shapely.geometry import shape
    from shapely.ops import unary_union

    t = _mgrs(item)
    if not t:
        return tile_geoms
    k = normalize_mgrs_tile(t)
    g = _unify_footprint(shape(item.geometry))
    out = dict(tile_geoms)
    if k in out:
        out[k] = unary_union([out[k], g])
    else:
        out[k] = g
    return out


def augment_mgrs_tiles_from_cardinal_searches(
    catalog,
    tile_geoms_stac: dict[str, object],
    center_id: str,
    date_start: str,
    date_end: str,
    cloud_lt: float,
) -> dict[str, object]:
    """Добавить тайлы вокруг центра (смещение bbox), чтобы поймать соседей за пределами union эталона."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if center_id not in tile_geoms_stac:
        return tile_geoms_stac
    g0 = mgrs_100km_cell_polygon_wgs84(center_id)
    c = g0.centroid
    lon, lat = float(c.x), float(c.y)
    out = dict(tile_geoms_stac)
    for dlon, dlat in (
        (0.55, 0.0),
        (-0.55, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
    ):
        bbox = [
            lon + dlon - 0.06,
            lat + dlat - 0.06,
            lon + dlon + 0.06,
            lat + dlat + 0.06,
        ]
        res = catalog.search(
            collections=[S2_COLLECTION],
            bbox=bbox,
            datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
            query={"eo:cloud_cover": {"lt": cloud_lt}},
            max_items=40,
        )
        for item in res.items():
            t = _mgrs(item)
            if not t:
                continue
            key = normalize_mgrs_tile(t)
            g = _unify_footprint(shape(item.geometry))
            if key in out:
                out[key] = unary_union([out[key], g])
            else:
                out[key] = g
    return out


def _geom_intersects_mask(gdf, tile_geom):
    """Пересечение геометрий: полигон ∩ (номинальная) ячейка MGRS 100×100 км."""
    return gdf.geometry.intersects(tile_geom)


def reference_count_and_fill_on_tile(gdf, tile_geom) -> tuple[int, float]:
    """Сколько полигонов пересекает ячейку MGRS и какой fill по площади.

    fill = площадь(union(эталон) ∩ ячейка) / площадь(ячейки) в EPSG:3857.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    mask = _geom_intersects_mask(gdf, tile_geom)
    n = int(mask.sum())
    if n == 0:
        return 0, 0.0
    ref_on = gdf.loc[mask]
    u = unary_union(ref_on.geometry.tolist())
    inter = u.intersection(tile_geom)
    if inter.is_empty:
        return n, 0.0
    gs = gpd.GeoSeries([inter, tile_geom], crs="EPSG:4326").to_crs(3857)
    a_ref = float(gs.iloc[0].area)
    a_tile = float(gs.iloc[1].area)
    fill = a_ref / a_tile if a_tile > 0 else 0.0
    return n, fill


def count_polygons_per_mgrs_tile(gdf, tile_geoms: dict[str, object]) -> dict[str, int]:
    """Сколько полигонов пересекает геометрию каждого тайла."""
    return {tid: int(_geom_intersects_mask(gdf, geom).sum()) for tid, geom in tile_geoms.items()}


def mgrs_tile_metrics(
    gdf, tile_geoms: dict[str, object]
) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    fills: dict[str, float] = {}
    for tid, geom in tile_geoms.items():
        n, fill = reference_count_and_fill_on_tile(gdf, geom)
        counts[tid] = n
        fills[tid] = fill
    return counts, fills


def pick_densest_tile_id(counts: dict[str, int], fills: dict[str, float]) -> str:
    return max(counts.keys(), key=lambda k: (fills[k], counts[k]))


def neighbor_mgrs_tiles_with_counts(
    tile_geoms: dict[str, object],
    center_id: str,
    gdf,
) -> list[tuple[str, int]]:
    g0 = tile_geoms.get(center_id)
    if g0 is None:
        return []
    neigh: list[tuple[str, int]] = []
    for tid, g in tile_geoms.items():
        if tid == center_id:
            continue
        if _tiles_are_neighbors(g0, g):
            n = int(_geom_intersects_mask(gdf, g).sum())
            neigh.append((tid, n))
    neigh.sort(key=lambda x: (-x[1], x[0]))
    return neigh


def compact_aoi_bbox_for_tile(
    tile_geom,
    ref_gdf,
    half_span_deg: float,
) -> tuple[float, float, float, float]:
    """Компактный AOI по эталону, пересекающему геометрию тайла (номинальная MGRS или ячейка сетки)."""
    from shapely.geometry import box

    on_tile = ref_gdf[_geom_intersects_mask(ref_gdf, tile_geom)]
    if on_tile.empty:
        c = tile_geom.centroid
        cx, cy = float(c.x), float(c.y)
        r = half_span_deg
        return (cx - r, cy - r, cx + r, cy + r)
    minx, miny, maxx, maxy = on_tile.total_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half_w = max(half_span_deg, (maxx - minx) / 2.0 + 1e-4)
    half_h = max(half_span_deg, (maxy - miny) / 2.0 + 1e-4)
    aw, as_, ae, an = cx - half_w, cy - half_h, cx + half_w, cy + half_h
    chip = box(aw, as_, ae, an).intersection(tile_geom)
    if chip.is_empty:
        r = half_span_deg
        return (cx - r, cy - r, cx + r, cy + r)
    return tuple(float(x) for x in chip.bounds)


def download_bbox_around_reference_on_tile(
    ref_gdf,
    tile_geom,
    pad_deg: float,
) -> tuple[float, float, float, float]:
    """Прямоугольник скачивания: bounds полигонов, пересекающих ячейку, + отступ."""
    on_tile = ref_gdf[_geom_intersects_mask(ref_gdf, tile_geom)]
    if on_tile.empty:
        return tuple(float(x) for x in tile_geom.bounds)
    minx, miny, maxx, maxy = on_tile.total_bounds
    p = max(pad_deg, 1e-4)
    return (minx - p, miny - p, maxx + p, maxy + p)


def adjacent_mgrs_manifest(
    neighbor_rows: list[tuple[str, int]],
    tile_geoms: dict[str, object],
    ref_gdf,
    half_span_deg: float,
) -> list[dict]:
    out: list[dict] = []
    for tid, n_poly in neighbor_rows:
        geom = tile_geoms.get(tid)
        if geom is None:
            continue
        w, s, e, n_ = compact_aoi_bbox_for_tile(
            geom, ref_gdf, half_span_deg
        )
        out.append(
            {
                "mgrs_tile": tid,
                "polygon_count_on_tile": n_poly,
                "bbox_wgs84": [w, s, e, n_],
            }
        )
    return out


def print_list_cells(
    csv_path: Path,
    by_year: dict[int, list],
    years: list[int],
    *,
    catalog,
    grid_deg: float,
    half_span_deg: float,
    cloud_lt: float,
    max_items: int,
) -> None:
    for year in years:
        boxes = by_year[year]
        gdf = load_reference_geodataframe_year(csv_path, year)
        if gdf.empty:
            print(f"### {year} — нет полигонов\n")
            continue
        main = bbox_from_percentiles(
            [tuple(b) for b in boxes], 5.0, 95.0, 0.02
        )
        gdf_aoi = filter_reference_gdf_to_main_bbox(gdf, main)
        if gdf_aoi.empty:
            gdf_aoi = gdf
        union_bounds = list(gdf.total_bounds)
        margin = 0.35
        bbox_pc = [
            union_bounds[0] - margin,
            union_bounds[1] - margin,
            union_bounds[2] + margin,
            union_bounds[3] + margin,
        ]
        ds, de = f"{year}-01-01", f"{year}-12-31"
        tile_geoms_stac = discover_mgrs_tile_geometries(
            catalog, bbox_pc, ds, de, cloud_lt, max_items
        )
        if not tile_geoms_stac:
            print(
                f"### {year} — нет сцен Sentinel-2 в PC для bbox эталона (проверьте сеть и --cloud-lt).\n"
            )
            continue
        tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
        counts, fills = mgrs_tile_metrics(gdf_aoi, tile_geoms)
        if not counts or max(counts.values()) == 0:
            print(
                f"### {year} — эталон (в main_bbox) не пересекает номинальную ячейку MGRS "
                f"ни одного тайла из выборки STAC.\n"
            )
            continue
        dense_tid = pick_densest_tile_id(counts, fills)
        dense_n = counts[dense_tid]
        dense_fill = fills[dense_tid]
        tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
            catalog, tile_geoms_stac, dense_tid, ds, de, cloud_lt
        )
        tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
        counts, fills = mgrs_tile_metrics(gdf_aoi, tile_geoms)
        dense_tid = pick_densest_tile_id(counts, fills)
        dense_n = counts[dense_tid]
        dense_fill = fills[dense_tid]
        neigh = neighbor_mgrs_tiles_with_counts(tile_geoms, dense_tid, gdf_aoi)
        neigh = [(t, n) for t, n in neigh if n > 0]
        neigh_keys = {t for t, _ in neigh}

        print(
            f"### {year}  — полигонов в CSV: {len(boxes)}; в main_bbox (перц. 5–95%): {len(gdf_aoi)}"
        )
        print()
        default_dir = output_file_stem(year, "", mgrs_tile=dense_tid)
        print(
            "  Прогон по умолчанию (самый плотный тайл Sentinel MGRS, без --tile / --cell):"
        )
        print(f"    uv run python ftw_oneshot.py --years {year} --run")
        print(
            f"    uv run python validate_iou.py --year {year} "
            f"--manifest outputs/ftw/{default_dir}/manifest.json"
        )
        print()
        db = compact_aoi_bbox_for_tile(tile_geoms[dense_tid], gdf_aoi, half_span_deg)
        print(
            f"  Плотнейший тайл MGRS: {dense_tid}  fill={dense_fill:.1%}  "
            f"(полигонов: bbox полигона ∩ bbox ячейки тайла, эталон в main_bbox): {dense_n})"
        )
        print(f"  Компактный AOI bbox: {fmt_bbox(db)}")
        print()

        if neigh:
            print(
                "  --- Соседние тайлы (номинальная сетка MGRS, общая граница ячейки) ---"
            )
            for tid, nc in neigh:
                geom = tile_geoms[tid]
                bb = compact_aoi_bbox_for_tile(geom, gdf_aoi, half_span_deg)
                _, nf = reference_count_and_fill_on_tile(gdf_aoi, geom)
                print(
                    f"  # {tid}  полигонов на тайле: {nc}  fill={nf:.1%}  bbox {fmt_bbox(bb)}"
                )
                print(
                    f"  uv run python ftw_oneshot.py --years {year} --tile {tid} --run"
                )
                neigh_dir = output_file_stem(year, "", mgrs_tile=tid)
                print(
                    f"  uv run python validate_iou.py --year {year} "
                    f"--manifest outputs/ftw/{neigh_dir}/manifest.json"
                )
                print()
        else:
            print("  Соседних тайлов в выборке STAC не найдено (расширьте bbox или max_items).")
            print()

        top_n = 20
        rows = sorted(
            counts.items(),
            key=lambda x: (-fills[x[0]], -x[1], x[0]),
        )[:top_n]
        print(
            f"  --- Таблица: топ-{len(rows)} тайлов по fill, затем по числу полигонов на тайле ---"
        )
        hdr = f"  {'#':>3}  {'MGRS':<8}  {'fill':>6}  {'полиг.':>6}  {'примечание':<18}  bbox"
        print(hdr)
        print(f"  {'-'*3}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*18}  {'-'*48}")
        for rank, (tid, nc) in enumerate(rows, start=1):
            bb = fmt_bbox(
                compact_aoi_bbox_for_tile(
                    tile_geoms[tid], gdf_aoi, half_span_deg
                )
            )
            fl = fills.get(tid, 0.0)
            if tid == dense_tid:
                note = "densest"
            elif tid in neigh_keys:
                note = "сосед тайла"
            else:
                note = ""
            print(f"  {rank:3d}  {tid:<8}  {fl:6.1%}  {nc:5d}  {note:<18}  {bb}")
        if len(counts) > top_n:
            print(f"  ... всего тайлов с данными: {len(counts)} (показаны первые {top_n})")
        else:
            print(f"  всего тайлов с данными: {len(counts)}")
        print()

        grid_counts = count_polygons_intersecting_grid_cells(
            gdf_aoi, grid_deg=grid_deg, half_span_deg=half_span_deg
        )
        if grid_counts:
            gc, gn = grid_counts.most_common(1)[0]
            print(
                f"  Справочно: сетка {grid_deg}° / компактный AOI — плотнейшая ячейка ({gc[0]},{gc[1]}), "
                f"полигонов в ячейке: {gn} (это не тайл Sentinel).\n"
            )

    print(
        "Шаблон для тайла из таблицы:\n"
        "  uv run python ftw_oneshot.py --years <год> --tile <MGRS> --run\n"
        "  uv run python validate_iou.py --year <год> --manifest outputs/ftw/<год>_tile_<MGRS>/manifest.json\n"
        "Сетка ячейки (gi,gj) — только с --cell (см. --help)."
    )


def ftw_inference_device_args(cpu: bool) -> tuple[list[str], str, str]:
    """
    Аргументы для `ftw inference run`, человекочитаемая подпись и короткий код для manifest.
    На Apple Silicon по умолчанию включается MPS (Metal), не CUDA.
    """
    if cpu:
        return (["--gpu", "-1"], "CPU (принудительно)", "cpu")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                return (
                    ["--mps_mode"],
                    "Apple GPU — MPS (`--mps_mode` у ftw inference run)",
                    "mps",
                )
        except Exception:
            pass
        return (
            ["--gpu", "-1"],
            "CPU (на этом Mac MPS недоступен — проверьте macOS и PyTorch)",
            "cpu",
        )
    return (["--gpu", "0"], "NVIDIA CUDA, устройство 0", "cuda:0")


def inference_run_extra_args(cpu: bool) -> list[str]:
    return ftw_inference_device_args(cpu)[0]


def run_ftw(ftw: Path, args_list: list[str], cwd: Path, *, check: bool = True) -> int:
    """Запуск ftw. Возвращает exit code. При check=True кидает CalledProcessError если != 0."""
    cmd = [str(ftw)] + args_list

    def _disp(x: str) -> str:
        try:
            p = Path(x)
            if p.is_absolute():
                return _rel_to_project(p)
        except Exception:
            pass
        return x

    disp = [_disp(str(x)) for x in cmd]
    print("  →", " ".join(disp[:6]), "..." if len(disp) > 6 else "")
    r = subprocess.run(cmd, cwd=cwd, check=check, env=os.environ.copy())
    return r.returncode


def gpkg_set_layer_name(gpkg: Path, layer_name: str) -> None:
    """Один слой в GPKG — записать под именем layer_name (удобно в QGIS)."""
    import geopandas as gpd

    gdf = gpd.read_file(gpkg)
    tmp = gpkg.with_suffix(".gpkg.tmp")
    if tmp.exists():
        tmp.unlink()
    gdf.to_file(tmp, driver="GPKG", layer=layer_name)
    gpkg.unlink()
    tmp.rename(gpkg)


def label_ftw_gpkg(
    gpkg: Path,
    *,
    stem: str,
    year: int,
    csv_path: Path,
    win_a: str,
    win_b: str,
    tile: str | None,
    bbox: tuple[float, float, float, float],
    model: str,
) -> Path:
    try:
        import geopandas as gpd
    except ImportError:
        print("  (geopandas недоступен — пропуск разметки колонок)", file=sys.stderr)
        return gpkg

    gdf = gpd.read_file(gpkg)
    gdf["fields_reference_year"] = year
    gdf["fields_reference_csv"] = _rel_to_project(csv_path.resolve())
    gdf["fields_sentinel_win_a"] = win_a
    gdf["fields_sentinel_win_b"] = win_b
    gdf["fields_mgrs_tile"] = tile or ""
    gdf["fields_aoi_bbox_wgs84"] = json.dumps(
        {"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]}
    )
    gdf["fields_ftw_model"] = model
    gdf["fields_pipeline"] = "ftw_oneshot"
    out = gpkg.parent / f"{gpkg.stem}_labeled.gpkg"
    gdf.to_file(out, driver="GPKG", layer=f"{stem}_ftw_boundaries_labeled")
    return out


def win_dates(year: int, spec: str) -> tuple[str, str]:
    a, b = spec.split("/", 1)
    sm, sd = a.strip().split("-")
    em, ed = b.strip().split("-")
    return f"{year}-{sm}-{sd}", f"{year}-{em}-{ed}"


def unique_kml_source_names_in_aoi(
    csv_path: Path,
    year: int,
    aoi: tuple[float, float, float, float],
) -> tuple[list[str], int, list[str]]:
    """Пересечение эталона с AOI: культуры, число полигонов, уникальные имена файлов из source_file."""
    from shapely.geometry import box

    w, s, e, n = aoi
    aoi_poly = box(w, s, e, n)
    ref_all = load_reference_geodataframe_year(csv_path, year)
    ref_aoi = ref_all[ref_all.geometry.intersects(aoi_poly)].copy()
    cultures = sorted({str(c).strip() for c in ref_aoi["culture"] if str(c).strip()})
    seen: set[str] = set()
    names: list[str] = []
    for sf in ref_aoi["source_file"]:
        raw = str(sf).strip()
        if not raw:
            continue
        name = Path(raw).name
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    names.sort()
    return cultures, len(ref_aoi), names


def write_report_md(
    path: Path,
    *,
    year: int,
    stem: str,
    yrdir_rel: str,
    aoi_mode: str,
    mgrs_tile: str | None,
    cell: tuple[int, int],
    cell_count: int,
    bbox_download_line: str,
    main_bbox_line: str,
    mgrs_fill_ratio: float | None,
    id_a: str,
    id_b: str,
    ta: str | None,
    tb: str | None,
    window_a: str,
    window_b: str,
    ca: float | None,
    cb: float | None,
    model: str,
    inference_note: str,
    csv_rel: str,
    cultures_line: str,
    kml_qgis_block: str,
    ran_ftw: bool,
    tag_note: str = "",
) -> None:
    fr = f"{mgrs_fill_ratio:.1%}" if mgrs_fill_ratio is not None else "—"
    ca_s = f"{ca:.1f}%" if ca is not None else "—"
    cb_s = f"{cb:.1f}%" if cb is not None else "—"
    vec = f"{stem}_ftw_boundaries.gpkg"
    stk = f"{stem}_stack_8band.tif"
    infn = f"{stem}_inference.tif"
    if aoi_mode == "mgrs_tile":
        tile_line = (
            f"**Тайл MGRS:** `{mgrs_tile}` — полигонов эталона на тайле: **{cell_count}**, fill **{fr}**."
        )
    else:
        tile_line = (
            f"**Ячейка сетки:** ({cell[0]},{cell[1]}) — полигонов эталона в AOI: **{cell_count}**."
        )
    status = "Инференс уже выполнен (`--run`)." if ran_ftw else "Без `--run` только manifest и этот отчёт."
    if aoi_mode == "mgrs_tile":
        scenes_title = "Снимки Sentinel-2 (оба — **с выбранным MGRS тайла**, не случайный тайл)"
    else:
        scenes_title = "Снимки Sentinel-2 (в bbox ячейки сетки)"
    path.write_text(
        f"""# Oneshot {year}

{tile_line}
**Bbox скачивания (WGS84):** `{bbox_download_line}` — по нему же метрики в `validate_iou`.
**main_bbox года (перц. 5–95%):** `{main_bbox_line}`{tag_note}

## {scenes_title}

| | Окно | Scene id | Облачность | MGRS |
|--|------|----------|------------|------|
| A | {window_a} | `{id_a}` | {ca_s} | {ta or "—"} |
| B | {window_b} | `{id_b}` | {cb_s} | {tb or "—"} |

## Модель и устройство

- **Модель:** `{model}` (локальный .ckpt в репозитории, если есть — см. код по умолчанию).
- **Инференс:** {inference_note}

## Эталон и QGIS

- CSV: `{csv_rel}`
- KML в AOI: {kml_qgis_block}
- Культуры: {cultures_line}

## Файлы в `{yrdir_rel}/`

- `manifest.json` — для `validate_iou.py`
- `{stk}`, `{infn}`, `{vec}` — после `--run`

## Метрики

```bash
uv run python validate_iou.py --year {year} --manifest {yrdir_rel}/manifest.json
```

{status}
""",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Oneshot FTW по CSV полей: автоматически год + тайл MGRS, "
            "где есть полигоны эталона (пересечение main_bbox и ячейки MGRS 100×100 км), "
            "два окна Sentinel-2, опционально скачивание и модель. "
            "Один типовой запуск:  uv run python ftw_oneshot.py --years 2021 --run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Сеть нужна для STAC (кроме --dry-run / только --cell). "
            "На Mac с Apple Silicon инференс: MPS (--mps_mode); на Linux с NVIDIA: CUDA GPU 0."
        ),
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--years", type=str, default=None, help="2020,2021 или пусто = все годы")
    ap.add_argument("--grid-deg", type=float, default=0.35)
    ap.add_argument("--half-span-deg", type=float, default=0.06)
    ap.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Доп. суффикс к имени папки при том же годе и ячейке (напр. второй эксперимент): "
        "outputs/ftw/<год>_tile_<MGRS>_<тег>/ или cell_<gi>_<gj>_<тег>/",
    )
    ap.add_argument("--window-a", type=str, default="04-15/06-20")
    ap.add_argument("--window-b", type=str, default="07-01/09-15")
    ap.add_argument("--cloud-lt", type=float, default=25.0)
    ap.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="STAC: сколько сцен перебрать на каждое окно дат (нужно, чтобы найти сцены с нужным MGRS).",
    )
    ap.add_argument(
        "--discovery-max-items",
        type=int,
        default=600,
        help="STAC: сколько сцен перебрать при сборе тайлов MGRS (меньше — быстрее; при нехватке тайлов увеличьте).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Имя из `ftw model list` или путь к .ckpt. По умолчанию: "
        "ftw-baselines/prue_efnetb7_ccby_checkpoint.ckpt если файл есть, иначе FTW_PRUE_EFNET_B5.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT,
        help="Корень для outputs/ftw/<год>_tile_<MGRS>/ или <год>_cell_<gi>_<gj>/",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Только bbox и команды, без STAC",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="Выполнить ftw download / run / polygonize в этом же окружении",
    )
    ap.add_argument(
        "--no-label-parquet",
        action="store_true",
        help="Не писать ftw_boundaries_labeled.gpkg с колонками fields_*",
    )
    ap.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Ячейка сетки (gi,gj): компактный AOI как раньше; эталон — пересечение с AOI (не тайл Sentinel).",
    )
    ap.add_argument(
        "--tile",
        type=str,
        default=None,
        help="Тайл MGRS Sentinel-2 (как в STAC s2:mgrs_tile), вместо плотнейшего; эталон — main_bbox ∩ номинальная ячейка MGRS 100×100 км.",
    )
    ap.add_argument(
        "--list-cells",
        action="store_true",
        help="Таблица тайлов MGRS (эталон в main_bbox ∩ ячейка 100×100 км) и соседи; нужен STAC (сеть).",
    )
    args = ap.parse_args()

    if args.model is None:
        _ckpt = PROJECT_ROOT / "ftw-baselines" / "prue_efnetb7_ccby_checkpoint.ckpt"
        args.model = str(_ckpt.resolve()) if _ckpt.is_file() else "FTW_PRUE_EFNET_B5"

    run_tag = re.sub(r"[^\w\-]+", "_", (args.run_tag or "").strip())
    run_tag = run_tag[:64].strip("_") if run_tag else ""

    csv_path = args.csv if args.csv is not None else default_polygons_csv()
    if not csv_path.is_file():
        print(f"Нет CSV: {_rel_to_project(csv_path.resolve())}", file=sys.stderr)
        return 1

    by_year, _skipped = load_polygon_boxes_by_year(csv_path)
    if not by_year:
        print("Нет полигонов с годом", file=sys.stderr)
        return 1

    years = sorted(by_year.keys())
    if args.years:
        want = {int(x.strip()) for x in args.years.split(",") if x.strip()}
        years = [y for y in years if y in want]
    if not years:
        print("Нет подходящих годов", file=sys.stderr)
        return 1

    if args.cell and args.tile:
        print("Укажите только --cell или только --tile, не оба.", file=sys.stderr)
        return 1

    manual_cell: tuple[int, int] | None = None
    if args.cell:
        try:
            manual_cell = parse_cell_arg(args.cell)
        except ValueError as ex:
            print(f"--cell: {ex}", file=sys.stderr)
            return 1

    manual_tile: str | None = None
    if args.tile:
        manual_tile = normalize_mgrs_tile(args.tile)
        if not manual_tile:
            print("--tile: пустой тайл", file=sys.stderr)
            return 1

    if args.list_cells:
        catalog = pystac_client.Client.open(PC_STAC)
        print_list_cells(
            csv_path,
            by_year,
            years,
            catalog=catalog,
            grid_deg=args.grid_deg,
            half_span_deg=args.half_span_deg,
            cloud_lt=args.cloud_lt,
            max_items=args.discovery_max_items,
        )
        return 0

    catalog = None
    if not args.dry_run:
        catalog = pystac_client.Client.open(PC_STAC)

    ftw_exe = resolve_ftw_executable()
    if args.run and not ftw_exe:
        print(
            "Команда `ftw` не найдена. Выполните в корне проекта: uv sync",
            file=sys.stderr,
        )
        return 1

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run and manual_tile:
        print("--tile требует STAC; уберите --dry-run.", file=sys.stderr)
        return 1

    completed_dirs: list[tuple[int, str, str]] = []

    for year in years:
        boxes = by_year[year]
        gdf = load_reference_geodataframe_year(csv_path, year)
        if gdf.empty:
            print(f"### {year}: нет полигонов в CSV", file=sys.stderr)
            return 1
        main = bbox_from_percentiles(boxes, 5.0, 95.0, 0.02)
        gdf_aoi = filter_reference_gdf_to_main_bbox(gdf, main)
        if gdf_aoi.empty:
            gdf_aoi = gdf
        grid_counts = count_polygons_intersecting_grid_cells(
            gdf_aoi, grid_deg=args.grid_deg, half_span_deg=args.half_span_deg
        )
        if grid_counts:
            dense_cell_g, dense_n_g = grid_counts.most_common(1)[0]
        else:
            dense_cell_g, dense_n_g = (0, 0), 0

        union_bounds = list(gdf.total_bounds)
        margin = 0.35
        bbox_pc = [
            union_bounds[0] - margin,
            union_bounds[1] - margin,
            union_bounds[2] + margin,
            union_bounds[3] + margin,
        ]
        ds, de = f"{year}-01-01", f"{year}-12-31"

        tile_geoms_stac: dict[str, object] = {}
        tile_geoms: dict[str, object] = {}
        aoi_mode = "grid_cell"
        cell_selection = "densest"
        selected_mgrs: str | None = None
        mgrs_fill_ratio: float | None = None
        stem: str = ""
        cell: tuple[int, int] = (dense_cell_g[0], dense_cell_g[1])
        cell_count = 0
        w = s = e = n = 0.0
        tight = (0.0, 0.0, 0.0, 0.0)

        if manual_cell is not None:
            gi, gj = manual_cell
            c_here = grid_counts.get((gi, gj), 0)
            if c_here == 0:
                print(
                    f"### {year}: в ячейке {manual_cell} нет эталона "
                    "(пересечение с компактным AOI ячейки). "
                    f"См. uv run python ftw_oneshot.py --list-cells",
                    file=sys.stderr,
                )
                return 1
            w, s, e, n = bbox_for_grid_cell(
                gi,
                gj,
                grid_deg=args.grid_deg,
                half_span_deg=args.half_span_deg,
            )
            tight = (w, s, e, n)
            cell = (gi, gj)
            cell_count = c_here
            cell_selection = "manual_cell"
            aoi_mode = "grid_cell"
            stem = output_file_stem(year, run_tag, grid_cell=cell)
            if catalog is not None:
                tile_geoms_stac = discover_mgrs_tile_geometries(
                    catalog,
                    bbox_pc,
                    ds,
                    de,
                    args.cloud_lt,
                    args.discovery_max_items,
                )
                tile_geoms = (
                    nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
                    if tile_geoms_stac
                    else {}
                )
        elif manual_tile is not None:
            assert catalog is not None
            tile_geoms_stac = discover_mgrs_tile_geometries(
                catalog,
                bbox_pc,
                ds,
                de,
                args.cloud_lt,
                args.discovery_max_items,
            )
            if manual_tile not in tile_geoms_stac and tile_geoms_stac:
                tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
                    catalog,
                    tile_geoms_stac,
                    next(iter(tile_geoms_stac)),
                    ds,
                    de,
                    args.cloud_lt,
                )
            if manual_tile not in tile_geoms_stac:
                margin2 = 2.0
                bbox_wide = [
                    union_bounds[0] - margin2,
                    union_bounds[1] - margin2,
                    union_bounds[2] + margin2,
                    union_bounds[3] + margin2,
                ]
                tile_geoms_stac = discover_mgrs_tile_geometries(
                    catalog,
                    bbox_wide,
                    ds,
                    de,
                    args.cloud_lt,
                    args.discovery_max_items * 2,
                )
            if manual_tile not in tile_geoms_stac:
                print(
                    f"### {year}: тайл MGRS «{manual_tile}» не найден в STAC для bbox эталона.",
                    file=sys.stderr,
                )
                return 1
            tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
                catalog, tile_geoms_stac, manual_tile, ds, de, args.cloud_lt
            )
            tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
            counts = count_polygons_per_mgrs_tile(gdf_aoi, tile_geoms)
            if counts.get(manual_tile, 0) == 0:
                print(
                    f"### {year}: эталон в main_bbox не пересекает номинальную ячейку MGRS {manual_tile}.",
                    file=sys.stderr,
                )
                return 1
            geom = tile_geoms[manual_tile]
            _, mgrs_fill_ratio = reference_count_and_fill_on_tile(gdf_aoi, geom)
            w, s, e, n = download_bbox_around_reference_on_tile(
                gdf_aoi, geom, max(0.02, args.half_span_deg)
            )
            tight = (w, s, e, n)
            cell_count = counts.get(manual_tile, 0)
            selected_mgrs = manual_tile
            aoi_mode = "mgrs_tile"
            cell_selection = "manual_tile"
            stem = output_file_stem(year, run_tag, mgrs_tile=manual_tile)
        elif catalog is None or args.dry_run:
            w, s, e, n, cell, cell_count = densest_grid_cell_bbox(
                gdf_aoi, grid_deg=args.grid_deg, half_span_deg=args.half_span_deg
            )
            tight = (w, s, e, n)
            cell_selection = "densest_grid_cell"
            aoi_mode = "grid_cell"
            stem = output_file_stem(year, run_tag, grid_cell=cell)
            if args.dry_run:
                print(
                    "    (--dry-run: AOI по плотнейшей ячейке сетки; "
                    "для режима тайла MGRS запустите без --dry-run)",
                    file=sys.stderr,
                )
        else:
            tile_geoms_stac = discover_mgrs_tile_geometries(
                catalog,
                bbox_pc,
                ds,
                de,
                args.cloud_lt,
                args.discovery_max_items,
            )
            if not tile_geoms_stac:
                print(
                    f"### {year}: нет сцен Sentinel-2 в PC для bbox эталона.",
                    file=sys.stderr,
                )
                return 1
            tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
            counts, fills = mgrs_tile_metrics(gdf_aoi, tile_geoms)
            if max(counts.values()) == 0:
                print(
                    f"### {year}: эталон в main_bbox не пересекает номинальную ячейку MGRS "
                    f"ни одного тайла из выборки STAC.",
                    file=sys.stderr,
                )
                return 1
            dense_tid = pick_densest_tile_id(counts, fills)
            tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
                catalog, tile_geoms_stac, dense_tid, ds, de, args.cloud_lt
            )
            tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
            counts, fills = mgrs_tile_metrics(gdf_aoi, tile_geoms)
            dense_tid = pick_densest_tile_id(counts, fills)
            mgrs_fill_ratio = fills[dense_tid]
            geom = tile_geoms[dense_tid]
            w, s, e, n = download_bbox_around_reference_on_tile(
                gdf_aoi, geom, max(0.02, args.half_span_deg)
            )
            tight = (w, s, e, n)
            cell_count = counts[dense_tid]
            selected_mgrs = dense_tid
            aoi_mode = "mgrs_tile"
            cell_selection = "densest_mgrs_tile"
            stem = output_file_stem(year, run_tag, mgrs_tile=dense_tid)

        bbox_list = [w, s, e, n]
        adjacent_manifest: list[dict] = []

        mgrs_for_scenes = (
            selected_mgrs if (aoi_mode == "mgrs_tile" and selected_mgrs) else None
        )
        strict_tile_scenes = bool(mgrs_for_scenes)

        win_a = win_b = None
        if catalog is not None:
            a0, a1 = win_dates(year, args.window_a)
            b0, b1 = win_dates(year, args.window_b)
            win_a, win_b = search_scene_pair(
                catalog,
                bbox_list,
                date_a=(a0, a1),
                date_b=(b0, b1),
                cloud_lt=args.cloud_lt,
                max_items=args.max_items,
                prefer_mgrs=mgrs_for_scenes,
                require_mgrs_match=strict_tile_scenes,
            )
            if win_a and win_b:
                orbit_a = _orbit(win_a)
                orbit_b = _orbit(win_b)
                orb_info = ""
                if orbit_a:
                    orb_info += f"  orbit=R{orbit_a:03d}"
                if orbit_b and orbit_b != orbit_a:
                    orb_info += f" / R{orbit_b:03d}"
                    print(
                        f"    (!) Орбиты различаются — footprints проверены, но возможны артефакты на краях.",
                        file=sys.stderr,
                    )
                print(f"    Выбрана пара сцен{orb_info}, footprints пересекаются.")
            if strict_tile_scenes and (win_a is None or win_b is None):
                print(
                    f"    ! Нет пары сцен с MGRS {mgrs_for_scenes} в этом bbox за окна "
                    f"{args.window_a} / {args.window_b} (облачность <{args.cloud_lt}%, "
                    f"до {args.max_items} кандидатов на окно). "
                    "Увеличьте --max-items или --cloud-lt.",
                    file=sys.stderr,
                )

        data_bbox: tuple[float, float, float, float] | None = None
        n_ref_in_data = 0
        if win_a is None or win_b is None:
            if args.dry_run:
                print(
                    "    (--dry-run: STAC не вызывался; в manifest — плейсхолдеры id сцен)"
                )
            else:
                print(
                    "    ! Нет сцен в PC (сеть, облачность или сузить --cloud-lt)."
                )
            id_a = "S2X_MSIL2A_..._TxxXXX_..."
            id_b = "S2X_MSIL2A_..._TxxXXX_..."
            ca = cb = None
            ta = tb = None
        else:
            id_a = win_a.id
            id_b = win_b.id
            ca = _cloud(win_a)
            cb = _cloud(win_b)
            ta = _mgrs(win_a)
            tb = _mgrs(win_b)
            print(f"    win_a: {id_a}  cloud≈{ca:.1f}%  tile={ta}")
            print(f"    win_b: {id_b}  cloud≈{cb:.1f}%  tile={tb}")

            data_bbox = footprint_intersection_bbox(win_a, win_b)
            if data_bbox is not None:
                n_ref_in_data, _ = count_reference_in_footprint(gdf_aoi, win_a, win_b)
                dw, ds_, de_, dn = data_bbox
                print(
                    f"    Пересечение footprints: {dw:.5f},{ds_:.5f},{de_:.5f},{dn:.5f}"
                    f"  ({de_ - dw:.2f}° × {dn - ds_:.2f}°)"
                )
                print(f"    Эталонных полигонов в зоне данных: {n_ref_in_data}")
                tight = data_bbox
                w, s, e, n = data_bbox
                bbox_list = [w, s, e, n]
                if n_ref_in_data == 0:
                    print(
                        "    (!) В зоне пересечения footprints нет эталонных полигонов.\n"
                        "    Попробуйте другой тайл (--tile) или расширьте окна дат.",
                        file=sys.stderr,
                    )
            else:
                print(
                    "    (!) Footprints сцен не пересекаются — bbox скачивания остаётся по полигонам.",
                    file=sys.stderr,
                )
        print()

        if catalog is not None:
            if win_a:
                tile_geoms_stac = _merge_item_tile_geom(tile_geoms_stac, win_a)
            if win_b:
                tile_geoms_stac = _merge_item_tile_geom(tile_geoms_stac, win_b)
        if tile_geoms_stac:
            tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
        neighbor_center: str | None = None
        if aoi_mode == "mgrs_tile" and selected_mgrs:
            neighbor_center = selected_mgrs
        elif ta:
            neighbor_center = normalize_mgrs_tile(ta)
        if catalog is not None and neighbor_center and tile_geoms_stac:
            if neighbor_center not in tile_geoms_stac:
                tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
                    catalog,
                    tile_geoms_stac,
                    next(iter(tile_geoms_stac)),
                    ds,
                    de,
                    args.cloud_lt,
                )
            if neighbor_center in tile_geoms_stac:
                tile_geoms_stac = augment_mgrs_tiles_from_cardinal_searches(
                    catalog, tile_geoms_stac, neighbor_center, ds, de, args.cloud_lt
                )
                tile_geoms = nominal_tile_geoms_from_stac_keys(tile_geoms_stac)
                neigh = neighbor_mgrs_tiles_with_counts(
                    tile_geoms, neighbor_center, gdf_aoi
                )
                neigh = [(t, n) for t, n in neigh if n > 0]
                adjacent_manifest = adjacent_mgrs_manifest(
                    neigh, tile_geoms, gdf_aoi, args.half_span_deg
                )

        yrdir = out_root / stem
        yrdir.mkdir(parents=True, exist_ok=True)
        completed_dirs.append((year, _rel_to_project(yrdir.resolve()), stem))

        cultures: list[str] = []
        cultures_line = "—"
        n_in_aoi = 0
        kml_names: list[str] = []
        try:
            cultures, n_in_aoi, kml_names = unique_kml_source_names_in_aoi(
                csv_path, year, tight
            )
            cultures_line = ", ".join(cultures) if cultures else "—"
            print(
                f"    Эталон в AOI: {n_in_aoi} полигонов, уникальных KML: {len(kml_names)}"
            )
        except Exception as ex:
            print(f"    ! Не удалось сформировать список имён KML: {ex}", file=sys.stderr)

        stack = yrdir / f"{stem}_stack_8band.tif"
        inf = yrdir / f"{stem}_inference.tif"
        vec_gpkg = yrdir / f"{stem}_ftw_boundaries.gpkg"
        labeled_rel = (yrdir / f"{stem}_ftw_boundaries_labeled.gpkg").relative_to(
            PROJECT_ROOT
        )
        bbox_arg = fmt_bbox(tight)

        _ftw_dev_args, _ftw_dev_human, _ftw_dev_code = ftw_inference_device_args(
            False
        )

        manifest = {
            "schema": "fields.ftw_oneshot.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "year": year,
            "reference_polygons_csv": _rel_to_project(csv_path.resolve()),
            "reference_polygon_count": len(boxes),
            "aoi_mode": aoi_mode,
            "mgrs_tile": selected_mgrs,
            "reference_polygon_count_on_mgrs_tile": cell_count
            if aoi_mode == "mgrs_tile"
            else None,
            "mgrs_tile_fill_ratio": mgrs_fill_ratio
            if aoi_mode == "mgrs_tile"
            else None,
            "densest_mgrs_selection": "fill_ratio_then_polygon_count",
            "dense_grid_cell_index": [dense_cell_g[0], dense_cell_g[1]],
            "densest_polygon_count_in_grid_cell": dense_n_g,
            "aoi_grid_cell_index": [cell[0], cell[1]]
            if aoi_mode == "grid_cell"
            else None,
            "polygon_count_in_selected_aoi_cell": cell_count
            if aoi_mode == "grid_cell"
            else None,
            "cell_selection": cell_selection,
            "adjacent_mgrs_tiles": adjacent_manifest,
            "grid_deg": args.grid_deg,
            "run_tag": run_tag or None,
            "aoi_bbox_wgs84": [w, s, e, n],
            "footprint_intersection_bbox_wgs84": list(data_bbox) if data_bbox else None,
            "reference_polygons_in_data_area": n_ref_in_data if data_bbox else None,
            "main_bbox_wgs84": [main[0], main[1], main[2], main[3]],
            "sentinel": {
                "win_a": id_a,
                "win_b": id_b,
                "mgrs_tile_a": ta,
                "mgrs_tile_b": tb,
                "cloud_cover_a": ca,
                "cloud_cover_b": cb,
            },
            "model": args.model,
            "inference_device": _ftw_dev_code,
            "inference_device_note": _ftw_dev_human,
            "cultures_in_aoi": cultures,
            "reference_polygons_intersecting_aoi": n_in_aoi,
            "kml_source_filenames": kml_names,
            "output_file_stem": stem,
            "outputs": {
                "stack_8band": str(stack.relative_to(PROJECT_ROOT)),
                "inference": str(inf.relative_to(PROJECT_ROOT)),
                "ftw_boundaries_gpkg": str(vec_gpkg.relative_to(PROJECT_ROOT)),
                "labeled_gpkg": str(labeled_rel),
                "stack_8band.tif": str(stack.relative_to(PROJECT_ROOT)),
                "inference.tif": str(inf.relative_to(PROJECT_ROOT)),
                "ftw_boundaries.gpkg": str(vec_gpkg.relative_to(PROJECT_ROOT)),
            },
        }

        manifest_path = yrdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        yrdir_rel = _rel_to_project(yrdir.resolve())
        summary_path = yrdir / "REPORT.md"
        csv_rel = _rel_to_project(csv_path.resolve())
        if kml_names:
            kml_qgis_block = "\n".join(f"- `{p}`" for p in kml_names)
        else:
            kml_qgis_block = "— (нет полигонов эталона в AOI или не удалось сформировать список)"

        ran_ftw = False
        if args.run and ftw_exe:
            print(f"    --- Запуск ftw ({year}) ---")
            print(f"    Инференс: {_ftw_dev_human}")
            dl_rc = run_ftw(
                ftw_exe,
                [
                    "inference",
                    "download",
                    f"--win_a={id_a}",
                    f"--win_b={id_b}",
                    "-o",
                    _ftw_path_arg(stack),
                    f"--bbox={bbox_arg}",
                    "-f",
                ],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if dl_rc != 0 or not stack.is_file():
                print(
                    f"    (!) ftw inference download завершился с кодом {dl_rc}.",
                    file=sys.stderr,
                )
                if not stack.is_file():
                    print(
                        f"    Стек не создан: {_rel_to_project(stack)}.\n"
                        "    Вероятная причина: «The provided images do not intersect» — "
                        "два снимка (win_a, win_b) не перекрываются в заданном bbox.\n"
                        "    Попробуйте другой тайл (--tile) или расширьте --cloud-lt / --max-items,\n"
                        "    чтобы подобрать пару сцен с одной орбиты.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"    Стек создан ({_rel_to_project(stack)}), но download вернул ошибку. "
                        "Проверьте вывод выше.",
                        file=sys.stderr,
                    )
                    print("    Пропуск inference run / polygonize для этого года.", file=sys.stderr)
            else:
                run_ftw(
                    ftw_exe,
                    [
                        "inference",
                        "run",
                        _ftw_path_arg(stack),
                        "-m",
                        args.model,
                        "-o",
                        _ftw_path_arg(inf),
                        *inference_run_extra_args(False),
                        "-f",
                    ],
                    cwd=PROJECT_ROOT,
                )
                run_ftw(
                    ftw_exe,
                    [
                        "inference",
                        "polygonize",
                        _ftw_path_arg(inf),
                        "-o",
                        _ftw_path_arg(vec_gpkg),
                        "-f",
                    ],
                    cwd=PROJECT_ROOT,
                )
                ran_ftw = True
                if vec_gpkg.is_file():
                    try:
                        gpkg_set_layer_name(vec_gpkg, f"{stem}_ftw_boundaries")
                    except Exception as ex:
                        print(
                            f"    (!) Не удалось переименовать слой в GPKG (QGIS): {ex}",
                            file=sys.stderr,
                        )
                if not args.no_label_parquet and vec_gpkg.is_file():
                    labeled = label_ftw_gpkg(
                        vec_gpkg,
                        stem=stem,
                        year=year,
                        csv_path=csv_path,
                        win_a=id_a,
                        win_b=id_b,
                        tile=ta or tb,
                        bbox=tight,
                        model=args.model,
                    )
                    print(f"    Сохранено: {_rel_to_project(labeled.resolve())}")
            print(f"    manifest: {_rel_to_project(manifest_path)}")
        else:
            print(f"    Сохранено: {_rel_to_project(manifest_path)}")
            print(
                f"    Повтор с инференсом: uv run python ftw_oneshot.py --years {year} --run",
            )
            print()

        tag_note = ""
        if run_tag:
            tag_note = f"\n\nПапка: **`{yrdir_rel}`** (`--run-tag`)."

        write_report_md(
            summary_path,
            year=year,
            stem=stem,
            yrdir_rel=yrdir_rel,
            aoi_mode=aoi_mode,
            mgrs_tile=selected_mgrs,
            cell=cell,
            cell_count=cell_count,
            bbox_download_line=bbox_arg,
            main_bbox_line=fmt_bbox(main),
            mgrs_fill_ratio=mgrs_fill_ratio,
            id_a=id_a,
            id_b=id_b,
            ta=ta,
            tb=tb,
            window_a=args.window_a,
            window_b=args.window_b,
            ca=ca,
            cb=cb,
            model=args.model,
            inference_note=_ftw_dev_human,
            csv_rel=csv_rel,
            cultures_line=cultures_line,
            kml_qgis_block=kml_qgis_block,
            ran_ftw=ran_ftw,
            tag_note=tag_note,
        )
        print(f"    REPORT.md → {_rel_to_project(summary_path)}")

    print("Готово. Каталоги:")
    for _y, rel, _st in completed_dirs:
        print(f"  {rel}/  → manifest.json, REPORT.md")
    if args.run:
        for _y, rel, st in completed_dirs:
            print(
                f"    + растры/GPKG: {st}_stack_8band.tif, {st}_inference.tif, "
                f"{st}_ftw_boundaries.gpkg"
            )
    print("Метрики:")
    for y, rel, _st in completed_dirs:
        print(
            f"  uv run python validate_iou.py --year {y} --manifest {rel}/manifest.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
