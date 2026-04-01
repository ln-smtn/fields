"""Разбор WKT, bbox полигонов, центроид, компактный AOI вокруг самой плотной сетки."""

from __future__ import annotations

import functools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def default_polygons_csv() -> Path:
    """Канон: data/cultures_polygons_dataset.csv; корень и ftw-baselines — только для старых копий."""
    for p in (
        PROJECT_ROOT / "data" / "cultures_polygons_dataset.csv",
        PROJECT_ROOT / "cultures_polygons_dataset.csv",
        PROJECT_ROOT / "ftw-baselines" / "cultures_polygons_dataset.csv",
    ):
        if p.is_file():
            return p
    return PROJECT_ROOT / "data" / "cultures_polygons_dataset.csv"


def parse_wkt_polygon_ring(wkt: str) -> list[tuple[float, float]]:
    s = wkt.strip()
    if not s.upper().startswith("POLYGON"):
        raise ValueError("ожидался POLYGON")
    lo = s.index("((") + 2
    hi = s.rindex("))")
    inner = s[lo:hi]
    verts: list[tuple[float, float]] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        if len(bits) < 2:
            continue
        verts.append((float(bits[0]), float(bits[1])))
    if len(verts) < 3:
        raise ValueError("мало вершин")
    return verts


def ring_bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def ring_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(ring)
    if ring[0] == ring[-1]:
        n -= 1
    if n < 1:
        raise ValueError("пустое кольцо")
    sx = sum(ring[i][0] for i in range(n))
    sy = sum(ring[i][1] for i in range(n))
    return sx / n, sy / n


def percentile_linear(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = min(f + 1, n - 1)
    t = k - f
    return sorted_vals[f] + t * (sorted_vals[c] - sorted_vals[f])


def bbox_from_percentiles(
    boxes: list[tuple[float, float, float, float]],
    p_low: float,
    p_high: float,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    mins_lon = sorted(b[0] for b in boxes)
    mins_lat = sorted(b[1] for b in boxes)
    maxs_lon = sorted(b[2] for b in boxes)
    maxs_lat = sorted(b[3] for b in boxes)
    return (
        percentile_linear(mins_lon, p_low) - margin_deg,
        percentile_linear(mins_lat, p_low) - margin_deg,
        percentile_linear(maxs_lon, p_high) + margin_deg,
        percentile_linear(maxs_lat, p_high) + margin_deg,
    )


def polygons_fully_inside(
    boxes: list[tuple[float, float, float, float]],
    outer: tuple[float, float, float, float],
) -> float:
    w, s, e, n = outer
    ok = 0
    for bw, bs, be, bn in boxes:
        if bw >= w and be <= e and bs >= s and bn <= n:
            ok += 1
    return ok / len(boxes) if boxes else 0.0


def bbox_to_wkt(b: tuple[float, float, float, float]) -> str:
    w, s, e, n = b
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def load_polygon_boxes_by_year(
    csv_path: Path,
) -> tuple[dict[int, list[tuple[float, float, float, float]]], int]:
    import csv

    by_year: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    skipped = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "geometry_wkt" not in r.fieldnames:
            raise SystemExit("В CSV нет geometry_wkt")
        if "year" not in r.fieldnames:
            raise SystemExit("В CSV нет year")
        for row in r:
            y_raw = row.get("year")
            if y_raw is None or str(y_raw).strip() == "":
                skipped += 1
                continue
            try:
                year = int(str(y_raw).strip())
            except ValueError:
                skipped += 1
                continue
            wkt = (row.get("geometry_wkt") or "").strip()
            if not wkt:
                skipped += 1
                continue
            try:
                ring = parse_wkt_polygon_ring(wkt)
                by_year[year].append(ring_bbox(ring))
            except Exception:
                skipped += 1
                continue
    return dict(by_year), skipped


def load_reference_geodataframe_year(csv_path: Path, year: int):
    """Все полигоны за год из CSV с колонками row_id, culture, source_file (для QGIS)."""
    import csv

    from shapely import wkt as shp_wkt

    import geopandas as gpd

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "geometry_wkt" not in r.fieldnames:
            raise SystemExit("В CSV нет geometry_wkt")
        if "year" not in r.fieldnames:
            raise SystemExit("В CSV нет year")
        for row in r:
            y_raw = row.get("year")
            if y_raw is None or str(y_raw).strip() == "":
                continue
            try:
                yi = int(str(y_raw).strip())
            except ValueError:
                continue
            if yi != year:
                continue
            wkt = (row.get("geometry_wkt") or "").strip()
            if not wkt:
                continue
            try:
                geom = shp_wkt.loads(wkt)
            except Exception:
                continue
            rows.append(
                {
                    "row_id": (row.get("row_id") or "").strip(),
                    "culture": (row.get("culture") or "").strip(),
                    "source_file": (row.get("source_file") or "").strip(),
                    "geometry": geom,
                }
            )
    if not rows:
        return gpd.GeoDataFrame(
            columns=["row_id", "culture", "source_file", "geometry"], crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def normalize_mgrs_tile(s: str) -> str:
    """Как в STAC / имени сцены: T38TLR → 38TLR."""
    t = s.strip().upper()
    if t.startswith("T") and len(t) >= 4:
        t = t[1:]
    return t


@functools.lru_cache(maxsize=512)
def _mgrs_100km_cell_polygon_cached(key5: str):
    """Кэш по 5 символам (38TLR), чтобы не строить полигон сотни раз за один прогон."""
    import mgrs
    from pyproj import Transformer
    from shapely.geometry import Polygon

    m = mgrs.MGRS()
    zone, hem, e0, n0 = m.MGRSToUTM(key5)
    z = int(zone)
    epsg = 32600 + z if hem == "N" else 32700 + z
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    e1, n1 = e0 + 100_000.0, n0 + 100_000.0
    ring = [
        t.transform(e0, n0),
        t.transform(e1, n0),
        t.transform(e1, n1),
        t.transform(e0, n1),
        t.transform(e0, n0),
    ]
    return Polygon(ring)


def mgrs_100km_cell_polygon_wgs84(mgrs_id: str):
    """
    Номинальная граница ячейки MGRS 100×100 km (5 символов), WGS84.
    Совпадает с типичной сеткой MGRS в QGIS; не footprint сцены STAC.
    """
    key = normalize_mgrs_tile(mgrs_id)
    if len(key) < 5:
        raise ValueError(f"MGRS id слишком короткий: {mgrs_id!r}")
    return _mgrs_100km_cell_polygon_cached(key[:5])


def filter_reference_gdf_to_main_bbox(gdf, main: tuple[float, float, float, float]):
    """Полигоны эталона, пересекающие main_bbox (перцентили по bbox полей года)."""
    from shapely.geometry import box

    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    b = box(*main)
    return gdf[gdf.geometry.intersects(b)].copy()


def count_polygons_intersecting_grid_cells(
    gdf,
    *,
    grid_deg: float,
    half_span_deg: float,
) -> Counter[tuple[int, int]]:
    """Сколько полигонов эталона пересекает компактный AOI ячейки (gi,gj) — тот же bbox, что в прогоне."""
    from shapely.geometry import box

    if gdf is None or getattr(gdf, "empty", True):
        return Counter()
    b = gdf.total_bounds
    gi_min = int(b[0] // grid_deg) - 1
    gi_max = int(b[2] // grid_deg) + 1
    gj_min = int(b[1] // grid_deg) - 1
    gj_max = int(b[3] // grid_deg) + 1
    cell_counts: Counter[tuple[int, int]] = Counter()
    for gi in range(gi_min, gi_max + 1):
        for gj in range(gj_min, gj_max + 1):
            w, s, e, n = bbox_for_grid_cell(
                gi, gj, grid_deg=grid_deg, half_span_deg=half_span_deg
            )
            cell_poly = box(w, s, e, n)
            n_hit = int(gdf.geometry.intersects(cell_poly).sum())
            if n_hit > 0:
                cell_counts[(gi, gj)] = n_hit
    return cell_counts


def bbox_for_grid_cell(
    gi: int,
    gj: int,
    *,
    grid_deg: float,
    half_span_deg: float,
) -> tuple[float, float, float, float]:
    """Компактный AOI: центр ячейки (gi, gj) ± half_span_deg (как в densest_tile_bbox)."""
    center_lon = (gi + 0.5) * grid_deg
    center_lat = (gj + 0.5) * grid_deg
    w = center_lon - half_span_deg
    e = center_lon + half_span_deg
    s = center_lat - half_span_deg
    n = center_lat + half_span_deg
    return (w, s, e, n)


def densest_grid_cell_bbox(
    gdf,
    *,
    grid_deg: float,
    half_span_deg: float,
) -> tuple[float, float, float, float, tuple[int, int], int]:
    """
    Ячейка сетки с максимальным числом эталонных полигонов, пересекающих компактный AOI ячейки.
    Возвращает компактный квадрат ±half_span_deg вокруг центра ячейки.
    """
    if gdf is None or getattr(gdf, "empty", True):
        raise ValueError("нет полигонов")
    cell_counts = count_polygons_intersecting_grid_cells(
        gdf, grid_deg=grid_deg, half_span_deg=half_span_deg
    )
    if not cell_counts:
        raise ValueError("нет полигонов")
    (gi, gj), cnt = cell_counts.most_common(1)[0]
    w, s, e, n = bbox_for_grid_cell(
        gi, gj, grid_deg=grid_deg, half_span_deg=half_span_deg
    )
    return (w, s, e, n, (gi, gj), cnt)


def _bbox_dict(b: tuple[float, float, float, float]) -> dict:
    w, s, e, n = b
    return {"min_lon": w, "min_lat": s, "max_lon": e, "max_lat": n, "crs": "EPSG:4326"}


def cli_bbox_table(argv: list[str] | None = None) -> int:
    """Таблица main/full bbox по годам (без FTW). Запуск: python site_geo.py bbox"""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python site_geo.py bbox",
        description="Bbox по годам из cultures_polygons_dataset.csv",
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--p-low", type=float, default=5.0)
    ap.add_argument("--p-high", type=float, default=95.0)
    ap.add_argument("--margin-deg", type=float, default=0.02)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    csv_path = args.csv if args.csv is not None else default_polygons_csv()

    if not csv_path.is_file():
        print(f"Нет файла: {csv_path}", file=sys.stderr)
        return 1

    by_year, skipped = load_polygon_boxes_by_year(csv_path)
    if not by_year:
        print("Нет данных", file=sys.stderr)
        return 1

    yearly = []
    for year in sorted(by_year.keys()):
        boxes = by_year[year]
        full = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        main = bbox_from_percentiles(boxes, args.p_low, args.p_high, args.margin_deg)
        frac = polygons_fully_inside(boxes, main)
        yearly.append(
            {
                "year": year,
                "polygon_count": len(boxes),
                "full_bbox": _bbox_dict(full),
                "main_bbox": _bbox_dict(main),
                "main_wkt": bbox_to_wkt(main),
                "fraction_inside_main": round(frac, 4),
            }
        )

    if args.json:
        print(
            json.dumps(
                {"csv": str(csv_path), "skipped": skipped, "years": yearly},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"CSV: {csv_path}  пропусков: {skipped}\n")
    for y in yearly:
        print(f"=== {y['year']}  n={y['polygon_count']} ===")
        m = y["main_bbox"]
        print(
            f"  main: {m['min_lon']:.6f},{m['min_lat']:.6f},{m['max_lon']:.6f},{m['max_lat']:.6f}"
        )
    print("\nДальше: uv run python ftw_oneshot.py")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "bbox":
        return cli_bbox_table(sys.argv[2:])
    print("Использование: python site_geo.py bbox [--csv ПУТЬ] [--json]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
