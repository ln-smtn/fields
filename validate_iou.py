#!/usr/bin/env python3
"""
Сопоставление эталонных полигонов (CSV) с полигонами FTW (GeoPackage) внутри AOI oneshot.

Пишет CSV/JSON и эталон в AOI (**`reference_aoi.gpkg`** или **`<stem>_reference_aoi.gpkg`**). Полигоны FTW из manifest / **`*_ftw_boundaries.gpkg`**.

Запуск (после ftw_oneshot --run):
  uv run python validate_iou.py --year 2020
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely import wkt as shp_wkt

from site_geo import default_polygons_csv

PROJECT_ROOT = Path(__file__).resolve().parent


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.name


def _to_single_polygon(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return max(g.geoms, key=lambda x: x.area)
    return None


def geom_iou_m2(a, b) -> float:
    a = _to_single_polygon(a)
    b = _to_single_polygon(b)
    if a is None or b is None:
        return 0.0
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0
    ia = inter.area
    ua = a.area + b.area - ia
    return float(ia / ua) if ua > 1e-6 else 0.0


def load_reference_year(csv_path: Path, year: int) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "geometry_wkt" not in r.fieldnames:
            raise SystemExit("В CSV нет geometry_wkt")
        if "year" not in r.fieldnames:
            raise SystemExit("В CSV нет year")
        seq = 0
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
            geom = _to_single_polygon(geom)
            if geom is None:
                continue
            rid = (row.get("row_id") or "").strip()
            if not rid:
                rid = f"ref_{seq}"
            seq += 1
            rows.append({"reference_row_id": rid, "geometry": geom})
    if not rows:
        return gpd.GeoDataFrame(columns=["reference_row_id", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def greedy_match_iou(
    ref_m: gpd.GeoDataFrame,
    pred_m: gpd.GeoDataFrame,
    min_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Пары (idx_ref, idx_pred, iou), множества использованных индексов."""
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(ref_m)):
        gi = ref_m.geometry.iloc[i]
        for j in range(len(pred_m)):
            gj = pred_m.geometry.iloc[j]
            v = geom_iou_m2(gi, gj)
            if v >= min_iou:
                pairs.append((v, i, j))
    pairs.sort(key=lambda x: -x[0])
    used_r: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for v, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        out.append((i, j, v))
    all_r = set(range(len(ref_m)))
    all_p = set(range(len(pred_m)))
    return out, all_r - used_r, all_p - used_p


def _read_pred(path: Path) -> gpd.GeoDataFrame:
    suf = path.suffix.lower()
    if suf == ".gpkg":
        return gpd.read_file(path)
    if suf == ".parquet":
        return gpd.read_parquet(path)
    raise SystemExit(f"Неподдерживаемый формат: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="IoU эталон vs FTW внутри AOI oneshot")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Если не задан: единственный outputs/ftw/<год>_cell_*/ или <год>_tile_*/manifest.json",
    )
    ap.add_argument("--ref-csv", type=Path, default=None)
    ap.add_argument(
        "--pred",
        "--pred-parquet",
        type=Path,
        default=None,
        dest="pred",
        help="Полигоны FTW (.gpkg); по умолчанию путь из manifest.outputs или *_ftw_boundaries.gpkg",
    )
    ap.add_argument(
        "--min-iou",
        type=float,
        default=0.05,
        help="Минимальный IoU для пары «нашёлся матч»",
    )
    args = ap.parse_args()

    year = args.year
    ftw_out = PROJECT_ROOT / "outputs" / "ftw"
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        yrdir = manifest_path.parent
    else:
        legacy = ftw_out / str(year) / "manifest.json"
        matches = sorted(
            ftw_out.glob(f"{year}_cell_*/manifest.json")
        ) + sorted(ftw_out.glob(f"{year}_tile_*/manifest.json"))
        if len(matches) == 1:
            manifest_path = matches[0]
            yrdir = manifest_path.parent
        elif len(matches) > 1:
            print(
                f"Несколько прогонов за {year} (outputs/ftw/{year}_cell_*/ или {year}_tile_*/). "
                f"Укажите явно: --manifest путь/к/manifest.json",
                file=sys.stderr,
            )
            for p in matches:
                print(f"  {_rel(p)}", file=sys.stderr)
            return 1
        elif legacy.is_file():
            manifest_path = legacy
            yrdir = manifest_path.parent
        else:
            print(
                f"Нет manifest: ни {_rel(legacy)}, ни outputs/ftw/{year}_cell_*/ или {year}_tile_*/",
                file=sys.stderr,
            )
            print("  Запустите oneshot или укажите --manifest вручную.", file=sys.stderr)
            return 1
    if not manifest_path.is_file():
        print(f"Нет manifest: {_rel(manifest_path)}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref_csv = args.ref_csv or Path(manifest["reference_polygons_csv"])
    if not ref_csv.is_absolute():
        ref_csv = (PROJECT_ROOT / ref_csv).resolve()
    if not ref_csv.is_file():
        ref_csv = default_polygons_csv()
    if not ref_csv.is_file():
        print(f"Нет эталонного CSV: {_rel(ref_csv)}", file=sys.stderr)
        return 1

    outs = manifest.get("outputs") or {}
    stem = manifest.get("output_file_stem")
    pred_rel = (
        outs.get("ftw_boundaries_gpkg")
        or outs.get("ftw_boundaries.gpkg")
    )
    if pred_rel:
        pred_path = (PROJECT_ROOT / pred_rel).resolve()
    elif stem:
        cand = yrdir / f"{stem}_ftw_boundaries.gpkg"
        pred_path = cand if cand.is_file() else yrdir / "ftw_boundaries.gpkg"
    else:
        pred_path = yrdir / "ftw_boundaries.gpkg"
    if not pred_path.is_file():
        for alt in sorted(yrdir.glob("*_ftw_boundaries_labeled.gpkg")) + sorted(
            yrdir.glob("*_ftw_boundaries.gpkg")
        ):
            if alt.is_file():
                pred_path = alt
                break
        else:
            for alt in (
                yrdir / "ftw_boundaries_labeled.gpkg",
                yrdir / "ftw_boundaries.parquet",
                yrdir / "ftw_boundaries_labeled.parquet",
            ):
                if alt.is_file():
                    pred_path = alt
                    break
            else:
                print(
                    f"Нет файла полигонов FTW (gpkg/parquet): {_rel(yrdir)}",
                    file=sys.stderr,
                )
                return 1

    aoi = manifest.get("aoi_bbox_wgs84")
    if not aoi or len(aoi) != 4:
        print("В manifest нет aoi_bbox_wgs84", file=sys.stderr)
        return 1
    w, s, e, n = aoi
    aoi_poly = box(w, s, e, n)

    ref = load_reference_year(ref_csv, year)
    if ref.empty:
        print(f"Нет эталонных полигонов за {year} в {_rel(ref_csv)}", file=sys.stderr)
        return 1

    ref = ref[ref.geometry.intersects(aoi_poly)].copy()
    if ref.empty:
        print("После пересечения с AOI эталон пуст — проверьте год и bbox.", file=sys.stderr)
        return 1

    pred = _read_pred(pred_path)
    if "geometry" not in pred.columns:
        print("В слое FTW нет колонки geometry", file=sys.stderr)
        return 1
    pred = pred[pred.geometry.notna()].copy()
    pred["geometry"] = pred.geometry.map(_to_single_polygon)
    pred = pred[pred.geometry.notna()]
    # AOI в WGS84 (градусы); полигоны FTW часто в CRS растра (например UTM). Без to_crs пересечение с box в градусах даёт 0 объектов.
    if pred.crs is None:
        print(
            "У слоя FTW не задан CRS — для отбора по AOI принимается EPSG:4326",
            file=sys.stderr,
        )
        pred = pred.set_crs("EPSG:4326")
    pred_wgs = pred.to_crs("EPSG:4326")
    keep = pred_wgs.geometry.intersects(aoi_poly)
    pred = pred.loc[keep].copy()
    if pred.empty:
        print("В AOI нет полигонов FTW — сначала выполните oneshot с --run.", file=sys.stderr)
        return 1

    crs_m = "EPSG:3857"
    ref_m = ref.to_crs(crs_m)
    pred_m = pred.to_crs(crs_m)

    matches, unmatched_r, unmatched_p = greedy_match_iou(
        ref_m, pred_m, args.min_iou
    )

    rows_out = []
    ious = []
    for i_ref, j_pred, iou_v in matches:
        ious.append(iou_v)
        pred_idx_label = pred.index[j_pred]
        rows_out.append(
            {
                "reference_row_id": ref["reference_row_id"].iloc[i_ref],
                "ftw_parquet_row_index": pred_idx_label,
                "iou": round(iou_v, 6),
                "ref_area_m2": round(float(ref_m.geometry.iloc[i_ref].area), 2),
                "pred_area_m2": round(float(pred_m.geometry.iloc[j_pred].area), 2),
            }
        )

    mean_iou = sum(ious) / len(ious) if ious else 0.0

    summary = {
        "year": year,
        "reference_csv": _rel(ref_csv),
        "pred_vector": _rel(pred_path),
        "manifest": _rel(manifest_path),
        "aoi_bbox_wgs84": aoi,
        "reference_polygons_in_aoi": len(ref),
        "ftw_polygons_in_aoi": len(pred),
        "matched_pairs": len(matches),
        "unmatched_reference": len(unmatched_r),
        "unmatched_ftw": len(unmatched_p),
        "mean_iou_matched": round(mean_iou, 6),
        "min_iou_threshold": args.min_iou,
    }

    yrdir.mkdir(parents=True, exist_ok=True)
    if stem:
        out_csv = yrdir / f"{stem}_validation_iou_per_field.csv"
        out_json = yrdir / f"{stem}_validation_iou_summary.json"
        gpkg_ref = yrdir / f"{stem}_reference_aoi.gpkg"
    else:
        out_csv = yrdir / "validation_iou_per_field.csv"
        out_json = yrdir / "validation_iou_summary.json"
        gpkg_ref = yrdir / "reference_aoi.gpkg"

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ref.to_file(gpkg_ref, driver="GPKG", layer="reference")

    print("=" * 72)
    print(f"Валидация IoU — год {year}, AOI из manifest")
    print(f"  Эталон в AOI: {len(ref)} полигонов | FTW в AOI: {len(pred)}")
    print(f"  Пар с IoU ≥ {args.min_iou}: {len(matches)}")
    print(f"  Без пары (эталон): {len(unmatched_r)} | Без пары (FTW): {len(unmatched_p)}")
    print(f"  Средний IoU по найденным парам: {mean_iou:.4f}")
    print(f"  → {_rel(out_csv)}  |  {_rel(out_json)}")
    print(
        f"  QGIS (эталон в AOI): {_rel(gpkg_ref)}  |  модель: {_rel(pred_path)}"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
