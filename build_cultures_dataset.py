#!/usr/bin/env python3
"""
Собирает табличный датасет из KML в cultures_kmls/:
одна строка на полигон, координаты в WGS84 (EPSG:4326) как WKT POLYGON.

Имя файла: <культура>_<год>.kml — культура и год берутся отсюда.
Месяца в именах файлов нет; колонка month остаётся пустой (можно заполнить вручную).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

KML_NS = "http://www.opengis.net/kml/2.2"


def _tag(local: str) -> str:
    return f"{{{KML_NS}}}{local}"


FILENAME_RE = re.compile(r"^(.+)_(\d{4})$")


def parse_kml_filename(stem: str) -> tuple[str, int] | None:
    """Разбор «культура_2019» из имени файла без расширения."""
    m = FILENAME_RE.match(stem.strip())
    if not m:
        return None
    culture = m.group(1).strip()
    year = int(m.group(2))
    return culture, year


def coordinates_to_wkt_polygon(coords_text: str) -> str:
    """
    KML: пробел между вершинами, в вершине «lon,lat,alt».
    WKT POLYGON: одно кольцо, первая точка = последняя (в KML уже замкнуто).
    """
    raw = coords_text.strip().split()
    if len(raw) < 4:
        raise ValueError("слишком мало вершин в LinearRing")

    verts: list[tuple[float, float]] = []
    for token in raw:
        parts = token.split(",")
        if len(parts) < 2:
            continue
        lon, lat = float(parts[0]), float(parts[1])
        verts.append((lon, lat))

    if len(verts) < 4:
        raise ValueError("недостаточно вершин после разбора")

    # Замыкание для WKT
    if verts[0] != verts[-1]:
        verts.append(verts[0])

    ring = ", ".join(f"{lon} {lat}" for lon, lat in verts)
    return f"POLYGON(({ring}))"


def extract_polygons_from_placemark(pm: ET.Element) -> list[tuple[str | None, str]]:
    """
    Возвращает список (placemark_name, wkt) для каждого Polygon в Placemark.
    """
    name_el = pm.find(_tag("name"))
    pname = (name_el.text or "").strip() if name_el is not None else None

    out: list[tuple[str | None, str]] = []
    for poly in pm.findall(f".//{_tag('Polygon')}"):
        ring_el = poly.find(f"{_tag('outerBoundaryIs')}/{_tag('LinearRing')}/{_tag('coordinates')}")
        if ring_el is None or not (ring_el.text and ring_el.text.strip()):
            continue
        wkt = coordinates_to_wkt_polygon(ring_el.text)
        out.append((pname, wkt))
    return out


def parse_kml_file(path: Path) -> list[tuple[str | None, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[tuple[str | None, str]] = []
    for pm in root.iter(_tag("Placemark")):
        rows.extend(extract_polygons_from_placemark(pm))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="KML → таблица (один полигон = одна строка)")
    root = Path(__file__).resolve().parent
    data_kml = root / "data" / "cultures_kmls"
    legacy_kml = root / "cultures_kmls"
    default_kml = data_kml if data_kml.is_dir() else legacy_kml

    ap.add_argument(
        "--input-dir",
        type=Path,
        default=default_kml,
        help="Папка с .kml (по умолчанию data/cultures_kmls или cultures_kmls)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "cultures_polygons_dataset.csv",
        help="Выходной CSV (по умолчанию data/cultures_polygons_dataset.csv)",
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.is_dir():
        print(f"Нет папки: {args.input_dir}", file=sys.stderr)
        return 1

    kml_files = sorted(args.input_dir.glob("*.kml"))
    if not kml_files:
        print(f"В {args.input_dir} не найдено .kml", file=sys.stderr)
        return 1

    fieldnames = [
        "row_id",
        "source_file",
        "culture",
        "year",
        "month",
        "crs",
        "placemark_name",
        "geometry_wkt",
    ]

    n_ok = 0
    n_skip = 0
    row_id = 0

    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for kml_path in kml_files:
            parsed = parse_kml_filename(kml_path.stem)
            if parsed is None:
                print(f"Пропуск (имя не вида культура_ГГГГ): {kml_path.name}", file=sys.stderr)
                n_skip += 1
                continue
            culture, year = parsed

            try:
                polys = parse_kml_file(kml_path)
            except Exception as e:
                print(f"Ошибка {kml_path.name}: {e}", file=sys.stderr)
                n_skip += 1
                continue

            if not polys:
                print(f"Нет полигонов: {kml_path.name}", file=sys.stderr)
                n_skip += 1
                continue

            for placemark_name, wkt in polys:
                row_id += 1
                w.writerow(
                    {
                        "row_id": row_id,
                        "source_file": kml_path.name,
                        "culture": culture,
                        "year": year,
                        "month": "",
                        "crs": "EPSG:4326",
                        "placemark_name": placemark_name or "",
                        "geometry_wkt": wkt,
                    }
                )
            n_ok += 1

    print(
        f"Готово: {args.output} — файлов обработано: {n_ok}, "
        f"строк (полигонов): {row_id}, пропусков: {n_skip}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
