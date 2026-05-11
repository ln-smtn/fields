"""
Проверка реального распределения полигонов по регионам и годам.

Запуск:
  uv run python tools/check_regions.py
"""
import pandas as pd
from shapely import wkt


def region_for(lat: float, lon: float) -> str:
    """Грубая привязка полигона к региону РФ по координатам центроида."""
    if 42 <= lat <= 48 and 35 <= lon <= 46:
        return "south"        # Юг РФ
    if 50 <= lat <= 55 and 34 <= lon <= 41:
        return "chernozem"    # Чернозёмье
    if 48 <= lat <= 55 and 40 <= lon <= 50:
        return "volga"        # Поволжье
    if 42 <= lat <= 46 and 130 <= lon <= 135:
        return "primorye"     # Приморье
    return "other"


def main():
    df = pd.read_csv("data/cultures_polygons_dataset.csv", encoding="utf-8-sig")
    df["geom"] = df["geometry_wkt"].apply(wkt.loads)
    df["lat"] = df["geom"].apply(lambda g: g.centroid.y)
    df["lon"] = df["geom"].apply(lambda g: g.centroid.x)

    print("Диапазон координат:")
    print(f"  Широта:  {df['lat'].min():.2f} — {df['lat'].max():.2f}")
    print(f"  Долгота: {df['lon'].min():.2f} — {df['lon'].max():.2f}")
    print()

    df["region"] = df.apply(lambda r: region_for(r["lat"], r["lon"]), axis=1)

    print("Распределение по регионам:")
    print(df["region"].value_counts())
    print()
    print("Регион × Год:")
    print(df.groupby(["region", "year"]).size().unstack(fill_value=0))
    print()
    print("Регион × Культура (топ-5):")
    print(df.groupby(["region", "culture"]).size().unstack(fill_value=0).T)


if __name__ == "__main__":
    main()
