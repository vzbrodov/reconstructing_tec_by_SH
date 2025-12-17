import numpy as np

def select_na_stations(coord_file, n_select=50,
                       lon_min=-150, lon_max=-40,
                       lat_min=5, lat_max=80,
                       excluded=None, seed=42):
    """
    coord_file: tec_suite_coords.txt
    n_select  : сколько станций выбрать
    lon_min..lon_max, lat_min..lat_max: окно по Северной Америке
    excluded  : множество уже исключённых имён (чтобы не повторяться)
    """
    if excluded is None:
        excluded = set()

    names = []
    lons = []
    lats = []

    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            name = parts[2]

            # фильтр по NA и по уже исключённым
            if (lon_min <= lon <= lon_max and
                lat_min <= lat <= lat_max and
                name not in excluded):
                names.append(name)
                lons.append(lon)
                lats.append(lat)

    names = np.array(names)
    lons = np.array(lons)
    lats = np.array(lats)

    print(f"Найдено {len(names)} станций в окне NA после исключения уже выкинутых.")

    if len(names) == 0:
        return []

    n_select = min(n_select, len(names))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(names), size=n_select, replace=False)

    selected = names[idx]

    print("\nВыбранные станции для удаления:")
    for name in selected:
        print(name)

    # удобно сразу сгенерировать кусок для EXCLUDED_STATIONS
    print("\nФрагмент для EXCLUDED_STATIONS = { ... }:")
    print("EXCLUDED_STATIONS = {")
    for name in selected:
        print(f"    '{name}',")
    print("}")

    return selected.tolist()


if __name__ == "__main__":
    # сюда можно подать уже имеющийся список исключённых станций
    already_excluded = {
        # 'DRAO', 'KOKB', ...
    }

    select_na_stations("tec_suite_coords.txt",
                       n_select=150,
                       lon_min=-150, lon_max=-40,
                       lat_min=5, lat_max=80,
                       excluded=already_excluded,
                       seed=42)
