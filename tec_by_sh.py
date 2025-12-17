import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import lpmv
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import warnings

from PlotTECGlobal import read_ionex_fixed, create_tec_matrix

warnings.filterwarnings('ignore')

# ------- сюда впиши свои станции, которые надо выкинуть -------
EXCLUDED_STATIONS = {
    'STPM',    'MGO4',    'THU2',    'SCH2',    'SNI1',    'GODN',    'WSLB',    'NRC1',    'LMMF',
    'LPOC',
    'TFNO',    'ABMF',    'ALGO',    'VNDP',    'BCOV',    'NANO',    'QAQ1',    'BCRK',    'COSO',    'CMP9',    'PTRF',    'WHC1',    'SFDM',    'NLIB',    'NCSB',
    'GODE',       'GOLD',    'JEDY',    'WES2',    'SSIA',    'NIST',    'ALB4',    'GOL2',    'ROCK',    'ALBH',    'DIBT',    'MGRB',    'MGO6',
    'SGPO',    'CIT1',    'EIL3',    'YELL',    'QUIN',     'EIL4',
    'WIDC',    'SEUS',    'MGO2',    'BARH',    'GODS',    'PRD3',    'ATLI',    'P053',   'PTAL',    'AMC4',    'YEL3',    'TORP',
    'SHPB',    'QUAD',    'KOUG',    'IQAL',    'STFU',    'RDSD',    'CHUR',    'PICL',    'LBCH',    'CHWK',    'USN8',    'KLSQ',
    'GCGO',       'PRDS',    'DUBO',    'JPLM',
    'KSU1',    'MGO3',    'DR2O',    'CLRS',    'WILL',    'WDC6',    'GODR',    'STJ2',    'MRIB',    'HOLB',    'BILL',    'SC04',
    'INVK',    'RTS2',    'BREW',    'P043',    'UCLP',    'NTKA',    'P389',    'PRD2',    'BAIE',    'BAMF',    'AL2H',    'FAIR',    'CHU2',
    'WHIT',    'SABY',
    'PIE1',    'JORD',    'EUR2',    'UCLU',    'ESCU',    'TAHB',    'APO1',    'STJO',    'DRAO',    'ALG3',    'TRAK',
}


def load_station_coords(coord_file):
    """Загрузка координат станций"""
    stations = {}
    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                lon, lat, name = float(parts[0]), float(parts[1]), parts[2]
                stations[name] = (lon, lat)
    return stations


def load_vtec_data(data_dir, hour_start, hour_end, stations_coords,
                   exclude_stations=None):
    """
    Загрузка данных VTEC за указанный временной интервал.
    exclude_stations: множество имён станций, которые НЕ использовать.
    """
    if exclude_stations is None:
        exclude_stations = set()

    data = []

    for station_path in Path(data_dir).iterdir():
        if not station_path.is_dir():
            continue

        station_name = station_path.name
        if station_name not in stations_coords:
            continue
        if station_name in exclude_stations:
            # эту станцию специально выкидываем
            continue

        station_lon, station_lat = stations_coords[station_name]
        data_file = station_path / f"{station_name}_{data_dir}_2025.dat"
        if not data_file.exists():
            continue

        df = pd.read_csv(
            data_file,
            delim_whitespace=True,
            comment='#',
            names=['UT', 'I_v', 'G_lon', 'G_lat',
                   'G_q_lon', 'G_q_lat', 'G_t', 'G_q_t']
        )

        mask = (df['UT'] >= hour_start) & (df['UT'] <= hour_end) & (df['I_v'] > 0)
        df_filtered = df[mask]

        if len(df_filtered) > 0:
            avg_vtec = df_filtered['I_v'].mean()
            data.append([station_lon, station_lat, avg_vtec])

    return np.array(data)


def create_distance_weight_matrix(lats, lons, k_neighbors=5):
    """Весовая матрица по плотности станций"""
    colat = 90.0 - lats
    theta = np.radians(colat)
    phi = np.radians(lons)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords_3d = np.column_stack([x, y, z])

    tree = cKDTree(coords_3d)
    k = min(k_neighbors, max(1, len(lats) // 10))
    if k > 0:
        distances, _ = tree.query(coords_3d, k=k + 1)
        mean_distances = distances[:, 1:].mean(axis=1)
        weights = 1.0 / (mean_distances + 1e-10)
        weights = weights / weights.mean()
    else:
        weights = np.ones(len(lats))

    return weights


def create_vtec_map_with_ocean_constraint(data, lmax=5, ocean_smooth_factor=4.0):
    """
    Создание карты VTEC по SH + сглаживание океанов
    data: [lon, lat, vtec]
    """
    if len(data) == 0:
        return None

    lons = data[:, 0]
    lats = data[:, 1]
    values = data[:, 2]

    print(f"Создание карты VTEC (lmax={lmax})")
    print(f"Количество станций: {len(values)}")
    print(
        f"VTEC: mean={values.mean():.2f}, std={values.std():.2f}, "
        f"min={values.min():.2f}, max={values.max():.2f}"
    )

    weights = create_distance_weight_matrix(lats, lons, k_neighbors=5)

    n_points = len(values)
    n_coeffs = (lmax + 1) ** 2

    A = np.zeros((n_points, n_coeffs))

    # используем широту (sin(phi)) и долготу в радианах
    sin_phi = np.sin(np.radians(lats))
    lambda_rad = np.radians(lons)

    idx = 0
    for n in range(lmax + 1):
        for m in range(n + 1):
            P_nm = lpmv(m, n, sin_phi)

            if m == 0:
                norm = np.sqrt(2 * n + 1)
            else:
                from math import factorial
                norm = np.sqrt(
                    (2 * n + 1) * factorial(n - m) / factorial(n + m)
                )
            P_nm_norm = P_nm * norm

            if m == 0:
                A[:, idx] = P_nm_norm
                idx += 1
            else:
                A[:, idx] = P_nm_norm * np.cos(m * lambda_rad)
                idx += 1
                A[:, idx] = P_nm_norm * np.sin(m * lambda_rad)
                idx += 1

    # Регуляризация версии 1.0
    L = np.zeros((n_coeffs, n_coeffs))
    idx = 0
    for n in range(lmax + 1):
        if n <= 2:
            reg_strength = 0.01 * (1 + n**2)
        elif n == 3:
            reg_strength = 0.1 * (1 + n**3)
        elif n == 4:
            reg_strength = 0.3 * (1 + n**3)
        else:  # n >= 5
            reg_strength = 0.6 * (1 + n**4)
        for m in range(n + 1):
            L[idx, idx] = reg_strength
            idx += 1 if m == 0 else 2

    W = np.diag(weights)
    lhs = A.T @ W @ A + L
    rhs = A.T @ W @ values

    try:
        coeffs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    print(f"C00 (средний уровень): {coeffs[0]:.4f}")

    # Сетка по широте: от -90 до +80 с шагом 2°
    res = 2.0
    grid_lats = np.arange(-90, 90 + res, res)
    grid_lons = np.arange(0, 360 + res, res)

    nlat = len(grid_lats)
    nlon = len(grid_lons)
    grid = np.zeros((nlat, nlon))

    sin_phi_grid = np.sin(np.radians(grid_lats))

    for i in range(nlat):
        sin_phi_i = sin_phi_grid[i]
        for j in range(nlon):
            tec_value = 0.0
            lambda_grid = np.radians(grid_lons[j])

            idx = 0
            for n in range(lmax + 1):
                for m in range(n + 1):
                    P_nm = lpmv(m, n, sin_phi_i)
                    if m == 0:
                        norm = np.sqrt(2 * n + 1)
                    else:
                        from math import factorial
                        norm = np.sqrt(
                            (2 * n + 1) * factorial(n - m) / factorial(n + m)
                        )
                    P_nm_norm = P_nm * norm

                    if m == 0:
                        tec_value += coeffs[idx] * P_nm_norm
                        idx += 1
                    else:
                        tec_value += coeffs[idx] * P_nm_norm * np.cos(m * lambda_grid)
                        idx += 1
                        tec_value += coeffs[idx] * P_nm_norm * np.sin(m * lambda_grid)
                        idx += 1

            grid[i, j] = tec_value

    # Сглаживание океанов
    grid = apply_ocean_smoothing(grid, grid_lats, grid_lons, data, ocean_smooth_factor)

    # Ограничение диапазона
    vtec_mean = values.mean()
    vtec_std = values.std()
    grid[grid < 0] = 0
    upper_limit = min(100, vtec_mean + 3 * vtec_std)
    grid[grid > upper_limit] = upper_limit

    print(
        f"Итоговая карта: mean={grid.mean():.2f}, std={grid.std():.2f}, "
        f"min={grid.min():.2f}, max={grid.max():.2f}"
    )

    return grid, grid_lats, grid_lons


def apply_ocean_smoothing(grid, grid_lats, grid_lons,
                          station_data, smooth_factor=4.0):
    """Сглаживание в областях далеко от станций"""
    station_mask = np.zeros_like(grid, dtype=bool)
    radius_deg = 15.0

    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            distances = np.sqrt(
                (station_data[:, 1] - lat) ** 2 +
                (((station_data[:, 0] - lon + 180) % 360) - 180) ** 2
            )
            if distances.min() < radius_deg:
                station_mask[i, j] = True

    from scipy.ndimage import distance_transform_edt
    binary_mask = station_mask.astype(float)
    distances_px = distance_transform_edt(1 - binary_mask)

    smoothed_grid = grid.copy()
    max_sigma = smooth_factor

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not station_mask[i, j] and distances_px[i, j] > 0:
                sigma = min(max_sigma, distances_px[i, j] / 5.0)
                i_min = max(0, int(i - 2 * sigma))
                i_max = min(grid.shape[0], int(i + 2 * sigma) + 1)
                j_min = max(0, int(j - 2 * sigma))
                j_max = min(grid.shape[1], int(j + 2 * sigma) + 1)
                window = grid[i_min:i_max, j_min:j_max]
                if window.size > 0:
                    smoothed_grid[i, j] = np.mean(window)

    smoothed_grid = gaussian_filter(smoothed_grid, sigma=0.5)
    return smoothed_grid


def plot_vtec_map(grid, lat_grid, lon_grid, title, out_path, station_data=None):
    """Глобальная карта VTEC + (опционально) точки станций"""
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    vmin, vmax = 0, 80

    contour = ax.contourf(
        lon_mesh, lat_mesh, grid,
        transform=ccrs.PlateCarree(),
        cmap='jet',
        levels=np.linspace(vmin, vmax, 45),
        vmin=vmin, vmax=vmax,
        extend='both'
    )

    # чёрные точки — станции
    if station_data is not None and len(station_data) > 0:
        lons_st = station_data[:, 0]
        lats_st = station_data[:, 1]
        ax.scatter(
            lons_st, lats_st,
            c='k', s=10, marker='o',
            transform=ccrs.PlateCarree(),
            zorder=3
        )

    ax.coastlines(linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('Vertical TEC (TECU)', fontsize=12)

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_na_diff_self(diff_grid, lat_grid, lon_grid, hour_end):
    """Разность full − cut для Северной Америки"""
    na_lon_min, na_lon_max = -150, -40
    na_lat_min, na_lat_max = 5, 80

    lon_180 = (lon_grid % 360 + 180) % 360 - 180
    Lon, Lat = np.meshgrid(lon_180, lat_grid)

    lat_mask = (lat_grid >= na_lat_min) & (lat_grid <= na_lat_max)
    lon_mask = (lon_180 >= na_lon_min) & (lon_180 <= na_lon_max)

    diff_na = diff_grid[np.ix_(lat_mask, lon_mask)]
    print(
        "SELF NA shape:", diff_na.shape,
        "nan frac:", np.isnan(diff_na).mean() if diff_na.size else None
    )

    if diff_na.size == 0 or np.isnan(diff_na).all():
        print(" SELF NA diff пустая/NaN, карту не рисую")
        return

    lat_na = lat_grid[lat_mask]
    lon_na = lon_180[lon_mask]
    Lon_na, Lat_na = np.meshgrid(lon_na, lat_na)

    print(
        f" ΔVTEC SELF NA: min={np.nanmin(diff_na):.2f}, "
        f"max={np.nanmax(diff_na):.2f}, "
        f"mean={np.nanmean(diff_na):.2f}, "
        f"std={np.nanstd(diff_na):.2f}"
    )

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([na_lon_min, na_lon_max, na_lat_min, na_lat_max],
                  crs=ccrs.PlateCarree())

    vmax = 10.0
    im = ax.contourf(
        Lon_na, Lat_na, diff_na,
        levels=np.linspace(-vmax, vmax, 41),
        cmap='bwr',
        vmin=-vmax, vmax=vmax,
        transform=ccrs.PlateCarree(),
        extend='both'
    )

    ax.coastlines(linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('ΔVTEC (full − cut), TECU')

    Path("diff_self_na").mkdir(exist_ok=True)
    out = f"diff_self_na/diff_self_na_{hour_end:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(" SELF NA diff карта сохранена:", out)


def plot_global_diff_self(diff_grid, lat_grid, lon_grid, hour_end):
    """Глобальная разность full − cut"""
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    print(
        f" ΔVTEC SELF GLOBAL: min={np.nanmin(diff_grid):.2f}, "
        f"max={np.nanmax(diff_grid):.2f}, "
        f"mean={np.nanmean(diff_grid):.2f}, "
        f"std={np.nanstd(diff_grid):.2f}"
    )

    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    vmax = 10.0
    im = ax.contourf(
        Lon, Lat, diff_grid,
        levels=np.linspace(-vmax, vmax, 41),
        cmap='bwr',
        vmin=-vmax, vmax=vmax,
        transform=ccrs.PlateCarree(),
        extend='both'
    )

    ax.coastlines(linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('ΔVTEC (full − cut), TECU')

    Path("diff_self_global").mkdir(exist_ok=True)
    out = f"diff_self_global/diff_self_global_{hour_end:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(" SELF global diff карта сохранена:", out)


def main():
    data_dir = "320"
    coord_file = "tec_suite_coords.txt"

    lmax_val = 5
    smooth_factor = 4.0

    print("Построение карт VTEC (full и cut)...")
    stations_coords = load_station_coords(coord_file)
    print(f"Станций в координатах: {len(stations_coords)}")

    intervals = [(i, i + 2) for i in range(0, 24, 2)]

    Path("global_maps_full").mkdir(exist_ok=True)
    Path("global_maps_cut").mkdir(exist_ok=True)

    successful = 0

    ionex_file = "code_Data/COD0OPSFIN_20253200000_01D_01H_GIM.INX"
    tec_maps = read_ionex_fixed(ionex_file)

    for hour_start, hour_end in intervals:
        print(f"\nИнтервал: {hour_start:02d}-{hour_end:02d} UT")

        # Полная сеть
        vtec_full = load_vtec_data(
            data_dir, hour_start, hour_end, stations_coords,
            exclude_stations=set()
        )
        # Усечённая сеть (без выбранных станций)
        vtec_cut = load_vtec_data(
            data_dir, hour_start, hour_end, stations_coords,
            exclude_stations=EXCLUDED_STATIONS
        )

        print(f" Данных FULL: {len(vtec_full)} точек")
        print(f" Данных CUT : {len(vtec_cut)} точек")

        if len(vtec_full) < 10 or len(vtec_cut) < 10:
            print(" Мало данных для full или cut, пропускаю интервал")
            continue

        try:
            res_full = create_vtec_map_with_ocean_constraint(
                vtec_full, lmax=lmax_val, ocean_smooth_factor=smooth_factor
            )
            res_cut = create_vtec_map_with_ocean_constraint(
                vtec_cut, lmax=lmax_val, ocean_smooth_factor=smooth_factor
            )

            if res_full and res_cut:
                grid_full, lat_grid, lon_grid = res_full
                grid_cut, lat_grid2, lon_grid2 = res_cut

                # сетки должны совпадать
                if not (np.array_equal(lat_grid, lat_grid2) and
                        np.array_equal(lon_grid, lon_grid2)):
                    print(" Предупреждение: lat/lon сетки full и cut не совпадают, пропускаю diff")
                    continue

                # сохраняем обе карты с точками станций
                plot_vtec_map(
                    grid_full, lat_grid, lon_grid,
                    f"VTEC FULL: {hour_end:02d} UT",
                    f"global_maps_full/vtec_full_{hour_end:02d}.png",
                    station_data=vtec_full
                )
                plot_vtec_map(
                    grid_cut, lat_grid, lon_grid,
                    f"VTEC CUT : {hour_end:02d} UT",
                    f"global_maps_cut/vtec_cut_{hour_end:02d}.png",
                    station_data=vtec_cut
                )

                # разность full − cut
                diff_grid = grid_full - grid_cut

                plot_global_diff_self(diff_grid, lat_grid, lon_grid, hour_end)
                plot_na_diff_self(diff_grid, lat_grid, lon_grid, hour_end)

                successful += 1
                print("Карты построены")

        except Exception as e:
            print(f"Ошибка: {e}")

    print(f"\nГотово! Успешно: {successful}/{len(intervals)} интервалов")


if __name__ == "__main__":
    main()
