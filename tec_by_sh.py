import numpy as np
import pandas as pd
import pyshtools as sh
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import lpmv
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

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


def load_vtec_data(data_dir, hour_start, hour_end, stations_coords):
    """Загрузка данных VTEC за указанный временной интервал"""
    data = []

    for station_path in Path(data_dir).iterdir():
        if not station_path.is_dir():
            continue

        station_name = station_path.name
        if station_name not in stations_coords:
            continue

        station_lon, station_lat = stations_coords[station_name]
        data_file = station_path / f"{station_name}_{data_dir}_2025.dat"

        if not data_file.exists():
            continue

        df = pd.read_csv(data_file, delim_whitespace=True, comment='#',
                         names=['UT', 'I_v', 'G_lon', 'G_lat', 'G_q_lon',
                                'G_q_lat', 'G_t', 'G_q_t'])

        # Фильтрация по времени и удаление нулевых значений
        mask = (df['UT'] >= hour_start) & (df['UT'] <= hour_end) & (df['I_v'] > 0)
        df_filtered = df[mask]

        if len(df_filtered) > 0:
            # Используем среднее значение за интервал
            avg_vtec = df_filtered['I_v'].mean()
            data.append([station_lon, station_lat, avg_vtec])

    return np.array(data)


def create_distance_weight_matrix(lats, lons, k_neighbors=5):
    """Создание весовой матрицы на основе плотности станций"""
    # Преобразуем в 3D координаты на единичной сфере
    colat = 90.0 - lats
    theta = np.radians(colat)
    phi = np.radians(lons)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    coords_3d = np.column_stack([x, y, z])
    tree = cKDTree(coords_3d)

    # Расстояние до k-го соседа как мера плотности
    k = min(k_neighbors, len(lats) // 10)
    if k > 0:
        distances, _ = tree.query(coords_3d, k=k + 1)  # +1 для исключения самой точки
        # Среднее расстояние до k ближайших соседей
        mean_distances = distances[:, 1:].mean(axis=1)
        # Веса обратно пропорциональны плотности
        weights = 1.0 / (mean_distances + 1e-10)
        # Нормализуем так, чтобы средний вес был 1
        weights = weights / weights.mean()
    else:
        weights = np.ones(len(lats))

    return weights


def create_vtec_map_with_ocean_constraint(data, lmax=8, ocean_smooth_factor=3.0):
    """
    Создание карты VTEC с ограничениями для океанических областей

    Parameters:
    -----------
    data : ndarray
        Массив [lon, lat, vtec]
    lmax : int
        Максимальная степень сферических гармоник
    ocean_smooth_factor : float
        Коэффициент сглаживания для океанических областей (чем больше, тем сильнее)
    """

    if len(data) == 0:
        return None

    lons = data[:, 0]
    lats = data[:, 1]
    values = data[:, 2]

    print(f"Создание карты VTEC (lmax={lmax})")
    print(f"Количество станций: {len(values)}")
    print(f"VTEC: mean={values.mean():.2f}, std={values.std():.2f}, "
          f"min={values.min():.2f}, max={values.max():.2f}")

    # 1. Создаем веса для компенсации неравномерного распределения
    weights = create_distance_weight_matrix(lats, lons, k_neighbors=5)

    # 2. Создаем матрицу дизайна согласно формуле (1)
    n_points = len(values)

    # Количество коэффициентов: для каждой степени n от 0 до lmax
    # и для каждого порядка m от 0 до n (C_nm и S_nm для m>0)
    n_coeffs = (lmax + 1) ** 2

    A = np.zeros((n_points, n_coeffs))

    sin_phi = np.sin(np.radians(lats))
    lambda_rad = np.radians(lons)

    idx = 0
    for n in range(lmax + 1):
        for m in range(n + 1):
            # Нормализованная присоединенная функция Лежандра P_n^m(sin φ)
            P_nm = lpmv(m, n, sin_phi)

            # Нормализация для геодезических приложений
            if m == 0:
                norm = np.sqrt(2 * n + 1)
            else:
                from math import factorial
                norm = np.sqrt((2 * n + 1) * factorial(n - m) / factorial(n + m))

            P_nm_norm = P_nm * norm

            if m == 0:
                # Только косинусная часть для m=0: C_n0 * P_n0
                A[:, idx] = P_nm_norm
                idx += 1
            else:
                # C_nm * P_nm * cos(mλ)
                A[:, idx] = P_nm_norm * np.cos(m * lambda_rad)
                idx += 1

                # S_nm * P_nm * sin(mλ)
                A[:, idx] = P_nm_norm * np.sin(m * lambda_rad)
                idx += 1

    # 3. Создаем матрицу регуляризации для подавления артефактов в океанах
    # Сильнее штрафуем высокие степени (быстрые колебания)
    L = np.zeros((n_coeffs, n_coeffs))

    idx = 0
    for n in range(lmax + 1):
        # Регуляризация растет как n^3 для высоких гармоник
        reg_strength = 0.001 * (1 + n ** 3)

        for m in range(n + 1):
            L[idx, idx] = reg_strength
            idx += 1 if m == 0 else 2

    # 4. Взвешенные наименьшие квадраты с регуляризацией
    W = np.diag(weights)
    lhs = A.T @ W @ A + L
    rhs = A.T @ W @ values

    # Решаем систему
    try:
        coeffs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    print(f"C00 (средний уровень): {coeffs[0]:.4f}")

    # 5. Восстанавливаем на глобальной сетке
    res = 2.0  # разрешение в градусах
    grid_lats = np.arange(-90, 90 + res, res)
    grid_lons = np.arange(0, 360 + res, res)

    nlat = len(grid_lats)
    nlon = len(grid_lons)
    grid = np.zeros((nlat, nlon))

    sin_phi_grid = np.sin(np.radians(grid_lats))

    # Восстановление значений на сетке
    for i in range(nlat):
        for j in range(nlon):
            tec_value = 0.0
            lambda_grid = np.radians(grid_lons[j])
            idx = 0

            for n in range(lmax + 1):
                for m in range(n + 1):
                    P_nm = lpmv(m, n, sin_phi_grid[i])

                    if m == 0:
                        norm = np.sqrt(2 * n + 1)
                    else:
                        from math import factorial
                        norm = np.sqrt((2 * n + 1) * factorial(n - m) / factorial(n + m))

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

    # 6. Применяем сглаживание в областях без станций
    grid = apply_ocean_smoothing(grid, grid_lats, grid_lons, data, ocean_smooth_factor)

    # 7. Ограничиваем физически возможные значения
    vtec_mean = values.mean()
    vtec_std = values.std()

    # VTEC не может быть отрицательным
    grid[grid < 0] = 0

    # Ограничиваем сверху разумным значением
    upper_limit = min(100, vtec_mean + 3 * vtec_std)
    grid[grid > upper_limit] = upper_limit

    print(f"Итоговая карта: mean={grid.mean():.2f}, std={grid.std():.2f}, "
          f"min={grid.min():.2f}, max={grid.max():.2f}")

    return grid, grid_lats, grid_lons


def apply_ocean_smoothing(grid, grid_lats, grid_lons, station_data, smooth_factor=3.0):
    """
    Применяет сглаживание в областях далеких от станций

    Parameters:
    -----------
    grid : ndarray
        Исходная сетка VTEC
    grid_lats, grid_lons : ndarray
        Координаты сетки
    station_data : ndarray
        Данные станций [lon, lat, vtec]
    smooth_factor : float
        Коэффициент сглаживания
    """

    # Создаем маску: 1 где есть станции поблизости, 0 где нет
    station_mask = np.zeros_like(grid, dtype=bool)
    radius_deg = 15.0  # радиус влияния станции

    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            # Находим расстояние до ближайшей станции
            distances = np.sqrt(
                (station_data[:, 1] - lat) ** 2 +
                ((station_data[:, 0] - lon + 180) % 360 - 180) ** 2
            )
            min_distance = distances.min()

            # Если есть станция в пределах радиуса
            if min_distance < radius_deg:
                station_mask[i, j] = True

    # Расстояние до ближайшей станции (в градусах)
    from scipy.ndimage import distance_transform_edt

    # Для distance_transform нужно преобразовать в бинарную маску
    binary_mask = station_mask.astype(float)

    # Расстояние в пикселях
    # 1 градус ≈ 111 км, но для сглаживания используем пиксели
    distances_px = distance_transform_edt(1 - binary_mask)

    # Применяем сглаживание, зависящее от расстояния
    smoothed_grid = grid.copy()

    # Области без станций сглаживаем сильнее
    max_sigma = smooth_factor
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not station_mask[i, j] and distances_px[i, j] > 0:
                # Сила сглаживания зависит от расстояния до станций
                sigma = min(max_sigma, distances_px[i, j] / 5.0)

                # Определяем окно для сглаживания
                i_min = max(0, int(i - 2 * sigma))
                i_max = min(grid.shape[0], int(i + 2 * sigma) + 1)
                j_min = max(0, int(j - 2 * sigma))
                j_max = min(grid.shape[1], int(j + 2 * sigma) + 1)

                # Локальное среднее
                window = grid[i_min:i_max, j_min:j_max]
                if window.size > 0:
                    smoothed_grid[i, j] = np.mean(window)

    # Слегка сглаживаем всю карту для устранения резких переходов
    smoothed_grid = gaussian_filter(smoothed_grid, sigma=0.5)

    return smoothed_grid


def plot_vtec_map_with_stations(grid, lat_grid, lon_grid, station_data, title):
    """Визуализация карты VTEC со станциями"""

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Сетка для отображения
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # ФИКСИРОВАННАЯ шкала от 0 до 100 TECU
    vmin = 0
    vmax = 80

    cmap = 'jet'

    # Отображаем интерполированную карту VTEC
    contour = ax.contourf(lon_mesh, lat_mesh, grid,
                          transform=ccrs.PlateCarree(),
                          cmap=cmap,  # Используем выбранную цветовую карту
                          levels=np.linspace(vmin, vmax, 45),  # 50 интервалов
                          vmin=vmin, vmax=vmax,
                          extend='both')  # Показываем стрелки если значения вне диапазона


    # Отображаем станции (раскомментируйте если нужно)
    # scatter = ax.scatter(station_data[:, 0], station_data[:, 1],
    #                     c=station_data[:, 2],
    #                     s=40,
    #                     cmap=cmap,  # Та же цветовая карта
    #                     vmin=vmin, vmax=vmax,
    #                     edgecolors='black', linewidth=0.5,
    #                     transform=ccrs.PlateCarree(),
    #                     zorder=10,
    #                     label='GNSS станции')

    # Добавляем границы
    ax.coastlines(linewidth=0.8)
    ax.add_feature(ccrs.cartopy.feature.OCEAN, alpha=0.1)
    ax.add_feature(ccrs.cartopy.feature.LAND, alpha=0.1)

    # Добавляем границы стран
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, alpha=0.5)

    # Сетка
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5,
                      xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 30))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Цветовая шкала с фиксированными делениями
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('Vertical TEC (TECU)', fontsize=12, weight='bold')

    # Устанавливаем фиксированные деления на цветовой шкале
    cbar.set_ticks(np.arange(0, 81, 10))  # Деления каждые 10 TECU
    cbar.set_ticklabels([f'{i}' for i in range(0, 81, 10)])

    plt.title(title, fontsize=14, pad=20, weight='bold')
    plt.tight_layout()

    return fig, ax


def plot_north_america_vtec(grid, lat_grid, lon_grid, station_data, title, hour_start, hour_end):
    """Построение карты VTEC только для Северной Америки"""

    # Область Северной Америки
    na_extent = [-150, -40, 5, 80]

    # Создаем фигуру
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(na_extent, crs=ccrs.PlateCarree())

    # Отображаем карту VTEC
    vmin, vmax = 0, 80
    contour = ax.contourf(lon_grid, lat_grid, grid,
                          transform=ccrs.PlateCarree(),
                          cmap='jet',
                          levels=np.linspace(vmin, vmax, 40),
                          vmin=vmin, vmax=vmax,
                          extend='both')

    # Картографические элементы
    ax.coastlines(linewidth=0.8)
    ax.add_feature(ccrs.cartopy.feature.OCEAN, alpha=0.1)
    ax.add_feature(ccrs.cartopy.feature.LAND, alpha=0.1)

    # Сетка
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
    cbar.set_label('Vertical TEC (TECU)')
    cbar.set_ticks(range(0, 81, 10))

    plt.title(f'North America VTEC: {hour_end:02d} UT', fontsize=12, pad=15)
    plt.tight_layout()

    output_dir = "north_america_maps"
    Path(output_dir).mkdir(exist_ok=True)
    output_file = f"{output_dir}/vtec_na_{hour_end:02d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_file


def main():
    """Основная функция - все интервалы"""
    data_dir = "321"
    coord_file = "tec_suite_coords.txt"
    lmax_val = 5
    smooth_factor = 3.0

    print("Построение карт VTEC...")

    stations_coords = load_station_coords(coord_file)
    print(f"Станций: {len(stations_coords)}")

    intervals = [(i, i + 2) for i in range(0, 24, 2)]

    Path("global_maps").mkdir(exist_ok=True)
    Path("north_america_maps").mkdir(exist_ok=True)

    successful = 0

    for hour_start, hour_end in intervals:
        print(f"\nИнтервал: {hour_start:02d}-{hour_end:02d} UT")

        # Данные
        vtec_data = load_vtec_data(data_dir, hour_start, hour_end, stations_coords)

        if len(vtec_data) < 10:
            print(f"  Мало данных: {len(vtec_data)} точек")
            continue

        print(f"  Данных: {len(vtec_data)} точек")

        try:
            result = create_vtec_map_with_ocean_constraint(
                vtec_data, lmax=lmax_val, ocean_smooth_factor=smooth_factor)

            if result:
                grid, lat_grid, lon_grid = result

                fig, ax = plot_vtec_map_with_stations(
                    grid, lat_grid, lon_grid, vtec_data,
                    f"VTEC: {hour_end:02d} UT"
                )
                plt.savefig(f"global_maps/vtec_{hour_end:02d}.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Северная Америка
                plot_north_america_vtec(
                    grid, lat_grid, lon_grid, vtec_data,
                    "", hour_start, hour_end
                )

                successful += 1
                print(f"Карты построены")

        except Exception as e:
            print(f"Ошибка: {e}")

    print(f"\nГотово! Успешно: {successful}/{len(intervals)} интервалов")


def save_grid_data(grid, lat_grid, lon_grid, hour_start, hour_end, output_dir):
    """Сохранение данных карты в текстовый файл"""
    try:
        # Создаем файл для данных
        data_file = f"{output_dir}/vtec_{hour_end:02d}_data.txt"

        with open(data_file, 'w') as f:
            f.write(f"# VTEC Data: {hour_start:02d}:00 - {hour_end:02d}:00 UT\n")
            f.write(f"# Latitudes: {len(lat_grid)} points from {lat_grid[0]} to {lat_grid[-1]}\n")
            f.write(f"# Longitudes: {len(lon_grid)} points from {lon_grid[0]} to {lon_grid[-1]}\n")
            f.write(f"# Grid shape: {grid.shape}\n")
            f.write(f"# Statistics: min={grid.min():.2f}, max={grid.max():.2f}, "
                    f"mean={grid.mean():.2f}, std={grid.std():.2f}\n")
            f.write("#" * 50 + "\n")
            f.write("# Longitude Latitude VTEC\n")

            # Записываем данные построчно
            for i, lat in enumerate(lat_grid):
                for j, lon in enumerate(lon_grid):
                    f.write(f"{lon:8.2f} {lat:8.2f} {grid[i, j]:8.2f}\n")

        print(f"  Данные сохранены в {data_file}")

    except Exception as e:
        print(f" Не удалось сохранить данные: {e}")

if __name__ == "__main__":
    main()