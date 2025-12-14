import numpy as np
import pandas as pd
import pyshtools as sh
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
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
        data_file = station_path / f"{station_name}_319_2025.dat"

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


def create_vtec_map(data, lmax=15):
    """Создание глобальной карты VTEC с использованием сферических гармоник"""
    if len(data) == 0:
        return None

    lons = data[:, 0]
    lats = data[:, 1]
    values = data[:, 2]

    # Преобразование координат
    colat = 90 - lats
    lon_rad = np.radians(lons)
    colat_rad = np.radians(colat)

    # Используем SHExpandLSQ
    # SHExpandLSQ возвращает (cilm, chi2) где cilm это коэффициенты
    coeffs_tuple = sh.expand.SHExpandLSQ(values, colat_rad, lon_rad, lmax=lmax)

    # Распаковываем результат
    if isinstance(coeffs_tuple, tuple):
        cilm = coeffs_tuple[0]  # Это массив коэффициентов
        chi2 = coeffs_tuple[1]  # Это chi-squared

        print(f"Форма коэффициентов: {cilm.shape}")
        print(f"chi2: {chi2}")
    else:
        cilm = coeffs_tuple

    # Преобразуем в объект SHCoeffs
    # cilm имеет форму (2, lmax+1, lmax+1)
    coeffs = sh.SHCoeffs.from_array(cilm)

    # Расширяем на сетку
    grid_object = coeffs.expand(grid='DH2')

    # Извлекаем данные и координаты
    grid = grid_object.data
    lat_grid = np.linspace(-90, 90, grid.shape[0])
    lon_grid = np.linspace(0, 360, grid.shape[1])

    return grid, lat_grid, lon_grid


def create_vtec_map_correct(data, lmax=10, use_weights=True):
    """Реализация формулы (1) с правильной нормализацией"""
    if len(data) == 0:
        return None

    lons = data[:, 0]  # λ - геомагнитная долгота
    lats = data[:, 1]  # φ - геомагнитная широта
    values = data[:, 2]  # VTEC

    print(f"Создание карты VTEC по формуле (1), lmax={lmax}")
    print(f"Диапазон VTEC: [{values.min():.2f}, {values.max():.2f}] TECU")

    # Нормализуем широту: sin(φ) ∈ [-1, 1]
    sin_phi = np.sin(np.radians(lats))

    # Вычисляем веса если нужно
    if use_weights:
        from scipy.spatial import cKDTree

        # Преобразуем в 3D координаты для вычисления плотности
        theta = np.radians(90 - lats)
        phi_rad = np.radians(lons)
        x = np.sin(theta) * np.cos(phi_rad)
        y = np.sin(theta) * np.sin(phi_rad)
        z = np.cos(theta)

        coords_3d = np.column_stack([x, y, z])
        tree = cKDTree(coords_3d)
        k = min(5, len(data) // 10)

        if k > 0:
            distances, _ = tree.query(coords_3d, k=k + 1)
            densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
            weights = 1.0 / densities
            weights = weights / weights.mean()  # средний вес = 1
        else:
            weights = np.ones(len(data))
    else:
        weights = np.ones(len(data))

    # Строим матрицу дизайна согласно формуле (1)
    n_points = len(values)
    n_coeffs = (lmax + 1) ** 2  # C00 + (C10, C11, S11) + ...

    A = np.zeros((n_points, n_coeffs))

    from scipy.special import lpmv  # Associated Legendre functions

    idx = 0
    for n in range(lmax + 1):
        for m in range(n + 1):
            # Нормализованная присоединенная функция Лежандра P_n^m(sin φ)
            P_nm = lpmv(m, n, sin_phi)

            # Нормализация (обычно 1/√(4π) для n=m=0)
            # Для геодезии: sqrt((2n+1)*(n-m)!/(n+m)!)
            if m == 0:
                norm = np.sqrt(2 * n + 1)
            else:
                from math import factorial
                norm = np.sqrt((2 * n + 1) * factorial(n - m) / factorial(n + m))

            P_nm_norm = P_nm * norm

            # Косинус и синус части
            if m == 0:
                # Только косинусная часть для m=0
                A[:, idx] = P_nm_norm  # C_n0
                idx += 1
            else:
                # C_nm * cos(mλ)
                A[:, idx] = P_nm_norm * np.cos(m * np.radians(lons))
                idx += 1

                # S_nm * sin(mλ)
                A[:, idx] = P_nm_norm * np.sin(m * np.radians(lons))
                idx += 1

    # Взвешенные наименьшие квадраты: A^T * W * A * x = A^T * W * b
    W = np.diag(weights)

    # Решаем систему с небольшой регуляризацией
    lhs = A.T @ W @ A + 1e-6 * np.eye(n_coeffs)
    rhs = A.T @ W @ values

    try:
        coeffs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    print(f"Коэффициенты: C00={coeffs[0]:.4f} (должно быть ~{values.mean():.2f})")

    # Восстанавливаем на глобальной сетке
    res = 2.0  # разрешение в градусах
    grid_lats = np.arange(-90, 90 + res, res)
    grid_lons = np.arange(0, 360 + res, res)

    nlat = len(grid_lats)
    nlon = len(grid_lons)
    grid = np.zeros((nlat, nlon))

    sin_phi_grid = np.sin(np.radians(grid_lats))

    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            tec_value = 0.0
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
                        tec_value += coeffs[idx] * P_nm_norm * np.cos(m * np.radians(lon))
                        idx += 1
                        tec_value += coeffs[idx] * P_nm_norm * np.sin(m * np.radians(lon))
                        idx += 1

            grid[i, j] = tec_value

    # VTEC не может быть отрицательным
    grid[grid < 0] = 0

    # Сглаживание
    #from scipy.ndimage import gaussian_filter
    #grid = gaussian_filter(grid, sigma=0.5)

    print(f"Результат: [{grid.min():.2f}, {grid.max():.2f}] TECU")

    return grid, grid_lats, grid_lons

def plot_vtec_map(grid, lat_grid, lon_grid, data_time, data_points=None):
    """Визуализация карты VTEC"""
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Создание сетки для отображения
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Отображение VTEC
    contour = ax.contourf(lon_mesh, lat_mesh, grid,
                          transform=ccrs.PlateCarree(),
                          cmap='plasma', levels=30)

    # Добавление станций
    if data_points is not None:
        ax.scatter(data_points[:, 0], data_points[:, 1],
                   c='white', s=20, alpha=0.7,
                   transform=ccrs.PlateCarree(),
                   edgecolors='black', linewidth=0.5)

    # Настройки карты
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_global()

    # Цветовая шкала
    plt.colorbar(contour, ax=ax, orientation='horizontal',
                 pad=0.05, label='VTEC (TECU)')
    filename = f'Глобальная карта ПЭС {data_time}'
    plt.title(filename)
    plt.tight_layout()
    return fig, ax


def main():
    """Основная функция"""
    # Параметры
    data_dir = "319"  # Папка с данными
    coord_file = "tec_suite_coords.txt"  # Файл с координатами
    hour_start = 6  # Начало интервала (часы)
    hour_end = 8 # Конец интервала (часы)
    lmax = 5  # Максимальная степень гармоник
    data_time = f"319d {hour_end}UT"
    # 1. Загрузка координат станций
    print("Загрузка координат станций...")
    stations_coords = load_station_coords(coord_file)
    print(f"Загружено {len(stations_coords)} станций")

    # 2. Загрузка данных VTEC
    print(f"Загрузка данных VTEC ({hour_start}-{hour_end} UT)...")
    vtec_data = load_vtec_data(data_dir, hour_start, hour_end, stations_coords)
    print(f"Загружено {len(vtec_data)} измерений")

    if len(vtec_data) == 0:
        print("Нет данных для построения карты")
        return

    # 3. Создание карты VTEC
    print("Построение карты сферическими гармониками...")
    result = create_vtec_map(vtec_data, lmax=lmax)
    result1 = create_vtec_map_correct(vtec_data, lmax=lmax, use_weights=True)

    if result1 is None:
        print("Ошибка при построении карты")
        return

    grid, lat_grid, lon_grid = result1

    # 4. Визуализация
    print("Визуализация...")
    fig, ax = plot_vtec_map(grid, lat_grid, lon_grid, data_time, vtec_data)

    # 5. Сохранение
    output_file = f"vtec_map_{hour_start}_{hour_end}h.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Карта сохранена как {output_file}")

    plt.show()

    # Вывод статистики
    print(f"\nСтатистика VTEC:")
    print(f"  Минимум: {grid.min():.2f} TECU")
    print(f"  Максимум: {grid.max():.2f} TECU")
    print(f"  Среднее: {grid.mean():.2f} TECU")
    print(f"  Степень гармоник: lmax = {lmax}")


if __name__ == "__main__":
    main()