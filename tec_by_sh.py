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

    colat = 90 - lats  # широта -> дополнение до колatitude
    lon_rad = np.radians(lons)
    colat_rad = np.radians(colat)

    # Аппроксимация сферическими гармониками
    coeffs = sh.expand.SHExpandLSQ(values, colat_rad, lon_rad, lmax=lmax)

    # Восстановление на регулярной сетке
    grid = sh.expand.MakeGridDH(coeffs, sampling=2)

    # Преобразование в градусы
    nlat, nlon = grid.shape
    lat_grid = np.linspace(-90, 90, nlat)
    lon_grid = np.linspace(0, 360, nlon)

    return grid, lat_grid, lon_grid


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
    hour_start = 0  # Начало интервала (часы)
    hour_end = 2  # Конец интервала (часы)
    lmax = 15  # Максимальная степень гармоник
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

    if result is None:
        print("Ошибка при построении карты")
        return

    grid, lat_grid, lon_grid = result

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