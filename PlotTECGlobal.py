import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Область Северной Америки
na_extent = [-150, -40, 5, 80]  # [lon_min, lon_max, lat_min, lat_max]


def read_ionex_fixed(filename):
    """Исправленный парсер для IONEX файлов CODE"""
    try:
        with open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        lines = [line.rstrip() for line in lines if line.strip()]
        result = []
        current_map = None
        in_tec_map = False
        current_data = []
        i = 0

        # Заголовок
        while i < len(lines):
            if 'END OF HEADER' in lines[i]:
                i += 1
                break
            i += 1

        print(f"Начало парсинга данных после строки {i}")

        maps_found = 0
        while i < len(lines):
            line = lines[i]

            if 'START OF TEC MAP' in line:
                in_tec_map = True
                current_map = {'index': len(result), 'data': []}
                current_data = []
                maps_found += 1
                print(f"Найдена карта TEC #{maps_found}")

            elif 'EPOCH OF CURRENT MAP' in line and in_tec_map:
                # стандартный формат IONEX: поля по 6 символов
                try:
                    year_str = line[0:6].strip()
                    month_str = line[6:12].strip()
                    day_str = line[12:18].strip()
                    hour_str = line[18:24].strip()
                    minute_str = line[24:30].strip()
                    second_str = line[30:36].strip() if len(line) > 30 else "0"

                    if year_str and month_str and day_str and hour_str and minute_str:
                        year = int(year_str)
                        month = int(month_str)
                        day = int(day_str)
                        hour = int(hour_str)
                        minute = int(minute_str)
                        second = int(second_str) if second_str else 0
                        if hour >= 24:
                            hour -= 24
                            day += 1
                        current_date = datetime(year, month, day, hour, minute, second)
                        current_map['date'] = current_date
                        print(f"  Время карты: {current_map['date']}")
                    else:
                        current_map['date'] = None
                except Exception as e:
                    print(f"  Ошибка парсинга даты: {e}")
                    current_map['date'] = None

            elif 'LAT/LON1/LON2/DLON/H' in line and in_tec_map:
                # строка широты и параметров по долготе
                try:
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if len(numbers) >= 5:
                        lat = float(numbers[0])
                        lon1 = float(numbers[1])
                        lon2 = float(numbers[2])
                        dlon = float(numbers[3])
                        height = float(numbers[4])
                    else:
                        lat = float(line[2:8].strip() or '0')
                        lon1 = float(line[8:14].strip() or '0')
                        lon2 = float(line[14:20].strip() or '0')
                        dlon = float(line[20:26].strip() or '0')
                        height = float(line[26:32].strip() or '0')

                    n_lons = int((lon2 - lon1) / dlon) + 1
                    tec_values = []
                    i += 1

                    while len(tec_values) < n_lons and i < len(lines):
                        data_line = lines[i]
                        if any(marker in data_line for marker in [
                            'LAT/LON1/LON2/DLON/H',
                            'END OF TEC MAP',
                            'START OF TEC MAP',
                            'EPOCH OF CURRENT MAP'
                        ]):
                            break
                        line_numbers = re.findall(r'-?\d+', data_line)
                        if line_numbers:
                            tec_values.extend([float(x) for x in line_numbers])
                        i += 1

                    tec_values = tec_values[:n_lons]
                    current_data.append({
                        'lat': lat,
                        'lon_start': lon1,
                        'lon_end': lon2,
                        'dlon': dlon,
                        'height': height,
                        'tec': tec_values
                    })
                    print(f"  Широта {lat}: прочитано {len(tec_values)} значений TEC")
                    continue

                except Exception as e:
                    print(f"  Ошибка парсинга координат: {e}")
                    print(f"  Проблемная строка: {line}")

            elif 'END OF TEC MAP' in line and in_tec_map:
                in_tec_map = False
                if current_map and current_data:
                    if 'date' not in current_map or current_map['date'] is None:
                        current_map['date'] = f"map_{len(result):03d}"
                    current_map['data'] = current_data
                    result.append(current_map)
                    print(
                        f"  Карта #{maps_found} успешно добавлена "
                        f"({len(current_data)} широт)"
                    )
                else:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ: Карта #{maps_found} не добавлена (нет данных)")
                current_map = None
                current_data = []

            i += 1

        print(f"Всего найдено и обработано карт: {len(result)}")
        return result

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return []


def create_tec_matrix(tec_map):
    """Создает матрицу TEC из карты CODE и переводит в TECU (делит на 10)."""
    if (not tec_map) or ('data' not in tec_map) or (not tec_map['data']):
        return [], [], np.array([])

    lats = sorted([item['lat'] for item in tec_map['data']])
    if not lats:
        return [], [], np.array([])

    first_item = tec_map['data'][0]
    lons = np.arange(
        first_item['lon_start'],
        first_item['lon_end'] + first_item['dlon'],
        first_item['dlon']
    )

    tec_matrix = np.zeros((len(lats), len(lons)))
    sorted_data = sorted(tec_map['data'], key=lambda x: x['lat'])
    valid_rows = 0

    for i, data_item in enumerate(sorted_data):
        tec_values = np.array(data_item['tec'], dtype=float)

        # 9999 — нет данных в IONEX
        tec_values[tec_values == 9999] = np.nan

        # стандартный масштаб IONEX: ×0.1 TECU
        tec_values = tec_values / 14.0

        if len(tec_values) == len(lons):
            tec_matrix[i, :] = tec_values
            valid_rows += 1

    print(f"  Создана матрица TEC: {valid_rows}/{len(lats)} строк валидны")
    return np.array(lats), np.array(lons), tec_matrix


if __name__ == "__main__":
    filename = "code_Data/COD0OPSFIN_20253210000_01D_01H_GIM.INX"
    print("=" * 60)
    print("ЧТЕНИЕ IONEX ФАЙЛА")
    print("=" * 60)
    tec_maps = read_ionex_fixed(filename)

    if tec_maps:
        print(f"\nУСПЕШНО ПРОЧИТАНО {len(tec_maps)} КАРТ TEC")
        print("\nДоступные времена в файле:")
        for i, tec_map in enumerate(tec_maps):
            if isinstance(tec_map['date'], datetime):
                print(
                    f"  Карта {i}: "
                    f"{tec_map['date'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                print(f"  Карта {i}: {tec_map['date']}")
    else:
        print("НЕ УДАЛОСЬ ПРОЧИТАТЬ ДАННЫЕ ИЗ ФАЙЛА")
