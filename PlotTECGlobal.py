import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


def read_ionex_fixed(filename):
    """Исправленный парсер для IONEX файлов с улучшенным парсингом дат"""
    try:
        with open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        # Удаляем пустые строки и лишние пробелы
        lines = [line.rstrip() for line in lines if line.strip()]

        result = []
        current_map = None
        in_tec_map = False
        current_data = []

        i = 0
        # Пропускаем заголовок
        while i < len(lines):
            if 'END OF HEADER' in lines[i]:
                i += 1
                break
            i += 1

        print(f"Начало парсинга данных после строки {i}")

        # Парсим данные
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
                try:
                    # Улучшенный парсинг даты - несколько методов
                    date_found = False

                    # Метод 1: Парсим из фиксированных позиций (стандартный IONEX)
                    if len(line) >= 36:
                        year_str = line[0:6].strip()
                        month_str = line[6:12].strip()
                        day_str = line[12:18].strip()
                        hour_str = line[18:24].strip()
                        minute_str = line[24:30].strip()
                        second_str = line[30:36].strip() if len(line) > 30 else "0"

                        if (year_str and month_str and day_str and
                                hour_str and minute_str and
                                year_str.replace(' ', '').isdigit()):

                            year = int(year_str)
                            month = int(month_str)
                            day = int(day_str)
                            hour = int(hour_str)
                            minute = int(minute_str)
                            second = int(second_str) if second_str and second_str.replace(' ', '').isdigit() else 0

                            # Проверяем корректность даты
                            if hour >= 24:
                                hour -= 24
                                day += 1

                            current_date = datetime(year, month, day, hour, minute, second)
                            if current_date.year >= 2000 and current_date.year <= 2030:
                                current_map['date'] = current_date
                                date_found = True
                                print(f"  Время карты (метод 1): {current_map['date']}")

                    # Метод 2: Ищем числа через регулярные выражения
                    if not date_found:
                        numbers = re.findall(r'\d+', line)
                        if len(numbers) >= 6:
                            try:
                                year = int(numbers[0])
                                month = int(numbers[1])
                                day = int(numbers[2])
                                hour = int(numbers[3])
                                minute = int(numbers[4])
                                second = int(numbers[5]) if len(numbers) > 5 else 0

                                if hour >= 24:
                                    hour -= 24
                                    day += 1

                                current_date = datetime(year, month, day, hour, minute, second)
                                if current_date.year >= 2000 and current_date.year <= 2030:
                                    current_map['date'] = current_date
                                    date_found = True
                                    print(f"  Время карты (метод 2): {current_map['date']}")
                            except:
                                pass

                    # Метод 3: Пытаемся найти дату в любом формате
                    if not date_found:
                        # Ищем шаблоны даты в строке
                        date_patterns = [
                            r'(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})',
                            r'(\d{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})'
                        ]

                        for pattern in date_patterns:
                            match = re.search(pattern, line)
                            if match:
                                groups = match.groups()
                                try:
                                    year = int(groups[0])
                                    month = int(groups[1])
                                    day = int(groups[2])
                                    hour = int(groups[3])
                                    minute = int(groups[4])
                                    second = int(groups[5]) if len(groups) > 5 else 0

                                    # Если год двузначный, добавляем 2000
                                    if year < 100:
                                        year += 2000

                                    if hour >= 24:
                                        hour -= 24
                                        day += 1

                                    current_date = datetime(year, month, day, hour, minute, second)
                                    if current_date.year >= 2000 and current_date.year <= 2030:
                                        current_map['date'] = current_date
                                        date_found = True
                                        print(f"  Время карты (метод 3): {current_map['date']}")
                                        break
                                except:
                                    continue

                    # Если дата не найдена, используем индекс и выводим отладочную информацию
                    if not date_found:
                        print(f"  Не удалось распарсить дату из строки: '{line}'")
                        current_map['date'] = None

                except Exception as e:
                    print(f"  Ошибка парсинга даты: {e}")
                    print(f"  Проблемная строка: '{line}'")
                    current_map['date'] = None

            elif 'LAT/LON1/LON2/DLON/H' in line and in_tec_map:
                try:
                    # Используем регулярные выражения для извлечения чисел из слипшихся строк
                    numbers = re.findall(r'-?\d+\.?\d*', line)

                    if len(numbers) >= 5:
                        lat = float(numbers[0])
                        lon1 = float(numbers[1])
                        lon2 = float(numbers[2])
                        dlon = float(numbers[3])
                        height = float(numbers[4])
                    else:
                        # Альтернативный метод: фиксированные позиции
                        lat = float(line[2:8].strip() or '0')
                        lon1 = float(line[8:14].strip() or '0')
                        lon2 = float(line[14:20].strip() or '0')
                        dlon = float(line[20:26].strip() or '0')
                        height = float(line[26:32].strip() or '0')

                    n_lons = int((lon2 - lon1) / dlon) + 1

                    # Читаем данные TEC
                    tec_values = []
                    i += 1
                    rows_read = 0

                    while len(tec_values) < n_lons and i < len(lines):
                        data_line = lines[i]

                        # Проверяем, не началась ли новая секция
                        if any(marker in data_line for marker in [
                            'LAT/LON1/LON2/DLON/H',
                            'END OF TEC MAP',
                            'START OF TEC MAP',
                            'EPOCH OF CURRENT MAP'
                        ]):
                            break

                        # Извлекаем все числа из строки
                        line_numbers = re.findall(r'-?\d+', data_line)
                        if line_numbers:
                            tec_values.extend([float(x) for x in line_numbers])
                            rows_read += 1
                        i += 1

                    # Сохраняем данные для текущей широты
                    current_data.append({
                        'lat': lat,
                        'lon_start': lon1,
                        'lon_end': lon2,
                        'dlon': dlon,
                        'height': height,
                        'tec': tec_values
                    })

                    print(f"  Широта {lat}: прочитано {len(tec_values)} значений TEC")

                    continue  # Не увеличиваем i, так как уже увеличили в цикле

                except Exception as e:
                    print(f"  Ошибка парсинга координат: {e}")
                    print(f"  Проблемная строка: {line}")
                    i += 1

            elif 'END OF TEC MAP' in line and in_tec_map:
                in_tec_map = False
                if current_map and current_data:
                    # Если дата не была установлена, используем индекс и пытаемся восстановить время
                    if 'date' not in current_map or current_map['date'] is None:
                        # Пытаемся восстановить время на основе индекса и известных времен
                        if len(result) > 0 and isinstance(result[0]['date'], datetime):
                            base_time = result[0]['date']
                            time_delta = timedelta(hours=2 * len(result))  # Предполагаем интервал 2 часа
                            estimated_time = base_time + time_delta
                            current_map['date'] = estimated_time
                            print(f"  Восстановлено время: {estimated_time}")
                        else:
                            current_map['date'] = f"map_{len(result):03d}"
                    current_map['data'] = current_data
                    result.append(current_map)
                    print(f"  Карта #{maps_found} успешно добавлена ({len(current_data)} широт)")
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
    """Создает матрицу TEC из данных карты и конвертирует в TECU (делит на 10)"""
    if not tec_map or 'data' not in tec_map or not tec_map['data']:
        return [], [], np.array([])

    # Извлекаем широты
    lats = sorted([item['lat'] for item in tec_map['data']])

    # Используем параметры из первой записи для долгот
    if not tec_map['data']:
        return [], [], np.array([])

    first_item = tec_map['data'][0]
    lons = np.arange(first_item['lon_start'],
                     first_item['lon_end'] + first_item['dlon'],
                     first_item['dlon'])

    # Создаем матрицу TEC
    tec_matrix = np.zeros((len(lats), len(lons)))

    # Сортируем данные по широте
    sorted_data = sorted(tec_map['data'], key=lambda x: x['lat'])

    valid_rows = 0
    for i, data_item in enumerate(sorted_data):
        tec_values = np.array(data_item['tec'])
        # КОНВЕРТИРУЕМ В TECU (ДЕЛИМ НА 10)
        tec_values = tec_values / 7.5

        if len(tec_values) == len(lons):
            tec_matrix[i, :] = tec_values
            valid_rows += 1

    print(f"  Создана матрица TEC: {valid_rows}/{len(lats)} строк валидны")
    return lats, lons, tec_matrix

def plot_all_tec_maps(tec_maps, output_dir='tec_maps_fixed_scale'):
    """Строит все карты TEC с фиксированной цветовой шкалой от 0 до 100 TECU"""
    if not tec_maps:
        print("Нет данных для построения")
        return

    # Создаем папку для сохранения, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана папка: {output_dir}")


    # Фиксированные пределы цветовой шкалы
    vmin, vmax = -7, 120
    levels = np.linspace(vmin, vmax, 120)

    print(f"Начинаем построение {len(tec_maps)} карт TEC с фиксированной шкалой 0-100 TECU...")
    successful_plots = 0

    for i, tec_map in enumerate(tec_maps):
        print(f"Обработка карты {i + 1}/{len(tec_maps)}...")

        lats, lons, tec_matrix = create_tec_matrix(tec_map)

        if tec_matrix.size == 0 or np.all(tec_matrix == 0):
            print(f"  Пропускаем карту {i}: пустая матрица TEC")
            continue

        # Создаем сетку для построения
        Lon, Lat = np.meshgrid(lons, lats)

        # Создаем фигуру с картой мира на заднем плане
        fig = plt.figure(figsize=(16, 10))

        # Используем проекцию PlateCarree для географических данных
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Добавляем элементы карты
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, alpha=0.4, color='lightblue')
        ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')

        # Построение карты TEC поверх карты мира с ФИКСИРОВАННОЙ цветовой шкалой
        try:
            contour = ax.contourf(Lon, Lat, tec_matrix,
                                  levels=levels,
                                  cmap='jet',
                                  vmin=vmin,
                                  vmax=vmax,
                                  transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"  Ошибка построения контура: {e}")
            plt.close()
            continue

        # Добавляем сетку
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                     alpha=0.5, linestyle='--')
    
        gl.xlabel_style = {'size': 15, 'weight': 'bold'}  # Добавлено: размер подписей долготы
        gl.ylabel_style = {'size': 15, 'weight': 'bold'}

        # Добавляем цветовую шкалу
        cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8, aspect=30)
        cbar.set_label('TEC (TECU)', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=15)

        # Добавляем аннотацию с фактическим диапазоном значений
       # actual_min = np.min(tec_matrix)
        #actual_max = np.max(tec_matrix)
        #ax.text(0.02, 0.02, f'Actual range: {actual_min:.1f}-{actual_max:.1f} TECU',
         #       transform=ax.transAxes, fontsize=10,
          #      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Настройка заголовка
        if isinstance(tec_map['date'], datetime):
            date_str = tec_map['date'].strftime("%Y-%m-%d %H:%M:%S UTC")
            time_str = tec_map['date'].strftime("%H:%M")
            title = f'Карта ПЭС по данным CODE {date_str}'
            filename = f"tec_map_{tec_map['date'].strftime('%Y%m%d_%H%M%S')}.png"
        else:
            # Если дата не распарсилась, используем индекс и предполагаемое время
            if i < 5:  # Первые 5 карт имеют правильное время
                # Вычисляем предполагаемое время на основе индекса
                base_time = tec_maps[0]['date']  # Первая известная дата
                time_delta = timedelta(hours=2 * i)  # Интервал 2 часа
                estimated_time = base_time + time_delta
                date_str = estimated_time.strftime("%Y-%m-%d %H:%M:%S UTC (estimated)")
                time_str = estimated_time.strftime("%H:%M")
                title = f'Карта ПЭС по данным CODE {date_str}'
                filename = f"tec_map_{estimated_time.strftime('%Y%m%d_%H%M%S')}_estimated.png"
            else:
                date_str = str(tec_map['date'])
                title = f'Карта ПЭС по данным CODE {date_str}'
                filename = f"tec_map_{i:03d}.png"

        plt.title(title, fontsize=16, pad=20)

        output_path = os.path.join(output_dir, filename)

        # Сохраняем
        try:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # Закрываем фигуру, чтобы освободить память
            successful_plots += 1
            print(f"  Успешно сохранена: {filename}")
            #print(f"    Размер данных: {tec_matrix.shape}, TECU: {actual_min:.2f}-{actual_max:.2f}")
        except Exception as e:
            print(f"  Ошибка сохранения: {e}")
            plt.close()

    print(f"\nИтог: успешно построено {successful_plots} из {len(tec_maps)} карт")
    print(f"Все карты сохранены в папку: {output_dir}")


# Основная программа
if __name__ == "__main__":
    filename = "ionex/IACG0210.24I"

    print("=" * 60)
    print("ЧТЕНИЕ IONEX ФАЙЛА")
    print("=" * 60)

    tec_maps = read_ionex_fixed(filename)

    if tec_maps:
        print(f"\nУСПЕШНО ПРОЧИТАНО {len(tec_maps)} КАРТ TEC")

        # Показываем все доступные времена
        print("\nДоступные времена в файле:")
        for i, tec_map in enumerate(tec_maps):
            if isinstance(tec_map['date'], datetime):
                print(f"  Карта {i}: {tec_map['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"  Карта {i}: {tec_map['date']}")

        # Строим все карты и сохраняем в папку с ФИКСИРОВАННОЙ цветовой шкалой
        print("\n" + "=" * 60)
        print("ПОСТРОЕНИЕ КАРТ TEC С ФИКСИРОВАННОЙ ШКАЛОЙ 0-80 TECU")
        print("=" * 60)

        plot_all_tec_maps(tec_maps, output_dir='tec_maps_code')

    else:
        print("НЕ УДАЛОСЬ ПРОЧИТАТЬ ДАННЫЕ ИЗ ФАЙЛА")