"""МКР з Python for Data Science — наскрізний кейс «Метеослужба»."""

# ====================================================================
# Прізвище, ім'я, по батькові: Гогола Вікторія Володимирівна
# Група:                       ЗК-32
# Дата виконання:              2026-05-10
# ====================================================================

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

# Налаштування підключення
DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 33306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 15, delay: float = 3.0) -> pd.DataFrame:
    url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except (OperationalError, Exception):
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})... чекаємо...")
            time.sleep(delay)
    raise RuntimeError("Не вдалося підключитися до БД")


# ====================================================================
# БЛОК 1. NumPy (15 балів)
# ====================================================================

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    t = df_raw['temperature_c'].to_numpy()
    rh = df_raw['humidity_pct'].to_numpy()
    ws = df_raw['wind_speed_ms'].to_numpy()
    ids = df_raw['obs_id'].to_numpy()
    dt = df_raw['datetime'].to_numpy()

    # 1) Apparent temperature
    apparent = t - (100 - rh) / 5
    print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

    # 2) Заміна викидів
    t_clean = np.where((t > 60) | (t < -60), np.nan, t)
    ws_clean = np.where(ws > 100, np.nan, ws)

    t_out_cnt = np.sum(((t > 60) | (t < -60)) & ~np.isnan(t))
    ws_out_cnt = np.sum((ws > 100) & ~np.isnan(ws))
    print(f"2) Викидів температури замінено: {t_out_cnt}")
    print(f"   Викидів вітру замінено:       {ws_out_cnt}")

    # 3) Статистика вручну
    mean_t = np.nanmean(t_clean)
    median_t = np.nanmedian(t_clean)
    std_t = np.nanstd(t_clean)
    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # 4) Маски
    n_frost = np.sum(t_clean < 0)
    n_hot = np.sum(t_clean > 30)
    print(f"4) морозних: {n_frost}    жарких: {n_hot}")

    # 5) Argmax / Argmin
    idx_max = np.nanargmax(t_clean)
    idx_min = np.nanargmin(t_clean)
    print(f"5) Max T: {t_clean[idx_max]} (ID: {ids[idx_max]})")
    print(f"   Min T: {t_clean[idx_min]} (ID: {ids[idx_min]})")


# ====================================================================
# БЛОК 2. Pandas — очищення (20 балів)
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")
    rows_before = len(df_raw)
    df = df_raw.copy()

    # 2) Datetime index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    # 3) Дублікати
    n_dups = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнення NaN humidity
    df['month'] = df.index.month
    before_nan = df['humidity_pct'].isna().sum()
    df['humidity_pct'] = df.groupby(['city', 'month'])['humidity_pct'].transform(lambda s: s.fillna(s.median()))
    n_filled = before_nan - df['humidity_pct'].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Фізичні викиди
    mask = (df['temperature_c'].between(-60, 60)) & \
           ((df['wind_speed_ms'].isna()) | (df['wind_speed_ms'].between(0, 60)))
    df_new = df[mask].copy()
    n_outliers = len(df) - len(df_new)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    print(f"\n   Звіт: {rows_before} → {len(df_new)} рядків")
    return df_new


# ====================================================================
# БЛОК 3. Pandas — аналітика (30 балів)
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # 1) Середня T
    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values()
    print("1) Середня T по містах:\n", by_city_temp.round(2))

    # 2) Опади
    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values(ascending=False)
    print("\n2) Сумарні опади:\n", by_city_precip.round(1))

    # 3) Місячна динаміка (ME = Month End)
    monthly_mean = df.resample('ME')['temperature_c'].mean()

    # 4) Pivot
    pivot = df.pivot_table(index='city', columns='month', values='temperature_c', aggfunc='mean')

    # 5) Дні з опадами > 5мм
    rainy_days = df.groupby(['city', df.index.date])['precipitation_mm'].sum()
    rainy_days_count = rainy_days[rainy_days > 5].groupby('city').count()
    print("\n5) Дні з опадами > 5 мм:\n", rainy_days_count)

    # 6) Аномальний місяць
    # Рахуємо норму для кожного місяця (1-12)
    monthly_norm = df.groupby(df.index.month)['temperature_c'].mean()
    # Рахуємо середню для кожного конкретного (рік, місяць)
    actual_monthly = df.groupby([df.index.year, df.index.month])['temperature_c'].mean()

    deviations = []
    for (yr, mo), val in actual_monthly.items():
        dev = val - monthly_norm[mo]
        deviations.append(((yr, mo), dev))

    anomaly_month, anomaly_dev = max(deviations, key=lambda x: abs(x[1]))
    print(f"\n6) Аномальний місяць: {anomaly_month} відхилення = {anomaly_dev:+.2f}°C")

    return {"by_city_temp": by_city_temp, "by_city_precip": by_city_precip,
            "monthly_mean": monthly_mean, "pivot": pivot, "monthly_norm": monthly_norm}


# ====================================================================
# БЛОК 4. Matplotlib (35 балів)
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")
    plt.style.use('ggplot')

    # 1. Line Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    cities = df['city'].unique()[:3]
    for city in cities:
        city_data = df[df['city'] == city].resample('ME')['temperature_c'].mean()
        ax.plot(city_data.index, city_data.values, label=city, marker='o', markersize=4)
    ax.set_title("Місячна динаміка температури (обрані міста)")
    ax.set_ylabel("Температура (°C)")
    ax.legend()
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120)

    # 2. Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    analytics['by_city_precip'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Сумарні опади по містах")
    ax.set_ylabel("Опади (мм)")
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120)

    # 3. Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df['temperature_c'].dropna(), bins=30, color='salmon', edgecolor='black', alpha=0.7)
    mean_val = df['temperature_c'].mean()
    median_val = df['temperature_c'].median()
    ax.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.1f}')
    ax.set_title("Розподіл температур")
    ax.legend()
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120)

    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(analytics['pivot'], cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(analytics['pivot'].columns)))
    ax.set_xticklabels(analytics['pivot'].columns)
    ax.set_yticks(np.arange(len(analytics['pivot'].index)))
    ax.set_yticklabels(analytics['pivot'].index)
    plt.colorbar(im, label='Середня T (°C)')
    ax.set_title("Теплокарта: Місто vs Місяць")
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120)

    print(f"Усі графіки збережено в {PLOTS_DIR}/")


def main() -> None:
    try:
        df_raw = load_observations()
        block_1_numpy(df_raw)
        df_clean = block_2_cleaning(df_raw)
        analytics = block_3_analytics(df_clean)
        block_4_plots(df_clean, analytics)
    except Exception as e:
        print(f"Помилка під час виконання: {e}")


if __name__ == "__main__":
    main()

"""
ВИСНОВКИ (5–8 речень).

Проведений аналіз метеорологічних даних за 2023–2024 роки демонструє характерні кліматичні особливості регіонів України. 
Найхолоднішим містом прогнозовано виявився Харків (8.08°C), що пояснюється його північно-східним розташуванням, 
тоді як Львів продемонстрував найвищу середню температуру (12.65°C), що може бути пов'язано зі специфікою 
локального рельєфу та загальними тенденціями потепління.

Сезонність чітко виражена, проте спостерігаються значні температурні аномалії. Зокрема, травень 2023 року виявився 
аномально теплим із відхиленням +3.69°C від норми, що підтверджує вплив глобальних чинників, таких як Ель-Ніньо. 
У питанні опадів спостерігається значний контраст: Київ є найвологішим містом (561.3 мм), маючи 30 днів з інтенсивними 
опадами, тоді як Дніпро демонструє посушливу тенденцію — лише 4 дні з опадами понад 5 мм за весь період. 

На основі цих даних можна рекомендувати Дніпру посилення систем зрошення для агросектору, а Києву та Львову — 
модернізацію зливових каналізацій через високу частоту інтенсивних дощів.
"""
