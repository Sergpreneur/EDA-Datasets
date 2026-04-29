"""
EDA TEMPLATE — копипаст в начало DSI-ноутбука.

Покрывает первые 10–15 минут осмотра данных:
shape → dtypes → NaN → describe → distributions → target → correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')

# =========================================================
# 1. ЗАГРУЗКА И ПЕРВЫЙ ОСМОТР
# =========================================================
df = pd.read_csv('data.csv')   # или pd.read_parquet, pd.read_pickle

print('=' * 60)
print(f'SHAPE: {df.shape}')
print('=' * 60)
print('\nDTYPES:')
print(df.dtypes)
print('\nMEMORY:')
df.info(memory_usage='deep')

# Первые/последние/случайные строки — часто там сюрпризы
print('\n--- HEAD ---')
print(df.head())
print('\n--- TAIL ---')
print(df.tail())
print('\n--- SAMPLE ---')
print(df.sample(min(10, len(df)), random_state=42))


# =========================================================
# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# =========================================================
print('\n--- DESCRIBE (numeric) ---')
print(df.describe().T)
print('\n--- DESCRIBE (object/category) ---')
print(df.describe(include=['object', 'category']).T)


# =========================================================
# 3. ДУБЛИКАТЫ
# =========================================================
print(f'\nFull duplicates: {df.duplicated().sum()}')

# Если есть ключ:
# print(f'Duplicates by id: {df.duplicated(subset=["id"]).sum()}')
# Если time + entity:
# print(f'Duplicates by (date, ticker): {df.duplicated(subset=["date","ticker"]).sum()}')


# =========================================================
# 4. ПРОПУСКИ
# =========================================================
nan_pct = (df.isna().mean() * 100).sort_values(ascending=False)
print('\n--- NaN % per column (top 20) ---')
print(nan_pct[nan_pct > 0].head(20))

# Heatmap пропусков (если установлен missingno):
# import missingno as msno
# msno.matrix(df.sample(min(1000, len(df))))
# msno.heatmap(df)


# =========================================================
# 5. РАЗДЕЛЕНИЕ КОЛОНОК
# =========================================================
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

print(f'\nNumeric: {len(num_cols)}')
print(f'Categorical: {len(cat_cols)}')
print(f'Datetime: {len(date_cols)}')


# =========================================================
# 6. РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ
# =========================================================
print('\n--- SKEW (sorted) ---')
print(df[num_cols].skew().sort_values())
print('\n--- KURTOSIS (sorted) ---')
print(df[num_cols].kurtosis().sort_values())

# Hist + box для топ-N числовых:
# top_num = num_cols[:8]
# fig, axes = plt.subplots(len(top_num), 2, figsize=(12, 3 * len(top_num)))
# for i, col in enumerate(top_num):
#     df[col].hist(bins=50, ax=axes[i, 0])
#     axes[i, 0].set_title(f'{col} — hist')
#     df[col].plot.box(ax=axes[i, 1])
#     axes[i, 1].set_title(f'{col} — box')
# plt.tight_layout()


# =========================================================
# 7. КАТЕГОРИАЛЬНЫЕ
# =========================================================
print('\n--- CATEGORICAL CARDINALITY ---')
for col in cat_cols:
    n = df[col].nunique()
    print(f'  {col}: {n} unique')
    if n <= 20:
        print(df[col].value_counts(normalize=True).head(10).to_string())
    print('-' * 40)


# =========================================================
# 8. АНАЛИЗ ТАРГЕТА
# =========================================================
TARGET = 'y'  # ← ЗАМЕНИ на имя таргета

if TARGET in df.columns:
    y = df[TARGET]
    print(f'\n--- TARGET: {TARGET} ---')
    print(f'dtype: {y.dtype}')
    print(f'NaN: {y.isna().sum()}')

    if y.dtype in ('object', 'category') or y.nunique() < 20:
        # Классификация:
        print('\nClass distribution:')
        print(y.value_counts(normalize=True))
        majority = y.value_counts(normalize=True).iloc[0]
        print(f'\nMajority class baseline: {majority:.4f}')
    else:
        # Регрессия:
        print(f'\nMean: {y.mean():.4f}, Std: {y.std():.4f}')
        print(f'Min: {y.min():.4f}, Max: {y.max():.4f}')
        print(f'Skew: {y.skew():.4f}, Kurt: {y.kurtosis():.4f}')


# =========================================================
# 9. КОРРЕЛЯЦИИ (Spearman — стандарт для финансов)
# =========================================================
if TARGET in num_cols:
    corr_with_target = df[num_cols].corr(method='spearman')[TARGET].drop(TARGET)
    print('\n--- TOP 15 |Spearman| with target ---')
    print(corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index).head(15))


# =========================================================
# 10. ВЫСОКОКОРРЕЛИРОВАННЫЕ ПАРЫ ФИЧ
# =========================================================
if len(num_cols) > 1:
    corr = df[num_cols].corr(method='spearman').abs()
    pairs = corr.unstack().sort_values(ascending=False)
    pairs = pairs[pairs < 1].drop_duplicates()
    print('\n--- TOP 10 most correlated feature pairs ---')
    print(pairs.head(10))


# =========================================================
# QUICK DECISIONS:
# =========================================================
# - Пропусков много (>20%) в важной фиче? → indicator + median impute
# - |skew| > 1 в числовой? → log1p или Yeo-Johnson
# - High cardinality (>15) в категориальной? → target encoding или native
# - |corr| > 0.95 фичи с таргетом? → проверь leakage!
# - Дисбаланс таргета > 10:1? → class_weight='balanced', PR-AUC, Stratified CV
