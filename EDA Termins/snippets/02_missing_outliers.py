"""
ОБРАБОТКА ПРОПУСКОВ И ВЫБРОСОВ.

Готовые функции для копипаста.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest


# =========================================================
# 1. ПРОПУСКИ — INDICATOR TRICK
# =========================================================
def add_missing_indicators(df, threshold=0.0):
    """
    Для каждой колонки с пропусками > threshold добавляет
    индикатор 'col_was_nan' и заполняет медианой/модой.

    Полезно когда missingness сама по себе информативна (MNAR).
    """
    df = df.copy()
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > threshold:
            df[f'{col}_was_nan'] = df[col].isna().astype(int)
            if df[col].dtype in (np.float64, np.int64):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'missing')
    return df


# =========================================================
# 2. PIPELINE-БЕЗОПАСНАЯ ИМПУТАЦИЯ
# =========================================================
# Эти imputers нужно засунуть в Pipeline — иначе утечка через статистики.

simple_imputer = SimpleImputer(strategy='median')         # для числовых
mode_imputer = SimpleImputer(strategy='most_frequent')    # для категориальных
knn_imputer = KNNImputer(n_neighbors=5)                   # медленно, для коррелирующих фич
iter_imputer = IterativeImputer(max_iter=10, random_state=42)  # MICE


# =========================================================
# 3. ВЫБРОСЫ — IQR МЕТОД
# =========================================================
def iqr_outliers(series, k=1.5):
    """Возвращает boolean mask: True для outliers."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def iqr_bounds(series, k=1.5):
    """Возвращает (lower, upper) для clip."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# =========================================================
# 4. WINSORIZATION
# =========================================================
def winsorize_quantiles(series, lower=0.01, upper=0.99):
    """Clip values at percentiles."""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def winsorize_df(df, cols, lower=0.01, upper=0.99):
    """Apply winsorization к нескольким колонкам."""
    df = df.copy()
    for col in cols:
        df[col] = winsorize_quantiles(df[col], lower, upper)
    return df


# =========================================================
# 5. Z-SCORE OUTLIERS (только если ~normal)
# =========================================================
def zscore_outliers(series, threshold=3):
    z = np.abs(zscore(series.dropna()))
    return z > threshold


# =========================================================
# 6. ISOLATION FOREST (multivariate)
# =========================================================
def isolation_forest_outliers(X, contamination=0.01, random_state=42):
    """Returns -1 for outlier, 1 for inlier."""
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    return iso.fit_predict(X.fillna(0))


# =========================================================
# 7. CROSS-SECTIONAL WINSORIZE (для quant)
# =========================================================
def cross_sectional_winsorize(df, factor_col, date_col='date', lower=0.01, upper=0.99):
    """Winsorize внутри каждой даты."""
    df = df.copy()
    df[factor_col] = df.groupby(date_col)[factor_col].transform(
        lambda x: winsorize_quantiles(x, lower, upper)
    )
    return df


# =========================================================
# 8. CROSS-SECTIONAL Z-SCORE
# =========================================================
def cross_sectional_zscore(df, factor_col, date_col='date'):
    """z-score внутри каждой даты."""
    df = df.copy()
    df[f'{factor_col}_z'] = df.groupby(date_col)[factor_col].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df


def cross_sectional_rank(df, factor_col, date_col='date', pct=True):
    """Rank внутри каждой даты."""
    df = df.copy()
    df[f'{factor_col}_rank'] = df.groupby(date_col)[factor_col].rank(pct=pct)
    return df


# =========================================================
# 9. RANK-NORMAL TRANSFORM
# =========================================================
def rank_normal(series):
    """Преобразует в normal-distributed ranks."""
    from scipy.stats import norm
    ranks = series.rank()
    return norm.ppf((ranks - 0.5) / len(series))


# =========================================================
# ИСПОЛЬЗОВАНИЕ:
# =========================================================
"""
# Пропуски с индикатором:
df = add_missing_indicators(df, threshold=0.05)

# Winsorize одну колонку:
df['x_clean'] = winsorize_quantiles(df['x'], 0.01, 0.99)

# Winsorize несколько:
df = winsorize_df(df, num_cols, lower=0.01, upper=0.99)

# Cross-sectional standardization (quant):
df = cross_sectional_zscore(df, 'factor', 'date')

# Pipeline (защита от leakage):
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])
"""
