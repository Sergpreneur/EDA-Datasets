"""
ОЦЕНКА ALPHA-ФАКТОРА — полный pipeline.

Стандарт quant-research для DSI:
1. Рассчитать фактор
2. Cross-sectional standardization
3. IC, ICIR, t-stat
4. Quintile portfolios
5. Long-short performance
6. Sanity checks
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


# =========================================================
# 1. ПОДГОТОВКА ДАННЫХ
# =========================================================
def prepare_panel(df, date_col='date', entity_col='ticker', factor_col='factor', return_col='fwd_return'):
    """Sanity checks для panel-данных."""
    print(f'Shape: {df.shape}')
    print(f'Dates: {df[date_col].nunique()} ({df[date_col].min()} → {df[date_col].max()})')
    print(f'Entities: {df[entity_col].nunique()}')
    print(f'Factor NaN: {df[factor_col].isna().sum()} ({df[factor_col].isna().mean():.2%})')
    print(f'Return NaN: {df[return_col].isna().sum()} ({df[return_col].isna().mean():.2%})')

    # Дубликаты:
    dups = df.duplicated(subset=[date_col, entity_col]).sum()
    print(f'Duplicates (date, entity): {dups}')


# =========================================================
# 2. CROSS-SECTIONAL TRANSFORMATIONS
# =========================================================
def cs_zscore(df, col, date_col='date'):
    return df.groupby(date_col)[col].transform(lambda x: (x - x.mean()) / x.std())


def cs_rank(df, col, date_col='date', pct=True):
    return df.groupby(date_col)[col].rank(pct=pct)


def cs_winsorize(df, col, date_col='date', lower=0.01, upper=0.99):
    def w(x):
        lo, hi = x.quantile([lower, upper])
        return x.clip(lo, hi)
    return df.groupby(date_col)[col].transform(w)


def cs_demean(df, col, date_col='date'):
    """Subtract cross-sectional mean — useful for market-neutral factors."""
    return df.groupby(date_col)[col].transform(lambda x: x - x.mean())


# =========================================================
# 3. INFORMATION COEFFICIENT
# =========================================================
def calculate_ic(df, factor_col, return_col, date_col='date', method='spearman'):
    """Returns Series of daily ICs."""
    def daily_corr(group):
        if method == 'spearman':
            return spearmanr(group[factor_col], group[return_col]).correlation
        else:
            return group[[factor_col, return_col]].corr().iloc[0, 1]

    return df.groupby(date_col).apply(daily_corr).dropna()


def ic_report(ic_series, periods_per_year=252):
    """Полная сводка IC."""
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    icir_annual = icir * np.sqrt(periods_per_year)
    t_stat, p_val = ttest_1samp(ic_series, 0)

    print('=' * 50)
    print('IC REPORT')
    print('=' * 50)
    print(f'Mean IC:        {mean_ic:.4f}')
    print(f'Std IC:         {std_ic:.4f}')
    print(f'ICIR:           {icir:.3f}')
    print(f'ICIR (annual):  {icir_annual:.3f}')
    print(f'IC > 0 rate:    {(ic_series > 0).mean():.2%}')
    print(f't-statistic:    {t_stat:.2f}')
    print(f'p-value:        {p_val:.4f}')
    print(f'N periods:      {len(ic_series)}')

    if abs(t_stat) > 2:
        print('✓ IC is statistically significant (|t| > 2)')
    else:
        print('✗ IC NOT significant (|t| ≤ 2)')


# =========================================================
# 4. QUINTILE ANALYSIS
# =========================================================
def quintile_returns(df, factor_col, return_col, date_col='date', n_quantiles=5):
    """
    Returns DataFrame: rows = dates, columns = quintiles, values = mean returns.
    """
    df = df.copy()
    df['_q'] = df.groupby(date_col)[factor_col].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )
    return df.groupby([date_col, '_q'])[return_col].mean().unstack()


def quintile_report(q_returns, periods_per_year=252):
    """Сводка по квинтилям + long-short."""
    n_q = q_returns.shape[1]

    summary = pd.DataFrame(index=q_returns.columns)
    summary['Mean'] = q_returns.mean()
    summary['Std'] = q_returns.std()
    summary['Sharpe'] = q_returns.mean() / q_returns.std() * np.sqrt(periods_per_year)
    summary['Cumulative'] = (1 + q_returns).prod() - 1

    print('=' * 60)
    print('QUINTILE PERFORMANCE')
    print('=' * 60)
    print(summary.round(4))

    # Long-short:
    ls = q_returns[n_q - 1] - q_returns[0]
    print(f'\n{"=" * 60}\nLONG-SHORT (Q{n_q - 1} - Q0)')
    print('=' * 60)
    print(f'Mean:           {ls.mean():.4f}')
    print(f'Std:            {ls.std():.4f}')
    print(f'Sharpe:         {ls.mean() / ls.std() * np.sqrt(periods_per_year):.2f}')
    print(f'Hit rate:       {(ls > 0).mean():.2%}')

    # Монотонность:
    means = summary['Mean'].values
    is_monotonic_inc = all(means[i] <= means[i+1] for i in range(len(means)-1))
    is_monotonic_dec = all(means[i] >= means[i+1] for i in range(len(means)-1))
    if is_monotonic_inc:
        print('✓ Monotonic increasing across quintiles')
    elif is_monotonic_dec:
        print('✓ Monotonic decreasing across quintiles')
    else:
        print('⚠ Non-monotonic — factor may be noisy or non-linear')

    return summary, ls


# =========================================================
# 5. ПОЛНЫЙ PIPELINE ОЦЕНКИ ФАКТОРА
# =========================================================
def evaluate_factor(df, factor_col, return_col, date_col='date',
                    standardize='zscore', winsorize_pct=0.01, n_quantiles=5,
                    periods_per_year=252):
    """
    End-to-end оценка alpha-фактора:
    1. Cross-sectional winsorize + standardize
    2. IC report
    3. Quintile analysis
    """
    df = df.copy()

    # 1. Winsorize:
    if winsorize_pct > 0:
        df[factor_col] = cs_winsorize(df, factor_col, date_col, winsorize_pct, 1 - winsorize_pct)

    # 2. Standardize:
    if standardize == 'zscore':
        df[factor_col] = cs_zscore(df, factor_col, date_col)
    elif standardize == 'rank':
        df[factor_col] = cs_rank(df, factor_col, date_col)

    # 3. IC:
    ic_series = calculate_ic(df, factor_col, return_col, date_col)
    ic_report(ic_series, periods_per_year)

    # 4. Quintiles:
    print()
    q_returns = quintile_returns(df, factor_col, return_col, date_col, n_quantiles)
    summary, ls = quintile_report(q_returns, periods_per_year)

    return {
        'ic_series': ic_series,
        'quintile_returns': q_returns,
        'long_short': ls,
        'summary': summary,
    }


# =========================================================
# 6. STABILITY CHECKS
# =========================================================
def factor_stability(df, factor_col, return_col, date_col='date', n_buckets=4):
    """
    Проверяет стабильность фактора во времени:
    делит период на N равных подпериодов и считает IC в каждом.
    """
    df = df.copy()
    dates = sorted(df[date_col].unique())
    bucket_size = len(dates) // n_buckets

    print('=' * 60)
    print(f'FACTOR STABILITY ({n_buckets} sub-periods)')
    print('=' * 60)

    results = []
    for i in range(n_buckets):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_buckets - 1 else len(dates)
        bucket_dates = dates[start:end]
        sub_df = df[df[date_col].isin(bucket_dates)]

        ic = calculate_ic(sub_df, factor_col, return_col, date_col)
        mean_ic = ic.mean()
        icir = mean_ic / ic.std() if ic.std() > 0 else 0

        period_str = f'{bucket_dates[0]} → {bucket_dates[-1]}'
        print(f'Period {i+1}: {period_str}')
        print(f'  Mean IC: {mean_ic:+.4f}, ICIR: {icir:+.2f}, N: {len(ic)}')
        results.append({'period': period_str, 'mean_ic': mean_ic, 'icir': icir})

    return pd.DataFrame(results)


# =========================================================
# 7. ИСПОЛЬЗОВАНИЕ
# =========================================================
"""
# Полный pipeline:
results = evaluate_factor(
    df,
    factor_col='my_factor',
    return_col='fwd_return_5d',
    date_col='date',
    standardize='zscore',     # 'zscore' | 'rank' | None
    winsorize_pct=0.01,
    n_quantiles=5,
)

# Stability check:
stability = factor_stability(df, 'my_factor', 'fwd_return_5d', 'date', n_buckets=4)

# Если ICIR > 0.5 во всех периодах → крепкий фактор
# Если IC сильно колеблется → нестабильный, возможно overfit к одному режиму
"""
