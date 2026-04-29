"""
BOOTSTRAP — для CI любой статистики.

Когда применять:
- Sharpe ratio CI (для него нет аналитической формулы)
- Медиана, max drawdown, IR — любая custom статистика
- Маленькие выборки, где параметрические тесты ненадёжны
- Нет уверенности в распределении
"""

import numpy as np
import pandas as pd


# =========================================================
# 1. БАЗОВЫЙ BOOTSTRAP
# =========================================================
def bootstrap_ci(data, statistic=np.mean, B=10000, alpha=0.05, random_state=42):
    """
    Bootstrap confidence interval для любой статистики.

    Args:
        data: array-like
        statistic: callable, принимает array, возвращает scalar
        B: число бутстрап-итераций
        alpha: 1 − confidence_level (0.05 для 95% CI)
        random_state: seed

    Returns:
        (point_estimate, lower_bound, upper_bound, std)
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)

    boots = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(B)
    ])

    point = statistic(data)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = np.std(boots, ddof=1)

    return {'point': point, 'lower': lo, 'upper': hi, 'se': se, 'samples': boots}


# =========================================================
# 2. SCIPY BOOTSTRAP (>= 1.8)
# =========================================================
def scipy_bootstrap(data, statistic=np.mean, n_resamples=10000, alpha=0.05):
    """Через scipy.stats.bootstrap (рекомендуется)."""
    from scipy.stats import bootstrap

    res = bootstrap(
        (data,), statistic,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method='percentile',  # или 'BCa' (bias-corrected accelerated)
    )
    return {
        'lower': res.confidence_interval.low,
        'upper': res.confidence_interval.high,
        'se': res.standard_error,
    }


# =========================================================
# 3. PERMUTATION TEST
# =========================================================
def permutation_test_2samp(x, y, statistic=lambda a, b: a.mean() - b.mean(),
                           n_perm=10000, alternative='two-sided', random_state=42):
    """
    Permutation test для разности двух выборок.

    Args:
        x, y: array-like
        statistic: callable(a, b) → scalar
        n_perm: число пермутаций
        alternative: 'two-sided' | 'greater' | 'less'

    Returns:
        (observed_stat, p_value, perm_distribution)
    """
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)

    obs = statistic(x, y)
    combined = np.concatenate([x, y])
    n_x = len(x)

    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        perm_stats[i] = statistic(perm[:n_x], perm[n_x:])

    if alternative == 'two-sided':
        p = np.mean(np.abs(perm_stats) >= np.abs(obs))
    elif alternative == 'greater':
        p = np.mean(perm_stats >= obs)
    else:
        p = np.mean(perm_stats <= obs)

    return {'observed': obs, 'p_value': p, 'perm_dist': perm_stats}


# =========================================================
# 4. BLOCK BOOTSTRAP (для time series)
# =========================================================
def block_bootstrap(series, statistic, block_size=20, B=10000, random_state=42):
    """
    Stationary block bootstrap (Politis-Romano).
    Сохраняет автокорреляционную структуру.

    Args:
        series: 1D array (time series)
        statistic: callable(array) → scalar
        block_size: средняя длина блока
        B: число итераций
    """
    rng = np.random.default_rng(random_state)
    series = np.asarray(series)
    n = len(series)
    p = 1.0 / block_size  # geometric distribution parameter

    boots = np.empty(B)
    for b in range(B):
        # Generate stationary bootstrap sample:
        sample = []
        i = rng.integers(n)
        while len(sample) < n:
            sample.append(series[i % n])
            if rng.random() < p:
                i = rng.integers(n)  # restart from new block
            else:
                i += 1
        boots[b] = statistic(np.array(sample[:n]))

    return boots


# =========================================================
# 5. SHARPE RATIO CI ЧЕРЕЗ BOOTSTRAP
# =========================================================
def sharpe_ratio_ci(returns, periods_per_year=252, B=10000, alpha=0.05,
                    block_size=None, random_state=42):
    """
    Bootstrap CI для Sharpe ratio.
    Если block_size задан — block bootstrap (для autocorrelated returns).
    """
    def sharpe(x):
        if x.std() == 0:
            return 0
        return x.mean() / x.std() * np.sqrt(periods_per_year)

    if block_size is None:
        result = bootstrap_ci(returns, statistic=sharpe, B=B, alpha=alpha, random_state=random_state)
    else:
        boots = block_bootstrap(returns, sharpe, block_size=block_size, B=B, random_state=random_state)
        point = sharpe(np.asarray(returns))
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result = {'point': point, 'lower': lo, 'upper': hi, 'se': boots.std(), 'samples': boots}

    print(f'Sharpe: {result["point"]:.3f}')
    print(f'{int(100*(1-alpha))}% CI: [{result["lower"]:.3f}, {result["upper"]:.3f}]')
    print(f'SE: {result["se"]:.3f}')
    return result


# =========================================================
# 6. ИСПОЛЬЗОВАНИЕ
# =========================================================
"""
# CI для медианы:
result = bootstrap_ci(data, statistic=np.median, B=10000)
print(f'Median: {result["point"]:.3f} [{result["lower"]:.3f}, {result["upper"]:.3f}]')

# CI для Sharpe:
sharpe_ratio_ci(returns, periods_per_year=252)

# Block bootstrap для autocorrelated returns:
sharpe_ratio_ci(returns, periods_per_year=252, block_size=20)

# Permutation test (вместо t-test):
result = permutation_test_2samp(group_a, group_b)
print(f'Observed diff: {result["observed"]:.4f}')
print(f'p-value: {result["p_value"]:.4f}')

# CI для max drawdown:
def mdd(returns):
    cum = (1 + returns).cumprod()
    return (cum / np.maximum.accumulate(cum) - 1).min()

result = bootstrap_ci(returns, statistic=mdd, B=10000)
"""
