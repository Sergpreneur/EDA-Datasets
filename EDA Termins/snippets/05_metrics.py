"""
МЕТРИКИ — регрессия, классификация, quant.

Готовые функции для копипаста.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =========================================================
# 1. РЕГРЕССИЯ
# =========================================================
def regression_metrics(y_true, y_pred):
    """Полный набор метрик регрессии."""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error,
        r2_score, mean_absolute_percentage_error
    )
    return pd.Series({
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Pearson': np.corrcoef(y_true, y_pred)[0, 1],
        'Spearman': spearmanr(y_true, y_pred).correlation,
    })


# =========================================================
# 2. КЛАССИФИКАЦИЯ
# =========================================================
def classification_metrics(y_true, y_pred, y_proba=None):
    """Полный набор метрик классификации."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, log_loss, brier_score_loss,
        confusion_matrix, classification_report,
    )

    metrics = pd.Series({
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
    })

    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        metrics['PR-AUC'] = average_precision_score(y_true, y_proba)
        metrics['LogLoss'] = log_loss(y_true, y_proba)
        metrics['Brier'] = brier_score_loss(y_true, y_proba)

    return metrics


def confusion_summary(y_true, y_pred):
    """Confusion matrix + classification report."""
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:')
    print(pd.DataFrame(cm,
                       index=['Actual Neg', 'Actual Pos'],
                       columns=['Pred Neg', 'Pred Pos']))
    print('\n' + classification_report(y_true, y_pred))


# =========================================================
# 3. QUANT МЕТРИКИ
# =========================================================
def sharpe(returns, periods_per_year=252, rf=0):
    """Annualized Sharpe ratio."""
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return 0
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def sortino(returns, periods_per_year=252, rf=0):
    """Sortino ratio (downside-only volatility)."""
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf
    return excess.mean() / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(returns):
    """Max drawdown of cumulative returns."""
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    return drawdown.min()


def calmar(returns, periods_per_year=252):
    """Annual return / |max drawdown|."""
    annual_return = returns.mean() * periods_per_year
    mdd = abs(max_drawdown(returns))
    return annual_return / mdd if mdd > 0 else 0


def hit_rate(y_pred, y_true):
    """Доля правильных направлений."""
    return (np.sign(y_pred) == np.sign(y_true)).mean()


def information_ratio(portfolio_returns, benchmark_returns, periods_per_year=252):
    """Sharpe vs benchmark."""
    active = portfolio_returns - benchmark_returns
    if active.std() == 0:
        return 0
    return active.mean() / active.std() * np.sqrt(periods_per_year)


def strategy_summary(returns, periods_per_year=252):
    """Полная сводка стратегии."""
    return pd.Series({
        'Annual return': returns.mean() * periods_per_year,
        'Annual volatility': returns.std() * np.sqrt(periods_per_year),
        'Sharpe': sharpe(returns, periods_per_year),
        'Sortino': sortino(returns, periods_per_year),
        'Max drawdown': max_drawdown(returns),
        'Calmar': calmar(returns, periods_per_year),
        'Hit rate': (returns > 0).mean(),
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Best day': returns.max(),
        'Worst day': returns.min(),
    })


# =========================================================
# 4. INFORMATION COEFFICIENT
# =========================================================
def daily_ic(df, factor_col, return_col, date_col='date'):
    """Возвращает Series ежедневного Spearman IC."""
    return df.groupby(date_col).apply(
        lambda x: spearmanr(x[factor_col], x[return_col]).correlation
    ).dropna()


def ic_summary(ic_series, periods_per_year=252):
    """Сводка по IC: mean, std, ICIR, t-stat."""
    from scipy.stats import ttest_1samp

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    icir_annual = icir * np.sqrt(periods_per_year)
    t_stat, p_val = ttest_1samp(ic_series.dropna(), 0)

    return pd.Series({
        'Mean IC': mean_ic,
        'Std IC': std_ic,
        'ICIR': icir,
        'ICIR (annual)': icir_annual,
        'IC > 0 rate': (ic_series > 0).mean(),
        't-stat': t_stat,
        'p-value': p_val,
        'N periods': len(ic_series),
    })


# =========================================================
# 5. QUINTILE ANALYSIS
# =========================================================
def quintile_analysis(df, factor_col, return_col, date_col='date', n_quantiles=5):
    """
    Делит на квинтили внутри даты, возвращает mean return по квинтилям.
    """
    df = df.copy()
    df['quantile'] = df.groupby(date_col)[factor_col].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )

    quantile_returns = df.groupby(['date', 'quantile'])[return_col].mean().unstack()

    summary = pd.DataFrame({
        'Mean return': quantile_returns.mean(),
        'Std': quantile_returns.std(),
        'Sharpe': [sharpe(quantile_returns[q].dropna()) for q in quantile_returns.columns],
    })
    summary.index.name = f'Quintile (0=lowest, {n_quantiles-1}=highest)'

    # Long-short:
    ls = quantile_returns[n_quantiles - 1] - quantile_returns[0]
    print(f'\nLong-Short (Q{n_quantiles-1} - Q0):')
    print(f'  Mean: {ls.mean():.4f}')
    print(f'  Sharpe: {sharpe(ls):.2f}')

    return summary, ls


# =========================================================
# 6. ИСПОЛЬЗОВАНИЕ
# =========================================================
"""
# Регрессия:
print(regression_metrics(y_true, y_pred))

# Классификация:
print(classification_metrics(y_true, y_pred, y_proba))
confusion_summary(y_true, y_pred)

# Стратегия:
print(strategy_summary(daily_returns))

# IC анализ фактора:
ic = daily_ic(df, 'factor_z', 'fwd_return_5d', 'date')
print(ic_summary(ic))

# Quintile spread:
summary, ls_returns = quintile_analysis(df, 'factor_z', 'fwd_return', 'date')
print(summary)
"""
