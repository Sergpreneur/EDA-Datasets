"""
CROSS-VALIDATION SPLITS — все варианты для DSI.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)


# =========================================================
# 1. STANDARD K-FOLD (iid данные)
# =========================================================
def kfold_cv(model, X, y, n_splits=5, scoring='neg_mean_squared_error', random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


# =========================================================
# 2. STRATIFIED K-FOLD (классификация)
# =========================================================
def stratified_cv(model, X, y, n_splits=5, scoring='average_precision', random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


# =========================================================
# 3. GROUP K-FOLD
# =========================================================
def group_cv(model, X, y, groups, n_splits=5, scoring='neg_mean_squared_error'):
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


# =========================================================
# 4. TIME SERIES SPLIT (с embargo)
# =========================================================
def timeseries_cv(model, X, y, n_splits=5, gap=20, test_size=None, scoring='neg_mean_squared_error'):
    """
    gap = embargo period (anti-leak)
    test_size = размер val в каждом фолде
    """
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


# =========================================================
# 5. WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_split(n_samples, train_size, test_size, step=None):
    """
    Generator: yield (train_idx, test_idx) tuples.

    train_size, test_size — в количестве наблюдений.
    step — шаг сдвига (default: test_size).
    """
    if step is None:
        step = test_size

    starts = range(0, n_samples - train_size - test_size + 1, step)
    for s in starts:
        train_idx = np.arange(s, s + train_size)
        test_idx = np.arange(s + train_size, s + train_size + test_size)
        yield train_idx, test_idx


def walk_forward_cv(model_factory, X, y, train_size, test_size, step=None,
                    metric=None, refit=True):
    """
    model_factory — callable, возвращающий новый instance модели.
    metric — функция (y_true, y_pred) → score. Default MSE.
    """
    from sklearn.metrics import mean_squared_error
    if metric is None:
        metric = lambda y_t, y_p: mean_squared_error(y_t, y_p)

    scores = []
    for tr_idx, te_idx in walk_forward_split(len(X), train_size, test_size, step):
        X_tr = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
        X_te = X.iloc[te_idx] if hasattr(X, 'iloc') else X[te_idx]
        y_te = y.iloc[te_idx] if hasattr(y, 'iloc') else y[te_idx]

        model = model_factory() if refit else model_factory
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        scores.append(metric(y_te, pred))

    return np.array(scores)


# =========================================================
# 6. PURGED K-FOLD (de Prado, для overlapping labels)
# =========================================================
def purged_kfold_indices(t1: pd.Series, n_splits=5, embargo_pct=0.01):
    """
    t1: Series, индексирована точками наблюдений, значения = когда наблюдение «закончилось»
        (например, для 5-day forward return: t1 = entry_date + 5 days)

    Возвращает: list of (train_idx, test_idx).
    """
    n = len(t1)
    embargo = int(n * embargo_pct)
    indices = np.arange(n)

    test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, n_splits)]

    splits = []
    for start, end in test_ranges:
        test_idx = indices[start:end]
        test_t1_max = t1.iloc[test_idx].max()
        test_t0_min = t1.index[start]

        # PURGE: удалить из train точки с label_period ∩ test_period ≠ ∅
        # EMBARGO: удалить точки в embargo после test
        train_mask = np.ones(n, dtype=bool)
        train_mask[start:end] = False

        # Purge before
        for i in range(start):
            if t1.iloc[i] >= test_t0_min:
                train_mask[i] = False

        # Embargo after
        embargo_end = min(end + embargo, n)
        train_mask[end:embargo_end] = False

        train_idx = indices[train_mask]
        splits.append((train_idx, test_idx))

    return splits


# =========================================================
# 7. ШАБЛОНЫ ДЛЯ DSI
# =========================================================
"""
# Регрессия (iid):
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])
kfold_cv(pipe, X, y, n_splits=5, scoring='neg_root_mean_squared_error')

# Дисбаланс классов:
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced')),
])
stratified_cv(pipe, X, y, scoring='average_precision')  # PR-AUC

# Time series:
import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
timeseries_cv(model, X, y, n_splits=5, gap=20)

# Walk-forward для quant фактора:
from scipy.stats import spearmanr

def make_model():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)

ic_scores = walk_forward_cv(
    make_model, X, y,
    train_size=504, test_size=21, step=21,
    metric=lambda y_t, y_p: spearmanr(y_p, y_t).correlation
)
print(f'Mean IC: {ic_scores.mean():.4f}')
print(f'ICIR: {ic_scores.mean() / ic_scores.std():.2f}')
"""
