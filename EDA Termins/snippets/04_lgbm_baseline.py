"""
LIGHTGBM BASELINE — готовые шаблоны для регрессии и классификации.

Дефолты подобраны для tabular данных среднего размера (10K – 1M).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold


# =========================================================
# 1. РЕГРЕССИЯ — БАЗОВЫЙ ШАБЛОН
# =========================================================
def lgbm_regression_baseline(X_train, y_train, X_val, y_val, params=None):
    """
    Baseline LightGBM для регрессии с early stopping.
    """
    default_params = dict(
        objective='regression',     # 'regression_l1' для MAE, 'huber' для робастности
        metric='rmse',
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,              # главный complexity param
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# =========================================================
# 2. КЛАССИФИКАЦИЯ — БАЗОВЫЙ ШАБЛОН
# =========================================================
def lgbm_classification_baseline(X_train, y_train, X_val, y_val, is_multiclass=False, params=None):
    default_params = dict(
        objective='binary' if not is_multiclass else 'multiclass',
        metric='binary_logloss' if not is_multiclass else 'multi_logloss',
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',    # для дисбалансных классов
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# =========================================================
# 3. K-FOLD OUT-OF-FOLD PREDICTIONS
# =========================================================
def kfold_oof(X, y, n_splits=5, params=None, is_classifier=False, seed=42):
    """
    K-Fold с out-of-fold predictions. Стандартный подход для Kaggle и DSI.

    Возвращает:
      oof_preds: np.array (len(X),) — предсказания на out-of-fold
      models: list — обученные модели по фолдам
      mean_score: float — усреднённая метрика
    """
    from sklearn.metrics import mean_squared_error, log_loss

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if is_classifier:
        oof_preds = np.zeros(len(X))
    else:
        oof_preds = np.zeros(len(X))

    models = []
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if is_classifier:
            model = lgbm_classification_baseline(X_tr, y_tr, X_va, y_va, params=params)
            preds = model.predict_proba(X_va)[:, 1]
            score = log_loss(y_va, preds)
        else:
            model = lgbm_regression_baseline(X_tr, y_tr, X_va, y_va, params=params)
            preds = model.predict(X_va)
            score = mean_squared_error(y_va, preds, squared=False)

        oof_preds[va_idx] = preds
        models.append(model)
        scores.append(score)
        print(f'Fold {fold}: {score:.4f}')

    mean_score = np.mean(scores)
    print(f'\nMean: {mean_score:.4f} ± {np.std(scores):.4f}')
    return oof_preds, models, mean_score


# =========================================================
# 4. FEATURE IMPORTANCE
# =========================================================
def plot_importance(model, feature_names=None, top_n=30, importance_type='gain'):
    """Визуализация feature importance."""
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = model.booster_.feature_name()

    importance = model.booster_.feature_importance(importance_type=importance_type)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.25)))
    plt.barh(imp_df['feature'][::-1], imp_df['importance'][::-1])
    plt.xlabel(f'Importance ({importance_type})')
    plt.tight_layout()
    plt.show()
    return imp_df


# =========================================================
# 5. КАСТОМНАЯ МЕТРИКА (например, для quant: Spearman IC)
# =========================================================
def custom_ic_metric(y_pred, y_true):
    """LightGBM custom metric: Spearman IC."""
    from scipy.stats import spearmanr
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()
    ic = spearmanr(y_pred, y_true).correlation
    return 'ic', ic, True  # name, value, is_higher_better


# Использование:
# model = lgb.LGBMRegressor(...)
# model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=custom_ic_metric)


# =========================================================
# 6. ОБЪЕКТИВЫ ДЛЯ РАЗНЫХ ЗАДАЧ
# =========================================================
"""
Регрессия:
  'regression'       — MSE (default)
  'regression_l1'    — MAE
  'huber'            — Huber loss (робастно к выбросам)
  'quantile'         — Quantile regression (alpha=0.5 = MAE, alpha=0.9 = upper bound)
  'poisson'          — счётчики
  'gamma'            — для тяжёлых хвостов

Классификация:
  'binary'           — бинарная
  'multiclass'       — мультиклассовая (нужен num_class)

Ranking:
  'lambdarank'       — обучение ранжированию (для рекомендательных систем)
"""


# =========================================================
# 7. ТЮНИНГ (Optuna)
# =========================================================
"""
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    }
    _, _, score = kfold_oof(X_train, y_train, n_splits=5, params=params)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(study.best_params)
"""
