# DSI Quant Cheatsheet

Быстрый справочник по статистике, EDA, моделям и quant-метрикам для DSI.

## Навигация

| # | Файл | Что внутри |
|---|------|------------|
| 1 | [01-statistical-tests.md](01-statistical-tests.md) | Все тесты: t-test, Mann-Whitney, ANOVA, KS, Shapiro, χ², Spearman, ADF, Ljung-Box, bootstrap, FDR |
| 2 | [02-eda.md](02-eda.md) | Workflow первого осмотра, пропуски, выбросы, преобразования, утечки данных |
| 3 | [03-models.md](03-models.md) | Linear (OLS, Ridge, Lasso), Logistic, Trees (RF, LGBM, XGBoost, CatBoost) |
| 4 | [04-validation.md](04-validation.md) | KFold, Stratified, GroupKFold, TimeSeriesSplit, walk-forward, Purged K-Fold |
| 5 | [05-metrics.md](05-metrics.md) | MSE/MAE/R², Precision/Recall/F1, ROC-AUC/PR-AUC, IC, ICIR, Sharpe, drawdown |
| 6 | [06-quant-specific.md](06-quant-specific.md) | Факторные модели, IC, ICIR, long-short, biases, walk-forward в финансах |
| 7 | [99-glossary.md](99-glossary.md) | Глоссарий всех терминов |

## Готовые сниппеты для копипаста

| Файл | Когда использовать |
|------|--------------------|
| [snippets/01_eda_template.py](snippets/01_eda_template.py) | Первые 5 минут после загрузки данных |
| [snippets/02_missing_outliers.py](snippets/02_missing_outliers.py) | Обработка пропусков и выбросов |
| [snippets/03_cv_splits.py](snippets/03_cv_splits.py) | Все варианты CV (включая time series) |
| [snippets/04_lgbm_baseline.py](snippets/04_lgbm_baseline.py) | LightGBM с правильными дефолтами |
| [snippets/05_metrics.py](snippets/05_metrics.py) | Метрики регрессии, классификации, quant |
| [snippets/06_factor_eval.py](snippets/06_factor_eval.py) | Оценка фактора: IC, ICIR, quintiles |
| [snippets/07_bootstrap.py](snippets/07_bootstrap.py) | Bootstrap CI для любой статистики |

## Дерево решений: какой тест выбрать

### Сравнение средних

| Сценарий | Параметрический | Непараметрический |
|----------|----------------|--------------------|
| 1 выборка vs константа | `ttest_1samp` | `wilcoxon` |
| 2 независимые | **`ttest_ind(equal_var=False)`** ← Welch | `mannwhitneyu` |
| 2 зависимые (paired) | `ttest_rel` | `wilcoxon` |
| > 2 независимых | `f_oneway` (ANOVA) | `kruskal` |
| > 2 зависимых | repeated-measures ANOVA | `friedmanchisquare` |

### Связь между переменными

| Тип | Тест |
|-----|------|
| числовая ↔ числовая (линейная, нормальные) | Pearson |
| числовая ↔ числовая (любая монотонная) | **Spearman** ← дефолт для финансов |
| числовая ↔ числовая (малая n) | Kendall τ |
| кат. ↔ кат. | χ² или Fisher exact |
| числовая ↔ бинарная | t-test / Mann-Whitney |
| числовая ↔ k категорий | ANOVA / Kruskal-Wallis |

### Распределения и временные ряды

| Вопрос | Тест |
|--------|------|
| Это нормальное распределение? | Shapiro-Wilk (n<5000), Jarque-Bera, D'Agostino |
| Два распределения одинаковы? | Kolmogorov-Smirnov 2-sample |
| Дисперсии равны? | Levene (median) |
| Ряд стационарен? | ADF + KPSS (оба) |
| Автокорреляция в остатках? | Ljung-Box |
| ARCH-эффект (кластеризация волатильности)? | Engle ARCH-LM |

## Дефолты для DSI

- **Сравнить две группы** → Welch's t-test (не Student's)
- **Корреляция фактор↔доходность** → Spearman (не Pearson)
- **Time series CV** → TimeSeriesSplit с gap, либо walk-forward
- **Скрининг факторов** → Benjamini-Hochberg (не Bonferroni)
- **Tree-based модель** → LightGBM с early stopping
- **Дисбаланс классов** → `class_weight='balanced'`, метрика PR-AUC
- **Когда нет уверенности в распределении** → Bootstrap

## Workflow на DSI

1. **EDA**: shape → dtypes → NaN → describe → distributions → target → correlations
2. **Sanity baseline**: DummyRegressor / DummyClassifier
3. **Linear baseline**: Ridge / LogisticRegression в Pipeline
4. **Tree baseline**: LightGBM с дефолтами + early stopping
5. **Feature engineering**: lags, rolling, cross-sectional ranks
6. **Тюнинг**: 3–5 главных гиперпараметров, NEVER на test set
7. **Анализ ошибок**: где ошибается, есть ли паттерн
8. **Стабильность**: out-of-time, чувствительность к гиперпараметрам

## Главное правило

Простая модель + отличный EDA + честная валидация **бьёт** сложную модель + утечки + хайповый CV-score.

Думай вслух. Объясняй каждый выбор. Это и есть ценность.
