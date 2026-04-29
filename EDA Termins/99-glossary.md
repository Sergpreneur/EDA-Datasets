# Глоссарий

## Foundations

| Термин | Определение |
|--------|-------------|
| **H₀ (Null hypothesis)** | «Скучный» дефолт — эффекта нет. То, что мы пытаемся опровергнуть. |
| **H₁ (Alternative)** | То, что хотим показать. Two-sided: μ ≠ 0. One-sided: μ > 0. |
| **p-value** | `P(данные ≥ наблюдаемые \| H₀)`. НЕ `P(H₀ верна)`. |
| **α (Significance level)** | Допустимая вероятность ошибки I рода. Стандарт 0.05. |
| **Type I error (False positive)** | Отвергли H₀, хотя она верна. Контролируется через α. |
| **Type II error (False negative)** | Не отвергли H₀, хотя она ложна. Контролируется через n. |
| **Power (1−β)** | Вероятность отвергнуть H₀, когда H₁ верна. Цель ≥ 0.80. |
| **Effect size** | Размер эффекта (Cohen's d, r, odds ratio). |
| **Cohen's d** | `(μ₁ − μ₂) / σ_pooled`. Малый ≈ 0.2, средний ≈ 0.5, большой ≈ 0.8. |
| **Confidence interval** | Диапазон, содержащий истинный параметр с X% уверенностью при повторных выборках. |
| **Standard error** | Std сэмплинг-распределения статистики. SE(mean) = σ/√n. |
| **Central Limit Theorem** | Распределение средних → normal с ростом n, независимо от исходного. |

## Tests

| Термин | Определение |
|--------|-------------|
| **One-sample t-test** | Тест, что среднее = заданному значению μ₀. |
| **Welch's t-test** | Two-sample t-test, НЕ предполагающий равенства дисперсий. **Дефолт.** |
| **Student's t-test** | Two-sample, предполагает равные дисперсии. Старая школа. |
| **Paired t-test** | Для зависимых наблюдений (до/после). |
| **Mann-Whitney U** | Непараметрический аналог Welch. Аналог Wilcoxon rank-sum. |
| **Wilcoxon signed-rank** | Непараметрический paired test. |
| **Kruskal-Wallis** | Непараметрическая ANOVA для k > 2 групп. |
| **ANOVA** | F = (between-var) / (within-var). Для k > 2 групп. |
| **Shapiro-Wilk** | Тест нормальности для n < 5000. **Дефолт.** |
| **Jarque-Bera** | Тест нормальности по skew + kurtosis. Стандарт в эконометрике. |
| **Kolmogorov-Smirnov** | Сравнение CDF. Чувствителен к центру, слаб на хвостах. |
| **Anderson-Darling** | Модификация KS с весом на хвостах. |
| **Levene's test** | Тест равенства дисперсий, робастный к ненормальности. |
| **Bartlett's test** | Тест равенства дисперсий, требует нормальности. |
| **Chi-square** | Независимость категориальных переменных или goodness-of-fit. |
| **Fisher's exact** | Точный тест для 2×2 таблиц с малым n. |
| **McNemar** | Парный χ². Сравнение двух классификаторов на одном тесте. |

## Correlation

| Термин | Определение |
|--------|-------------|
| **Pearson r** | Линейная корреляция. Чувствителен к выбросам. |
| **Spearman ρ** | Pearson на рангах. Монотонная связь. **Дефолт для финансов.** |
| **Kendall τ** | На основе concordant minus discordant pairs. Лучше на малых n. |
| **Information Coefficient (IC)** | Spearman(predictions, forward returns). Метрика alpha-фактора. |
| **ICIR** | mean(IC) / std(IC). Аналог Sharpe для IC. |
| **Fisher z-transform** | arctanh(r). Стабилизирует дисперсию для CI и сравнения корреляций. |

## Time Series

| Термин | Определение |
|--------|-------------|
| **Stationarity** | Mean, var, autocov постоянны во времени. |
| **Unit root** | AR(1) coefficient = 1: random walk, нестационарен. |
| **ADF test** | H₀: unit root. Reject p<0.05 → stationary. |
| **KPSS test** | H₀: stationary. Reject p<0.05 → non-stationary. |
| **Ljung-Box** | H₀: остатки = white noise. Применяй к остаткам и квадратам остатков. |
| **Durbin-Watson** | Автокорреляция 1-го порядка. DW≈2 нет, <1.5 +, >2.5 −. |
| **ARCH effect** | Conditional heteroskedasticity (кластеризация волатильности). |
| **Granger causality** | «Прошлое X предсказывает Y?» Не настоящая причинность. |
| **Cointegration** | Линейная комбинация нестационарных рядов стационарна. Pairs trading. |
| **Autocorrelation (ACF)** | Корреляция ряда с собственными лагами. |
| **PACF** | Partial ACF: эффект лага k без вкладов промежуточных лагов. |
| **Heteroskedasticity** | Дисперсия различается между наблюдениями. |

## Resampling

| Термин | Определение |
|--------|-------------|
| **Bootstrap** | Resample с возвращением для оценки sampling distribution. |
| **Permutation test** | Перемешать метки под H₀, построить null distribution. |
| **Block bootstrap** | Ресэмпл блоками (длина L) для сохранения автокорреляции. |

## Multiple Testing

| Термин | Определение |
|--------|-------------|
| **FWER** | Family-wise error rate: P(хотя бы одна ошибка I рода). Консервативно. |
| **FDR** | False discovery rate: ожидаемая доля FP среди rejections. |
| **Bonferroni** | α/m. Контролирует FWER. Очень консервативный. |
| **Holm-Bonferroni** | Степенной Bonferroni. Всегда лучше. |
| **Benjamini-Hochberg** | Контролирует FDR. **Дефолт для скрининга.** |

## EDA

| Термин | Определение |
|--------|-------------|
| **Skewness** | Асимметрия. Positive = длинный правый хвост. |
| **Kurtosis** | «Tailedness». Excess > 0 = тяжелее нормального. |
| **IQR** | Q3 − Q1. Робастный spread. Outliers: вне [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. |
| **Winsorization** | Clip значений на percentiles вместо удаления. |
| **MCAR** | Missing Completely At Random. Безопасно дроп/импутировать. |
| **MAR** | Missing At Random (зависит от observed). Нужна модельная импутация. |
| **MNAR** | Missing Not At Random. Самый опасный. Indicator + impute. |
| **Mutual Information** | Любая зависимость (не только линейная). 0 ⟺ независимость. |
| **VIF** | 1/(1 − R²ᵢ). VIF > 5–10 → мультиколлинеарность. |
| **Box-Cox** | y^λ для λ≠0, log(y) для λ=0. Только для y > 0. |
| **Yeo-Johnson** | Generalization Box-Cox для negative values. |
| **Quantile transform** | Маппит на uniform или normal через empirical CDF. |

## Linear Models

| Термин | Определение |
|--------|-------------|
| **OLS** | Ordinary Least Squares. β = (XᵀX)⁻¹Xᵀy. |
| **Ridge (L2)** | OLS + α·Σβ². Сжимает все коэффициенты. |
| **Lasso (L1)** | OLS + α·Σ\|β\|. Зануляет часть → feature selection. |
| **ElasticNet** | L1 + L2. l1_ratio регулирует пропорцию. |
| **Logistic regression** | P(y=1\|x) = σ(Xβ). Калибрована из коробки. |
| **Sigmoid** | σ(z) = 1/(1+e⁻ᶻ). Maps ℝ → (0,1). |
| **Log loss (Cross-entropy)** | −Σ[y·log(p) + (1-y)·log(1-p)]. |
| **Bias** | Underfitting. Train и Val оба плохие. |
| **Variance (model)** | Overfitting. Train хороший, Val плохой. |
| **Bias-variance tradeoff** | Total error = bias² + variance + noise. |
| **Regularization** | L1, L2, dropout, max_depth, early stopping. |

## Trees

| Термин | Определение |
|--------|-------------|
| **Bagging** | Bootstrap Aggregating. Тренируем на ресэмплах, усредняем. |
| **Boosting** | Sequential: каждый исправляет ошибки предыдущего. |
| **Random Forest** | Bagging + случайные фичи на сплитах. |
| **Gradient Boosting** | Каждое новое дерево фитит negative gradient loss. |
| **LightGBM** | MS GBM. Histogram-based, leaf-wise, native categorical. **Дефолт.** |
| **XGBoost** | Original modern GBM. Стабильный. |
| **CatBoost** | Yandex GBM. Лучший для categorical. |
| **learning_rate** | Шаг градиентного спуска. 0.01–0.1. |
| **num_leaves** | LightGBM main complexity param. 15–255 (default 31). |
| **Early stopping** | Стоп когда val score перестаёт улучшаться N раундов. |

## Validation

| Термин | Определение |
|--------|-------------|
| **K-Fold CV** | Split на K фолдов, ротация train/val. |
| **Stratified K-Fold** | Сохраняет class proportions в фолдах. Дефолт для классификации. |
| **Group K-Fold** | Группа никогда не делится между train и val. |
| **TimeSeriesSplit** | Train всегда строго до val. Без shuffle. |
| **Walk-forward** | Rolling window: train на N, test на M, advance. |
| **Purged K-Fold** | de Prado: removes overlapping labels + embargo. |
| **Embargo** | Gap после val для anti-leak (для serial correlation). |

## Metrics — Regression

| Термин | Определение |
|--------|-------------|
| **MSE** | mean((y − ŷ)²). Penalizes outliers. |
| **RMSE** | √MSE. В единицах y. |
| **MAE** | mean(\|y − ŷ\|). Робастно к выбросам. |
| **R²** | 1 − SS_res / SS_tot. Может быть < 0 на test. |
| **Huber loss** | MSE для малых, MAE для больших. Робастный. |

## Metrics — Classification

| Термин | Определение |
|--------|-------------|
| **Accuracy** | (TP + TN) / N. Misleading на дисбалансе. |
| **Precision** | TP / (TP + FP). |
| **Recall (Sensitivity)** | TP / (TP + FN). |
| **F1** | Harmonic mean P, R. |
| **ROC-AUC** | Area under TPR vs FPR. Threshold-independent. |
| **PR-AUC** | Area under P vs R. **Лучше для дисбаланса.** |
| **Brier score** | mean((y − p)²). Калибровка. |
| **Calibration** | Predicted p = observed frequency. |

## Quant

| Термин | Определение |
|--------|-------------|
| **Sharpe ratio** | (mean − rf) / std × √252. > 1 хорошо, > 2 отлично. |
| **Information Ratio (IR)** | Sharpe vs benchmark. Active return / TE. |
| **Sortino ratio** | Like Sharpe но только downside std. |
| **Max drawdown** | Max peak-to-trough в cum equity. |
| **Hit rate** | Доля правильных направлений. > 52% уже хорошо. |
| **Quintile spread** | Mean(top 20% pred) − mean(bottom 20%). |
| **Alpha** | Excess return не объяснённый known factors. |
| **Beta** | Sensitivity to a factor. β = 1 → moves with market. |
| **Factor** | Variable hypothesized to explain cross-section returns. |
| **Long-short portfolio** | Long top, short bottom. Approx market-neutral. |
| **Factor decay** | Predictive power weakens over time. |

## Biases

| Термин | Определение |
|--------|-------------|
| **Look-ahead bias** | Использование информации, недоступной в момент решения. |
| **Survivorship bias** | Анализ только выживших, без failed. |
| **Selection bias** | Выборка нерепрезентативна. |
| **p-hacking / Data snooping** | Multi-testing до p < 0.05 случайно. |
| **Data leakage** | Future/test info попадает в training. |
| **Multiple comparisons** | m тестов с α → ожидаем m·α false positives. |

## Engineering

| Термин | Определение |
|--------|-------------|
| **One-hot encoding** | Категория → бинарные колонки. cardinality < 15. |
| **Target encoding** | Категория → mean(target). MUST be out-of-fold. |
| **Frequency encoding** | Категория → её частота. |
| **Standardization (z-score)** | (x − μ) / σ. Нужно для L1/L2, KNN, NN, PCA. |
| **Cross-sectional standardization** | z-score внутри даты. **Стандарт quant.** |
| **sklearn Pipeline** | Chain preprocessing + model. Защита от leakage в CV. |
| **PCA** | Dim reduction через SVD. Стандартизуй перед. |
