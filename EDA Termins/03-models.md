# Модели

## Workflow

1. **Sanity baseline**: `DummyRegressor(strategy='mean')` или `DummyClassifier(strategy='most_frequent')`
2. **Linear baseline**: Ridge / LogisticRegression в Pipeline
3. **Tree baseline**: LightGBM с дефолтами + early stopping
4. **Feature engineering**: lags, rolling, cross-sectional ranks
5. **Тюнинг**: 3–5 главных гиперпараметров через Optuna/grid

> Всегда строй простой baseline ПЕРВЫМ. Это: (1) проверка пайплайна, (2) sanity-check на утечки, (3) бенчмарк.
> Если LinearRegression даёт R² = 0.99 — почти наверняка leakage.

---

## 1. Линейные модели

### OLS

```python
# scikit-learn:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# statsmodels — даёт p-values:
import statsmodels.api as sm
X_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_const).fit()
print(model.summary())
```

Минимизирует `Σ(y - Xβ)²`. Решение: `β = (XᵀX)⁻¹Xᵀy`.

**Предположения**:
- Линейность
- Независимость остатков
- Гомоскедастичность
- Нормальность остатков
- Отсутствие мультиколлинеарности

**Проверка**: residuals plot (облако без паттернов), QQ-plot остатков. Если VIF > 5–10 — мультиколлинеарность.

### Ridge (L2)

```python
from sklearn.linear_model import Ridge, RidgeCV
model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
model.fit(X_train, y_train)
print('Best alpha:', model.alpha_)
```

Минимизирует `Σ(y - Xβ)² + α·Σβ²`. Сжимает все коэффициенты. Помогает при мультиколлинеарности и переобучении.

- Коэффициенты остаются ненулевыми
- ⚠️ Стандартизуй фичи перед Ridge
- Дефолт для quant с многими коррелирующими факторами

### Lasso (L1)

```python
from sklearn.linear_model import LassoCV
model = LassoCV(cv=5, n_alphas=100, random_state=42)
model.fit(X_train, y_train)
selected = X.columns[model.coef_ != 0]
```

`Σ(y - Xβ)² + α·Σ|β|`. Зануляет часть коэффициентов → автоматический feature selection.

### ElasticNet (L1 + L2)

```python
from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
```

Хорош когда есть группы коррелирующих фич — Lasso выкидывает случайно одну, ElasticNet оставляет группу.

### Сравнение

| Модель | Штраф | Feature selection | Когда |
|--------|-------|-------------------|-------|
| OLS | — | Нет | Мало фич, нет мультиколлинеарности |
| Ridge | L2 | Нет (сжимает) | Много фич, мультиколлинеарность |
| Lasso | L1 | Да (зануляет) | Sparse модель |
| ElasticNet | L1+L2 | Да | Группы коррелирующих фич |

---

## 2. Логистическая регрессия

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,                   # обратный к alpha
    penalty='l2',            # 'l1', 'elasticnet', 'none'
    solver='lbfgs',          # для l1: 'liblinear' или 'saga'
    class_weight='balanced', # автокомпенсация дисбаланса
    max_iter=1000,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_val)[:, 1]
```

Предсказывает `P(y=1 | x) = sigmoid(Xβ)`. Минимизирует log loss = `−Σ[y·log(p) + (1-y)·log(1-p)]`.

**Интерпретация**: `exp(βᵢ)` — odds ratio для фичи i.

### Калибровка вероятностей

Логистическая регрессия даёт калиброванные вероятности из коробки. Tree-based — нет.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    base_estimator=LGBMClassifier(),
    method='isotonic',       # или 'sigmoid' для маленьких выборок
    cv=5
)
```

### Дисбаланс классов

| Соотношение | Стратегия |
|-------------|-----------|
| До 1:10 | `class_weight='balanced'` |
| 1:100 | undersampling или SMOTE |
| 1:1000+ | Anomaly detection задача |

- ВСЕГДА `StratifiedKFold` для CV
- Метрики: НЕ accuracy. PR-AUC, F1, recall@k, log loss

---

## 3. Tree-based модели

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20)
```

Прозрачно, но катастрофически переобучается. Использовать только в ансамблях.

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,         # обычно None или большое
    min_samples_leaf=5,
    max_features='sqrt',    # стандарт для классификации
    n_jobs=-1,
    random_state=42,
)
```

Bagging + случайная подвыборка фич. Снижает дисперсию. Хорош из коробки.

### LightGBM (стандарт для DSI)

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    objective='regression',  # 'binary', 'multiclass', 'quantile', 'huber'
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,           # главный параметр сложности
    max_depth=-1,            # -1 = не ограничено
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,           # L1
    reg_lambda=0.1,          # L2
    random_state=42,
    n_jobs=-1,
)

# С early stopping (стандарт):
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```

### Гиперпараметры LightGBM

| Параметр | Эффект | Диапазон |
|----------|--------|----------|
| `n_estimators` | Кол-во деревьев | 500–5000 + early_stopping |
| `learning_rate` | Скорость обучения | 0.01–0.1 |
| `num_leaves` | Сложность дерева | 15–255 |
| `min_child_samples` | Мин. примеров в листе | 5–100 |
| `subsample` | Row sampling | 0.6–1.0 |
| `colsample_bytree` | Feature sampling | 0.6–1.0 |
| `reg_alpha` / `reg_lambda` | L1 / L2 | 0.0–10.0 |
| `max_depth` | Макс. глубина | -1 или 4–12 |

> **Дефолты для DSI**: `n_estimators=2000, learning_rate=0.05, num_leaves=31` + early_stopping.
> Маленький датасет (<10K): `num_leaves=15, min_child_samples=50`.
> Зашумлённые данные: `reg_alpha > 0, reg_lambda > 0`.

### XGBoost

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.05,
    max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method='hist',  # быстро как LGBM
)
```

### CatBoost

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=2000, learning_rate=0.05, depth=6,
    cat_features=['cat1', 'cat2'],   # native handling
    verbose=200,
)
```

Лучший дефолт для категориальных фич.

### Сравнение

| Модель | Скорость | Категориальные | Тюнинг |
|--------|----------|----------------|--------|
| RandomForest | Средняя | Нужен encoding | Минимум |
| LightGBM | Очень быстро | Через `category` dtype | Средний |
| XGBoost | Быстро | Нужен encoding | Средний |
| CatBoost | Средне | Нативно (лучший) | Минимум |

---

## 4. Bias-variance tradeoff

| | Train score | Val score | Диагноз |
|--|-------------|-----------|---------|
| Underfit | плохой | плохой | Усложни модель/фичи |
| **Sweet spot** | хороший | хороший | ✓ |
| Overfit | отличный | плохой | Регуляризация, упрощение |

`Total error = bias² + variance + irreducible noise`

Регуляризация (L1, L2, max_depth, dropout, early_stopping) — все способы движения по этой кривой.

---

## 5. Feature engineering

### Числовые

- Преобразования: log1p, sqrt, ², квантильное
- Биннинг: `pd.qcut`, `pd.cut`
- Полиномиальные: `PolynomialFeatures(degree=2)` — осторожно, взрывает размерность
- Взаимодействия: A*B, A/B (домен-специфично)

### Категориальные

| Метод | Когда |
|-------|-------|
| One-hot | cardinality < 15 |
| Target encoding | High cardinality, нужна регуляризация и CV-fitting |
| Frequency encoding | Cardinality сама по себе важна |
| Hash encoding | Очень high cardinality (>10K) |
| Native (LGBM/CatBoost) | Когда модель умеет — используй |

### Target encoding — главная ловушка

> Если `mean(target) by category` на всём датасете → leakage.
> Правильно: out-of-fold (k-fold target encoding) или `category_encoders.TargetEncoder` со smoothing.
> Лучше: CatBoost или LightGBM с `category` dtype — они делают это сами.

### Time series фичи

```python
# Лаги:
for lag in [1, 5, 10, 20]:
    df[f'y_lag_{lag}'] = df['y'].shift(lag)

# Rolling:
for w in [5, 20, 60]:
    df[f'y_mean_{w}'] = df['y'].rolling(w).mean()
    df[f'y_std_{w}'] = df['y'].rolling(w).std()
    df[f'y_zscore_{w}'] = (df['y'] - df[f'y_mean_{w}']) / df[f'y_std_{w}']

# Календарные:
df['dow'] = df.index.dayofweek
df['month'] = df.index.month
df['is_month_end'] = df.index.is_month_end.astype(int)

# Cross-sectional ranks (для quant):
df['rank'] = df.groupby('date')['feature'].rank(pct=True)
```

### Standardization для финансов

> **Cross-sectional** (внутри одной даты) — каждый день z-score фичи по всем тикерам. Фича безразмерна и сравнима между датами.
>
> **Time-series** (rolling z-score) — для фич, где важна динамика во времени.
>
> ⚠️ Не стандартизуй на всех данных сразу — это утечка из будущего.

---

## 6. Подводные камни

### Переобучение

- Train >> Val → overfit. Усиль регуляризацию, упрости модель, добавь данные
- Train ≈ Val, но оба плохие → underfit. Усложни
- Val прыгает между фолдами → высокая дисперсия. Усреднение или больше данных

### Look-ahead bias

- Стандартизация/импутация на всех данных
- Целевая переменная вычисляется с использованием будущих данных
- Survivorship bias: исторический список акций без делистнутых
- Снэпшот, отражающий ревизии (ВВП, прибыль)

### Selection bias

- Тренировка только на «успешных» наблюдениях
- Несбалансированный split
- p-hacking: подбор гиперпараметров до получения хорошего результата

### Утечка через CV

- Нестратифицированный split при дисбалансе
- Неучтённые группы (один пациент в train и test)
- Time series без TimeSeriesSplit

---

## Final workflow для DSI

1. **EDA**: shape → dtypes → NaN → distributions → target → correlations
2. **Sanity baseline**: DummyRegressor (среднее) или DummyClassifier (мажорный класс)
3. **Linear baseline**: Ridge/Logistic с правильным CV
4. **Tree baseline**: LightGBM с дефолтами
5. **Feature engineering**: лаги, rolling, кросс-секционные ранги
6. **Тюнинг**: early_stopping → grid/Optuna по 3–5 главным гиперпараметрам
7. **Анализ ошибок**: где модель ошибается? есть ли паттерн?
8. **Стабильность**: out-of-time, разные сэмплы, чувствительность к гиперпараметрам

> Простая модель + отличный EDA + честная валидация **бьёт** сложную модель + утечки + хайповый CV-score.
