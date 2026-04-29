# EDA — Exploratory Data Analysis

## Главное правило

> shape → dtypes → head → describe → NaN → distributions → target → correlations → leakage check

Никогда не моделируй, пока не сделал EDA. Комментируй каждое наблюдение вслух.

---

## 1. Первый осмотр

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
pd.set_option('display.max_columns', 100)

df = pd.read_csv('data.csv')

print(df.shape)
print(df.dtypes)
df.head()                    # первые 5
df.tail()                    # часто там сюрпризы
df.sample(10)
df.info(memory_usage='deep')
df.describe(include='all').T
```

### Что искать в head/tail/sample

- Колонки-идентификаторы (`id`, `ticker`, `date`) — НЕ должны быть фичами
- Числа в строках (`'1,234.56'`, `'12%'`, `'$100'`)
- Даты как строки → нужна `pd.to_datetime`
- Подозрительные значения: `-999`, `9999`, `-1` — часто закодированные пропуски
- Смешанные типы в колонке (часть int, часть string)

---

## 2. Дубликаты

```python
# Полные:
df.duplicated().sum()
df = df.drop_duplicates()

# По ключу:
df.duplicated(subset=['id']).sum()

# Time + entity:
df.duplicated(subset=['date', 'ticker']).sum()
```

---

## 3. Пропуски

```python
# Доля NaN:
(df.isna().mean() * 100).sort_values(ascending=False).head(20)

# Тепловая карта пропусков:
import missingno as msno
msno.matrix(df)
msno.heatmap(df)
```

### Паттерны пропусков

| Паттерн | Что значит | Стратегия |
|---------|-----------|-----------|
| **MCAR** | Missing Completely At Random | Удаление или простая импутация |
| **MAR** | Missing At Random (зависит от других) | Модельная импутация (MICE) |
| **MNAR** | Missing Not At Random (зависит от значения) | Indicator + impute |

> На DSI всегда добавляй индикаторную колонку `is_missing_X` — она часто становится сильным предиктором.

### Стратегии импутации

| Стратегия | Когда | Подвох |
|-----------|-------|--------|
| Удалить строки | <5% NaN, MCAR | Теряем данные |
| Удалить колонки | >50% NaN | Теряем потенциально ценную инфу |
| Mean/Median | MCAR, числовые | Уменьшает дисперсию |
| Mode | Категориальные | Усиливает доминирующую |
| Forward fill | Time series | Только если разумно |
| KNN imputation | Корреляции между фичами | Медленно |
| MICE (Iterative) | MAR, сложные паттерны | Медленно, нужен тюнинг |
| Indicator + median | MNAR | Удваивает фичи |

### Indicator trick (универсальное решение)

```python
for col in df.columns[df.isna().any()]:
    df[f'{col}_was_nan'] = df[col].isna().astype(int)
    df[col] = df[col].fillna(df[col].median())
```

> ⚠️ Импутация должна fit-иться на train и применяться к val/test. Иначе утечка через статистики.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('model', LinearRegression()),
])
pipe.fit(X_train, y_train)
```

---

## 4. Univariate analysis

### Числовые

```python
num_cols = df.select_dtypes(include=np.number).columns

# Hist + boxplot для каждой:
fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 3*len(num_cols)))
for i, col in enumerate(num_cols):
    df[col].hist(bins=50, ax=axes[i, 0])
    df[col].plot.box(ax=axes[i, 1])

# Skewness и kurtosis:
df[num_cols].skew().sort_values()
df[num_cols].kurtosis().sort_values()
```

### Что искать

- **Тяжёлые хвосты** (kurtosis > 3): лог или Box-Cox
- **Сильная асимметрия** (|skew| > 1): лог, sqrt, ранги
- **Бимодальность**: часто скрытая категориальная
- **Спайки**: часто закодированные «нет данных»
- **Постоянные колонки**: удалить

### Категориальные

```python
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    print(f'{col}: {df[col].nunique()} unique')
    print(df[col].value_counts(normalize=True).head(10))
```

### Что искать

- High cardinality (>50): нужна target encoding
- Редкие категории (<1%): объединить в `other`
- Дисбаланс (99% Yes, 1% No): почти-константа
- Опечатки: `'Male'` vs `'male'` vs `'M'`

### Преобразования

| Проблема | Решение | Код |
|----------|---------|-----|
| Положительная асимметрия | log1p | `np.log1p(x)` |
| Тяжёлые хвосты в обе стороны | Yeo-Johnson | `PowerTransformer(method='yeo-johnson')` |
| Только положительные | Box-Cox | `PowerTransformer(method='box-cox')` |
| Хочется ранги | rankdata | `scipy.stats.rankdata(x)` |
| Сделать ~uniform | QuantileTransformer | `output_distribution='uniform'` |
| Сделать ~normal | QuantileTransformer | `output_distribution='normal'` |

---

## 5. Bivariate

### Корреляционная матрица

```python
# Spearman — стандарт для финансов:
corr = df[num_cols].corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True)

# Топ-10 пар:
pairs = corr.abs().unstack().sort_values(ascending=False)
pairs = pairs[pairs < 1].drop_duplicates()
pairs.head(10)
```

### Что искать

- Высокие |r| > 0.9 между фичами → мультиколлинеарность
- |r| > 0.95 с таргетом → почти наверняка leakage
- Кластеры коррелирующих фич → можно агрегировать

### Scatter plots

```python
target = 'y'
features = [c for c in num_cols if c != target]

n = len(features)
fig, axes = plt.subplots((n+2)//3, 3, figsize=(15, 4*((n+2)//3)))
for i, col in enumerate(features):
    ax = axes.flat[i]
    ax.scatter(df[col], df[target], alpha=0.3, s=8)
    ax.set_xlabel(col); ax.set_ylabel(target)
```

### Группированный анализ

```python
df.boxplot(column='y', by='category')
df.groupby('category')['y'].agg(['mean', 'median', 'std', 'count'])
```

---

## 6. Выбросы

### IQR (универсальный, робастный)

```python
Q1, Q3 = df[col].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = df[(df[col] < lower) | (df[col] > upper)]
```

### Z-score (только если ~normal)

```python
from scipy.stats import zscore
z = np.abs(zscore(df[col].dropna()))
mask = z > 3
```

### Isolation Forest (multivariate)

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=42)
pred = iso.fit_predict(df[num_cols].fillna(0))
# -1 = outlier, 1 = inlier
```

### Что делать

| Подход | Когда |
|--------|-------|
| Удалить | Точно ошибка ввода |
| Winsorize | Сохранить, но ограничить влияние |
| Лог | Тяжёлый правый хвост |
| Робастная модель | Tree-based, Huber, Quantile |
| Оставить | В финансах часто это сигнал |

```python
# Winsorize:
from scipy.stats.mstats import winsorize
df['x_w'] = winsorize(df['x'], limits=[0.01, 0.01])

# Или clip:
lo, hi = df['x'].quantile([0.01, 0.99])
df['x_w'] = df['x'].clip(lo, hi)
```

> В финансах «выбросы» (события > 5σ) — это часто и есть сигнал (кризисы, скачки). Удалять нельзя. Дефолт: rank-based фичи или Winsorize в [1%, 99%] + tree-based модели.

---

## 7. Time Series EDA

```python
df = df.set_index('date').sort_index()

# 1. Сам ряд:
df['y'].plot(figsize=(14, 4))

# 2. Rolling статистики:
df['y'].rolling(30).mean().plot(label='30-day MA')
df['y'].rolling(30).std().plot(label='30-day std')

# 3. Декомпозиция:
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(df['y'], model='additive', period=252)
decomp.plot()

# 4. ACF/PACF:
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['y'].dropna(), lags=40)
plot_pacf(df['y'].dropna(), lags=40)
```

### Чек-лист

- [ ] Стационарность: ADF + KPSS
- [ ] Сезонность: пики ACF на одинаковых лагах
- [ ] Автокорреляция: PACF показывает порядок AR
- [ ] Структурные сдвиги: rolling mean/std стабильны?
- [ ] Пропуски в датах: `pd.date_range` vs `df.index`
- [ ] Кластеризация волатильности: пики `|returns|.rolling(20).mean()` = ARCH

### Frequency check

```python
df.index.to_series().diff().value_counts()

# Заполнить пропущенные даты:
full_range = pd.date_range(df.index.min(), df.index.max(), freq='B')
df = df.reindex(full_range)
```

### Главные ловушки

- **Сравнение rolling без выравнивания** → ложный «тренд»
- **`.corr()` на нестационарных рядах** → spurious correlation
- **Усреднение по разным режимам** → бессмыслица
- **Лог-цены — стандарт**: `np.log(price).diff()`, не `pct_change()`

---

## 8. Анализ таргета

### Distribution

```python
# Регрессия:
df['y'].plot.hist(bins=50)
df['y'].describe()
df['y'].skew(), df['y'].kurtosis()

# Классификация:
df['y'].value_counts(normalize=True)
# Дисбаланс > 10:1 → нужны class_weight или resampling
```

### Quick screening связи фич с таргетом

```python
# Регрессия:
ic = df.corr(method='spearman')['y'].drop('y').sort_values(key=abs, ascending=False)
print(ic.head(20))

# Классификация — target rate по бинам:
for col in num_cols[:10]:
    bins = pd.qcut(df[col], 10, duplicates='drop')
    print(col, df.groupby(bins)['y'].mean().values)

# Mutual information (нелинейные):
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y)
pd.Series(mi, index=X.columns).sort_values(ascending=False)
```

### Sanity checks

- [ ] Диапазоны разумны (возраст не -5 и не 200)
- [ ] Уникальность ключей: `id`, `(date, ticker)`
- [ ] Совпадение типов с описанием
- [ ] Категориальные значения соответствуют ожидаемому списку

---

## 9. Поиск утечек (data leakage)

### Симптомы

- Одна фича с |r| > 0.9 с таргетом
- Модель из одной фичи даёт почти идеальный score
- Train ≈ Test, оба сильно лучше, чем должно быть

### Типы

| Тип | Пример | Как искать |
|-----|--------|------------|
| **Target leakage** | Колонка `churned_amount` для предсказания `churn` | Корреляции > 0.9 |
| **Time leakage** | Будущие значения в train | `train.date.max() < test.date.min()`? |
| **Group leakage** | Один пациент в train и test | GroupKFold |
| **Preprocessing leakage** | `scaler.fit` на всех данных | Pipeline |
| **Selection leakage** | Feature selection на всех данных | Selection внутри CV |

### Защита: Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('select', SelectKBest(k=20)),
    ('model', LogisticRegression()),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

### Time series — особый случай

```python
# НЕЛЬЗЯ: train_test_split, KFold
# НАДО: TimeSeriesSplit с gap (embargo)

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=20)

for train_idx, val_idx in tscv.split(X):
    X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
    # train always strictly before val
```
