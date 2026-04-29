# Метрики

## 1. Регрессия

| Метрика | Формула | Когда |
|---------|---------|-------|
| **MSE** | `mean((y - ŷ)²)` | Default. Штрафует выбросы |
| **RMSE** | `√MSE` | MSE в единицах y, удобнее интерпретировать |
| **MAE** | `mean(|y - ŷ|)` | Робастно к выбросам |
| **MAPE** | `mean(|y - ŷ| / |y|)` | Процентная ошибка, не работает с y≈0 |
| **R²** | `1 - SS_res/SS_tot` | Доля объяснённой дисперсии. Может быть < 0! |
| **Huber** | MSE для малых, MAE для больших | Робастный default |

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

> На finance returns: R² = 0.01 уже сильно. R² > 0.05 → проверь leakage.

---

## 2. Классификация

| Метрика | Формула | Когда |
|---------|---------|-------|
| **Accuracy** | `(TP+TN)/N` | Только при балансе классов |
| **Precision** | `TP/(TP+FP)` | Когда дорого FP |
| **Recall** | `TP/(TP+FN)` | Когда дорого пропустить позитив |
| **F1** | Harmonic mean P, R | Дисбаланс, нужен баланс P/R |
| **ROC-AUC** | Area under TPR/FPR | Дисбаланс не критичный |
| **PR-AUC** | Area under P/R | **Стандарт для сильного дисбаланса** |
| **Log loss** | `−Σy·log(p)` | Калиброванность вероятностей |
| **Brier** | `mean((y - p)²)` | Калиброванность |

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)

# Классы:
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Вероятности:
roc = roc_auc_score(y_true, y_proba)
pr = average_precision_score(y_true, y_proba)  # = PR-AUC
ll = log_loss(y_true, y_proba)
brier = brier_score_loss(y_true, y_proba)
```

### Confusion matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

### Какую метрику выбрать

- **Сбалансированные классы** → Accuracy, F1
- **Сильный дисбаланс (1:100+)** → PR-AUC, не ROC-AUC
- **Важна калибровка** → Log loss, Brier
- **Бизнесовая стоимость FP/FN различна** → custom cost-sensitive metric

---

## 3. Quant метрики

### Information Coefficient (IC)

Корреляция предсказаний с реальными доходностями. Spearman — стандарт.

```python
from scipy.stats import spearmanr

# Single period IC:
ic = spearmanr(y_pred, y_true).correlation

# Daily IC over time:
ic_history = []
for date in dates:
    mask = df['date'] == date
    ic_t = spearmanr(df.loc[mask, 'pred'], df.loc[mask, 'fwd_return']).correlation
    ic_history.append(ic_t)
```

### ICIR

Стабильность IC во времени. Аналог Sharpe для информационного коэффициента.

```python
mean_ic = np.mean(ic_history)
icir = mean_ic / np.std(ic_history)

# Annualized (для дневной частоты):
icir_annual = icir * np.sqrt(252)
```

| ICIR | Оценка |
|------|--------|
| > 0.5 | Сильный фактор |
| > 1.0 | Очень редко в реальности |
| > 2.0 | Подозрительно — проверь leakage |

### Sharpe ratio

```python
def sharpe(returns, periods_per_year=252, rf=0):
    excess = returns - rf / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)
```

| Sharpe | Оценка |
|--------|--------|
| > 1 | Хорошо |
| > 2 | Отлично (или подозрительно) |
| > 3 | Почти наверняка leakage |

### Sortino ratio

Penalize только downside volatility.

```python
def sortino(returns, periods_per_year=252, rf=0):
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    return excess.mean() / downside.std() * np.sqrt(periods_per_year)
```

### Max drawdown

```python
def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    return drawdown.min()
```

### Quintile spread

Long top 20%, short bottom 20%.

```python
df['quintile'] = df.groupby('date')['pred'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)
spread = df.groupby('quintile')['fwd_return'].mean()
long_short = spread.iloc[-1] - spread.iloc[0]
print(f'Q5 - Q1: {long_short:.4f}')
```

### Hit rate

Доля правильных направлений.

```python
hit_rate = (np.sign(y_pred) == np.sign(y_true)).mean()
# > 52% на дневной частоте — уже хорошо
```

### Information Ratio (IR)

Sharpe vs benchmark. Active return = portfolio - benchmark.

```python
def information_ratio(portfolio_returns, benchmark_returns, periods=252):
    active = portfolio_returns - benchmark_returns
    return active.mean() / active.std() * np.sqrt(periods)
```

---

## 4. Калибровка

### Что это

Хорошо откалиброванная модель: если предсказывает `p=0.7`, то 70% таких объектов реально в классе 1.

### Visual check

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('Predicted probability')
plt.ylabel('True frequency')
```

### Метрики калибровки

```python
from sklearn.metrics import brier_score_loss, log_loss

brier = brier_score_loss(y_true, y_proba)  # ниже = лучше
ll = log_loss(y_true, y_proba)
```

### Калибрация модели

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    base_estimator=LGBMClassifier(),
    method='isotonic',  # для большого n
    # method='sigmoid',  # для маленького n (Platt scaling)
    cv=5
)
```

> Логистическая регрессия — калиброванная по построению.
> Tree-based модели — нет, для них нужен `CalibratedClassifierCV`.

---

## 5. Резюме

### По задаче

| Задача | Дефолтная метрика |
|--------|-------------------|
| Регрессия | RMSE |
| Регрессия с выбросами | MAE или Huber |
| Бинарная классификация (баланс) | F1 или ROC-AUC |
| Бинарная классификация (дисбаланс) | **PR-AUC** |
| Калибровка вероятностей | Log loss или Brier |
| Multi-class | Macro-F1 |
| Quant alpha factor | **IC + ICIR** |
| Стратегия | Sharpe + max drawdown |
| Long-short portfolio | IR + quintile spread |

### Главное

> Никогда не оценивай quant модель только по MSE/R².
> Считай IC, ICIR, hit rate, quintile spread.
> Sharpe > 2 на out-of-sample → ищи leakage.
