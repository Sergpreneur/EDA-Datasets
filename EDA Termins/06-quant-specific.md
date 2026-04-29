# Quant-specific

## 1. Факторные модели

### Что такое фактор

Фактор — вычисляемая переменная, которая (гипотетически) предсказывает кросс-секцию доходностей.

**Классические факторы**:
- **Value**: P/E, P/B, EV/EBITDA
- **Momentum**: 12-1 month return
- **Size**: log(market cap)
- **Quality**: ROE, debt/equity, earnings stability
- **Low volatility**: trailing volatility (-1)

### Workflow

1. **Гипотеза**: «Высокая P/E (overvalued) → низкая будущая доходность»
2. **Расчёт фактора**: cross-sectional, на каждый период
3. **Стандартизация**: rank-normal или z-score внутри даты
4. **Оценка предсказательной силы**: IC, ICIR
5. **Quintile portfolios**: long top 20%, short bottom 20%
6. **Out-of-sample backtest**: walk-forward

### Cross-sectional standardization (стандарт)

```python
# z-score внутри даты:
df['factor_z'] = df.groupby('date')['factor'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank-normal (более робастно):
from scipy.stats import norm
df['factor_rn'] = df.groupby('date')['factor'].transform(
    lambda x: norm.ppf((x.rank() - 0.5) / len(x))
)

# Простой rank (от 0 до 1):
df['factor_rank'] = df.groupby('date')['factor'].rank(pct=True)
```

> ⚠️ Никогда не стандартизуй на всех данных сразу — это утечка из будущего.

---

## 2. Information Coefficient (IC)

### Что это

`IC_t = Spearman(factor_t, return_{t+k})` — корреляция предсказаний с будущими доходностями.

**Spearman**, не Pearson. Доходности имеют тяжёлые хвосты.

### Расчёт

```python
from scipy.stats import spearmanr

def daily_ic(df, factor_col, return_col, date_col='date'):
    """Возвращает Series IC по датам"""
    return df.groupby(date_col).apply(
        lambda x: spearmanr(x[factor_col], x[return_col]).correlation
    )

ic_series = daily_ic(df, 'factor_z', 'fwd_return_5d')
```

### Интерпретация

| Mean IC | Оценка |
|---------|--------|
| 0.01–0.02 | Реалистично для дневной частоты |
| 0.03–0.05 | Хороший фактор |
| > 0.05 | Сильный (проверь, нет ли leakage) |
| > 0.10 | Почти наверняка leakage или look-ahead |

### t-статистика IC

```python
from scipy.stats import ttest_1samp

t_stat, p_val = ttest_1samp(ic_series.dropna(), 0)
# t > 2 → IC статистически значимо ≠ 0
```

---

## 3. ICIR

`ICIR = mean(IC_t) / std(IC_t)` — стабильность IC во времени. Аналог Sharpe для IC.

```python
mean_ic = ic_series.mean()
std_ic = ic_series.std()
icir = mean_ic / std_ic
icir_annual = icir * np.sqrt(252)  # для дневной частоты
```

### Интерпретация

| ICIR (annual) | Оценка |
|---------------|--------|
| 0.3–0.5 | Слабый |
| 0.5–1.0 | **Хороший** |
| 1.0–1.5 | Отличный |
| > 1.5 | Подозрительно |

> **На DSI**: ICIR > 0.5 — отличный сигнал, что фактор стабилен.

---

## 4. Quintile portfolios

### Концепция

1. Каждый период ранжируем все имена по фактору
2. Делим на 5 квинтилей
3. Считаем mean(forward return) для каждого квинтиля
4. Q5 - Q1 = long-short spread

### Расчёт

```python
df['quintile'] = df.groupby('date')['factor'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)

quintile_returns = df.groupby(['date', 'quintile'])['fwd_return'].mean().unstack()
# Каждая колонка = доходность квинтиля во времени

# Long-short:
long_short = quintile_returns[4] - quintile_returns[0]

# Метрики:
sharpe_ls = long_short.mean() / long_short.std() * np.sqrt(252)
print(f'Long-short Sharpe: {sharpe_ls:.2f}')
```

### Что искать

- **Монотонность**: Q1 < Q2 < Q3 < Q4 < Q5 (или наоборот). Если нет — фактор нелинеен или шумный
- **Спред Q5 - Q1**: чем больше, тем сильнее фактор
- **Стабильность спреда**: Sharpe долгосрочно > 1 → крепкий фактор

---

## 5. Long-short portfolio

Approximately market-neutral: long top names, short bottom names.

```python
def long_short_returns(df, factor_col, return_col, top_pct=0.2, bottom_pct=0.2):
    """Дневные доходности equal-weight long-short портфеля"""
    def daily(group):
        n = len(group)
        n_top = int(n * top_pct)
        n_bot = int(n * bottom_pct)
        sorted_g = group.sort_values(factor_col)
        bot_ret = sorted_g.iloc[:n_bot][return_col].mean()
        top_ret = sorted_g.iloc[-n_top:][return_col].mean()
        return top_ret - bot_ret

    return df.groupby('date').apply(daily)

ls_returns = long_short_returns(df, 'factor', 'fwd_return_1d')
sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(252)
```

---

## 6. Walk-forward backtest

Стандарт quant. Имитирует реальный деплой модели.

```python
def walk_forward_backtest(df, model, train_days=504, test_days=21, step=21):
    results = []
    dates = sorted(df['date'].unique())

    for start in range(0, len(dates) - train_days - test_days + 1, step):
        train_dates = dates[start : start + train_days]
        test_dates = dates[start + train_days : start + train_days + test_days]

        train = df[df['date'].isin(train_dates)]
        test = df[df['date'].isin(test_dates)]

        model.fit(train[features], train['target'])
        test = test.copy()
        test['pred'] = model.predict(test[features])

        # IC по тест-периоду:
        ic = test.groupby('date').apply(
            lambda x: spearmanr(x['pred'], x['target']).correlation
        )
        results.append({'period': start, 'mean_ic': ic.mean(), 'std_ic': ic.std()})

    return pd.DataFrame(results)
```

---

## 7. Главные biases

### Survivorship bias

> Изучаем только выжившие компании. Делистнутые исчезают.

**Решение**: использовать point-in-time universe (список тикеров на каждую дату, включая делистнутые).

### Look-ahead bias

> Используем информацию, которая не была доступна в момент решения.

**Примеры**:
- Earnings, объявленные после close, для предсказания return того же дня
- Ребалансированный composition индекса
- Ревизированный ВВП / макро-данные

**Решение**: lag всех фундаментальных данных на разумный период; использовать данные in their original release form.

### Selection bias

> Выборка нерепрезентативна.

**Примеры**:
- Только highly-liquid names
- Только S&P 500 (большие cap)

### Data snooping / p-hacking

> Многократное тестирование до p < 0.05 чисто случайно.

**Решение**:
- Pre-registration гипотез
- Strict OOS testing
- Benjamini-Hochberg FDR correction
- Bonferroni для FWER

### Multiple comparisons

100 тестов с α=0.05 → ожидаем 5 ложных позитивов чисто случайно.

```python
from statsmodels.stats.multitest import multipletests
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## 8. Анализ результатов

### Sharpe ratio

```python
def sharpe(returns, periods=252, rf=0):
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods)
```

### Max drawdown

```python
def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    return (cum / cum.cummax() - 1).min()
```

### Calmar ratio

`return / max_drawdown` — return на единицу риска.

```python
def calmar(returns, periods=252):
    annual_return = returns.mean() * periods
    mdd = abs(max_drawdown(returns))
    return annual_return / mdd if mdd > 0 else 0
```

### Полная сводка стратегии

```python
def strategy_summary(returns, periods=252):
    return pd.Series({
        'Annual return': returns.mean() * periods,
        'Annual volatility': returns.std() * np.sqrt(periods),
        'Sharpe': sharpe(returns, periods),
        'Sortino': returns.mean() / returns[returns < 0].std() * np.sqrt(periods),
        'Max drawdown': max_drawdown(returns),
        'Calmar': calmar(returns, periods),
        'Hit rate': (returns > 0).mean(),
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis(),
    })
```

---

## 9. Чек-лист quant модели

### До разработки

- [ ] Понимаю, какой реальный экономический эффект хочу поймать?
- [ ] Данные point-in-time (нет ревизий из будущего)?
- [ ] Universe включает делистнутые компании?

### EDA

- [ ] Распределение доходностей: skew, kurtosis, тяжёлые хвосты?
- [ ] Стационарность ряда: ADF + KPSS?
- [ ] Cross-sectional vs time-series вариация фичи?

### Моделирование

- [ ] Cross-sectional standardization (z-score или rank внутри даты)?
- [ ] CV: walk-forward или Purged K-Fold с embargo?
- [ ] Метрики: IC, ICIR, quintile spread, Sharpe?

### Анализ

- [ ] IC статистически значим (t > 2)?
- [ ] ICIR > 0.5?
- [ ] Quintile spread монотонен?
- [ ] Sharpe устойчив к разным окнам?
- [ ] Performance не сосредоточена в одном режиме (тогда переподгонка)?

### Sanity check

- [ ] Нет look-ahead bias?
- [ ] Нет survivorship bias?
- [ ] Transaction costs учтены?
- [ ] Capacity (slippage) разумна?
