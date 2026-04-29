# Статистические критерии

## 1. Базовая логика

Любой тест работает по схеме:

1. Формулируем `H₀` (нулевая, «эффекта нет») и `H₁` (альтернатива)
2. Считаем тестовую статистику
3. Получаем `p-value` = вероятность увидеть такие или более экстремальные данные при `H₀`
4. Если `p < α` (обычно 0.05) → отвергаем `H₀`

### Ключевые понятия

| Термин | Что это |
|--------|---------|
| `H₀` | Нулевая гипотеза («эффекта нет») |
| `H₁` | Альтернатива (то, что хотим показать) |
| `α` | Уровень значимости, обычно 0.05 |
| `p-value` | `P(данные ≥ наблюдаемые \| H₀)` |
| Power (1−β) | Вероятность отвергнуть `H₀`, когда она ложна |
| Type I error | Отвергли `H₀`, хотя она верна. Контролируется через α |
| Type II error | Не отвергли `H₀`, хотя она ложна. Контролируется через n |

### Подводные камни p-value

- p-value НЕ есть `P(H₀ верна)`. Это `P(данные \| H₀)`
- Маленький p-value не означает большой эффект — на огромных выборках любой микроэффект «значим»
- При множественных тестах p-value «гниют» (нужны поправки)
- p-hacking — подбор подвыборки/признаков до получения `p < 0.05`

### Размер эффекта

`p-value` говорит «эффект есть», `effect size` — «насколько он большой»:

- **Cohen's d** для разности средних: `d = (μ₁ − μ₂) / σ_pooled`. Малый ≈ 0.2, средний ≈ 0.5, большой ≈ 0.8
- **r или r²** для корреляции/регрессии
- **Odds ratio** для бинарных исходов
- **Cliff's delta** — непараметрический аналог d

---

## 2. Тесты на нормальность

### Shapiro-Wilk
- `H₀`: выборка из нормального распределения
- Самый мощный для `n < 5000`. Дефолт.

```python
from scipy import stats
stat, p = stats.shapiro(x)
# p > 0.05 → не отвергаем нормальность
```

### D'Agostino-Pearson (omnibus)
- На основе skewness и kurtosis
- Хорошо ловит асимметрию и тяжёлые хвосты, работает на больших выборках

```python
stat, p = stats.normaltest(x)  # требует n >= 20
```

### Jarque-Bera
- `H₀`: skewness = 0, excess kurtosis = 0
- Стандарт в эконометрике, особенно для остатков регрессии
- На финансовых данных почти всегда отвергнет H₀

```python
stat, p = stats.jarque_bera(returns)
```

### Kolmogorov-Smirnov
- Сравнивает CDF выборки с теоретической или с другой выборкой
- Чувствителен к центру, слаб на хвостах
- ⚠️ Если параметры оцениваются из той же выборки → используй Lilliefors или Shapiro

```python
stat, p = stats.kstest(x, 'norm')        # 1-sample
stat, p = stats.ks_2samp(x, y)           # 2-sample
```

### Anderson-Darling
- Модификация KS с большим весом на хвостах. Лучше KS для нормальности.

```python
result = stats.anderson(x, dist='norm')
# сравни result.statistic с result.critical_values
```

### Главное правило

> Сначала смотри график (histogram, QQ-plot), потом тест.
> Тест на нормальность при `n=10000` почти всегда отвергнет H₀ из-за крошечных отклонений.
> При `n=20` — мощности не хватит ничего обнаружить.

---

## 3. Параметрические тесты средних

### One-sample t-test
`H₀`: `μ = μ₀`. Например, отличается ли средняя доходность от нуля.

```python
stat, p = stats.ttest_1samp(returns, popmean=0)
```

Статистика: `t = (x̄ − μ₀) / (s/√n)`, df = `n − 1`.

### Welch's t-test (дефолт для двух выборок)
`H₀`: `μ₁ = μ₂`. НЕ предполагает равенство дисперсий.

```python
stat, p = stats.ttest_ind(x, y, equal_var=False)  # ← Welch (default)
stat, p = stats.ttest_ind(x, y, equal_var=True)   # ← Student (старая школа)
```

> **На DSI**: всегда используй Welch и упомяни:
> «I'm using Welch's t-test as default since it doesn't assume equal variances».

### Paired t-test
`H₀`: средняя разность пар = 0. Для зависимых наблюдений (до/после, две стратегии на одних датах).

```python
stat, p = stats.ttest_rel(before, after)
# эквивалентно:
stat, p = stats.ttest_1samp(before - after, 0)
```

Парность снижает дисперсию → выше мощность, чем у independent t-test.

### One-way ANOVA
`H₀`: все групповые средние равны. Для k > 2 групп.

```python
stat, p = stats.f_oneway(group1, group2, group3)
```

- F-статистика = (between-group var) / (within-group var)
- Предположения: нормальность в каждой группе, равные дисперсии, независимость
- Если ANOVA отверг H₀ — нужны post-hoc (Tukey HSD)
- Для неравных дисперсий — Welch's ANOVA

---

## 4. Непараметрические тесты

Не предполагают нормальности. Работают с рангами вместо значений. Менее мощные при выполнении параметрических предположений, но робастные к выбросам и форме распределения.

### Mann-Whitney U
Аналог Welch's t-test. `H₀`: `P(X > Y) = P(Y > X)` (распределения одинаковы).

```python
stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
```

- Идеален для порядковых данных и распределений с выбросами
- Более мощный, чем t-test, на тяжёлых хвостах

### Wilcoxon signed-rank
Аналог paired t-test. `H₀`: распределение разностей симметрично около 0.

```python
stat, p = stats.wilcoxon(before, after)
```

### Kruskal-Wallis
Аналог one-way ANOVA для k > 2 групп.

```python
stat, p = stats.kruskal(group1, group2, group3)
```

Post-hoc: Dunn's test с поправкой Bonferroni или BH.

### Friedman
Аналог repeated-measures ANOVA. Для k > 2 зависимых групп (например, k моделей на одном тестсете).

```python
stat, p = stats.friedmanchisquare(model_a, model_b, model_c)
```

---

## 5. Тесты на равенство дисперсий

### Levene (дефолт)
`H₀`: `σ₁² = σ₂² = ...`. Робастный к нарушению нормальности.

```python
stat, p = stats.levene(x, y, center='median')  # ← самая робастная
# 'mean' — оригинальная, требует нормальности
# 'trimmed' — для тяжёлых хвостов
```

### Bartlett
Более мощный, НО требует нормальности данных. Если её нет → ложные выводы.

```python
stat, p = stats.bartlett(x, y)
```

### Fligner-Killeen
Самый робастный, на основе рангов. Для сильно ненормальных данных.

```python
stat, p = stats.fligner(x, y)
```

> Если используешь Welch — проверять равенство дисперсий не обязательно. Welch робастен к этому.

---

## 6. Тесты для категориальных данных

### Chi-square (independence)
`H₀`: две категориальные переменные независимы.

```python
import pandas as pd
from scipy.stats import chi2_contingency

ct = pd.crosstab(df['gender'], df['churned'])
chi2, p, dof, expected = chi2_contingency(ct)
```

- Предположение: ожидаемые частоты ≥ 5 в каждой ячейке
- Effect size: Cramér's V

### Fisher exact
Точный тест для таблиц 2×2 при малых выборках, когда χ² не применим.

```python
odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
```

### McNemar
Парный аналог χ². Для бинарных результатов на одних объектах: «до/после», два классификатора на одном тесте.

```python
from statsmodels.stats.contingency_tables import mcnemar
result = mcnemar([[a, b], [c, d]], exact=True)
```

---

## 7. Тесты на корреляцию

### Pearson
Линейная связь. `H₀`: `r = 0`. Предполагает: обе переменные нормальны (для p-value), линейность, отсутствие выбросов.

```python
r, p = stats.pearsonr(x, y)
```

- Чувствителен к выбросам
- Видит только линейные связи; `r=0` не значит «нет связи»

### Spearman (дефолт для финансов)
Pearson на рангах. Монотонная связь, не обязательно линейная. Робастный к выбросам.

```python
rho, p = stats.spearmanr(x, y)
```

В факторных моделях:
- **rank IC** = `Spearman(factor_t, return_{t+1})` каждый период
- **ICIR** = `mean(IC_t) / std(IC_t)` — отношение, аналогичное Sharpe

### Kendall's tau
На основе согласованности пар. Лучше Spearman на маленьких выборках.

```python
tau, p = stats.kendalltau(x, y)
```

### Тест на разность корреляций (Fisher z)

```python
import numpy as np
from scipy.stats import norm

def fisher_z_test(r1, r2, n1, n2):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p
```

---

## 8. Тесты для временных рядов

### ADF (Augmented Dickey-Fuller)
`H₀`: ряд имеет единичный корень (нестационарен). `H₁`: стационарен.

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series, autolag='AIC')
stat, p = result[0], result[1]
# p < 0.05 → отвергаем единичный корень → ряд стационарен
```

- Цены акций обычно НЕ стационарны. Лог-доходности — стационарны
- Низкая мощность на коротких рядах (n ≥ 100 для надёжности)

### KPSS
«Зеркальный» тест: `H₀`: ряд СТАЦИОНАРЕН.

```python
from statsmodels.tsa.stattools import kpss
stat, p, lags, crit = kpss(series, regression='c')
# p > 0.05 → не отвергаем стационарность
```

### Совместная интерпретация ADF + KPSS

| ADF p | KPSS p | Вывод |
|-------|--------|-------|
| < 0.05 | > 0.05 | ✓ Стационарен |
| > 0.05 | < 0.05 | ✗ Нестационарен |
| < 0.05 | < 0.05 | Конфликт — возможно тренд-стационарный |
| > 0.05 | > 0.05 | Не хватает данных |

### Ljung-Box
`H₀`: остатки модели — белый шум.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
# p > 0.05 на всех лагах → автокорреляции нет
```

- После ARIMA всегда применяй к остаткам
- Также применяй к квадратам остатков — тест на ARCH-эффекты

### Durbin-Watson
Автокорреляция 1-го порядка в остатках регрессии.

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(model.resid)
# DW ≈ 2 — нет автокорреляции
# DW < 1.5 — позитивная, DW > 2.5 — негативная
```

### Engle's ARCH-LM
Условная гетероскедастичность. На финансовых данных почти всегда положителен.

```python
from statsmodels.stats.diagnostic import het_arch
stat, p, _, _ = het_arch(residuals, nlags=12)
```

### Granger causality
«Помогает ли прошлое X предсказывать Y?» Это НЕ настоящая причинность.

```python
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(df[['Y', 'X']], maxlag=5)
```

---

## 9. Bootstrap и пермутационные тесты

### Bootstrap CI
Универсальный метод для любой статистики (Sharpe, медиана, max drawdown).

```python
import numpy as np
from scipy.stats import bootstrap

# Ручная реализация:
def bootstrap_ci(data, statistic=np.mean, B=10000, alpha=0.05):
    n = len(data)
    boots = np.array([
        statistic(np.random.choice(data, size=n, replace=True))
        for _ in range(B)
    ])
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

# scipy:
res = bootstrap((data,), np.mean, n_resamples=10000)
ci = res.confidence_interval
```

### Permutation test
Если `H₀` верна → метки случайны → перемешиваем и получаем распределение статистики при `H₀`.

```python
from scipy.stats import permutation_test

res = permutation_test(
    (x, y),
    lambda a, b: np.mean(a) - np.mean(b),
    n_resamples=10000,
    alternative='two-sided'
)
```

### Block bootstrap для time series
Обычный bootstrap уничтожает временную структуру. Решение — ресэмплируем блоки длины L.

```python
from arch.bootstrap import StationaryBootstrap

bs = StationaryBootstrap(block_size=10, returns)
results = bs.apply(lambda x: x.mean(), reps=10000)
```

---

## 10. Поправки на множественные сравнения

Тестируешь 100 факторов с α=0.05 → ожидаешь ~5 ложных позитивов даже если все бесполезны.

| Метод | Контролирует | Когда |
|-------|--------------|-------|
| Bonferroni | FWER | Консервативный, m мало |
| Holm | FWER | Всегда лучше Bonferroni |
| **Benjamini-Hochberg** | **FDR** | **Дефолт для скрининга** |
| Benjamini-Yekutieli | FDR при зависимости | Зависимые тесты |

```python
from statsmodels.stats.multitest import multipletests

rejected, p_corrected, _, _ = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)
```

- **FWER**: вероятность хотя бы одной ошибки I рода (очень консервативно)
- **FDR**: ожидаемая доля ложных позитивов среди отвергнутых (менее консервативно, больше открытий)

---

## Чек-лист

### Перед тестом
- [ ] Сформулирована `H₀` и `H₁`?
- [ ] Тип данных: числовой / порядковый / категориальный?
- [ ] Сколько групп, зависимы ли?
- [ ] Размер выборки достаточен для ЦПТ?
- [ ] Есть ли выбросы?
- [ ] Это часть скрининга → нужна поправка на множественность?

### После теста
- [ ] p-value я понимаю правильно?
- [ ] Размер эффекта посчитан?
- [ ] Доверительный интервал посчитан?
- [ ] Если не отверг — хватило ли мощности?
- [ ] Результат проверен на устойчивость?
