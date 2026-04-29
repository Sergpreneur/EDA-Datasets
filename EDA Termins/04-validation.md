# Cross-Validation

## Какой CV выбрать

| Сценарий | CV |
|----------|-----|
| Случайный iid датасет | KFold |
| Классификация | StratifiedKFold |
| Есть группы (юзер, пациент, тикер) | GroupKFold |
| Time series | TimeSeriesSplit |
| Quant с overlapping labels | Purged K-Fold + embargo |

> На DSI всегда явно объясняй выбор CV. Это маркер компетентности.

---

## 1. KFold (для iid данных)

```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
print(f'CV: {-scores.mean():.4f} ± {scores.std():.4f}')
```

---

## 2. StratifiedKFold (для классификации)

Сохраняет пропорцию классов в каждом фолде. Дефолт для дисбалансных задач.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 3. GroupKFold

Когда есть группы (пациенты, пользователи, тикеры) — одна группа должна быть либо в train, либо в val, но не в обоих.

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
for tr, val in cv.split(X, y, groups=df['user_id']):
    ...
```

---

## 4. TimeSeriesSplit

Train всегда строго до val. Без перетасовки.

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=20, test_size=252)
# gap — embargo period (анти-leak)
# test_size — размер val в каждом сплите

for train_idx, val_idx in cv.split(X):
    X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
```

---

## 5. Walk-forward validation (стандарт quant)

```python
def walk_forward_split(dates, train_size, test_size, step):
    starts = range(0, len(dates) - train_size - test_size + 1, step)
    for s in starts:
        train_idx = range(s, s + train_size)
        test_idx = range(s + train_size, s + train_size + test_size)
        yield train_idx, test_idx

# Например: train=252 дня, test=21 день, шаг=21 день
# → каждый месяц переобучаем модель на последнем годе
for tr, te in walk_forward_split(dates, 252, 21, 21):
    ...
```

---

## 6. Purged K-Fold (de Prado)

Стандарт для quant ML с overlapping labels.

Концепция:
1. **PURGE**: удалить из train все наблюдения с label_period ∩ val_period ≠ ∅
2. **EMBARGO**: удалить из train точки сразу после val (на длину embargo)

Критично, когда `y_t` зависит от данных за следующие N дней (например, 5-day forward returns).

```python
# Через mlfinlab:
from mlfinlab.cross_validation import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    samples_info_sets=t1_series,  # когда наблюдение «закончилось»
    pct_embargo=0.01
)
```

---

## 7. Подводный камень: разрыв CV vs test

Если CV-score значительно лучше test-score → переобучение на CV (выбор модели/гиперпараметров «выгорает»).

Решения:
- **Nested CV** — внешний цикл для оценки, внутренний для тюнинга
- **Holdout test set** — держать до самого конца, не трогать
- **Не подбирать гиперпараметры по test**

---

## 8. Шаблон для DSI

### Регрессия (iid)

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error')
print(f'RMSE: {-scores.mean():.4f} ± {scores.std():.4f}')
```

### Классификация (дисбаланс)

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='average_precision')
# average_precision = PR-AUC
```

### Time series

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=20)
scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error')
```

### Quant с фактором (walk-forward)

```python
ic_history = []
for tr, te in walk_forward_split(dates, 504, 21, 21):
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    ic_t = spearmanr(pred, y.iloc[te]).correlation
    ic_history.append(ic_t)

mean_ic = np.mean(ic_history)
icir = mean_ic / np.std(ic_history)
print(f'IC: {mean_ic:.4f}, ICIR: {icir:.2f}')
```
