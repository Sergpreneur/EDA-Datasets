# Как залить на GitHub

## Вариант 1: через GitHub Desktop (самый простой)

1. Открой [github.com](https://github.com), создай новую репозиторию `dsi-cheatsheet` (Public или Private — на твой выбор)
2. Скачай эту папку себе на компьютер
3. Открой GitHub Desktop → File → Add Local Repository → выбери папку
4. Commit все файлы → Push

## Вариант 2: через терминал

```bash
cd path/to/dsi-cheatsheet

git init
git add .
git commit -m "DSI cheatsheet: stats, EDA, models, quant"

# Создай репу на github.com (без README), потом:
git branch -M main
git remote add origin https://github.com/<твой_username>/dsi-cheatsheet.git
git push -u origin main
```

## Что получишь

Репозиторий будет рендериться на GitHub так:
- README.md → красивая навигационная страница
- 01-statistical-tests.md, 02-eda.md, ... → документация со всем форматированием, таблицами, code blocks
- snippets/*.py → подсветка синтаксиса Python

## Использование на DSI

1. Открой репу в браузере на их машине: `github.com/<твой_username>/dsi-cheatsheet`
2. Используй `Ctrl+F` для поиска по странице
3. Файлы `.py` в `/snippets/` → клик «Raw» → копипаст в Jupyter

## Public vs Private

- **Public**: тебя могут увидеть рекрутёры (плюс к профилю), доступен без авторизации, можно показать как доказательство «это моё»
- **Private**: только ты и приглашённые. Открывать через залогиненный аккаунт. Безопаснее, если переживаешь.

Для DSI оба варианта работают — но **Public нагляднее как портфолио** на будущее.
