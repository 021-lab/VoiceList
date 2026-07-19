# VoiceList

`VoiceList` теперь оформлен как универсальная база для форков: одностраничный HTML-интерфейс для иерархических списков задач с touch-first UX, htmlpreview-деплоем и CI-проверкой готового превью.

## Что внутри

- вложенный список задач с drag, swipe и fullscreen-редактированием;
- статусы `Open`, `Done`, `Focus`, `Archive`, `Pause`;
- отдельные вкладки списка, фронтира и журнала действий;
- единый контракт ввода через интерпретатор команд;
- self-contained `list-manager.html` для деплоя через GitHub + `htmlpreview`.

## Бэкенд: taosmd (ветка `taosmd-backend`)

Ведётся интеграция с бэкендом [taosmd](https://github.com/021-lab/TaOS/tree/feat/user-statuses) (архив-первая память + граф задач + A2A): приложение продолжает работать на localStorage, модуль синхронизации односторонне синкает журнал действий в taosmd и восстанавливает состояние на пустом localStorage. См. `docs/CODER_TASK_sync-module.md` и `docs/INTEGRATION_taosmd.md` в ветке `taosmd-backend`.

## Как использовать как базу для форка

1. Форкните репозиторий.
2. Меняйте `list-data.js` под свой seed snapshot.
3. Переопределяйте тексты, цвета и интеграции в `src/` и шаблоне `list-manager.template.html`.
4. Генерируйте deployable HTML через `npm run prepare-preview`.
5. Пушьте изменения: CI сам проверит exact `htmlpreview` URL от коммита.

## Локальная разработка

```bash
npm ci
npm run test:unit
npm run prepare-preview
npm run test:e2e
```

## Preview

Финальный preview для каждой версии строится от точного `commit SHA`, а не от имени ветки. Это уменьшает проблемы с кэшем и делает CI-результат воспроизводимым.
